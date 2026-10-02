package engine

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestRerankRetryableTimeoutIsNotRetryable(t *testing.T) {
	timeoutErr := fmt.Errorf("rerank request failed: Post \"http://x/v1/rerank\": context deadline exceeded (Client.Timeout exceeded while awaiting headers)")
	if rerankRetryable(timeoutErr) {
		t.Fatal("a client timeout must not be retried (it doubles the stall)")
	}
	if !rerankRetryable(fmt.Errorf("rerank API error 503: busy")) {
		t.Fatal("503 must stay retryable")
	}
	if !rerankRetryable(fmt.Errorf("rerank request failed: connection refused")) {
		t.Fatal("transport failure must stay retryable")
	}
}

// After one failed rerank the breaker opens: the next Rerank call must return
// immediately without touching the upstream.
func TestRerankCircuitBreakerSkipsUpstream(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	c := NewRerankClient(srv.URL, "", "reranker", "", 500, 500)

	if _, err := c.Rerank(context.Background(), "q", []string{"a", "b"}, 2, time.Second); err == nil {
		t.Fatal("expected error from failing reranker")
	}
	// callWithRetry makes 2 attempts on a 500 before failing.
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("failed rerank attempts = %d, want 2", got)
	}
	if !c.isDown() {
		t.Fatal("breaker should be open after a failure")
	}

	// Second call is short-circuited: no new upstream request.
	_, err := c.Rerank(context.Background(), "q", []string{"a", "b"}, 2, time.Second)
	if err == nil || !strings.Contains(err.Error(), "temporarily disabled") {
		t.Fatalf("expected disabled-by-breaker error, got %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("upstream touched while breaker open: calls = %d, want 2", got)
	}

	c.markUp()
	if c.isDown() {
		t.Fatal("breaker should close after markUp")
	}
}

// A caller budget that elapses mid-request fails open WITHOUT latching the
// breaker. The upstream stays reachable afterwards — this is exactly the bug
// the wait-budget change fixes (a transient stall used to pin uptime ~70%).
func TestRerankFailOpenOnTimeoutDoesNotLatch(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) <= 1 {
			time.Sleep(300 * time.Millisecond) // slow first hit → deadline wins
		}
		fmt.Fprint(w, `{"results":[{"index":0,"relevance_score":0.9}]}`)
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	c := NewRerankClient(srv.URL, "", "reranker", "", 500, 500)

	if _, err := c.Rerank(ctx, "q", []string{"a", "b"}, 2, time.Second); err == nil {
		t.Fatal("expected a deadline-exceeded error on the slow first call")
	}
	if c.isDown() {
		t.Fatal("fail-open on timeout must NOT latch the breaker")
	}

	// After fail-open the upstream is still reachable: a generous budget gets
	// results back, proving no latch was applied.
	ctx2, cancel2 := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel2()
	results, err := c.Rerank(ctx2, "q", []string{"a", "b"}, 2, time.Second)
	if err != nil || results == nil {
		t.Fatalf("expected results on follow-up call after fail-open, got %v/%v", results, err)
	}
	if len(results) != 1 {
		t.Fatalf("follow-up results = %d, want 1", len(results))
	}
}

// Fatal misconfig (bad key / unparseable body) latches the breaker and is NOT
// cleared by later attempts: hammering a dead config must not keep retrying.
func TestRerankFatalErrorsLatchBreaker(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		http.Error(w, `{"error":"invalid api key"}`, http.StatusUnauthorized)
	}))
	defer srv.Close()

	c := NewRerankClient(srv.URL, "", "reranker", "", 500, 500)

	if _, err := c.Rerank(context.Background(), "q", []string{"a", "b"}, 2, time.Second); err == nil {
		t.Fatal("expected error from fatal 401")
	}
	if !c.isDown() {
		t.Fatal("fatal auth error must open the breaker")
	}

	// The breaker stays open: a subsequent call short-circuits with no new hit.
	before := atomic.LoadInt32(&calls)
	if _, err := c.Rerank(context.Background(), "q", []string{"a", "b"}, 2, time.Second); err == nil || !strings.Contains(err.Error(), "temporarily disabled") {
		t.Fatalf("expected temporarily disabled while breaker open, got %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != before {
		t.Fatalf("breaker did not short-circuit: calls went %d -> %d, want unchanged", before, got)
	}

	c.markUp()
	if c.isDown() {
		t.Fatal("breaker should close after markUp")
	}
}
