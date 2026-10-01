package engine

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"strings"
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

	if _, err := c.Rerank("q", []string{"a", "b"}, 2); err == nil {
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
	_, err := c.Rerank("q", []string{"a", "b"}, 2)
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
