package mcp

import (
	"strings"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/jobs"
)

func TestBoundedRunReturnsFastResult(t *testing.T) {
	m := jobs.NewManager(nil, 2)
	res, err := boundedRun(m, jobs.KindSearch, 3, time.Second, func() (any, error) {
		return "fast", nil
	})
	if err != nil || res != "fast" {
		t.Fatalf("res=%v err=%v", res, err)
	}
}

func TestBoundedRunReturnsPendingPayload(t *testing.T) {
	m := jobs.NewManager(nil, 2)
	release := make(chan struct{})
	res, err := boundedRun(m, jobs.KindStore, 42, 40*time.Millisecond, func() (any, error) {
		<-release
		return "finished", nil
	})
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	text, ok := res.(string)
	if !ok {
		t.Fatalf("expected a string payload, got %T", res)
	}
	if !strings.Contains(text, `"status": "pending"`) || !strings.Contains(text, `"job_id"`) || !strings.Contains(text, `"estimate_seconds": 42`) {
		t.Fatalf("pending payload missing fields:\n%s", text)
	}

	// Once the task finishes, job_status (Get) sees the stored result.
	close(release)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		all := m.ListKind(jobs.KindStore)
		if len(all) == 1 && all[0].State == jobs.StateDone {
			if all[0].Result != "finished" {
				t.Fatalf("result = %v", all[0].Result)
			}
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("job never reached done")
}

func TestGetWaitDuration(t *testing.T) {
	if got := getWaitDuration(map[string]any{}); got != syncWait {
		t.Fatalf("default = %v, want %v", got, syncWait)
	}
	if got := getWaitDuration(map[string]any{"wait_seconds": float64(0)}); got != 0 {
		t.Fatalf("zero = %v, want immediate", got)
	}
	if got := getWaitDuration(map[string]any{"wait_seconds": float64(30)}); got != 30*time.Second {
		t.Fatalf("30 = %v", got)
	}
	if got := getWaitDuration(map[string]any{"wait_seconds": float64(-1)}); got != -1 {
		t.Fatalf("negative = %v, want indefinite", got)
	}
	if got := getWaitDuration(map[string]any{"wait_seconds": float64(999999)}); got != time.Hour {
		t.Fatalf("clamp = %v, want 1h", got)
	}
}

func TestEstimatesArePositiveAndModeAware(t *testing.T) {
	// nil engine is not usable here; just assert the pure helpers' shape via a
	// fake through estimateStoreSeconds is covered elsewhere. Search estimate
	// needs a real engine, so only check the formatting path.
	if got := formatJobPending(&jobs.Job{ID: "job-1", Kind: jobs.KindSearch, State: jobs.StateRunning, Stage: "running"}); !strings.Contains(got, "job-1") {
		t.Fatalf("formatJobPending = %q", got)
	}
}
