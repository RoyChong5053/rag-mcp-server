package mcp

import (
	"strings"
	"testing"
)

func TestRejectUnknownArgs(t *testing.T) {
	if err := rejectUnknownArgs(map[string]any{"query": "x"}, "query"); err != nil {
		t.Fatalf("query alone must pass: %v", err)
	}
	err := rejectUnknownArgs(map[string]any{
		"query": "x",
		"count": float64(3),
		"store": true,
	}, "query")
	if err == nil {
		t.Fatal("old-style arguments must be rejected, not ignored")
	}
	msg := err.Error()
	for _, key := range []string{"count", "store"} {
		if !strings.Contains(msg, key) {
			t.Fatalf("error should name %q: %s", key, msg)
		}
	}
	if !strings.Contains(msg, "only query") {
		t.Fatalf("error should say query is the only argument: %s", msg)
	}
}
