package mcp

import (
	"strings"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/search"
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

// buildWebBundle must keep every byte the provider returned (no 400-char or
// length cut) and put the summaries before the raw results.
func TestBuildWebBundleKeepsFullText(t *testing.T) {
	longSummary := strings.Repeat("summary-sentence ", 120)  // ~1900 chars, well past 400
	longContent := strings.Repeat("full-article-body ", 500) // ~8.5k chars
	results := []search.Result{
		{
			Title:    "First hit",
			URL:      "https://example.com/a",
			Summary:  longSummary,
			Content:  longContent,
			Provider: "deepseek",
			Score:    1.5,
		},
		{
			Title:    "Second hit",
			URL:      "https://example.com/b",
			Snippet:  "second snippet",
			Provider: "deepseek",
		},
	}
	now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	title, url, body := buildWebBundle("malaysia news", results, now)

	if title != "web_search: malaysia news" {
		t.Fatalf("title = %q", title)
	}
	if !strings.Contains(url, "20261026T120000") || !strings.Contains(url, "malaysia news") {
		t.Fatalf("bundle url should carry the timestamp and query: %q", url)
	}
	// No truncation anywhere: the full summary and full body must survive.
	if !strings.Contains(body, strings.TrimSpace(longSummary)) {
		t.Fatal("summary was truncated")
	}
	if !strings.Contains(body, strings.TrimSpace(longContent)) {
		t.Fatal("content was truncated")
	}
	// Summaries first, results afterwards.
	si := strings.Index(body, "## Summary")
	ri := strings.Index(body, "## Results")
	if si < 0 || ri < 0 || si > ri {
		t.Fatalf("summary section must come before results (summary=%d results=%d)", si, ri)
	}
	// Both hits and their URLs are present.
	for _, want := range []string{"First hit", "https://example.com/a", "Second hit", "second snippet", "https://example.com/b"} {
		if !strings.Contains(body, want) {
			t.Fatalf("bundle missing %q", want)
		}
	}
}
