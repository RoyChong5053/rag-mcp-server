package mcp

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// A capped audit argument must still report its real length: a log line
// showing 120 runes of a 500-rune query reads like query_max_chars is broken.
func TestSummarizeArgsReportsFullLengthWhenCapped(t *testing.T) {
	q := strings.Repeat("你", argPreviewRunes+200)
	out := summarizeArgs(map[string]any{"query": q})

	if !strings.Contains(out, strings.Repeat("你", argPreviewRunes)) {
		t.Fatalf("expected the first %d runes in the preview", argPreviewRunes)
	}
	if !strings.Contains(out, "[700 runes total]") {
		t.Fatalf("capped value must report its real rune count: %s", out)
	}
	if !utf8.ValidString(out) {
		t.Fatal("audit line must stay valid UTF-8")
	}
	if strings.Contains(out, "\n") {
		t.Fatal("audit line must stay single-line")
	}
}
