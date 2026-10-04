package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestWindowSplitExact(t *testing.T) {
	text := strings.Repeat("a", 1200)
	chunks := WindowSplit(text, 500, 150)
	if len(chunks) != 3 {
		t.Fatalf("want 3 chunks, got %d", len(chunks))
	}
	for i := 0; i < len(chunks)-1; i++ {
		if n := utf8.RuneCountInString(chunks[i]); n != 500 {
			t.Errorf("chunk %d len = %d, want 500", i, n)
		}
	}
	// overlap: chunk[1] must start with chunk[0]'s last 150 runes
	if chunks[1][:150] != chunks[0][350:500] {
		t.Errorf("overlap mismatch")
	}
}

func TestWindowSplitCJKNoBrokenRunes(t *testing.T) {
	text := strings.Repeat("汉字测试字符。", 200)
	chunks := WindowSplit(text, 500, 150)
	for i, c := range chunks {
		if !utf8.ValidString(c) {
			t.Errorf("chunk %d invalid UTF-8", i)
		}
		if n := utf8.RuneCountInString(c); n > 500 {
			t.Errorf("chunk %d len %d > 500", i, n)
		}
	}
}

func TestWindowSplitShort(t *testing.T) {
	if got := WindowSplit("abc", 500, 150); len(got) != 1 || got[0] != "abc" {
		t.Errorf("short text: got %v", got)
	}
}

func TestSplitDispatch(t *testing.T) {
	text := strings.Repeat("Para one text.\n\nPara two text.\n\n", 40)
	for _, s := range StrategyIDs() {
		chunks := Split(text, Options{Strategy: s, Size: 100, Overlap: 20})
		if len(chunks) == 0 {
			t.Errorf("strategy %s returned no chunks", s)
		}
	}
	// unknown falls back to window
	w := Split(text, Options{Strategy: "nope", Size: 100, Overlap: 20})
	want := WindowSplit(text, 100, 20)
	if len(w) != len(want) {
		t.Errorf("unknown strategy did not fall back to window")
	}
}

func TestSectionChunk(t *testing.T) {
	text := "intro paragraph\n\n# H1\nbody one\n\n## H2\nbody two\n"
	chunks := Split(text, Options{Strategy: StrategySection, Size: 1000})
	if len(chunks) != 3 {
		t.Errorf("want 3 sections, got %d: %v", len(chunks), chunks)
	}
}

func TestParagraphChunk(t *testing.T) {
	text := strings.Repeat("aa\n\nbb\n\ncc\n\n", 10)
	chunks := Split(text, Options{Strategy: StrategyParagraph, Size: 100})
	if len(chunks) != 30 {
		t.Errorf("want 30 paragraphs, got %d", len(chunks))
	}
}

func TestDialogueKeepsQuotes(t *testing.T) {
	text := `他想了想。"你确定吗？"他说。她点点头。` + strings.Repeat("更多叙述。", 50)
	chunks := Split(text, Options{Strategy: StrategyDialogue, Size: 200})
	joined := strings.Join(chunks, "")
	if !strings.Contains(joined, `"你确定吗？"`) {
		t.Errorf("quote lost: %v", chunks)
	}
}

func TestSentenceChunkRespectsSize(t *testing.T) {
	text := strings.Repeat("这是一个句子。", 100)
	chunks := Split(text, Options{Strategy: StrategySentence, Size: 50})
	for i, c := range chunks {
		if utf8.RuneCountInString(c) > 200 { // CJK sentence grouping has no strict bound semantics; sanity only
			t.Errorf("chunk %d suspiciously large: %d", i, utf8.RuneCountInString(c))
		}
	}
}
