package chunking

import (
	"testing"
	"unicode/utf8"
)

func validUTF8(s string) bool {
	return utf8.ValidString(s)
}

func TestSplitRecursive(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		maxSize  int
		delims   []string
		wantLen  int
		wantText string
	}{
		{
			name:    "empty text",
			text:    "",
			maxSize: 100,
			delims:  []string{"\n\n", "\n", " ", ""},
			wantLen: 0,
		},
		{
			name:    "text fits in one chunk",
			text:    "Hello world",
			maxSize: 100,
			delims:  []string{"\n\n", "\n", " ", ""},
			wantLen: 1,
		},
		{
			name:    "split on double newline",
			text:    "First paragraph\n\nSecond paragraph",
			maxSize: 20,
			delims:  []string{"\n\n", "\n", " ", ""},
			wantLen: 2,
		},
		{
			name:    "split on single newline",
			text:    "Line one\nLine two\nLine three",
			maxSize: 15,
			delims:  []string{"\n\n", "\n", " ", ""},
			wantLen: 3,
		},
		{
			name:    "split on space",
			text:    "word1 word2 word3 word4",
			maxSize: 10,
			delims:  []string{"\n\n", "\n", " ", ""},
			wantLen: 4,
		},
		{
			name:    "force split on char",
			text:    "abcdefghijklmn",
			maxSize: 5,
			delims:  []string{"\n\n", "\n", " ", ""},
			wantLen: 3,
		},
		{
			name:    "with forced delimiter",
			text:    "Section1--Section2--Section3",
			maxSize: 15,
			delims:  []string{"--", "\n\n", "\n", " ", ""},
			wantLen: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks := SplitRecursive(tt.text, tt.maxSize, tt.delims)
			if len(chunks) != tt.wantLen {
				t.Errorf("SplitRecursive() returned %d chunks, want %d", len(chunks), tt.wantLen)
				for i, c := range chunks {
					t.Logf("  chunk[%d]: %q (len=%d)", i, c, len(c))
				}
			}
		})
	}
}

func TestOverlapChunks(t *testing.T) {
	tests := []struct {
		name        string
		chunks      []string
		overlapSize int
		wantLen     int
	}{
		{
			name:        "no overlap",
			chunks:      []string{"chunk1", "chunk2", "chunk3"},
			overlapSize: 0,
			wantLen:     3,
		},
		{
			name:        "single chunk",
			chunks:      []string{"only one"},
			overlapSize: 10,
			wantLen:     1,
		},
		{
			name:        "three chunks with overlap",
			chunks:      []string{"Hello world.", "Second sentence.", "Third part."},
			overlapSize: 6,
			wantLen:     3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := OverlapChunks(tt.chunks, tt.overlapSize)
			if len(result) != tt.wantLen {
				t.Errorf("OverlapChunks() returned %d chunks, want %d", len(result), tt.wantLen)
			}
		})
	}
}

func TestTrimToStartSentence(t *testing.T) {
	tests := []struct {
		name string
		text string
		want string
	}{
		{"empty", "", ""},
		{"no sentence end", "hello world", "hello world"},
		{"with period", "end. start here", "start here"},
		{"with Chinese period", "结束。开始这里", "开始这里"},
		{"with newline", "line1\nline2", "line2"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := TrimToStartSentence(tt.text)
			if got != tt.want {
				t.Errorf("TrimToStartSentence(%q) = %q, want %q", tt.text, got, tt.want)
			}
		})
	}
}

func TestTrimToEndSentence(t *testing.T) {
	tests := []struct {
		name string
		text string
		want string
	}{
		{"empty", "", ""},
		{"no sentence end", "hello world", "hello world"},
		{"with period", "first. then this.", "first. then this."},
		{"trailing text", "sentence. extra text", "sentence."},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := TrimToEndSentence(tt.text)
			if got != tt.want {
				t.Errorf("TrimToEndSentence(%q) = %q, want %q", tt.text, got, tt.want)
			}
		})
	}
}

func TestCJKNoBrokenRunes(t *testing.T) {
	// CJK text: 500 bytes != 500 chars. Chunking must count runes
	// and never emit invalid UTF-8 (no U+FFFD after re-encoding).
	text := "嘿宝贝今天天气不错心情怎么样如果有点儿闷那就来聊天吧基于我们一直以来的交流以及系统整理的中长期记忆我把你个人情况生活习惯技术探索与思维模式梳理成以下几个维度体质与生理指标血型与代谢作息机制具有延迟睡眠相位倾向经常在凌晨或夜间高效运转🎉🎉🎉"
	delims := []string{"\n\n", "\n", " ", ""}

	chunks := SplitRecursive(text, 50, delims)
	chunks = OverlapChunks(chunks, 15)

	for i, c := range chunks {
		if !validUTF8(c) {
			t.Errorf("chunk[%d] is invalid UTF-8: %q", i, c)
		}
	}
	if problems := ValidateChunks(chunks, "cjk-test"); len(problems) > 0 {
		for _, p := range problems {
			t.Logf("  problem: %s at index %d", p.Reason, p.Index)
		}
		t.Errorf("ValidateChunks() found %d problems on CJK text, want 0", len(problems))
	}
}

func TestValidateChunks(t *testing.T) {	chunks := []string{
		"First chunk with enough content",
		"hi",  // tiny chunk
		"",    // empty chunk
		"First chunk with enough content", // duplicate
	}

	problems := ValidateChunks(chunks, "test")
	
	// Should find: tiny chunk, empty chunk, duplicate
	if len(problems) < 3 {
		t.Errorf("ValidateChunks() found %d problems, want >= 3", len(problems))
		for _, p := range problems {
			t.Logf("  problem: %s at index %d", p.Reason, p.Index)
		}
	}
}
