package chunking

import (
	"errors"
	"log"
	"os"
	"path/filepath"
	"sync"

	"github.com/sugarme/tokenizer/pretrained"
)

// TokenizerPath customizes which HuggingFace tokenizer.json drives TokenLen.
// Empty = default search below. Wired from config.yaml chunking.tokenizer_path.
var TokenizerPath string

var (
	tokOnce sync.Once
	encode  func(s string) int
)

var errNotFound = errors.New("tokenizer.json not found")

func candidatePaths() []string {
	paths := []string{}
	if TokenizerPath != "" {
		paths = append(paths, TokenizerPath)
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		paths = append(paths,
			filepath.Join(dir, "tokenizers", "jina-v2.tokenizer.json"),
			filepath.Join(dir, "chunking", "tokenizers", "jina-v2.tokenizer.json"),
		)
	}
	paths = append(paths,
		filepath.Join("chunking", "tokenizers", "jina-v2.tokenizer.json"),
		filepath.Join("tokenizers", "jina-v2.tokenizer.json"),
		filepath.Join("..", "chunking", "tokenizers", "jina-v2.tokenizer.json"),
	)
	return paths
}

func loadOnce() {
	for _, p := range candidatePaths() {
		tk, err := pretrained.FromFile(p)
		if err != nil {
			continue
		}
		encode = func(s string) int {
			enc, err := tk.EncodeSingle(s)
			if err != nil {
				return -1
			}
			return len(enc.Ids)
		}
		log.Printf("chunking: tokenizer loaded from %s", p)
		return
	}
	log.Printf("chunking: tokenizer.json not found; falling back to rune estimate (tried %v)", candidatePaths())
}

// TokenLen returns the number of tokens the text costs under the jina-v2
// tokenizer, or a conservative rune-based estimate (2 runes per token,
// errs on the safe side) when the model file is unavailable. The estimate
// makes chunks smaller than the real budget, never larger, so the embed
// window is not overrun when the tokenizer is missing.
func TokenLen(s string) int {
	tokOnce.Do(loadOnce)
	if encode == nil {
		return runeLen(s) / 2
	}
	if n := encode(s); n >= 0 {
		return n
	}
	return runeLen(s) / 2
}

// SplitByTokens splits s into pieces each costing at most maxTokens, cutting
// at rune boundaries. Used for force-split and overlap-window extraction.
// Pieces' token cost is verified with the same TokenLen budget, so the output
// never exceeds maxTokens per piece.
func SplitByTokens(s string, maxTokens int) []string {
	if maxTokens <= 0 || s == "" {
		return []string{s}
	}
	if TokenLen(s) <= maxTokens {
		return []string{s}
	}
	runes := []rune(s)
	var out []string
	start := 0
	for start < len(runes) {
		best := start + 1
		lo, hi := start+1, len(runes)
		for lo <= hi {
			mid := (lo + hi) / 2
			if TokenLen(string(runes[start:mid])) <= maxTokens {
				best = mid
				lo = mid + 1
			} else {
				hi = mid - 1
			}
		}
		out = append(out, string(runes[start:best]))
		start = best
	}
	return out
}

// LastTokens returns the tail of s costing at most maxTokens.
func LastTokens(s string, maxTokens int) string {
	if maxTokens <= 0 || s == "" {
		return ""
	}
	if TokenLen(s) <= maxTokens {
		return s
	}
	parts := SplitByTokens(s, maxTokens)
	return parts[len(parts)-1]
}
