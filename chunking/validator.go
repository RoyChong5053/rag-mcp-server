package chunking

import (
	"fmt"
	"unicode/utf8"
)

// Problem represents a detected chunk issue
type Problem struct {
	Index  int    `json:"index"`
	Reason string `json:"reason"`
	Preview string `json:"preview,omitempty"`
}

// ValidateChunks checks chunks for common issues.
// Detection only - never mutates data.
// Matches Vector-Storage-5053's validateChunks behavior.
func ValidateChunks(chunks []string, sourceName string) []Problem {
	var problems []Problem
	seenHashes := make(map[uint32]int)

	for i, rawChunk := range chunks {
		chunk := rawChunk

		// Empty chunk
		if trimSpace(chunk) == "" {
			problems = append(problems, Problem{
				Index:  i,
				Reason: "empty chunk",
			})
			continue
		}

		// Broken emoji (lone surrogate pairs)
		if hasLoneSurrogate(chunk) {
			problems = append(problems, Problem{
				Index:   i,
				Reason:  "lone surrogate pair (broken emoji?)",
				Preview: truncate(chunk, 80),
			})
		}

		// Tiny chunk (less than 8 chars, but not the first chunk)
		if i > 0 && utf8.RuneCountInString(chunk) < 8 {
			problems = append(problems, Problem{
				Index:   i,
				Reason:  "suspiciously tiny chunk",
				Preview: truncate(chunk, 40),
			})
		}

		// Duplicate hash
		hash := StringHash(chunk)
		if prevIndex, exists := seenHashes[hash]; exists {
			problems = append(problems, Problem{
				Index:  i,
				Reason: fmt.Sprintf("duplicate hash with chunk #%d (identical content)", prevIndex),
			})
		}
		seenHashes[hash] = i
	}

	return problems
}

// hasLoneSurrogate checks for lone UTF-16 surrogate pairs.
func hasLoneSurrogate(s string) bool {
	for i := 0; i < len(s); i++ {
		r := rune(s[i])
		// Check for lone high surrogate (0xD800-0xDBFF) without low surrogate
		if r >= 0xD8 && r <= 0xDB {
			if i+1 < len(s) {
				next := rune(s[i+1])
				if next < 0xDC || next > 0xDF {
					return true
				}
				i++ // skip low surrogate
			} else {
				return true
			}
		}
	}
	return false
}

// StringHash computes a simple hash for deduplication.
// This is a placeholder - we'll need to match ST's actual hash function.
func StringHash(s string) uint32 {
	var hash uint32
	for _, r := range s {
		hash = hash*31 + uint32(r)
	}
	return hash
}

// truncate returns a truncated string with ellipsis
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// trimSpace is a simple whitespace trimmer
func trimSpace(s string) string {
	start := 0
	end := len(s)

	for start < end && isSpace(s[start]) {
		start++
	}
	for end > start && isSpace(s[end-1]) {
		end--
	}

	return s[start:end]
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}
