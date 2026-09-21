package chunking

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// runeLen returns the number of runes (characters), not bytes.
// ST's JS counts UTF-16 code units; runes are the closest correct Go equivalent
// and — critically — never split a multi-byte character in half.
func runeLen(s string) int {
	return utf8.RuneCountInString(s)
}

// sliceRunes safely slices a string by rune indices [start:end).
// Out-of-range indices are clamped; never produces invalid UTF-8.
func sliceRunes(s string, start, end int) string {
	runes := []rune(s)
	if start < 0 {
		start = 0
	}
	if end > len(runes) {
		end = len(runes)
	}
	if start >= end {
		return ""
	}
	return string(runes[start:end])
}

// SplitRecursive splits text into chunks, trying delimiters in priority order.
// Matches Vector-Storage-5053's splitRecursive behavior exactly.
//
// Delimiters are tried from largest to smallest. For each delimiter:
//   - Split text on that delimiter
//   - If any piece fits within maxChunkSize, keep the split
//   - Otherwise, try the next smaller delimiter
//
// The empty string delimiter means "split at every character" (last resort).
func SplitRecursive(text string, maxChunkSize int, delimiters []string) []string {
	if text == "" {
		return nil
	}
	if maxChunkSize <= 0 {
		return []string{text}
	}
	if runeLen(text) <= maxChunkSize {
		return []string{text}
	}

	// Try each delimiter
	for _, delim := range delimiters {
		chunks := trySplit(text, maxChunkSize, delim)
		if chunks != nil {
			return chunks
		}
	}

	// Last resort: force split at maxChunkSize
	return forceSplit(text, maxChunkSize)
}

// trySplit attempts to split text using a specific delimiter.
// Returns nil if any single piece exceeds maxChunkSize (caller tries
// a smaller delimiter). Pieces are greedily merged back up to
// maxChunkSize — this is ST splitRecursive parity: split, then pack.
// Without the merge step, inputs without big delimiters degenerate
// into one-chunk-per-word explosions.
func trySplit(text string, maxChunkSize int, delim string) []string {
	if delim == "" {
		return tryCharSplit(text, maxChunkSize)
	}

	parts := strings.Split(text, delim)
	if len(parts) <= 1 {
		return nil // delimiter not found
	}

	// Check if all parts fit within maxChunkSize (in runes, not bytes)
	for _, part := range parts {
		if runeLen(part) > maxChunkSize {
			return nil // this delimiter produces chunks that are too large
		}
	}

	// Greedily pack parts; Join restores the original text exactly.
	var chunks []string
	var cur []string
	curLen := 0
	flush := func() {
		if len(cur) > 0 {
			chunks = append(chunks, strings.Join(cur, delim))
		}
		cur = nil
		curLen = 0
	}
	for _, part := range parts {
		pl := runeLen(part)
		add := pl
		if len(cur) > 0 {
			add += runeLen(delim)
		}
		if len(cur) > 0 && curLen+add > maxChunkSize {
			flush()
		}
		cur = append(cur, part)
		if len(cur) == 1 {
			curLen = pl
		} else {
			curLen += add
		}
	}
	flush()

	return chunks
}

// tryCharSplit splits text at every character boundary (rune-safe).
func tryCharSplit(text string, maxChunkSize int) []string {
	runes := []rune(text)
	var chunks []string
	for i := 0; i < len(runes); i += maxChunkSize {
		end := i + maxChunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}
	return chunks
}

// forceSplit splits text at maxChunkSize boundaries, rune-safe
// (absolute last resort).
func forceSplit(text string, maxChunkSize int) []string {
	runes := []rune(text)
	var chunks []string
	for i := 0; i < len(runes); i += maxChunkSize {
		end := i + maxChunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}
	return chunks
}

// OverlapChunks adds sentence-boundary overlap between adjacent chunks.
// Matches Vector-Storage-5053's overlapChunks behavior exactly.
//
// For each chunk:
//   - Takes halfOverlap chars from the END of the previous chunk (trimmed to sentence start)
//   - Takes halfOverlap chars from the START of the next chunk (trimmed to sentence end)
//   - Joins: prevOverlap + " " + chunk + " " + nextOverlap
func OverlapChunks(chunks []string, overlapSize int) []string {
	if overlapSize <= 0 || len(chunks) <= 1 {
		return chunks
	}

	halfOverlap := overlapSize / 2
	result := make([]string, len(chunks))

	for i, chunk := range chunks {
		var prevOverlap, nextOverlap string

		// Get overlap from previous chunk's end (rune-safe)
		if i > 0 && halfOverlap > 0 {
			prev := chunks[i-1]
			if runeLen(prev) > halfOverlap {
				prev = sliceRunes(prev, runeLen(prev)-halfOverlap, runeLen(prev))
			}
			prevOverlap = TrimToStartSentence(prev)
		}

		// Get overlap from next chunk's start (rune-safe)
		if i < len(chunks)-1 && halfOverlap > 0 {
			next := chunks[i+1]
			if runeLen(next) > halfOverlap {
				next = sliceRunes(next, 0, halfOverlap)
			}
			nextOverlap = TrimToEndSentence(next)
		}

		// Join parts
		parts := make([]string, 0, 3)
		if prevOverlap != "" {
			parts = append(parts, prevOverlap)
		}
		parts = append(parts, chunk)
		if nextOverlap != "" {
			parts = append(parts, nextOverlap)
		}
		result[i] = strings.Join(parts, " ")
	}

	return result
}

// TrimToStartSentence trims text to start at a sentence boundary.
// Finds the first sentence-ending punctuation (。！？!?.\n) and returns from there.
// If no boundary found, returns the full text.
func TrimToStartSentence(text string) string {
	if text == "" {
		return ""
	}

	// Look for sentence-ending characters using rune iteration
	runes := []rune(text)
	for i, r := range runes {
		if isSentenceEnd(r) {
			// Return from the character AFTER the sentence end
			rest := string(runes[i+1:])
			rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
			if rest != "" {
				return rest
			}
		}
	}

	// No sentence boundary found, return full text
	return text
}

// TrimToEndSentence trims text to end at a sentence boundary.
// Finds the last sentence-ending punctuation and returns up to there.
// If no boundary found, returns the full text.
func TrimToEndSentence(text string) string {
	if text == "" {
		return ""
	}

	// Look for sentence-ending characters from the end
	runes := []rune(text)
	for i := len(runes) - 1; i >= 0; i-- {
		if isSentenceEnd(runes[i]) {
			return string(runes[:i+1])
		}
	}

	// No sentence boundary found, return full text
	return text
}

// isSentenceEnd checks if a rune is a sentence-ending punctuation.
func isSentenceEnd(r rune) bool {
	switch r {
	case '。', '！', '？', '!', '?', '.', '\n':
		return true
	}
	return false
}
