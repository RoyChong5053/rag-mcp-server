package chunking

import (
	"strings"
	"unicode"
)

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
	if len(text) <= maxChunkSize {
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
// Returns nil if any resulting chunk exceeds maxChunkSize.
func trySplit(text string, maxChunkSize int, delim string) []string {
	if delim == "" {
		return tryCharSplit(text, maxChunkSize)
	}

	parts := strings.Split(text, delim)
	if len(parts) <= 1 {
		return nil // delimiter not found
	}

	// Check if all parts fit within maxChunkSize
	for _, part := range parts {
		if len(part) > maxChunkSize {
			return nil // this delimiter produces chunks that are too large
		}
	}

	// Rejoin with delimiter to preserve original text
	var chunks []string
	for i, part := range parts {
		if i == 0 {
			chunks = append(chunks, part)
		} else {
			chunks = append(chunks, delim+part)
		}
	}

	return chunks
}

// tryCharSplit splits text at every character boundary.
func tryCharSplit(text string, maxChunkSize int) []string {
	var chunks []string
	for i := 0; i < len(text); i += maxChunkSize {
		end := i + maxChunkSize
		if end > len(text) {
			end = len(text)
		}
		chunks = append(chunks, text[i:end])
	}
	return chunks
}

// forceSplit splits text at maxChunkSize boundaries (absolute last resort).
func forceSplit(text string, maxChunkSize int) []string {
	var chunks []string
	for i := 0; i < len(text); i += maxChunkSize {
		end := i + maxChunkSize
		if end > len(text) {
			end = len(text)
		}
		chunks = append(chunks, text[i:end])
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

		// Get overlap from previous chunk's end
		if i > 0 && halfOverlap > 0 {
			prev := chunks[i-1]
			if len(prev) > halfOverlap {
				prev = prev[len(prev)-halfOverlap:]
			}
			prevOverlap = TrimToStartSentence(prev)
		}

		// Get overlap from next chunk's start
		if i < len(chunks)-1 && halfOverlap > 0 {
			next := chunks[i+1]
			if len(next) > halfOverlap {
				next = next[:halfOverlap]
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
