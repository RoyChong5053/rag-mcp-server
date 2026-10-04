package chunking

import (
	"regexp"
	"strings"
	"unicode/utf8"
)

// Options controls how Split cuts text into chunks.
type Options struct {
	Strategy string // window|adaptive|paragraph|section|sentence|dialogue
	Size     int    // target chunk length in runes (text strategies)
	Overlap  int    // shared runes between adjacent chunks; only "window" uses it
}

func runeLen(s string) int {
	return utf8.RuneCountInString(s)
}

const (
	StrategyWindow    = "window"
	StrategyAdaptive  = "adaptive"
	StrategyParagraph = "paragraph"
	StrategySection   = "section"
	StrategySentence  = "sentence"
	StrategyDialogue  = "dialogue"
)

// StrategyIDs lists valid selections for config validation and WebUI.
func StrategyIDs() []string {
	return []string{StrategyWindow, StrategyAdaptive, StrategyParagraph, StrategySection, StrategySentence, StrategyDialogue}
}

func IsValidStrategy(s string) bool {
	for _, id := range StrategyIDs() {
		if s == id {
			return true
		}
	}
	return false
}

// Split dispatches to the chosen strategy. Unknown or empty strategies fall
// back to "window" — the deterministic default.
func Split(text string, opts Options) []string {
	if text == "" {
		return nil
	}
	if opts.Size <= 0 {
		opts.Size = 500
	}
	if opts.Overlap < 0 {
		opts.Overlap = 0
	}
	if opts.Overlap >= opts.Size {
		opts.Overlap = opts.Size / 4
	}
	switch opts.Strategy {
	case StrategyAdaptive:
		return adaptiveChunk(text, opts.Size)
	case StrategyParagraph:
		return paragraphChunk(text, opts.Size)
	case StrategySection:
		return sectionChunk(text, opts.Size)
	case StrategySentence:
		return sentenceChunk(text, opts.Size)
	case StrategyDialogue:
		return dialogueChunk(text, opts.Size)
	default:
		return WindowSplit(text, opts.Size, opts.Overlap)
	}
}

// WindowSplit is the deterministic sliding-window splitter: every chunk is
// exactly `size` runes except possibly the last, consecutive chunks share
// `overlap` runes verbatim. Rune-boundary safe (never splits a multi-byte
// character). Chunking is fully determined by (size, overlap) — no boundary
// heuristics, no merging, offsets are predictable.
func WindowSplit(text string, size, overlap int) []string {
	if size <= 0 {
		return []string{text}
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= size {
		overlap = size / 4
	}
	runes := []rune(text)
	if len(runes) <= size {
		return []string{text}
	}
	step := size - overlap
	var out []string
	for start := 0; start < len(runes); start += step {
		end := start + size
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[start:end]))
		if end >= len(runes) {
			break
		}
	}
	return out
}

// adaptiveChunk splits at natural boundaries (paragraphs, then sentences,
// then words) and packs pieces up to maxSize. Ported from VectHare's
// adaptive strategy, rune-safe.
func adaptiveChunk(text string, maxSize int) []string {
	chunks := []string{}
	paragraphs := strings.Split(text, "\n\n")
	current := ""
	for _, para := range paragraphs {
		trimmed := strings.TrimSpace(para)
		if trimmed == "" {
			continue
		}
		if runeLen(current)+runeLen(trimmed)+2 > maxSize {
			if current != "" {
				chunks = append(chunks, strings.TrimSpace(current))
				current = ""
			}
			if runeLen(trimmed) > maxSize {
				chunks = append(chunks, splitLargeParagraph(trimmed, maxSize)...)
			} else {
				current = trimmed
			}
		} else {
			if current != "" {
				current += "\n\n" + trimmed
			} else {
				current = trimmed
			}
		}
	}
	if current != "" {
		chunks = append(chunks, strings.TrimSpace(current))
	}
	if len(chunks) == 0 {
		return []string{text}
	}
	return chunks
}

// splitLargeParagraph breaks an oversized paragraph at sentence boundaries,
// then at word boundaries, then hard-splits.
func splitLargeParagraph(text string, maxSize int) []string {
	sentences := splitSentences(text)
	var chunks []string
	current := ""
	flush := func() {
		if current != "" {
			chunks = append(chunks, current)
			current = ""
		}
	}
	for _, s := range sentences {
		if runeLen(current)+runeLen(s)+1 <= maxSize {
			if current != "" {
				current += " " + s
			} else {
				current = s
			}
			continue
		}
		flush()
		if runeLen(s) > maxSize {
			// Word-level fallback, then hard split.
			words := strings.Fields(s)
			wordChunk := ""
			for _, w := range words {
				if runeLen(wordChunk)+runeLen(w)+1 <= maxSize {
					if wordChunk != "" {
						wordChunk += " " + w
					} else {
						wordChunk = w
					}
				} else {
					if wordChunk != "" {
						chunks = append(chunks, wordChunk)
					}
					if runeLen(w) > maxSize {
						chunks = append(chunks, WindowSplit(w, maxSize, 0)...)
						wordChunk = ""
					} else {
						wordChunk = w
					}
				}
			}
			if wordChunk != "" {
				current = wordChunk
			}
		} else {
			current = s
		}
	}
	flush()
	return chunks
}

var sentenceEnd = map[rune]bool{'。': true, '！': true, '？': true, '.': true, '!': true, '?': true, '\n': true}

// splitSentences breaks after CJK/ASCII sentence-ending punctuation or
// newlines. Go's regexp has no lookahead, so this is manual.
func splitSentences(s string) []string {
	var out []string
	start := 0
	runes := []rune(s)
	for i, r := range runes {
		if sentenceEnd[r] && (r != '\n') {
			// Don't split between end punctuation and a closing quote/bracket —
			// `"你确定吗？"` must stay in one sentence.
			if i+1 < len(runes) && strings.ContainsRune(`"'」』）)】》〕»`, runes[i+1]) {
				continue
			}
			piece := strings.TrimSpace(string(runes[start : i+1]))
			if piece != "" {
				out = append(out, piece)
			}
			start = i + 1
		}
	}
	if rest := strings.TrimSpace(string(runes[start:])); rest != "" {
		out = append(out, rest)
	}
	return out
}

var hrRegex = regexp.MustCompile(`(?m)^---+$`)

// paragraphChunk splits on blank lines (and --- rules); oversized paragraphs
// are chopped with adaptiveChunk to respect maxSize.
func paragraphChunk(text string, maxSize int) []string {
	normalized := hrRegex.ReplaceAllString(text, "\n\n")
	parts := strings.Split(normalized, "\n\n")
	var out []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if runeLen(p) > maxSize {
			out = append(out, adaptiveChunk(p, maxSize)...)
		} else {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return []string{text}
	}
	return out
}

var headerRegex = regexp.MustCompile(`(?m)^#{1,6}\s+`)

// sectionChunk splits on markdown headers (keeping headers with their body).
// Falls back to paragraphChunk when no headers exist.
func sectionChunk(text string, maxSize int) []string {
	locs := headerRegex.FindAllStringIndex(text, -1)
	if len(locs) == 0 {
		return paragraphChunk(text, maxSize)
	}
	var out []string
	start := 0
	for i, loc := range locs {
		if i > 0 {
			end := loc[0]
			if section := strings.TrimSpace(text[start:end]); section != "" {
				out = append(out, section)
			}
			start = loc[0]
		} else if loc[0] > 0 {
			if pre := strings.TrimSpace(text[:loc[0]]); pre != "" {
				out = append(out, pre)
			}
			start = loc[0]
		}
	}
	if tail := strings.TrimSpace(text[start:]); tail != "" {
		out = append(out, tail)
	}
	var final []string
	for _, s := range out {
		if runeLen(s) > maxSize {
			final = append(final, adaptiveChunk(s, maxSize)...)
		} else {
			final = append(final, s)
		}
	}
	return final
}

// sentenceChunk groups sentences into chunks up to maxSize runes.
func sentenceChunk(text string, maxSize int) []string {
	sentences := splitSentences(text)
	var out []string
	current := ""
	for _, s := range sentences {
		if runeLen(current)+runeLen(s)+1 <= maxSize {
			if current != "" {
				current += " " + s
			} else {
				current = s
			}
		} else {
			if current != "" {
				out = append(out, current)
			}
			if runeLen(s) > maxSize {
				out = append(out, splitLargeParagraph(s, maxSize)...)
				current = ""
			} else {
				current = s
			}
		}
	}
	if current != "" {
		out = append(out, current)
	}
	if len(out) == 0 {
		return []string{text}
	}
	return out
}

var dialogueRegex = regexp.MustCompile(`"[^"]+"|「[^」]+」|『[^』]+』`)

// dialogueChunk keeps quoted speech intact: dialogue runs are never split
// from their surrounding narration when a chunk fits; oversized pieces are
// chopped with adaptiveChunk.
func dialogueChunk(text string, maxSize int) []string {
	var chunks []string
	current := ""
	last := 0
	for _, loc := range dialogueRegex.FindAllStringIndex(text, -1) {
		before := text[last:loc[0]]
		current += before
		dialogue := text[loc[0]:loc[1]]
		if runeLen(current)+utf8.RuneCountInString(dialogue) <= maxSize {
			current += dialogue
		} else {
			if strings.TrimSpace(current) != "" {
				chunks = append(chunks, strings.TrimSpace(current))
			}
			current = dialogue
		}
		last = loc[1]
	}
	current += text[last:]
	if strings.TrimSpace(current) != "" {
		chunks = append(chunks, strings.TrimSpace(current))
	}
	var final []string
	for _, c := range chunks {
		if runeLen(c) <= maxSize {
			final = append(final, c)
		} else {
			final = append(final, adaptiveChunk(c, maxSize)...)
		}
	}
	if len(final) == 0 {
		return []string{text}
	}
	return final
}
