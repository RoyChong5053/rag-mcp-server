package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"hash/fnv"
	"strings"
	"unicode"
)

// StringHash computes a hash for a string.
// This uses FNV-1a which is a common hash function.
// Note: Need to verify if this matches ST's calculateHash behavior.
// Kept for backward compatibility: existing Point IDs and payload.hash values
// were minted with this. New content addressing uses ContentSHA below.
func StringHash(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}

// NormalizeText canonicalizes chunk text before content hashing so trivial
// whitespace differences (trailing spaces, CRLF, double blank lines) do not
// mint distinct memory-card IDs.
func NormalizeText(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	lines := strings.Split(s, "\n")
	for i, ln := range lines {
		// Trim trailing spaces/tabs; keep leading indent (code blocks).
		lines[i] = strings.TrimRightFunc(ln, func(r rune) bool {
			return r == ' ' || r == '\t'
		})
	}
	s = strings.Join(lines, "\n")
	s = strings.TrimSpace(s)
	// Collapse 3+ consecutive newlines into 2.
	for strings.Contains(s, "\n\n\n") {
		s = strings.ReplaceAll(s, "\n\n\n", "\n\n")
	}
	return s
}

// ContentSHA returns the first 16 hex chars (64-bit) of SHA256 over the
// normalized text: the stable memory-card ID. CPU cost is ~2us per chunk,
// five orders of magnitude below one embedding call.
func ContentSHA(s string) string {
	sum := sha256.Sum256([]byte(NormalizeText(s)))
	return hex.EncodeToString(sum[:])[:16]
}

// ContentIDU64 maps the content SHA to a uint64 Point ID so the file backend
// (which stringifies uint64 IDs into index.json) needs no format migration.
// NOTE: this differs from the legacy FNV-based Point ID; both are accepted on
// read (see parsePointID + matchChunkID), but all NEW writes use this.
func ContentIDU64(collection, text string) uint64 {
	sum := sha256.Sum256([]byte(collection + "\x00" + NormalizeText(text)))
	var v uint64
	for i := 0; i < 8; i++ {
		v = v<<8 | uint64(sum[i])
	}
	if v == 0 {
		v = 1 // 0 is falsy in several JSON paths; never mint it
	}
	return v
}

// ExpandTags turns hierarchical tags like "生活/健康/作息" into their prefix
// closure: ["生活", "生活/健康", "生活/健康/作息"]. A prefix query is then a
// subtree select. Flat tags expand to themselves. Empty/whitespace entries
// are dropped. Leading '#' is stripped for storage uniformity.
func ExpandTags(tags []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, t := range tags {
		t = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(t), "#"))
		if t == "" {
			continue
		}
		parts := strings.Split(t, "/")
		var b strings.Builder
		for i, p := range parts {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			if i > 0 && b.Len() > 0 {
				b.WriteString("/")
			}
			b.WriteString(p)
			key := b.String()
			if !seen[key] {
				seen[key] = true
				out = append(out, key)
			}
		}
	}
	return out
}

// TagsText joins tags for BM25 indexing (slash/hash split into tokens by the
// existing tokenizer, so hierarchical tags contribute every level).
func TagsText(tags []string) string {
	clean := make([]string, 0, len(tags))
	for _, t := range tags {
		t = strings.TrimSpace(t)
		if t != "" {
			clean = append(clean, t)
		}
	}
	return strings.Join(clean, " ")
}

// isCJK reports whether r belongs to a CJK script block (mirror of bm25.go).
func isCJKTagRune(r rune) bool {
	return unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r)
}
