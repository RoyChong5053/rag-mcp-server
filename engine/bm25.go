package engine

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// BM25 (Okapi) keyword retrieval. Vector search misses exact tokens like
// "GT20" or "2026-09-25"; BM25 catches them. CJK text has no spaces, so
// Chinese runs are tokenized as overlapping rune bigrams on top of the
// whitespace-split Latin words (same bag, better recall on CJK).

const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

// tokenize splits text into a bag of terms: lowercased Latin/digit words
// plus bigrams over every CJK run (a run of length 1 yields one unigram).
func tokenize(s string) []string {
	var out []string
	var latin strings.Builder
	var cjk []rune
	flushLatin := func() {
		if latin.Len() > 0 {
			out = append(out, latin.String())
			latin.Reset()
		}
	}
	flushCJK := func() {
		if len(cjk) == 1 {
			out = append(out, string(cjk))
		} else {
			for i := 0; i+2 <= len(cjk); i++ {
				out = append(out, string(cjk[i:i+2]))
			}
		}
		cjk = cjk[:0]
	}
	for _, r := range strings.ToLower(s) {
		switch {
		case unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r):
			flushLatin()
			cjk = append(cjk, r)
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			flushCJK()
			latin.WriteRune(r)
		default:
			flushLatin()
			flushCJK()
		}
	}
	flushLatin()
	flushCJK()
	return out
}

// bm25Doc is one searchable item in the index.
type bm25Doc struct {
	id   string // collection + source + chunk index, used to fuse ranks
	text string
	tf   map[string]int
	dl   int
}

// bm25Index is a plain in-memory inverted index over one collection.
type bm25Index struct {
	docs   []*bm25Doc
	df     map[string]int
	n      int
	avgdl  float64
}

func buildBM25(docs []*bm25Doc) *bm25Index {
	b := &bm25Index{docs: docs, df: map[string]int{}, n: len(docs)}
	total := 0
	for _, d := range docs {
		d.tf = map[string]int{}
		d.dl = 0
		for _, t := range tokenize(d.text) {
			d.tf[t]++
			d.dl++
		}
		total += d.dl
		seen := map[string]bool{}
		for t := range d.tf {
			if !seen[t] {
				b.df[t]++
				seen[t] = true
			}
		}
	}
	if b.n > 0 {
		b.avgdl = float64(total) / float64(b.n)
	}
	return b
}

// Rank returns documents sorted by BM25 score for the query (highest first).
// limit <= 0 returns all with score > 0.
func (b *bm25Index) Rank(query string, limit int) []*bm25Doc {
	type scored struct {
		d *bm25Doc
		s float64
	}
	var out []scored
	for _, d := range b.docs {
		s := b.score(query, d)
		if s > 0 {
			out = append(out, scored{d, s})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].s > out[j].s })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	res := make([]*bm25Doc, len(out))
	for i := range out {
		res[i] = out[i].d
	}
	return res
}

func (b *bm25Index) score(query string, d *bm25Doc) float64 {
	var s float64
	for _, t := range tokenize(query) {
		tf := d.tf[t]
		if tf == 0 {
			continue
		}
		n := b.df[t]
		idf := math.Log(1 + (float64(b.n)-float64(n)+0.5)/(float64(n)+0.5))
		num := float64(tf) * (bm25K1 + 1)
		den := float64(tf) + bm25K1*(1-bm25B+bm25B*float64(d.dl)/b.avgdl)
		s += idf * num / den
	}
	return s
}
