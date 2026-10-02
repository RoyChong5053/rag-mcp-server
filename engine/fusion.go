package engine

import (
	"sort"
	"strconv"
)

// RRF (Reciprocal Rank Fusion) merges ranked lists without caring about
// absolute score scales: each item votes 1/(k+rank) in every list it appears
// in. k=60 is the Hindsight/BM25 literature default. Items are matched
// across lists by their stable result key.
const rrfK = 60

func resultKey(r SearchResult) string {
	if r.Source != "" || r.ChunkIndex != 0 {
		return r.Collection + "|" + r.Source + "#" + strconv.Itoa(r.ChunkIndex)
	}
	n := len(r.Text)
	if n > 80 {
		n = 80
	}
	return r.Collection + "|" + r.Text[:n]
}

// rrfFuse merges ranked result lists, replaces Score with the fused RRF
// score, and returns at most limit items (limit <= 0 = all).
func rrfFuse(lists [][]SearchResult, limit int) []SearchResult {
	type acc struct {
		r     SearchResult
		score float64
	}
	byKey := map[string]*acc{}
	for _, list := range lists {
		for rank, r := range list {
			k := resultKey(r)
			a, ok := byKey[k]
			if !ok {
				a = &acc{r: r}
				byKey[k] = a
			}
			a.score += 1.0 / float64(rrfK+rank+1)
		}
	}
	out := make([]SearchResult, 0, len(byKey))
	for _, a := range byKey {
		a.r.Score = a.score
		out = append(out, a.r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}
