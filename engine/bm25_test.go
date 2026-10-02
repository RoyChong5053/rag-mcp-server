package engine

import (
	"reflect"
	"testing"
)

func TestTokenizeCJK(t *testing.T) {
	got := tokenize("GT20 主力手机 2026")
	want := []string{"gt20", "主力", "力手", "手机", "2026"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokenize = %v, want %v", got, want)
	}
}

func TestBM25RanksExactTokenFirst(t *testing.T) {
	docs := []*bm25Doc{
		{id: "a", text: "今天讨论手机和电脑的日常使用"},
		{id: "b", text: "gt20 是一台便携备用机"},
		{id: "c", text: "pi-mcp 连通测试使用 gt20 作节点"},
	}
	ranked := buildBM25(docs).Rank("gt20", 0)
	if len(ranked) == 0 || ranked[0].id != "b" && ranked[0].id != "c" {
		t.Fatalf("expected gt20 doc first, got %v", ranked)
	}
	if len(ranked) > 2 {
		t.Fatalf("expected only gt20 docs, got %v", ranked)
	}
}

func TestRRFFuseDedupesAndBoosts(t *testing.T) {
	vec := []SearchResult{
		{Text: "alpha", Source: "a.md", Collection: "c"},
		{Text: "beta", Source: "b.md", Collection: "c"},
	}
	bm25 := []SearchResult{
		{Text: "beta", Source: "b.md", Collection: "c"},
		{Text: "gamma", Source: "d.md", Collection: "c"},
	}
	fused := rrfFuse([][]SearchResult{vec, bm25}, 0)
	if len(fused) != 3 {
		t.Fatalf("expected 3 unique, got %d", len(fused))
	}
	// "beta" appears in both lists, must outrank singleton hits
	if fused[0].Text != "beta" {
		t.Fatalf("beta should win RRF, got %s", fused[0].Text)
	}
}
