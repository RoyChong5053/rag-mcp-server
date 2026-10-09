package engine

import (
	"strings"
	"testing"
)

func TestContentSHAStable(t *testing.T) {
	a := ContentSHA("hello world")
	b := ContentSHA("  hello world  \n")
	if a != b {
		t.Fatalf("normalized SHA mismatch: %s vs %s", a, b)
	}
	if len(a) != 16 {
		t.Fatalf("SHA length = %d, want 16", len(a))
	}
	if ContentSHA("hello") == ContentSHA("world") {
		t.Fatal("distinct texts share SHA")
	}
}

func TestExpandTagsHierarchy(t *testing.T) {
	got := ExpandTags([]string{"#生活/健康/作息"})
	want := []string{"生活", "生活/健康", "生活/健康/作息"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestIDSetAndMatcherPrefix(t *testing.T) {
	set := idSetFromFilter(map[string]any{"ids": []any{"abc", "def"}})
	if !set["abc"] || !set["def"] {
		t.Fatalf("id set wrong: %v", set)
	}
	if !matchChunkID("abc", map[string]any{"text": "x"}, set) {
		t.Fatal("direct ID should match")
	}
	if !matchChunkID("999", map[string]any{"content_sha": "abc"}, set) {
		t.Fatal("content_sha should match")
	}
	m, err := newPayloadMatcher(map[string]any{
		"must": []any{map[string]any{
			"key":   "metadata.tags_expanded",
			"match": map[string]any{"prefix": "生活/健康"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !m(map[string]any{"metadata": map[string]any{"tags_expanded": "生活,生活/健康,生活/健康/作息"}}) {
		t.Fatal("prefix should match subtree")
	}
	if m(map[string]any{"metadata": map[string]any{"tags_expanded": "人物/偏好"}}) {
		t.Fatal("prefix must not match other tree")
	}
	_ = strings.TrimSpace
}

func TestSplitFilterANDSemantics(t *testing.T) {
	// ids-only must NOT match everything (regression: empty payload matcher
	// is match-all, so ids must gate first).
	idSet, mf, hasOther := splitFilter(map[string]any{"ids": []any{"abc"}})
	if hasOther {
		t.Fatal("ids-only must have no other clauses")
	}
	m, err := newPayloadMatcher(mf)
	if err != nil {
		t.Fatal(err)
	}
	if !matchItem("abc", nil, map[string]any{}, idSet, hasOther, m) {
		t.Fatal("listed id must match")
	}
	if matchItem("zzz", map[string]any{"text": "hi"}, map[string]any{"text": "hi"}, idSet, hasOther, m) {
		t.Fatal("unlisted id must NOT match (would wipe collection)")
	}
	// Empty filter = match all (whole-collection op preserved).
	idSet, mf, hasOther = splitFilter(map[string]any{})
	m, _ = newPayloadMatcher(mf)
	_ = idSet
	if !matchItem("anything", nil, map[string]any{}, nil, hasOther, m) {
		t.Fatal("empty filter must match all")
	}
}
