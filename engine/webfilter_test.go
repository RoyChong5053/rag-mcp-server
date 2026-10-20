package engine

import (
	"path/filepath"
	"testing"
	"time"
)

// metaPoint builds a point whose payload carries a metadata map in the same
// shape the indexer writes it: map[string]string nested under "metadata".
func metaPoint(id uint64, src, text string, vec []float32, metadata map[string]string) Point {
	md := make(map[string]string, len(metadata))
	for k, v := range metadata {
		md[k] = v
	}
	return Point{
		ID:     id,
		Vector: vec,
		Payload: map[string]any{
			"text":        text,
			"source_file": src,
			"source":      filepath.Base(src),
			"chunk_index": 0,
			"metadata":    md,
		},
	}
}

// TestFileStoreMetadataFilterMatches is a regression test: the chunk
// metadata is stored as map[string]string, and the payload matcher used to
// only understand map[string]any. Every "metadata.<key>" clause therefore
// silently matched nothing, which meant search_memory's documented metadata
// filter filtered nothing at all.
func TestFileStoreMetadataFilterMatches(t *testing.T) {
	fs, _, _ := newTestStore(t)
	if err := fs.CreateCollection("meta_col"); err != nil {
		t.Fatal(err)
	}
	vec := []float32{1, 0, 0}
	points := []Point{
		metaPoint(1, "a.md", "web page about deepseek", vec, map[string]string{"source_type": "web", "provider": "deepseek"}),
		metaPoint(2, "b.md", "personal note about blood type", vec, map[string]string{"source_type": "agent", "provider": "store_memory"}),
	}
	if err := fs.UpsertPoints("meta_col", points); err != nil {
		t.Fatal(err)
	}

	hits, err := fs.Search("meta_col", vec, 10, 0, map[string]any{"metadata.source_type": "web"})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected exactly the 1 web chunk, got %d", len(hits))
	}
	if got := hits[0].Payload["text"]; got != "web page about deepseek" {
		t.Fatalf("wrong chunk survived the filter: %v", got)
	}

	// A value that exists nowhere must exclude everything.
	hits, err = fs.Search("meta_col", vec, 10, 0, map[string]any{"metadata.source_type": "NO_SUCH_VALUE"})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Fatalf("expected no chunks for a non-matching metadata value, got %d", len(hits))
	}

	// The "must" form used by search_memory's metadata argument.
	hits, err = fs.Search("meta_col", vec, 10, 0, map[string]any{
		"must": []any{map[string]any{
			"key":   "metadata.source_type",
			"match": map[string]any{"value": "agent"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected 1 chunk for the must-form filter, got %d", len(hits))
	}
}

// TestFileStoreKeywordSearchHonoursFilter guards the BM25 half of a
// filtered search. The keyword leg used to ignore the filter entirely, so a
// filtered search pulled the unfiltered hits straight back in.
func TestFileStoreKeywordSearchHonoursFilter(t *testing.T) {
	fs, _, _ := newTestStore(t)
	if err := fs.CreateCollection("bm25_col"); err != nil {
		t.Fatal(err)
	}
	vec := []float32{1, 0, 0}
	points := []Point{
		metaPoint(1, "a.md", "deepseek model release notes", vec, map[string]string{"source_type": "web"}),
		metaPoint(2, "b.md", "deepseek model release notes", vec, map[string]string{"source_type": "agent"}),
	}
	if err := fs.UpsertPoints("bm25_col", points); err != nil {
		t.Fatal(err)
	}

	hits, err := fs.KeywordSearch("bm25_col", "deepseek release", 10, map[string]any{"metadata.source_type": "web"})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected 1 keyword hit after filtering, got %d", len(hits))
	}
}

// TestStoreToSearchResultReadsMetadata is a regression test for the second
// half of the same bug: results were built from a map[string]any view of the
// metadata, so every search result silently lost its tags and metadata.
func TestStoreToSearchResultReadsMetadata(t *testing.T) {
	res := storeToSearchResult(StoreSearchResult{
		ID: "42",
		Payload: map[string]any{
			"text": "hello",
			"metadata": map[string]string{
				"tags":        "web,news",
				"source_type": "web",
				"url":         "https://example.com/a",
			},
		},
	}, "col")

	if len(res.Tags) != 2 {
		t.Fatalf("expected 2 tags, got %v", res.Tags)
	}
	if res.Metadata["source_type"] != "web" {
		t.Fatalf("metadata not carried through: %v", res.Metadata)
	}
	if res.Metadata["url"] != "https://example.com/a" {
		t.Fatalf("url metadata not carried through: %v", res.Metadata)
	}
}

// TestWebDocumentAge covers the age rendering used next to a recalled chunk.
func TestWebDocumentAge(t *testing.T) {
	now := time.Date(2026, 10, 21, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		in   string
		want string
	}{
		{"", "unknown"},
		{"not a date", "unknown"},
		{"2026-10-21T10:00:00Z", "2 hours ago"},
		{"2026-10-20T12:00:00Z", "1 days ago"},
		{"2026-08-21T12:00:00Z", "2 months ago"},
	}
	for _, tc := range cases {
		if got := WebDocumentAge(tc.in, now); got != tc.want {
			t.Errorf("WebDocumentAge(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestWebDocumentFresh covers the TTL helper.
func TestWebDocumentFresh(t *testing.T) {
	if WebDocumentFresh("", 7) {
		t.Fatal("an unknown timestamp must count as stale")
	}
	if !WebDocumentFresh("2026-10-21T00:00:00Z", 0) {
		t.Fatal("a zero-day window must accept everything")
	}
}
