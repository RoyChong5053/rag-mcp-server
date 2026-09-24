package engine

import (
	"os"
	"path/filepath"
	"testing"
)

func newTestStore(t *testing.T) (*FileStore, string, string) {
	t.Helper()
	root := t.TempDir()
	docs := filepath.Join(root, "docs")
	if err := os.MkdirAll(filepath.Join(docs, "chat_history"), 0o755); err != nil {
		t.Fatal(err)
	}
	vectra := filepath.Join(root, "Vectra")
	return NewFileStore(vectra, docs), docs, vectra
}

func pt(id uint64, src string, vec []float32, text string) Point {
	return Point{
		ID:      id,
		Vector:  vec,
		Payload: map[string]any{"text": text, "source_file": src, "source": filepath.Base(src), "chunk_index": 0},
	}
}

func TestFileStoreCollectionLifecycle(t *testing.T) {
	fs, _, _ := newTestStore(t)

	if ok, _ := fs.CollectionExists("col_a"); ok {
		t.Fatal("collection should not exist yet")
	}
	if err := fs.CreateCollection("col_a"); err != nil {
		t.Fatal(err)
	}
	if ok, _ := fs.CollectionExists("col_a"); !ok {
		t.Fatal("collection should exist after create")
	}
	// Idempotent create
	if err := fs.CreateCollection("col_a"); err != nil {
		t.Fatalf("second create should be a no-op: %v", err)
	}

	info, err := fs.GetCollectionInfo("col_a")
	if err != nil || info.ChunkCount != 0 {
		t.Fatalf("fresh collection should be empty: %+v err=%v", info, err)
	}

	cols, err := fs.ListCollections()
	if err != nil || len(cols) != 1 || cols[0].Name != "col_a" {
		t.Fatalf("unexpected collections: %+v err=%v", cols, err)
	}

	if err := fs.DeleteCollection("col_a"); err != nil {
		t.Fatal(err)
	}
	if ok, _ := fs.CollectionExists("col_a"); ok {
		t.Fatal("collection should be gone after delete")
	}
}

func TestFileStoreUpsertSearchAndReload(t *testing.T) {
	fs, docs, vectra := newTestStore(t)
	if err := fs.CreateCollection("col"); err != nil {
		t.Fatal(err)
	}

	a := filepath.Join(docs, "chat_history", "A.md")
	b := filepath.Join(docs, "chat_history", "B.md")
	if err := fs.UpsertPoints("col", []Point{
		pt(1, a, []float32{1, 0, 0}, "alpha"),
		pt(2, b, []float32{0, 1, 0}, "beta"),
	}); err != nil {
		t.Fatal(err)
	}

	// index.json must exist for each source, in Vectra dir only
	if _, err := os.Stat(filepath.Join(vectra, "col", "chat_history", "A.md", "index.json")); err != nil {
		t.Fatalf("expected per-source index.json: %v", err)
	}

	// Reload from disk with a fresh instance: data must survive.
	fs2 := NewFileStore(vectra, docs)
	res, err := fs2.Search("col", []float32{1, 0, 0}, 10, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || res[0].Payload["text"] != "alpha" {
		t.Fatalf("unexpected search results: %+v", res)
	}
	if res[0].Score < 0.99 || res[0].Score > 1.001 {
		t.Fatalf("expected cosine ~1, got %v", res[0].Score)
	}

	// Threshold filters the orthogonal result.
	res, err = fs2.Search("col", []float32{1, 0, 0}, 10, 0.5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].Payload["text"] != "alpha" {
		t.Fatalf("threshold should drop beta: %+v", res)
	}

	info, _ := fs2.GetCollectionInfo("col")
	if info.ChunkCount != 2 {
		t.Fatalf("chunk count = %d, want 2", info.ChunkCount)
	}
}

func TestFileStoreUpsertIdempotent(t *testing.T) {
	fs, docs, _ := newTestStore(t)
	if err := fs.CreateCollection("col"); err != nil {
		t.Fatal(err)
	}
	a := filepath.Join(docs, "a.md")
	if err := fs.UpsertPoints("col", []Point{pt(7, a, []float32{1, 0}, "v1")}); err != nil {
		t.Fatal(err)
	}
	if err := fs.UpsertPoints("col", []Point{pt(7, a, []float32{1, 0}, "v2")}); err != nil {
		t.Fatal(err)
	}
	info, _ := fs.GetCollectionInfo("col")
	if info.ChunkCount != 1 {
		t.Fatalf("same ID must replace, count=%d want 1", info.ChunkCount)
	}
	res, _ := fs.Search("col", []float32{1, 0}, 5, 0, nil)
	if len(res) != 1 || res[0].Payload["text"] != "v2" {
		t.Fatalf("expected updated text: %+v", res)
	}
}

func TestFileStoreDeleteAndSetPayload(t *testing.T) {
	fs, docs, _ := newTestStore(t)
	_ = fs.CreateCollection("col")
	a := filepath.Join(docs, "a.md")
	b := filepath.Join(docs, "b.md")
	if err := fs.UpsertPoints("col", []Point{
		pt(1, a, []float32{1, 0}, "alpha"),
		pt(2, b, []float32{0, 1}, "beta"),
	}); err != nil {
		t.Fatal(err)
	}

	// Backfill a field on every item (nil filter = all).
	if err := fs.SetPayload("col", map[string]any{"indexed_at": "now"}, nil); err != nil {
		t.Fatal(err)
	}
	res, _ := fs.Search("col", []float32{1, 0}, 5, 0, nil)
	if res[0].Payload["indexed_at"] != "now" {
		t.Fatalf("backfill missing: %+v", res[0].Payload)
	}

	// Delete by match.value on source_file.
	filter := map[string]any{
		"must": []any{
			map[string]any{"key": "source_file", "match": map[string]any{"value": a}},
		},
	}
	if err := fs.DeletePoints("col", filter); err != nil {
		t.Fatal(err)
	}
	info, _ := fs.GetCollectionInfo("col")
	if info.ChunkCount != 1 {
		t.Fatalf("after delete count=%d want 1", info.ChunkCount)
	}
	res, _ = fs.Search("col", []float32{1, 0}, 5, 0, nil)
	for _, r := range res {
		if r.Payload["text"] == "alpha" {
			t.Fatalf("alpha should have been deleted")
		}
	}
}

func TestFileStoreSourceKey(t *testing.T) {
	fs, docs, _ := newTestStore(t)
	cases := []struct {
		src  string
		want string
	}{
		{filepath.Join(docs, "chat_history", "Leer.md"), "chat_history/Leer.md"},
		{filepath.Join(docs, "memory", "2026-09-23", "120000-abcd.md"), "memory/2026-09-23/120000-abcd.md"},
		{"direct_text", "direct_text"},
		{"", "direct_text"},
	}
	for _, c := range cases {
		if got := fs.sourceKey(c.src); got != c.want {
			t.Errorf("sourceKey(%q) = %q, want %q", c.src, got, c.want)
		}
	}
	// Unsafe names get sanitized and a hash disambiguator.
	weird := filepath.Join(docs, "a b", "c:d.md")
	key := fs.sourceKey(weird)
	if key == "" || filepath.IsAbs(key) {
		t.Fatalf("unexpected key for %q: %q", weird, key)
	}
	if key != "a_b/c_d.md__"+shortHash("a b/c:d.md") {
		t.Fatalf("unexpected sanitized key: %q", key)
	}
}

func TestFileStorePingAndDeleteAll(t *testing.T) {
	fs, docs, _ := newTestStore(t)
	if err := fs.Ping(); err != nil {
		t.Fatalf("ping should succeed once root is writable: %v", err)
	}
	_ = fs.CreateCollection("col")
	_ = fs.UpsertPoints("col", []Point{
		pt(1, filepath.Join(docs, "a.md"), []float32{1, 0}, "alpha"),
		pt(2, filepath.Join(docs, "b.md"), []float32{1, 0}, "beta"),
	})
	// Empty filter clears everything.
	if err := fs.DeletePoints("col", nil); err != nil {
		t.Fatal(err)
	}
	info, _ := fs.GetCollectionInfo("col")
	if info.ChunkCount != 0 {
		t.Fatalf("empty filter should clear all, count=%d", info.ChunkCount)
	}
}

func TestFileStoreIndexCacheServesRepeatedSearch(t *testing.T) {
	fs, docs, vectra := newTestStore(t)
	_ = fs.CreateCollection("col")
	_ = fs.UpsertPoints("col", []Point{pt(1, filepath.Join(docs, "a.md"), []float32{1, 0}, "alpha")})
	if _, err := fs.Search("col", []float32{1, 0}, 5, 0, nil); err != nil {
		t.Fatal(err)
	}
	var indexPath string
	_ = filepath.Walk(filepath.Join(vectra, "col"), func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && info.Name() == "index.json" {
			indexPath = path
		}
		return nil
	})
	if indexPath == "" {
		t.Fatal("index.json not found")
	}
	if err := os.Remove(indexPath); err != nil {
		t.Fatal(err)
	}
	res, err := fs.Search("col", []float32{1, 0}, 5, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].Payload["text"] != "alpha" {
		t.Fatalf("cache should serve search after file removal: %+v", res)
	}
}

func TestFileStorePrewarm(t *testing.T) {
	fs, docs, _ := newTestStore(t)
	_ = fs.CreateCollection("col")
	_ = fs.UpsertPoints("col", []Point{pt(1, filepath.Join(docs, "a.md"), []float32{1, 0}, "alpha")})
	if err := fs.Prewarm("col"); err != nil {
		t.Fatal(err)
	}
	if err := fs.PrewarmAll(); err != nil {
		t.Fatal(err)
	}
	res, err := fs.Search("col", []float32{1, 0}, 5, 0, nil)
	if err != nil || len(res) != 1 {
		t.Fatalf("search after prewarm: %v %+v", err, res)
	}
}

func TestFileStoreDeleteCollectionDropsCache(t *testing.T) {
	fs, docs, _ := newTestStore(t)
	_ = fs.CreateCollection("col")
	_ = fs.UpsertPoints("col", []Point{pt(1, filepath.Join(docs, "a.md"), []float32{1, 0}, "alpha")})
	if _, err := fs.Search("col", []float32{1, 0}, 5, 0, nil); err != nil {
		t.Fatal(err)
	}
	if err := fs.DeleteCollection("col"); err != nil {
		t.Fatal(err)
	}
	fs.cacheMu.Lock()
	n := len(fs.indexCache)
	fs.cacheMu.Unlock()
	if n != 0 {
		t.Fatalf("delete collection should drop cache entries, got %d", n)
	}
}

func TestFileStoreSearchFilter(t *testing.T) {
	fs, docs, _ := newTestStore(t)
	_ = fs.CreateCollection("col")
	src := filepath.Join(docs, "a.md")
	_ = fs.UpsertPoints("col", []Point{
		{ID: 1, Vector: []float32{1, 0}, Payload: map[string]any{"text": "alpha", "source_file": src, "metadata": map[string]any{"role": "user"}}},
		{ID: 2, Vector: []float32{0, 1}, Payload: map[string]any{"text": "beta", "source_file": src, "metadata": map[string]any{"role": "assistant"}}},
	})

	// Qdrant-style must clause on a nested metadata key.
	must := map[string]any{"must": []any{map[string]any{"key": "metadata.role", "match": map[string]any{"value": "user"}}}}
	res, err := fs.Search("col", []float32{1, 0}, 10, 0, must)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].Payload["text"] != "alpha" {
		t.Fatalf("nested metadata filter: %+v", res)
	}

	// Flat dotted-key form is equivalent.
	res, err = fs.Search("col", []float32{1, 0}, 10, 0, map[string]any{"metadata.role": "assistant"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].Payload["text"] != "beta" {
		t.Fatalf("flat metadata filter: %+v", res)
	}

	// No filter still returns everything.
	res, err = fs.Search("col", []float32{1, 0}, 10, 0, nil)
	if err != nil || len(res) != 2 {
		t.Fatalf("unfiltered search: %v %+v", err, res)
	}
}
