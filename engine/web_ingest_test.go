package engine

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// toggleEmbeddingServer answers with a fixed vector until it is told to fail.
func toggleEmbeddingServer(t *testing.T, fail *atomic.Bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fail.Load() {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"down"}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []any{
				map[string]any{"embedding": []float32{0.1, 0.2, 0.3}},
			},
		})
	}))
	t.Cleanup(srv.Close)
	return srv
}

func newWebEngine(t *testing.T, embedURL string) *Engine {
	t.Helper()
	dir := t.TempDir()
	cfg := &EngineConfig{
		VectraDir:        dir + "/Vectra",
		DocsDir:          dir + "/docs",
		OneAPIBaseURL:    embedURL,
		EmbedModel:       "embedding",
		EmbedProvider:    "one-api",
		VectorDim:        3,
		VectorDistance:   "Cosine",
		RegistryPath:     dir + "/collections.json",
		ChunkSize:        500,
		OverlapPercent:   30,
		ChunkStrategy:    "window",
		RerankEnabled:    false,
		EmbedTimeout:     10 * 1000000000,
		BulkEmbedTimeout: 10 * 1000000000,
	}
	e, err := NewEngine(cfg, settings.New("", settings.Settings{ActiveBackend: BackendVectra}))
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// The same page found twice is embedded once. Re-searching a topic is
// normal, and embedding the unchanged article again costs a full pass for
// nothing.
func TestIngestWebDocumentSkipsUnchangedContent(t *testing.T) {
	var fail atomic.Bool
	srv := toggleEmbeddingServer(t, &fail)
	e := newWebEngine(t, srv.URL)

	doc := WebDocument{
		URL:          "https://example.com/a",
		CanonicalURL: "https://example.com/a",
		Title:        "A",
		Text:         "the quick brown fox jumps over the lazy dog",
	}
	first, err := e.IngestWebDocument("web_cache", doc, WebIngestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if first.Unchanged {
		t.Fatal("the first ingest must not be reported as unchanged")
	}
	if first.ChunksIndexed == 0 {
		t.Fatal("the first ingest must index chunks")
	}

	second, err := e.IngestWebDocument("web_cache", doc, WebIngestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !second.Unchanged {
		t.Fatal("the same content must be reported as unchanged")
	}
	if second.ChunksIndexed != 0 {
		t.Fatalf("unchanged content must not be embedded again, got %d chunks", second.ChunksIndexed)
	}
}

// A failing embed must not cost the reader the page: the old version stays
// until the new one is ready. The delete used to happen before the embedding,
// so a slow or failed embed left the library with neither version.
func TestIngestWebDocumentKeepsOldVersionWhenEmbedFails(t *testing.T) {
	var fail atomic.Bool
	srv := toggleEmbeddingServer(t, &fail)
	e := newWebEngine(t, srv.URL)

	doc := WebDocument{
		URL:          "https://example.com/a",
		CanonicalURL: "https://example.com/a",
		Title:        "A",
		Text:         "the quick brown fox jumps over the lazy dog",
	}
	if _, err := e.IngestWebDocument("web_cache", doc, WebIngestOptions{}); err != nil {
		t.Fatal(err)
	}
	before, err := e.FileStore().GetCollectionInfo("web_cache")
	if err != nil {
		t.Fatal(err)
	}
	if before.ChunkCount == 0 {
		t.Fatal("expected the first version to be stored")
	}

	fail.Store(true)
	changed := doc
	changed.Text = "a completely different article body that has to be embedded again"
	if _, err := e.IngestWebDocument("web_cache", changed, WebIngestOptions{}); err == nil {
		t.Fatal("expected the ingest to fail while the embedding backend is down")
	}

	after, err := e.FileStore().GetCollectionInfo("web_cache")
	if err != nil {
		t.Fatal(err)
	}
	if after.ChunkCount != before.ChunkCount {
		t.Fatalf("the stored version changed on a failed embed: %d -> %d", before.ChunkCount, after.ChunkCount)
	}
}
