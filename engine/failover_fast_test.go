package engine

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// deadStore panics if the engine touches a backend that is cached as down.
// It embeds VectorStore (nil) and only overrides the methods the engine might
// call, so any accidental contact fails loudly instead of hanging.
type deadStore struct{ VectorStore }

func (deadStore) Ping() error { panic("Ping called on cached-down backend") }
func (deadStore) Search(collection string, vector []float32, limit int, threshold float64, filter map[string]any) ([]StoreSearchResult, error) {
	panic("Search called on cached-down backend")
}
func (deadStore) CollectionExists(name string) (bool, error) {
	panic("CollectionExists called on cached-down backend")
}

func fakeEmbeddingServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3],"index":0}]}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// A cached-down primary must be skipped outright: no Ping, no Search, no
// CollectionExists — the request goes straight to the same-name replica.
func TestCachedDownBackendSearchSkipsProbe(t *testing.T) {
	v := newFakeStore(true, "global_memory")
	v.results = []StoreSearchResult{{ID: 1, Score: 0.9, Payload: map[string]any{"text": "from vectra"}}}
	e := newTestEngine(t, settings.Settings{
		ActiveBackend:           BackendQdrant,
		DefaultCollectionQdrant: "global_memory",
		FailoverEnabled:         true,
	}, deadStore{}, v)
	srv := fakeEmbeddingServer(t)
	e.embedding = NewEmbeddingClient(srv.URL, "", "embedding", "")

	e.healthMu.Lock()
	e.downUntil[BackendQdrant] = time.Now().Add(time.Minute)
	e.healthMu.Unlock()

	res, err := e.SearchDefault("q", "global_memory", 5, nil, nil)
	if err != nil {
		t.Fatalf("SearchDefault: %v", err)
	}
	if len(res) != 1 || res[0].Backend != BackendVectra || res[0].Text != "from vectra" {
		t.Fatalf("expected vectra fallback, got %+v", res)
	}
}

// When both backends are cached down the engine must not probe either.
func TestBothBackendsDownSearchSkipsProbe(t *testing.T) {
	e := newTestEngine(t, settings.Settings{
		ActiveBackend:           BackendQdrant,
		DefaultCollectionQdrant: "global_memory",
		FailoverEnabled:         true,
	}, deadStore{}, deadStore{})
	srv := fakeEmbeddingServer(t)
	e.embedding = NewEmbeddingClient(srv.URL, "", "embedding", "")

	e.healthMu.Lock()
	e.downUntil[BackendQdrant] = time.Now().Add(time.Minute)
	e.downUntil[BackendVectra] = time.Now().Add(time.Minute)
	e.healthMu.Unlock()

	if _, err := e.SearchDefault("q", "global_memory", 5, nil, nil); err == nil {
		t.Fatal("expected error when both backends are down")
	}
}
