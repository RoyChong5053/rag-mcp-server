package engine

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

func fakeEmbeddingServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3],"index":0}]}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// A stale down mark must NOT block a query: the backend is still attempted, and
// a success clears the badge. This reverses the old "cached-down is skipped"
// contract, which is exactly what made a working vectra target unreachable.
func TestDownMarkDoesNotBlockQuery(t *testing.T) {
	q := newFakeStore(true, "global_memory")
	q.results = []StoreSearchResult{{ID: 1, Score: 0.9, Payload: map[string]any{"text": "from qdrant"}}}
	e := newTestEngine(t, settings.Settings{
		ActiveBackend:           BackendQdrant,
		DefaultCollectionQdrant: "global_memory",
		FailoverEnabled:         true,
	}, q, newFakeStore(true, "global_memory"))
	srv := fakeEmbeddingServer(t)
	e.embedding = NewEmbeddingClient(srv.URL, "", "embedding", "")

	e.healthMu.Lock()
	e.downUntil[BackendQdrant] = time.Now().Add(time.Minute)
	e.healthMu.Unlock()

	res, err := e.SearchDefault("q", "global_memory", 5, nil, nil)
	if err != nil {
		t.Fatalf("SearchDefault: %v", err)
	}
	if len(res) != 1 || res[0].Text != "from qdrant" {
		t.Fatalf("expected the qdrant result, got %+v", res)
	}
	if e.backendDown(BackendQdrant) {
		t.Fatal("a successful query must clear the down badge")
	}
}

// An explicit collection whose backend is genuinely unavailable must fail over
// to the same-name replica on the other backend.
func TestUnavailablePrimaryFailsOverToSameName(t *testing.T) {
	v := newFakeStore(true, "shared")
	v.results = []StoreSearchResult{{ID: 2, Score: 0.8, Payload: map[string]any{"text": "from vectra"}}}
	e := newTestEngine(t, settings.Settings{
		ActiveBackend:           BackendQdrant,
		DefaultCollectionQdrant: "global_memory",
		FailoverEnabled:         true,
	}, newFakeStore(false, "shared"), v)
	srv := fakeEmbeddingServer(t)
	e.embedding = NewEmbeddingClient(srv.URL, "", "embedding", "")

	res, err := e.SearchDefault("q", "shared", 5, nil, nil)
	if err != nil {
		t.Fatalf("SearchDefault: %v", err)
	}
	if len(res) != 1 || res[0].Backend != BackendVectra || res[0].Text != "from vectra" {
		t.Fatalf("expected vectra same-name fallback, got %+v", res)
	}
}

// When both backends are down the call reports the real error (the frontend
// then fails open) instead of silently returning nothing.
func TestBothBackendsDownReturnsError(t *testing.T) {
	e := newTestEngine(t, settings.Settings{
		ActiveBackend:           BackendQdrant,
		DefaultCollectionQdrant: "global_memory",
		FailoverEnabled:         true,
	}, newFakeStore(false, "global_memory"), newFakeStore(false, "global_memory"))
	srv := fakeEmbeddingServer(t)
	e.embedding = NewEmbeddingClient(srv.URL, "", "embedding", "")

	if _, err := e.SearchDefault("q", "global_memory", 5, nil, nil); err == nil {
		t.Fatal("expected an error when both backends are down")
	}
}

// A not-found explicit collection reports a case-insensitive did-you-mean hint
// (registry names are case-sensitive), and does not auto-correct.
func TestCollectionSuggestions(t *testing.T) {
	e := newTestEngine(t, settings.Settings{
		ActiveBackend:           BackendQdrant,
		DefaultCollectionQdrant: "global_memory",
	}, newFakeStore(true), newFakeStore(true))
	if err := e.registry.Update("leer-chat-jina-v5", func(en *registry.Entry) {
		en.Backend = BackendVectra
	}); err != nil {
		t.Fatalf("registry update: %v", err)
	}

	if hint := e.collectionSuggestions("Leer-chat-jina-v5"); !strings.Contains(hint, "leer-chat-jina-v5") || !strings.Contains(hint, BackendVectra) {
		t.Fatalf("hint should name the canonical id and backend, got: %q", hint)
	}
	if hint := e.collectionSuggestions("unrelated-name"); hint != "" {
		t.Fatalf("no match should yield no hint, got: %q", hint)
	}
}
