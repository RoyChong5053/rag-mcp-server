package engine

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// fakeStore is an in-memory VectorStore for unit tests.
type fakeStore struct {
	mu      sync.Mutex
	exists  map[string]bool
	results []StoreSearchResult
}

func newFakeStore(ok bool, cols ...string) *fakeStore {
	fs := &fakeStore{exists: map[string]bool{}}
	if ok {
		for _, c := range cols {
			fs.exists[c] = true
		}
		if len(cols) == 0 {
			fs.exists["good"] = true
		}
	}
	return fs
}

func (s *fakeStore) CreateCollection(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.exists[name] = true
	return nil
}

func (s *fakeStore) CollectionExists(name string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.exists[name], nil
}

func (s *fakeStore) GetCollectionInfo(name string) (*StoreCollectionInfo, error) {
	return &StoreCollectionInfo{Name: name, ChunkCount: len(s.results)}, nil
}

func (s *fakeStore) ListCollections() ([]StoreCollectionInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []StoreCollectionInfo
	for n := range s.exists {
		out = append(out, StoreCollectionInfo{Name: n})
	}
	return out, nil
}

func (s *fakeStore) UpsertPoints(collection string, points []Point) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.exists[collection] = true
	return nil
}

func (s *fakeStore) Search(collection string, vector []float32, limit int, threshold float64, filter map[string]any) ([]StoreSearchResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.exists[collection] {
		return nil, nil
	}
	out := s.results
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

func (s *fakeStore) DeletePoints(collection string, filter map[string]any) error {
	return nil
}

func (s *fakeStore) DeleteCollection(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.exists, name)
	return nil
}

func (s *fakeStore) SetPayload(collection string, payload map[string]any, filter map[string]any) error {
	return nil
}

func (s *fakeStore) Ping() error { return nil }

func (s *fakeStore) KeywordSearch(collection string, query string, limit int) ([]StoreSearchResult, error) {
	return nil, nil
}

// newTestEngine builds an Engine backed by the first fake store passed
// (extra positional stores are ignored; kept for caller compatibility).
func newTestEngine(t *testing.T, st settings.Settings, stores ...VectorStore) *Engine {
	t.Helper()
	var store VectorStore = newFakeStore(true)
	if len(stores) > 0 && stores[0] != nil {
		store = stores[0]
	}
	reg, err := registry.New(t.TempDir() + "/collections.json")
	if err != nil {
		t.Fatal(err)
	}
	ss := settings.New("", st)
	e := &Engine{
		stores:         map[string]VectorStore{BackendVectra: store},
		defaultBackend: BackendVectra,
		embedding:      NewEmbeddingClient("http://127.0.0.1:0", "", "embedding", ""),
		rerank:         NewRerankClient("http://127.0.0.1:0", "", "rerank", "", 0, 0),
		config: &EngineConfig{
			EmbedProvider: "one-api", EmbedModel: "embedding",
			VectorDim: 1024, VectorDistance: "Cosine",
		},
		registry:  reg,
		settings:  ss,
		downUntil: map[string]time.Time{},
	}
	return e
}

// fakeEmbeddingServer returns 3-d vectors for any input.
func fakeEmbeddingServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []any{
				map[string]any{"embedding": []float32{0.1, 0.2, 0.3}},
			},
		})
	}))
	t.Cleanup(srv.Close)
	return srv
}
