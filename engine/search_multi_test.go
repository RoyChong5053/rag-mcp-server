package engine

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// colStore fails Search only for the collections named in fail; everything
// else behaves like the underlying fake store.
type colStore struct {
	*fakeStore
	fail map[string]error
}

func (s *colStore) Search(collection string, vector []float32, limit int, threshold float64, filter map[string]any) ([]StoreSearchResult, error) {
	if err, ok := s.fail[collection]; ok {
		return nil, err
	}
	return s.fakeStore.Search(collection, vector, limit, threshold, filter)
}

func newMultiTestEngine(t *testing.T, fail map[string]error) *Engine {
	t.Helper()
	fs := newFakeStore(true, "good", "bad", "worse")
	fs.results = []StoreSearchResult{{ID: 1, Score: 0.9, Payload: map[string]any{"text": "hit"}}}
	e := newTestEngine(t, settings.Settings{ActiveBackend: BackendVectra}, &colStore{fakeStore: fs, fail: fail}, newFakeStore(true))
	srv := fakeEmbeddingServer(t)
	e.embedding = NewEmbeddingClient(srv.URL, "", "embedding", "")
	return e
}

// One broken library must not silence the others in a multi-collection call:
// the failure is skipped and reported in the log, healthy collections still
// answer.
func TestSearchMultiSkipsFailingCollection(t *testing.T) {
	e := newMultiTestEngine(t, map[string]error{"bad": errors.New("connection reset by peer")})

	res, err := e.SearchMulti("q", []string{"good", "bad"}, 5, false, 0, nil)
	if err != nil {
		t.Fatalf("a failing collection must not abort the batch: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hit" {
		t.Fatalf("expected the healthy result, got %+v", res)
	}
}

// When every requested collection fails, the batch reports loudly instead of
// returning a silently empty list.
func TestSearchMultiAllFailedIsLoudError(t *testing.T) {
	e := newMultiTestEngine(t, map[string]error{
		"bad":   errors.New("connection reset"),
		"worse": errors.New("timeout"),
	})

	_, err := e.SearchMulti("q", []string{"bad", "worse"}, 5, false, 0, nil)
	if err == nil {
		t.Fatal("expected an error when every requested collection fails")
	}
	if !strings.Contains(err.Error(), "all 2 requested collections failed") {
		t.Fatalf("error should report the whole batch, got: %v", err)
	}
}

// A collection whose stored dimension differs from the query vector is skipped
// with a dim-mismatch reason instead of scoring garbage or erroring the batch.
func TestSearchMultiSkipsDimMismatch(t *testing.T) {
	e := newMultiTestEngine(t, nil)
	if err := e.registry.Update("dimcol", func(en *registry.Entry) {
		en.Backend = BackendVectra
		en.VectorDim = 768 // fake embedding server returns 3-d
	}); err != nil {
		t.Fatalf("registry update: %v", err)
	}

	res, err := e.SearchMulti("q", []string{"dimcol", "good"}, 5, false, 0, nil)
	if err != nil {
		t.Fatalf("dim mismatch must not abort the batch: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hit" {
		t.Fatalf("expected the healthy result, got %+v", res)
	}

	_, err = e.SearchMulti("q", []string{"dimcol"}, 5, false, 0, nil)
	if err == nil || !strings.Contains(err.Error(), "dim mismatch") {
		t.Fatalf("all-failed error should carry the dim-mismatch reason, got: %v", err)
	}
}

// Disabled collections are skipped like any other non-searchable target.
func TestSearchMultiSkipsDisabledCollection(t *testing.T) {
	e := newMultiTestEngine(t, nil)
	if err := e.registry.Update("offcol", func(en *registry.Entry) {
		en.Backend = BackendVectra
		en.Enabled = false
	}); err != nil {
		t.Fatalf("registry update: %v", err)
	}

	res, err := e.SearchMulti("q", []string{"offcol", "good"}, 5, false, 0, nil)
	if err != nil {
		t.Fatalf("disabled collection must not abort the batch: %v", err)
	}
	if len(res) != 1 {
		t.Fatalf("expected the healthy result, got %+v", res)
	}

	_, err = e.SearchMulti("q", []string{"offcol"}, 5, false, 0, nil)
	if err == nil || !strings.Contains(err.Error(), "disabled") {
		t.Fatalf("all-failed error should carry the disabled reason, got: %v", err)
	}
}

// A rerank failure is an error, never a silent fall back to vector order —
// that silent swap is exactly what made recall mysteriously inconsistent.
func TestSearchRerankFailureIsError(t *testing.T) {
	failSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad rerank request", http.StatusBadRequest)
	}))
	t.Cleanup(failSrv.Close)

	e := newMultiTestEngine(t, nil)
	// Two candidates: the rerank leg only runs for merged lists > 1.
	e.stores[BackendVectra].(*colStore).results = []StoreSearchResult{
		{ID: 1, Score: 0.9, Payload: map[string]any{"text": "one"}},
		{ID: 2, Score: 0.8, Payload: map[string]any{"text": "two"}},
	}
	e.rerank = NewRerankClient(failSrv.URL, "", "rerank", "", 4000, 8000)
	e.settings.Update(settings.Patch{RerankEnabled: boolPtr(true)})

	_, err := e.SearchDefault("q", "good", 5, nil, nil)
	if err == nil {
		t.Fatal("expected the rerank failure to surface as an error")
	}
	if !strings.Contains(err.Error(), "rerank failed") {
		t.Fatalf("error should say rerank failed, got: %v", err)
	}
}

func boolPtr(b bool) *bool { return &b }

// The overall search budget exists only at stage boundaries: an expired
// deadline blocks the next stage, a zero one never does.
func TestCheckSearchBudget(t *testing.T) {
	if err := checkSearchBudget(time.Time{}); err != nil {
		t.Fatalf("zero deadline means no budget: %v", err)
	}
	if err := checkSearchBudget(time.Now().Add(time.Minute)); err != nil {
		t.Fatalf("fresh deadline must pass: %v", err)
	}
	if err := checkSearchBudget(time.Now().Add(-time.Second)); err == nil {
		t.Fatal("expired deadline must error")
	}

	e := &Engine{config: &EngineConfig{}}
	if d := time.Until(e.searchDeadline()); d < 160*time.Second || d > 171*time.Second {
		t.Fatalf("default search budget should be ~170s, got %v", d)
	}
}
