package engine

import (
	"errors"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

type fakeStore struct {
	up      bool
	cols    map[string]bool
	results []StoreSearchResult
}

func newFakeStore(up bool, cols ...string) *fakeStore {
	f := &fakeStore{up: up, cols: map[string]bool{}}
	for _, c := range cols {
		f.cols[c] = true
	}
	return f
}

func (f *fakeStore) down() error { return unavailable(errors.New("connection refused")) }

func (f *fakeStore) CreateCollection(name string) error { f.cols[name] = true; return nil }
func (f *fakeStore) CollectionExists(name string) (bool, error) {
	if !f.up {
		return false, f.down()
	}
	return f.cols[name], nil
}
func (f *fakeStore) GetCollectionInfo(name string) (*StoreCollectionInfo, error) {
	if !f.up {
		return nil, f.down()
	}
	return &StoreCollectionInfo{Name: name, ChunkCount: len(f.cols)}, nil
}
func (f *fakeStore) ListCollections() ([]StoreCollectionInfo, error) {
	if !f.up {
		return nil, f.down()
	}
	out := make([]StoreCollectionInfo, 0, len(f.cols))
	for c := range f.cols {
		out = append(out, StoreCollectionInfo{Name: c})
	}
	return out, nil
}
func (f *fakeStore) UpsertPoints(collection string, points []Point) error {
	if !f.up {
		return f.down()
	}
	return nil
}
func (f *fakeStore) Search(collection string, vector []float32, limit int, threshold float64, filter map[string]any) ([]StoreSearchResult, error) {
	if !f.up {
		return nil, f.down()
	}
	return f.results, nil
}
func (f *fakeStore) DeletePoints(collection string, filter map[string]any) error {
	if !f.up {
		return f.down()
	}
	return nil
}
func (f *fakeStore) DeleteCollection(name string) error { delete(f.cols, name); return nil }
func (f *fakeStore) SetPayload(collection string, payload map[string]any, filter map[string]any) error {
	if !f.up {
		return f.down()
	}
	return nil
}
func (f *fakeStore) Ping() error {
	if !f.up {
		return f.down()
	}
	return nil
}

func newTestEngine(t *testing.T, st settings.Settings, qdrant, vectra VectorStore) *Engine {
	t.Helper()
	reg, err := registry.New(filepath.Join(t.TempDir(), "collections.json"))
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	return &Engine{
		stores:         map[string]VectorStore{BackendQdrant: qdrant, BackendVectra: vectra},
		defaultBackend: BackendQdrant,
		settings:       settings.New("", st),
		registry:       reg,
		downUntil:      map[string]time.Time{},
	}
}

func TestIsUnavailable(t *testing.T) {
	if IsUnavailable(nil) {
		t.Fatal("nil must not be unavailable")
	}
	if IsUnavailable(errors.New("collection not found")) {
		t.Fatal("plain error must not be unavailable")
	}
	if !IsUnavailable(unavailable(errors.New("boom"))) {
		t.Fatal("wrapped must be unavailable")
	}
	if !IsUnavailable(fmt.Errorf("index failed: %w", unavailable(errors.New("boom")))) {
		t.Fatal("nested wrap must be unavailable")
	}
}

func TestDefaultSearchTargets(t *testing.T) {
	st := settings.Settings{
		ActiveBackend:           BackendQdrant,
		DefaultCollectionQdrant: "global_memory",
		DefaultCollectionVectra: "gm_vectra",
		FailoverEnabled:         true,
	}
	q := newFakeStore(true)
	v := newFakeStore(true)
	e := newTestEngine(t, st, q, v)

	got := e.defaultSearchTargets()
	want := []scopeTarget{
		{backend: BackendQdrant, collection: "global_memory"},
		{backend: BackendVectra, collection: "gm_vectra"},
		{backend: BackendVectra, collection: "global_memory"}, // same-name replica
	}
	if len(got) != len(want) {
		t.Fatalf("targets = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("target[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}

	// Failover off drops the vectra fallback.
	st.FailoverEnabled = false
	e.settings.Update(settings.Patch{FailoverEnabled: &st.FailoverEnabled})
	if n := len(e.defaultSearchTargets()); n != 1 {
		t.Fatalf("failover off should yield 1 target, got %d", n)
	}

	// Active vectra searches only the vectra default.
	vectra := BackendVectra
	e.settings.Update(settings.Patch{ActiveBackend: &vectra})
	targets := e.defaultSearchTargets()
	if len(targets) != 1 || targets[0].backend != BackendVectra || targets[0].collection != "gm_vectra" {
		t.Fatalf("active vectra targets = %+v", targets)
	}
}

func TestDefaultWriteTargetFailover(t *testing.T) {
	st := settings.Settings{
		ActiveBackend:           BackendQdrant,
		DefaultCollectionQdrant: "global_memory",
		DefaultCollectionVectra: "gm_vectra",
		FailoverEnabled:         true,
	}
	// qdrant down -> vectra fallback
	e := newTestEngine(t, st, newFakeStore(false), newFakeStore(true))
	target, err := e.defaultWriteTarget()
	if err != nil {
		t.Fatalf("write target: %v", err)
	}
	if target.backend != BackendVectra || target.collection != "gm_vectra" {
		t.Fatalf("down qdrant target = %+v, want vectra/gm_vectra", target)
	}

	// qdrant up -> primary
	e2 := newTestEngine(t, st, newFakeStore(true), newFakeStore(true))
	target2, err := e2.defaultWriteTarget()
	if err != nil {
		t.Fatalf("write target: %v", err)
	}
	if target2.backend != BackendQdrant || target2.collection != "global_memory" {
		t.Fatalf("up qdrant target = %+v, want qdrant/global_memory", target2)
	}

	// No configured vectra default and qdrant down -> automatic same-name
	// replica on vectra, no error (this is the zero-config failover path).
	st.DefaultCollectionVectra = ""
	e3 := newTestEngine(t, st, newFakeStore(false), newFakeStore(true))
	target3, err := e3.defaultWriteTarget()
	if err != nil {
		t.Fatalf("same-name failover should not error: %v", err)
	}
	if target3.backend != BackendVectra || target3.collection != "global_memory" {
		t.Fatalf("same-name target = %+v, want vectra/global_memory", target3)
	}

	// No defaults at all -> loud error (nothing to route to).
	e4 := newTestEngine(t, settings.Settings{FailoverEnabled: true}, newFakeStore(false), newFakeStore(true))
	if _, err := e4.defaultWriteTarget(); err == nil {
		t.Fatal("expected error when no default collection is configured")
	}
}

// A qdrant primary with no separate vectra default falls back to the
// same-name collection on vectra, so a qdrant outage needs no config change.
func TestDefaultSearchTargetsSameNameReplica(t *testing.T) {
	st := settings.Settings{
		ActiveBackend:           BackendQdrant,
		DefaultCollectionQdrant: "global_memory",
		FailoverEnabled:         true,
	}
	e := newTestEngine(t, st, newFakeStore(true), newFakeStore(true))
	got := e.defaultSearchTargets()
	want := []scopeTarget{
		{backend: BackendQdrant, collection: "global_memory"},
		{backend: BackendVectra, collection: "global_memory"},
	}
	if len(got) != len(want) {
		t.Fatalf("targets = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("target[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// With no qdrant default but a vectra default configured, the vectra default
// becomes primary instead of erroring.
func TestDefaultSearchTargetsVectraDefaultOnly(t *testing.T) {
	st := settings.Settings{
		ActiveBackend:           BackendQdrant,
		DefaultCollectionVectra: "gm_vectra",
		FailoverEnabled:         true,
	}
	e := newTestEngine(t, st, newFakeStore(true), newFakeStore(true))
	got := e.defaultSearchTargets()
	if len(got) != 1 || got[0].backend != BackendVectra || got[0].collection != "gm_vectra" {
		t.Fatalf("targets = %+v, want [vectra/gm_vectra]", got)
	}
}

func TestActiveBackendFallsBackToConfigDefault(t *testing.T) {
	e := newTestEngine(t, settings.Settings{}, newFakeStore(true), newFakeStore(true))
	if got := e.ActiveBackend(); got != BackendQdrant {
		t.Fatalf("active backend = %q, want config default %q", got, BackendQdrant)
	}
	bogus := "mysql"
	e.settings.Update(settings.Patch{ActiveBackend: &bogus})
	if got := e.ActiveBackend(); got != BackendQdrant {
		t.Fatalf("invalid active backend should fall back, got %q", got)
	}
}
