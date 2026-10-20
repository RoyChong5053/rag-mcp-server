package settings

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func base() Settings {
	return Settings{
		DefaultTopK:      10,
		DefaultThreshold: 0.25,
		RerankEnabled:    true,
		RerankRecall:     30,
		QueryMaxChars:    500,
		DocMaxChars:      1000,
		FailoverEnabled:  true,
	}
}

func TestNewSeedsDefaultsWhenFileMissing(t *testing.T) {
	s := New(filepath.Join(t.TempDir(), "nope.json"), base())
	got := s.Get()
	if got.DefaultTopK != 10 || got.DefaultThreshold != 0.25 || !got.RerankEnabled {
		t.Fatalf("unexpected defaults: %+v", got)
	}
	if !got.FailoverEnabled {
		t.Fatalf("failover should default to enabled: %+v", got)
	}
}

func TestUpdatePartialPatchKeepsOtherFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := New(path, base())

	col := "leer_chat_0824"
	if err := s.Update(Patch{DefaultCollectionQdrant: &col}); err != nil {
		t.Fatalf("update: %v", err)
	}
	got := s.Get()
	if got.DefaultCollectionQdrant != col {
		t.Fatalf("default_collection_qdrant = %q, want %q", got.DefaultCollectionQdrant, col)
	}
	if got.DefaultTopK != 10 || got.DefaultThreshold != 0.25 {
		t.Fatalf("partial patch clobbered fields: %+v", got)
	}
}

func TestUpdateActiveBackendAndFailover(t *testing.T) {
	s := New(filepath.Join(t.TempDir(), "settings.json"), base())
	vectra := "vectra"
	off := false
	if err := s.Update(Patch{ActiveBackend: &vectra, FailoverEnabled: &off}); err != nil {
		t.Fatalf("update: %v", err)
	}
	got := s.Get()
	if got.ActiveBackend != "vectra" || got.FailoverEnabled {
		t.Fatalf("active_backend/failover mismatch: %+v", got)
	}
}

func TestUpdateExplicitZeroThreshold(t *testing.T) {
	s := New(filepath.Join(t.TempDir(), "settings.json"), base())
	zero := 0.0
	if err := s.Update(Patch{DefaultThreshold: &zero}); err != nil {
		t.Fatalf("update: %v", err)
	}
	if got := s.Get().DefaultThreshold; got != 0 {
		t.Fatalf("threshold = %v, want 0 (explicit zero must apply)", got)
	}
}

func TestPersistenceRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := New(path, base())
	col := "obsidian"
	topK := 20
	off := false
	recall := 300
	if err := s.Update(Patch{DefaultCollectionQdrant: &col, DefaultTopK: &topK, RerankEnabled: &off, RerankRecall: &recall}); err != nil {
		t.Fatalf("update: %v", err)
	}

	reloaded := New(path, base())
	got := reloaded.Get()
	if got.DefaultCollectionQdrant != "obsidian" || got.DefaultTopK != 20 || got.RerankEnabled || got.RerankRecall != 300 {
		t.Fatalf("round trip mismatch: %+v", got)
	}
}

func TestClearDefaultCollection(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := New(path, base())
	col := "x"
	_ = s.Update(Patch{DefaultCollectionQdrant: &col})
	empty := ""
	if err := s.Update(Patch{DefaultCollectionQdrant: &empty}); err != nil {
		t.Fatalf("update: %v", err)
	}
	if got := s.Get().DefaultCollectionQdrant; got != "" {
		t.Fatalf("default_collection_qdrant = %q, want empty", got)
	}
}

func TestLegacyDefaultCollectionMigrates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	legacy := `{"default_collection":"global_memory","default_top_k":15}`
	if err := os.WriteFile(path, []byte(legacy), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	got := New(path, base()).Get()
	if got.DefaultCollectionQdrant != "global_memory" {
		t.Fatalf("legacy default_collection did not migrate: %+v", got)
	}
	if got.DefaultTopK != 15 {
		t.Fatalf("other legacy fields should still load: %+v", got)
	}
}

// TestPatchFieldsAreApplied guards the trap of declaring a field in Patch and
// forgetting to apply it in Update. The failure is quiet and nasty: the
// dashboard saves the value, settings.json shows it, the API echoes it back —
// and the server keeps using the old one, because the patch was decoded and
// then dropped on the floor.
//
// The check is structural: every scalar field of Patch must move the matching
// field of Settings.
func TestPatchFieldsAreApplied(t *testing.T) {
	pt := reflect.TypeOf(Patch{})
	st := reflect.TypeOf(Settings{})
	for i := 0; i < pt.NumField(); i++ {
		pf := pt.Field(i)
		if _, ok := st.FieldByName(pf.Name); !ok {
			t.Errorf("Patch.%s has no matching Settings field", pf.Name)
			continue
		}
		pv := reflect.New(pt).Elem()
		fv := pv.Field(i)
		// Every Patch field is a pointer, so the value to inject lives one
		// level down.
		if fv.Kind() != reflect.Ptr {
			continue
		}
		elem := fv.Type().Elem()
		nv := reflect.New(elem)
		switch elem.Kind() {
		case reflect.String:
			nv.Elem().SetString("sentinel-value")
		case reflect.Int:
			nv.Elem().SetInt(42)
		case reflect.Bool:
			nv.Elem().SetBool(true)
		case reflect.Float64:
			nv.Elem().SetFloat(0.42)
		default:
			continue // slices need a specific value to assert on
		}
		fv.Set(nv)
		store := New("", Settings{})
		if err := store.Update(pv.Interface().(Patch)); err != nil {
			t.Fatalf("Update with %s: %v", pf.Name, err)
		}
		got := reflect.ValueOf(store.Get()).FieldByName(pf.Name).Interface()
		if reflect.DeepEqual(got, reflect.Zero(elem).Interface()) {
			t.Errorf("Patch.%s was accepted but Settings.%s stayed at its zero value", pf.Name, pf.Name)
		}
	}
}
