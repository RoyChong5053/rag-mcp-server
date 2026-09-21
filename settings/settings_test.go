package settings

import (
	"path/filepath"
	"testing"
)

func base() Settings {
	return Settings{
		DefaultTopK:      10,
		DefaultThreshold: 0.25,
		RerankEnabled:    true,
		RerankRecall:     30,
		QueryMaxChars:    2000,
		DocMaxChars:      1000,
	}
}

func TestNewSeedsDefaultsWhenFileMissing(t *testing.T) {
	s := New(filepath.Join(t.TempDir(), "nope.json"), base())
	got := s.Get()
	if got.DefaultTopK != 10 || got.DefaultThreshold != 0.25 || !got.RerankEnabled {
		t.Fatalf("unexpected defaults: %+v", got)
	}
}

func TestUpdatePartialPatchKeepsOtherFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := New(path, base())

	col := "leer_chat_0824"
	if err := s.Update(Patch{DefaultCollection: &col}); err != nil {
		t.Fatalf("update: %v", err)
	}
	got := s.Get()
	if got.DefaultCollection != col {
		t.Fatalf("default_collection = %q, want %q", got.DefaultCollection, col)
	}
	if got.DefaultTopK != 10 || got.DefaultThreshold != 0.25 {
		t.Fatalf("partial patch clobbered fields: %+v", got)
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
	if err := s.Update(Patch{DefaultCollection: &col, DefaultTopK: &topK, RerankEnabled: &off, RerankRecall: &recall}); err != nil {
		t.Fatalf("update: %v", err)
	}

	reloaded := New(path, base())
	got := reloaded.Get()
	if got.DefaultCollection != "obsidian" || got.DefaultTopK != 20 || got.RerankEnabled || got.RerankRecall != 300 {
		t.Fatalf("round trip mismatch: %+v", got)
	}
}

func TestClearDefaultCollection(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := New(path, base())
	col := "x"
	_ = s.Update(Patch{DefaultCollection: &col})
	empty := ""
	if err := s.Update(Patch{DefaultCollection: &empty}); err != nil {
		t.Fatalf("update: %v", err)
	}
	if got := s.Get().DefaultCollection; got != "" {
		t.Fatalf("default_collection = %q, want empty", got)
	}
}
