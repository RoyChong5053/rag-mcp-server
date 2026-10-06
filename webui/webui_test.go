package webui

import (
	"encoding/json"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// Saving settings succeeds with warnings for unknown collections: a typo
// fails loudly at search time, never silently widens.
func TestSetSettingsSavesWithWarnings(t *testing.T) {
	dir := t.TempDir()
	cfg := &engine.EngineConfig{
		VectraDir:    filepath.Join(dir, "Vectra"),
		Backend:      engine.BackendVectra,
		DocsDir:      dir,
		RegistryPath: filepath.Join(dir, "collections.json"),
	}
	st := settings.New(filepath.Join(dir, "settings.json"), settings.Settings{
		ActiveBackend:           engine.BackendVectra,
		DefaultCollectionVectra: "global_memory",
	})
	eng, err := engine.NewEngine(cfg, st)
	if err != nil {
		t.Fatalf("engine: %v", err)
	}
	h := New(eng, filepath.Join(dir, "audit.log"), dir, BuildInfo{})

	body := `{"active_backend":"vectra","default_collection_vectra":"gm_vectra"}`
	req := httptest.NewRequest("POST", "/api/settings", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.setSettings(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	got := eng.Settings().Get()
	if got.ActiveBackend != engine.BackendVectra {
		t.Fatalf("active_backend = %q, want vectra", got.ActiveBackend)
	}
	if got.DefaultCollectionVectra != "gm_vectra" {
		t.Fatalf("default_collection_vectra = %q, want gm_vectra", got.DefaultCollectionVectra)
	}
	var resp struct {
		Warnings []string `json:"warnings"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Warnings) == 0 {
		t.Fatal("expected a warning that the default could not be verified")
	}
}

// An invalid active_backend is still a hard 400: that is a real input error,
// not a reachability problem.
func TestSetSettingsRejectsBadActiveBackend(t *testing.T) {
	dir := t.TempDir()
	cfg := &engine.EngineConfig{
		VectraDir:    filepath.Join(dir, "Vectra"),
		Backend:      engine.BackendVectra,
		DocsDir:      dir,
		RegistryPath: filepath.Join(dir, "collections.json"),
	}
	eng, err := engine.NewEngine(cfg, settings.New(filepath.Join(dir, "settings.json"), settings.Settings{}))
	if err != nil {
		t.Fatalf("engine: %v", err)
	}
	h := New(eng, filepath.Join(dir, "audit.log"), dir, BuildInfo{})

	req := httptest.NewRequest("POST", "/api/settings", strings.NewReader(`{"active_backend":"mysql"}`))
	rec := httptest.NewRecorder()
	h.setSettings(rec, req)
	if rec.Code != 400 {
		t.Fatalf("status = %d, want 400, body = %s", rec.Code, rec.Body.String())
	}
}
