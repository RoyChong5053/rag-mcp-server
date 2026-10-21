package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// registryNew opens a registry for the test.
func registryNew(t *testing.T, path string) (*registry.Registry, error) {
	t.Helper()
	return registry.New(path)
}

// settingsNew returns an empty runtime settings store.
func settingsNew() *settings.Store {
	return settings.New("", settings.Settings{})
}

// TestMigrateDocumentsMovesLegacyFiles covers the whole migration: the file
// is filed into the vault, the old file is put aside, and the vector index is
// re-pointed at the new file (a recall that points at a file that no longer
// exists is worse than no recall).
func TestMigrateDocumentsMovesLegacyFiles(t *testing.T) {
	root := t.TempDir()
	docs := filepath.Join(root, "docs")
	legacyDir := filepath.Join(docs, "memory", "web")
	if err := os.MkdirAll(legacyDir, 0o755); err != nil {
		t.Fatal(err)
	}
	legacyPath := filepath.Join(legacyDir, "abcd1234abcd1234.md")
	legacy := "# How to embed vectors\n\n- source: https://example.com/post\n- provider: tavily\n- fetched_at: 2026-10-20T08:00:00Z\n\nthe body of the page\n"
	if err := os.WriteFile(legacyPath, []byte(legacy), 0o644); err != nil {
		t.Fatal(err)
	}

	reg, err := registryNew(t, filepath.Join(root, "collections.json"))
	if err != nil {
		t.Fatal(err)
	}
	fs := NewFileStore(filepath.Join(root, "Vectra"), docs)
	e := &Engine{
		stores:         map[string]VectorStore{BackendVectra: fs},
		defaultBackend: BackendVectra,
		config: &EngineConfig{
			EmbedProvider: "one-api", EmbedModel: "embedding",
			VectorDim: 1024, VectorDistance: "Cosine",
			DocsDir: docs, DocumentsDir: filepath.Join(docs, "documents"),
		},
		registry:  reg,
		settings:  settingsNew(),
		downUntil: map[string]time.Time{},
	}
	ledger, err := newDocLedger(e.documentsDir(), 30)
	if err != nil {
		t.Fatal(err)
	}
	e.ledger = ledger

	// A collection that holds a chunk of the legacy file.
	if err := fs.CreateCollection("web_cache"); err != nil {
		t.Fatal(err)
	}
	if err := fs.UpsertPoints("web_cache", []Point{{
		ID: 42,
		Payload: map[string]any{
			"text":        "the body of the page",
			"source":      "abcd1234abcd1234.md",
			"source_file": legacyPath,
			"source_rel":  "memory/web/abcd1234abcd1234.md",
		},
		Vector: []float32{0.1, 0.2, 0.3},
	}}); err != nil {
		t.Fatal(err)
	}

	// Dry run: nothing moves.
	report, err := e.MigrateDocuments(true)
	if err != nil {
		t.Fatal(err)
	}
	if !report.DryRun || report.Migrated != 1 {
		t.Fatalf("dry run should report 1 document and move nothing: %+v", report)
	}
	if _, err := os.Stat(legacyPath); err != nil {
		t.Fatalf("the dry run must not touch the file: %v", err)
	}

	// The real thing.
	report, err = e.MigrateDocuments(false)
	if err != nil {
		t.Fatal(err)
	}
	if report.Migrated != 1 {
		t.Fatalf("expected 1 migrated document, got %d (%+v)", report.Migrated, report)
	}
	if _, err := os.Stat(legacyPath); err == nil {
		t.Fatalf("the legacy file should have been moved aside")
	}
	if len(report.Documents) != 1 {
		t.Fatalf("expected the new document to be reported: %+v", report)
	}

	// The document landed in the vault with its identity.
	doc, err := e.ReadDocument(report.Documents[0])
	if err != nil {
		t.Fatal(err)
	}
	if doc.Meta.Title != "How to embed vectors" {
		t.Errorf("the title should survive the migration, got %q", doc.Meta.Title)
	}
	if doc.Meta.URL != "https://example.com/post" {
		t.Errorf("the url should survive the migration, got %q", doc.Meta.URL)
	}
	if doc.Meta.Provider != "tavily" {
		t.Errorf("the provider should survive the migration, got %q", doc.Meta.Provider)
	}
	if !strings.Contains(doc.Body, "the body of the page") {
		t.Errorf("the body should survive the migration, got %q", doc.Body)
	}

	// The vector index points at the new file.
	newPath := filepath.Join(e.documentsDir(), filepath.FromSlash(report.Documents[0]))
	payload, _, ok := fs.GetChunk("web_cache", "42")
	if !ok {
		t.Fatalf("the chunk is gone")
	}
	if got := payloadString(payload, "source_file"); got != newPath {
		t.Fatalf("the chunk should point at the new file: %q", got)
	}
	if report.ChunksRepointed != 1 {
		t.Errorf("expected 1 re-pointed chunk, got %d", report.ChunksRepointed)
	}
}
