package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// sha256Hex is the hash the catalog and the vault ledger both use.
func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// TestRepairSourcesFindsMovedFiles covers the interrupted migration: the raw
// file is gone, the vault holds its text, and the index still points at the
// old path. The repair finds the new home by content hash.
func TestRepairSourcesFindsMovedFiles(t *testing.T) {
	root := t.TempDir()
	docs := filepath.Join(root, "docs")
	legacyDir := filepath.Join(docs, "memory", "web")
	if err := os.MkdirAll(legacyDir, 0o755); err != nil {
		t.Fatal(err)
	}
	legacyPath := filepath.Join(legacyDir, "abcd1234abcd1234.md")
	body := "the body of the page"
	if err := os.WriteFile(legacyPath, []byte("# Title\n\n- source: https://example.com/p\n\n"+body), 0o644); err != nil {
		t.Fatal(err)
	}

	reg, err := registry.New(filepath.Join(root, "collections.json"))
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
		settings:  settings.New("", settings.Settings{}),
		downUntil: map[string]time.Time{},
	}
	ledger, err := newDocLedger(e.documentsDir(), 30)
	if err != nil {
		t.Fatal(err)
	}
	e.ledger = ledger

	// The collection holds a chunk of the legacy file.
	if err := fs.CreateCollection("web_cache"); err != nil {
		t.Fatal(err)
	}
	if err := fs.UpsertPoints("web_cache", []Point{{
		ID: 7,
		Payload: map[string]any{
			"text":        body,
			"source":      "abcd1234abcd1234.md",
			"source_file": legacyPath,
			"source_rel":  "memory/web/abcd1234abcd1234.md",
		},
		Vector: []float32{0.1, 0.2, 0.3},
	}}); err != nil {
		t.Fatal(err)
	}
	// The catalog needs the hash of the indexed text, like a real index.
	fs.UpdateCatalogMeta("web_cache", legacyPath, sha256Hex(body), 500, 30, "window", "embedding")

	// The document is in the vault, the old file is gone.
	file, err := e.fileDocument(Document{
		Channel: "web", Title: "Title", URL: "https://example.com/p", Text: body,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(legacyPath); err != nil {
		t.Fatal(err)
	}

	rep, err := e.RepairSources()
	if err != nil {
		t.Fatal(err)
	}
	if rep.Missing != 1 {
		t.Fatalf("expected 1 missing source, got %d (%+v)", rep.Missing, rep)
	}
	if rep.Repaired != 1 {
		t.Fatalf("expected 1 repaired source, got %d (%+v)", rep.Repaired, rep)
	}
	if rep.ChunksRepointed != 1 {
		t.Fatalf("expected 1 re-pointed chunk, got %d", rep.ChunksRepointed)
	}
	payload, _, ok := fs.GetChunk("web_cache", "7")
	if !ok {
		t.Fatal("the chunk is gone")
	}
	if got := payloadString(payload, "source_file"); got != file.Path {
		t.Fatalf("the chunk should point at the vault document: %q", got)
	}

	// A second repair finds nothing to do: the pointer is correct now.
	rep2, err := e.RepairSources()
	if err != nil {
		t.Fatal(err)
	}
	if rep2.Repaired != 0 {
		t.Fatalf("the repair should be idempotent, got %+v", rep2)
	}
}

// TestRepairSourcesFindsRenamedDocument covers the case where the index
// points at a document that was renamed: the id of a document is part of its
// file name, so the document is still findable.
func TestRepairSourcesFindsRenamedDocument(t *testing.T) {
	root := t.TempDir()
	docs := filepath.Join(root, "docs")
	reg, err := registry.New(filepath.Join(root, "collections.json"))
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
		settings:  settings.New("", settings.Settings{}),
		downUntil: map[string]time.Time{},
	}
	ledger, err := newDocLedger(e.documentsDir(), 30)
	if err != nil {
		t.Fatal(err)
	}
	e.ledger = ledger

	file, err := e.fileDocument(Document{Channel: "web", Title: "A page", Text: "the body"})
	if err != nil {
		t.Fatal(err)
	}
	doc, err := e.ReadDocument(file.Rel)
	if err != nil {
		t.Fatal(err)
	}
	// The index points at the same document under an older name.
	oldPath := filepath.Join(e.documentsDir(), "2026-10-01/web/_/010101-"+doc.Meta.ID+"-older-name.md")
	if err := fs.CreateCollection("web_cache"); err != nil {
		t.Fatal(err)
	}
	if err := fs.UpsertPoints("web_cache", []Point{{
		ID:      9,
		Payload: map[string]any{"text": "the body", "source": "old.md", "source_file": oldPath},
		Vector:  []float32{0.1, 0.2, 0.3},
	}}); err != nil {
		t.Fatal(err)
	}

	rep, err := e.RepairSources()
	if err != nil {
		t.Fatal(err)
	}
	if rep.Repaired != 1 {
		t.Fatalf("expected 1 repaired source, got %+v", rep)
	}
	payload, _, ok := fs.GetChunk("web_cache", "9")
	if !ok {
		t.Fatal("the chunk is gone")
	}
	if got := payloadString(payload, "source_file"); got != file.Path {
		t.Fatalf("the chunk should point at the document: %q", got)
	}
}
