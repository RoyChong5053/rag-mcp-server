package engine

import (
	"strings"
	"testing"

	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// The dimension actually returned by the embedding model must win over the
// declared config, so a model swap behind the same alias is reflected in
// collection creation and provenance.
func TestActiveEmbedProvenancePrefersObservedDim(t *testing.T) {
	e := newTestEngine(t, settings.Settings{}, newFakeStore(true), newFakeStore(true))

	if _, _, dim, _ := e.activeEmbedProvenance(); dim != 1024 {
		t.Fatalf("declared fallback dim = %d, want 1024", dim)
	}

	e.noteEmbedDim(768)
	if _, _, dim, _ := e.activeEmbedProvenance(); dim != 768 {
		t.Fatalf("observed dim = %d, want 768", dim)
	}
}

// An append whose model now returns a different dimension must be refused by
// the provenance gate (with a clear vector_dim message), never mixed in.
func TestProvenanceConflictUsesObservedDim(t *testing.T) {
	e := newTestEngine(t, settings.Settings{}, newFakeStore(true, "myCol"), newFakeStore(true))
	if err := e.registry.Update("myCol", func(en *registry.Entry) {
		en.EmbedModel = "embedding"
		en.EmbedProvider = "one-api"
		en.VectorDim = 1024
	}); err != nil {
		t.Fatal(err)
	}

	// Same model as recorded: no conflict.
	if err := e.checkProvenanceConflict("myCol"); err != nil {
		t.Fatalf("matching dim must pass: %v", err)
	}

	// Model silently switched to a 768-d backend: gate must fire.
	e.noteEmbedDim(768)
	err := e.checkProvenanceConflict("myCol")
	if err == nil {
		t.Fatal("expected vector_dim conflict after model switch")
	}
	if !strings.Contains(err.Error(), "vector_dim: 1024 vs current 768") {
		t.Fatalf("conflict should report vector_dim mismatch, got: %v", err)
	}
}
