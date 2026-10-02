package engine

import (
	"strings"
	"testing"

	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// A collection whose recorded provenance (embed model/provider/vector space)
// differs from the running engine config must refuse an append, so two
// incompatible embedding setups never get mixed into one vector set.
func TestProvenanceConflictRejectsMixedWrite(t *testing.T) {
	q := newFakeStore(true, "myCol")
	e := newTestEngine(t, settings.Settings{}, q, newFakeStore(true))
	e.config.EmbedProvider = "openrouter"
	e.config.EmbedModel = "openai/text-embedding-3"
	e.config.VectorDim = 1536
	e.config.VectorDistance = "Cosine"

	// Existing collection was built with one-api provenance.
	if err := e.registry.Update("myCol", func(en *registry.Entry) {
		en.EmbedModel = "embedding"
		en.EmbedProvider = "one-api"
		en.VectorDim = 1024
	}); err != nil {
		t.Fatal(err)
	}

	err := e.checkProvenanceConflict("myCol")
	if err == nil {
		t.Fatal("expected conflict error")
	}
	for _, want := range []string{"embed_model", "embed_provider", "vector_dim"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error should name %s: %v", want, err)
		}
	}
}

func TestProvenanceConflictAllowsSameOrUnknown(t *testing.T) {
	q := newFakeStore(true, "ok")
	e := newTestEngine(t, settings.Settings{}, q, newFakeStore(true))

	if err := e.checkProvenanceConflict("unknown-col"); err != nil {
		t.Fatalf("unregistered collection must pass: %v", err)
	}
	if err := e.registry.Update("ok", func(en *registry.Entry) {
		en.EmbedModel = "embedding"
		en.EmbedProvider = "one-api"
		en.VectorDim = 1024
		en.VectorDistance = "cosine" // case-insensitive
	}); err != nil {
		t.Fatal(err)
	}
	if err := e.checkProvenanceConflict("ok"); err != nil {
		t.Fatalf("matching provenance must pass: %v", err)
	}
}
