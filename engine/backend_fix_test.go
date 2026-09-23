package engine

import (
	"testing"

	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// recordIndex must persist the backend the vectors actually landed in, so the
// dashboard can show it as fixed metadata.
func TestRecordIndexStampsBackend(t *testing.T) {
	e := newTestEngine(t, settings.Settings{}, newFakeStore(true), newFakeStore(true))
	e.recordIndex("c", Provenance{
		SourceFile:   "docs/a.md",
		Chunks:       3,
		ChunkSize:    500,
		ChunkOverlap: 150,
		EmbedModel:   "embedding",
	}, BackendVectra)

	en := e.registry.Get("c")
	if en == nil {
		t.Fatal("registry entry not created")
	}
	if en.Backend != BackendVectra {
		t.Fatalf("registry backend = %q, want %q", en.Backend, BackendVectra)
	}
	if en.ChunkCount != 3 {
		t.Fatalf("chunk count = %d, want 3", en.ChunkCount)
	}
}

// A collection that already holds vectors must not be switched to another
// backend (that would strand the data); an empty entry may still choose one.
func TestSetCollectionMetaBackendFixedOnceIndexed(t *testing.T) {
	e := newTestEngine(t, settings.Settings{}, newFakeStore(true, "col"), newFakeStore(true))
	vectra := BackendVectra
	if err := e.SetCollectionMeta("col", CollectionMetaUpdate{Backend: &vectra}); err == nil {
		t.Fatal("expected refusal to switch backend of an indexed collection")
	}

	name := "renamed"
	if err := e.SetCollectionMeta("col", CollectionMetaUpdate{DisplayName: &name}); err != nil {
		t.Fatalf("metadata-only update should succeed: %v", err)
	}

	e2 := newTestEngine(t, settings.Settings{}, newFakeStore(true), newFakeStore(true))
	if err := e2.SetCollectionMeta("fresh", CollectionMetaUpdate{Backend: &vectra}); err != nil {
		t.Fatalf("aspirational backend set: %v", err)
	}
	if en := e2.registry.Get("fresh"); en == nil || en.Backend != BackendVectra {
		t.Fatalf("fresh backend = %+v, want vectra", en)
	}
}

func TestCollectionExistsOnOther(t *testing.T) {
	e := newTestEngine(t, settings.Settings{}, newFakeStore(true, "shared"), newFakeStore(true, "only_vec"))
	if ok, err := e.CollectionExistsOnOther(BackendVectra, "shared"); err != nil || !ok {
		t.Fatalf("shared should exist on qdrant (other of vectra): ok=%v err=%v", ok, err)
	}
	if ok, _ := e.CollectionExistsOnOther(BackendVectra, "only_vec"); ok {
		t.Fatal("only_vec is on vectra, not the other backend")
	}
}
