package mcp

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/RoyChong5053/rag-mcp-server/engine"
)

// The MCP surface is the work surface: recall, store, delete, the document
// tools and the web tools. Maintenance (verifying the vector index, pruning
// it, restoring it from a backup) is operational work and lives in the
// server's CLI, not in an agent's tool list.
func TestRegisterToolsSurface(t *testing.T) {
	s := NewServer()
	RegisterTools(s, nil)
	RegisterDocumentTools(s, nil)

	want := []string{
		"search_memory",
		"store_memory",
		"delete_memory",
		"list_collections",
		"health_check",
		"collection_info",
		"set_collection_meta",
		"delete_collection",
		"get_chunk",
		"update_memory",
		"tag_tree",
		"search_document",
		"read_document",
		"list_documents",
		"documents_import",
	}
	if len(s.tools) != len(want) {
		t.Fatalf("registered %d tools, want %d", len(s.tools), len(want))
	}
	for _, name := range want {
		if _, ok := s.tools[name]; !ok {
			t.Fatalf("tool %q not registered", name)
		}
	}
	// The maintenance tools must not be reachable by an agent.
	for _, gone := range []string{"job_status", "job_list", "vault_verify", "vault_prune", "vault_import", "verify_deep"} {
		if _, ok := s.tools[gone]; ok {
			t.Fatalf("tool %q must be removed", gone)
		}
	}
}

// search_memory must point at search_document for exact strings: that is the
// whole point of having both.
func TestSearchMemoryPointsAtDocumentSearch(t *testing.T) {
	s := NewServer()
	RegisterTools(s, nil)
	if !strings.Contains(s.tools["search_memory"].Description, "search_document") {
		t.Fatal("search_memory should tell the caller when to use search_document")
	}
}

// search_memory must not expose wait_seconds anymore.
func TestSearchMemoryHasNoWaitSeconds(t *testing.T) {
	s := NewServer()
	RegisterTools(s, nil)
	props, _ := s.tools["search_memory"].InputSchema["properties"].(map[string]any)
	if _, ok := props["wait_seconds"]; ok {
		t.Fatal("search_memory still exposes wait_seconds")
	}
}

// Multi-collection search is exposed as collection_ids (array of strings).
func TestSearchMemoryHasCollectionIDs(t *testing.T) {
	s := NewServer()
	RegisterTools(s, nil)
	props, _ := s.tools["search_memory"].InputSchema["properties"].(map[string]any)
	v, ok := props["collection_ids"]
	if !ok {
		t.Fatal("search_memory must expose collection_ids")
	}
	m, ok := v.(map[string]any)
	if !ok || m["type"] != "array" {
		t.Fatalf("collection_ids must be an array, got %#v", v)
	}
}

// search_memory must return the stored chunk byte for byte. Truncating here
// once cut at 500 *bytes*, which split CJK runes and made the MCP payload
// shorter than the same hit in the dashboard.
func TestFormatSearchResultsReturnsFullText(t *testing.T) {
	text := strings.Repeat("记忆片段", 400) // 1600 runes / 4800 bytes
	out := formatSearchResults([]engine.SearchResult{{ID: "1", Hash: "abc", Text: text}})

	if !strings.Contains(out, text) {
		t.Fatalf("chunk text altered: got %d runes", utf8.RuneCountInString(out))
	}
	if strings.Contains(out, "…") {
		t.Fatal("chunk text must not be ellipsis-truncated")
	}
	if !utf8.ValidString(out) {
		t.Fatal("result payload must stay valid UTF-8")
	}
}
