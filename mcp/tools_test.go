package mcp

import "testing"

// The MCP surface is synchronous now: RAG tools plus vault maintenance, no
// job queue and no wait_seconds knob.
func TestRegisterToolsSurface(t *testing.T) {
	s := NewServer()
	RegisterTools(s, nil)

	want := []string{
		"search_memory",
		"store_memory",
		"delete_memory",
		"list_collections",
		"health_check",
		"collection_info",
		"set_collection_meta",
		"delete_collection",
		"vault_verify",
		"vault_prune",
		"vault_import",
		"get_chunk",
		"update_memory",
		"verify_deep",
		"tag_tree",
	}
	if len(s.tools) != len(want) {
		t.Fatalf("registered %d tools, want %d", len(s.tools), len(want))
	}
	for _, name := range want {
		if _, ok := s.tools[name]; !ok {
			t.Fatalf("tool %q not registered", name)
		}
	}
	for _, gone := range []string{"job_status", "job_list"} {
		if _, ok := s.tools[gone]; ok {
			t.Fatalf("tool %q must be removed", gone)
		}
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
