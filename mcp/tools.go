package mcp

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/RoyChong5053/rag-mcp-server/engine"
)

// RegisterTools registers all RAG MCP tools with the server. Search and store
// are synchronous: they block until the result is ready, bounded by the one-api
// embed/rerank budget. There is no background job queue or wait knob in the MCP
// surface — one-api owns channel failover, and a failure is reported so the
// caller can fail open (skip RAG and answer without retrieval).
func RegisterTools(server *Server, eng *engine.Engine) {
	server.RegisterTool(Tool{
		Name: "search_memory",
		Description: "Search your persistent memory using semantic similarity. Returns relevant chunks from your knowledge base. " +
			"Omit collection_id (and collection_ids) to use the configured default collection. " +
			"Pass collection_ids to search several collections in one call: recall is merged and globally reranked to top_k; a collection that fails (dim mismatch) is skipped and reported in the server log. " +
			"With no default configured it searches all enabled collections. " +
			"top_k, threshold and reranking default to server (WebUI) settings when omitted. " +
			"Long queries are truncated to the server's query_max_chars setting (head kept) before embedding. " +
			"Optionally restrict by metadata: pass metadata {\"key\":\"value\"} for exact matches on chunk metadata, " +
			"or a filter object with match.value clauses for advanced matching. " +
			"The call is synchronous and can take up to the server's one-api timeout; if retrieval is unavailable the tool returns an error and the caller should proceed without RAG.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "The search query",
				},
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Optional: a specific collection to search. Empty uses the server default.",
				},
				"collection_ids": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Optional: search several collections in one call (merged and globally reranked to top_k). A single element behaves like collection_id; wins over collection_id when both are given.",
				},
				"top_k": map[string]any{
					"type":        "integer",
					"description": "Optional: number of results. Omit for the server default.",
				},
				"threshold": map[string]any{
					"type":        "number",
					"description": "Optional: minimum similarity score 0-1. Omit for the server default (lower it when top_k is large).",
				},
				"metadata": map[string]any{
					"type":                 "object",
					"description":          "Optional: exact-match metadata filter, e.g. {\"role\":\"user\"} matches chunks whose metadata.role == user. Applied to both backends.",
					"additionalProperties": true,
				},
				"filter": map[string]any{
					"type":        "object",
					"description": "Optional: filter object with match.value clauses (advanced). Takes precedence over metadata.",
				},
			},
			"required": []string{"query"},
		},
	}, func(args map[string]any) (any, error) {
		query, _ := args["query"].(string)
		if strings.TrimSpace(query) == "" {
			return nil, fmt.Errorf("query is required")
		}
		collectionID, _ := args["collection_id"].(string)
		collectionIDs := getStringSliceArg(args, "collection_ids")
		topK := getIntArg(args, "top_k", 0) // 0 = server default
		var threshold *float64
		if v, ok := args["threshold"]; ok {
			if f, ok := v.(float64); ok {
				threshold = &f
			}
		}
		filter := getFilterArg(args)

		var results []engine.SearchResult
		var err error
		switch {
		case len(collectionIDs) > 1:
			results, err = eng.SearchMultiDefault(query, collectionIDs, topK, threshold, filter)
		default:
			id := collectionID
			if len(collectionIDs) == 1 {
				id = collectionIDs[0]
			}
			results, err = eng.SearchDefault(query, id, topK, threshold, filter)
		}
		if err != nil {
			return nil, err
		}
		return formatSearchResults(results), nil
	})

	server.RegisterTool(Tool{
		Name: "store_memory",
		Description: "Vectorize and store text into a collection so it can be recalled later with search_memory. " +
			"Omit collection_id to write to the configured default collection. " +
			"The raw text is appended to the daily agent journal (docs/memory/agent/YYYY-MM-DD.md) and only the new text is embedded, " +
			"so it can be browsed in the data bank. " +
			"The call is synchronous and can take up to the server's one-api timeout.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text": map[string]any{
					"type":        "string",
					"description": "Text to remember (chunked, embedded, stored).",
				},
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Optional: target collection. Empty uses the server default.",
				},
				"metadata": map[string]any{
					"type":        "object",
					"description": "Optional metadata to attach to the chunks",
				},
			},
			"required": []string{"text"},
		},
	}, func(args map[string]any) (any, error) {
		text, _ := args["text"].(string)
		collectionID, _ := args["collection_id"].(string)
		metadata := getStringMapArg(args, "metadata")

		result, err := eng.StoreMemory(text, collectionID, metadata)
		if err != nil {
			return nil, err
		}
		return fmt.Sprintf("Stored %d chunks into collection '%s'", result.ChunksIndexed, result.Collection), nil
	})

	server.RegisterTool(Tool{
		Name:        "delete_memory",
		Description: "Delete points from a collection based on a filter.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Collection to delete from",
				},
				"filter": map[string]any{
					"type":        "object",
					"description": "Filter to match points for deletion (match.value clauses)",
				},
			},
			"required": []string{"collection_id"},
		},
	}, func(args map[string]any) (any, error) {
		collectionID, _ := args["collection_id"].(string)
		filter, _ := args["filter"].(map[string]any)

		deleted, err := eng.DeleteMemory(collectionID, filter)
		if err != nil {
			return nil, err
		}

		return fmt.Sprintf("Deleted %d chunks from collection '%s'", deleted, collectionID), nil
	})

	server.RegisterTool(Tool{
		Name:        "list_collections",
		Description: "List all available memory collections with their chunk counts.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	}, func(args map[string]any) (any, error) {
		collections, err := eng.ListCollections()
		if err != nil {
			return nil, err
		}

		return formatCollectionList(collections), nil
	})

	server.RegisterTool(Tool{
		Name:        "health_check",
		Description: "Check component health (vectra, embedding, rerank) and report the active default backend. This is an on-demand probe; nothing in the query path depends on it.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	}, func(args map[string]any) (any, error) {
		status := eng.HealthCheck()
		return formatHealthCheck(status), nil
	})

	server.RegisterTool(Tool{
		Name:        "collection_info",
		Description: "Show management metadata and live chunk count for one collection (or all with collection_id empty).",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Collection to describe. Empty lists all with their registry metadata.",
				},
			},
		},
	}, func(args map[string]any) (any, error) {
		collectionID, _ := args["collection_id"].(string)
		info, err := eng.DescribeCollections(collectionID)
		if err != nil {
			return nil, err
		}
		return ConvertToJSON(info), nil
	})

	server.RegisterTool(Tool{
		Name:        "set_collection_meta",
		Description: "Set management metadata for a collection: display name, description, tags, enabled flag, consumers. Backend is always vectra.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Collection to update",
				},
				"display_name": map[string]any{
					"type":        "string",
					"description": "Human-readable name",
				},
				"description": map[string]any{
					"type":        "string",
					"description": "What this collection holds",
				},
				"tags": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Tags for grouping (replaces existing)",
				},
				"consumers": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Who queries this (e.g. st-leer, opencode-global). Replaces existing.",
				},
				"enabled": map[string]any{
					"type":        "boolean",
					"description": "Disabled collections are skipped by global search",
				},
				"backend": map[string]any{
					"type":        "string",
					"description": "Storage backend, always vectra. Accepted for compatibility.",
				},
			},
			"required": []string{"collection_id"},
		},
	}, func(args map[string]any) (any, error) {
		collectionID, _ := args["collection_id"].(string)
		meta := engine.CollectionMetaUpdate{}
		if v, ok := args["display_name"].(string); ok {
			meta.DisplayName = &v
		}
		if v, ok := args["description"].(string); ok {
			meta.Description = &v
		}
		if _, ok := args["tags"]; ok {
			v := getStringSliceArg(args, "tags")
			meta.Tags = &v
		}
		if _, ok := args["consumers"]; ok {
			v := getStringSliceArg(args, "consumers")
			meta.Consumers = &v
		}
		if v, ok := args["enabled"].(bool); ok {
			meta.Enabled = &v
		}
		if v, ok := args["backend"].(string); ok {
			meta.Backend = &v
		}
		if err := eng.SetCollectionMeta(collectionID, meta); err != nil {
			return nil, err
		}
		return fmt.Sprintf("Updated metadata for collection '%s'", collectionID), nil
	})

	server.RegisterTool(Tool{
		Name:        "delete_collection",
		Description: "Permanently drop an entire collection with all its points and registry entry. Requires confirm=true. Irreversible.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Collection to drop",
				},
				"confirm": map[string]any{
					"type":        "boolean",
					"description": "Must be true, otherwise the call is refused",
				},
			},
			"required": []string{"collection_id", "confirm"},
		},
	}, func(args map[string]any) (any, error) {
		collectionID, _ := args["collection_id"].(string)
		confirm := getBoolArg(args, "confirm", false)
		if err := eng.DeleteCollection(collectionID, confirm); err != nil {
			return nil, err
		}
		return fmt.Sprintf("Deleted collection '%s'", collectionID), nil
	})

	server.RegisterTool(Tool{
		Name:        "vault_verify",
		Description: "Read-only self-check of the file vault: catalog<->index.json consistency, counts, dims, orphans. Omit collection_id to check all.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Collection to verify. Empty checks all.",
				},
			},
		},
	}, func(args map[string]any) (any, error) {
		collectionID, _ := args["collection_id"].(string)
		if strings.TrimSpace(collectionID) != "" {
			return ConvertToJSON(eng.VerifyCollection(strings.TrimSpace(collectionID))), nil
		}
		return ConvertToJSON(eng.VerifyAll()), nil
	})

	server.RegisterTool(Tool{
		Name:        "vault_prune",
		Description: "Remove empty sources (0 items) from a collection and refresh its manifest. Safe cleanup for verify empty_source findings.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Collection to prune",
				},
			},
			"required": []string{"collection_id"},
		},
	}, func(args map[string]any) (any, error) {
		collectionID, _ := args["collection_id"].(string)
		n, err := eng.PruneCollection(collectionID)
		if err != nil {
			return nil, err
		}
		return fmt.Sprintf("Pruned %d empty sources from '%s'", n, collectionID), nil
	})

	server.RegisterTool(Tool{
		Name:        "vault_import",
		Description: "Make a vault folder live after rclone copy: scan collections, verify, register missing entries, refresh cache. Pass the vault root path (server-local).",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Vault root path on the server. Empty uses the configured store.",
				},
			},
		},
	}, func(args map[string]any) (any, error) {
		path, _ := args["path"].(string)
		res, err := eng.ImportVault(strings.TrimSpace(path))
		if err != nil {
			return nil, err
		}
		return ConvertToJSON(res), nil
	})
}

// Helper functions

func getIntArg(args map[string]any, key string, defaultVal int) int {
	if v, ok := args[key]; ok {
		if n, ok := v.(float64); ok {
			return int(n)
		}
	}
	return defaultVal
}

func getBoolArg(args map[string]any, key string, defaultVal bool) bool {
	if v, ok := args[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	return defaultVal
}

func getStringMapArg(args map[string]any, key string) map[string]string {
	if v, ok := args[key]; ok {
		if m, ok := v.(map[string]any); ok {
			result := make(map[string]string)
			for k, val := range m {
				result[k] = fmt.Sprintf("%v", val)
			}
			return result
		}
	}
	return nil
}

func getStringSliceArg(args map[string]any, key string) []string {
	v, ok := args[key]
	if !ok {
		return nil
	}
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(arr))
	for _, item := range arr {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// getFilterArg builds a store filter from search_memory args. An explicit
// "filter" wins; otherwise a flat "metadata" map becomes an AND of exact
// metadata.<key> matches.
func getFilterArg(args map[string]any) map[string]any {
	if f, ok := args["filter"].(map[string]any); ok && len(f) > 0 {
		return f
	}
	meta := getStringMapArg(args, "metadata")
	if len(meta) == 0 {
		return nil
	}
	must := make([]any, 0, len(meta))
	for k, v := range meta {
		must = append(must, map[string]any{
			"key":   "metadata." + k,
			"match": map[string]any{"value": v},
		})
	}
	return map[string]any{"must": must}
}

func formatSearchResults(results []engine.SearchResult) string {
	if len(results) == 0 {
		return "No results found."
	}

	result := fmt.Sprintf("Found %d results:\n\n", len(results))
	for i, r := range results {
		result += fmt.Sprintf("--- Result %d (score: %.3f) ---\n", i+1, r.Score)
		if r.Collection != "" {
			tag := ""
			if r.Backend != "" {
				tag = " [" + r.Backend + "]"
			}
			result += fmt.Sprintf("Collection: %s%s\n", r.Collection, tag)
		}
		if r.Source != "" {
			result += fmt.Sprintf("Source: %s\n", r.Source)
		}
		result += fmt.Sprintf("%s\n\n", r.Text)
	}
	return result
}

func formatCollectionList(collections []engine.CollectionInfo) string {
	if len(collections) == 0 {
		return "No collections found."
	}

	result := fmt.Sprintf("Found %d collections:\n\n", len(collections))
	for _, c := range collections {
		result += fmt.Sprintf("- %s (%d chunks)\n", c.Name, c.ChunkCount)
	}
	return result
}

func formatHealthCheck(status map[string]string) string {
	result := "Health Check:\n\n"
	for component, status := range status {
		result += fmt.Sprintf("- %s: %s\n", component, status)
	}
	return result
}

// ConvertToJSON converts any value to pretty JSON string
func ConvertToJSON(v any) string {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(data)
}
