package mcp

import (
	"encoding/json"
	"fmt"

	"github.com/RoyChong5053/rag-mcp-server/engine"
)

// RegisterTools registers all RAG MCP tools with the server
func RegisterTools(server *Server, eng *engine.Engine) {
	server.RegisterTool(Tool{
		Name:        "search_memory",
		Description: "Search your persistent memory using semantic similarity. Returns relevant chunks from your knowledge base.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "The search query",
				},
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Collection to search (e.g., 'obsidian', 'chat')",
				},
				"top_k": map[string]any{
					"type":        "integer",
					"description": "Number of results to return (default: 10)",
					"default":     10,
				},
				"rerank": map[string]any{
					"type":        "boolean",
					"description": "Apply reranker for precision boost (default: true)",
					"default":     true,
				},
				"threshold": map[string]any{
					"type":        "number",
					"description": "Minimum similarity score (0-1, default: 0.25)",
					"default":     0.25,
				},
			},
			"required": []string{"query", "collection_id"},
		},
	}, func(args map[string]any) (any, error) {
		query, _ := args["query"].(string)
		collectionID, _ := args["collection_id"].(string)
		topK := getIntArg(args, "top_k", 10)
		rerank := getBoolArg(args, "rerank", true)
		threshold := getFloatArg(args, "threshold", 0.25)

		results, err := eng.Search(query, collectionID, topK, rerank, threshold)
		if err != nil {
			return nil, err
		}

		return formatSearchResults(results), nil
	})

	server.RegisterTool(Tool{
		Name:        "index_document",
		Description: "Index a file into a Qdrant collection. The file will be chunked, embedded, and stored.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Path to the file to index",
				},
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Collection to index into (e.g., 'obsidian', 'notes')",
				},
				"metadata": map[string]any{
					"type":        "object",
					"description": "Optional metadata to attach to the chunks",
				},
			},
			"required": []string{"path", "collection_id"},
		},
	}, func(args map[string]any) (any, error) {
		path, _ := args["path"].(string)
		collectionID, _ := args["collection_id"].(string)
		metadata := getStringMapArg(args, "metadata")

		result, err := eng.IndexDocument(path, collectionID, metadata)
		if err != nil {
			return nil, err
		}

		return fmt.Sprintf("Indexed %d chunks into collection '%s'", result.ChunksIndexed, result.Collection), nil
	})

	server.RegisterTool(Tool{
		Name:        "index_text",
		Description: "Index raw text directly into a Qdrant collection.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text": map[string]any{
					"type":        "string",
					"description": "Text to index",
				},
				"collection_id": map[string]any{
					"type":        "string",
					"description": "Collection to index into",
				},
				"metadata": map[string]any{
					"type":        "object",
					"description": "Optional metadata to attach to the chunks",
				},
			},
			"required": []string{"text", "collection_id"},
		},
	}, func(args map[string]any) (any, error) {
		text, _ := args["text"].(string)
		collectionID, _ := args["collection_id"].(string)
		metadata := getStringMapArg(args, "metadata")

		result, err := eng.IndexText(text, collectionID, metadata)
		if err != nil {
			return nil, err
		}

		return fmt.Sprintf("Indexed %d chunks into collection '%s'", result.ChunksIndexed, result.Collection), nil
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
					"description": "Qdrant filter to match points for deletion",
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
		Description: "Check if all components (Qdrant, embedding, rerank) are healthy.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	}, func(args map[string]any) (any, error) {
		status := eng.HealthCheck()
		return formatHealthCheck(status), nil
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

func getFloatArg(args map[string]any, key string, defaultVal float64) float64 {
	if v, ok := args[key]; ok {
		if f, ok := v.(float64); ok {
			return f
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

func formatSearchResults(results []engine.SearchResult) string {
	if len(results) == 0 {
		return "No results found."
	}

	result := fmt.Sprintf("Found %d results:\n\n", len(results))
	for i, r := range results {
		result += fmt.Sprintf("--- Result %d (score: %.3f) ---\n", i+1, r.Score)
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
