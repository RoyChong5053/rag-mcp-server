package mcp

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/jobs"
)

// syncWait is how long a tool call blocks for a result before handing the
// caller a pollable job. On the GPU path most calls finish well inside this;
// on the CPU fallback the caller gets a job_id and an estimate instead of
// hanging past the MCP client's request timeout.
const syncWait = 10 * time.Second

// RegisterTools registers all RAG MCP tools with the server. jobsMgr is the
// shared background-task registry (same instance the dashboard uses), so a
// long operation has one identity no matter which surface started it.
func RegisterTools(server *Server, eng *engine.Engine, jobsMgr *jobs.Manager) {
	server.RegisterTool(Tool{
		Name: "search_memory",
		Description: "Search your persistent memory using semantic similarity. Returns relevant chunks from your knowledge base. " +
			"Omit collection_id to use the configured default collection; if qdrant is unreachable it automatically falls back to the vectra default, then the same-name vectra collection. " +
			"With no default configured it searches all enabled collections. " +
			"top_k, threshold and reranking default to server (WebUI) settings when omitted. " +
			"Long queries are truncated to the server's query_max_chars setting (head kept) before embedding. " +
			"Optionally restrict by metadata: pass metadata {\"key\":\"value\"} for exact matches on chunk metadata, " +
			"or a raw Qdrant filter for advanced clauses.",
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
					"description": "Optional: raw Qdrant filter (advanced). Takes precedence over metadata. Vectra supports match.value clauses only.",
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
		topK := getIntArg(args, "top_k", 0) // 0 = server default
		var threshold *float64
		if v, ok := args["threshold"]; ok {
			if f, ok := v.(float64); ok {
				threshold = &f
			}
		}
		filter := getFilterArg(args)

		return boundedRun(jobsMgr, jobs.KindSearch, estimateSearchSeconds(eng), syncWait, func() (any, error) {
			results, err := eng.SearchDefault(query, collectionID, topK, threshold, filter)
			if err != nil {
				return nil, err
			}
			return formatSearchResults(results), nil
		})
	})

	server.RegisterTool(Tool{
		Name: "store_memory",
		Description: "Vectorize and store text into a collection so it can be recalled later with search_memory. " +
			"Omit collection_id to write to the configured default collection, with automatic vectra failover (configured vectra default, then the same-name collection) when qdrant is unreachable. " +
			"The raw text is persisted on the server (under docs memory, by date) and indexed as a document, " +
			"so it can be browsed in the data bank and re-indexed.",
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

		return boundedRun(jobsMgr, jobs.KindStore, estimateStoreSeconds(eng, len(text)), syncWait, func() (any, error) {
			result, err := eng.StoreMemory(text, collectionID, metadata)
			if err != nil {
				return nil, err
			}
			return fmt.Sprintf("Stored %d chunks into collection '%s'", result.ChunksIndexed, result.Collection), nil
		})
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
		Description: "Check component health (qdrant, vectra, embedding, rerank) and report the active default backend.",
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
		Description: "Set management metadata for a collection: display name, description, tags, enabled flag, consumers, backend (qdrant|vectra).",
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
					"description": "Storage backend: qdrant or vectra. Empty follows the server default.",
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
		Name: "job_status",
		Description: "Check a background job returned by an earlier search_memory/store_memory call that exceeded the " +
			"synchronous window. Returns running progress, the final result once done, or the error. Poll with the " +
			"job_id after the estimate_seconds you were given; do not resubmit the original call.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"job_id": map[string]any{
					"type":        "string",
					"description": "Job id returned by search_memory/store_memory (e.g. job-7)",
				},
			},
			"required": []string{"job_id"},
		},
	}, func(args map[string]any) (any, error) {
		id, _ := args["job_id"].(string)
		if strings.TrimSpace(id) == "" {
			return nil, fmt.Errorf("job_id is required")
		}
		j := jobsMgr.Get(id)
		if j == nil {
			return nil, fmt.Errorf("unknown job %q (it may have been trimmed after completion)", id)
		}
		switch j.State {
		case jobs.StateDone:
			if j.Result != nil {
				return j.Result, nil
			}
			return fmt.Sprintf("Job %s done: indexed %d chunks into collection '%s'", j.ID, j.Chunks, j.Collection), nil
		case jobs.StateError:
			return nil, fmt.Errorf("job %s failed: %s", j.ID, j.Error)
		default:
			return formatJobPending(j), nil
		}
	})

	server.RegisterTool(Tool{
		Name:        "job_list",
		Description: "List recent background jobs (index jobs plus async search/store calls) with their state and progress.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	}, func(args map[string]any) (any, error) {
		all := jobsMgr.List()
		if len(all) == 0 {
			return "No background jobs.", nil
		}
		var b strings.Builder
		fmt.Fprintf(&b, "%d job(s):\n", len(all))
		for _, j := range all {
			switch j.Kind {
			case jobs.KindIndex:
				fmt.Fprintf(&b, "- %s [%s] %s/%s %s (%d/%d chunks)%s\n", j.ID, j.Kind, j.Collection, j.Backend, j.State, j.DoneChunks, j.TotalChunks, errSuffix(j))
			default:
				fmt.Fprintf(&b, "- %s [%s] %s%s\n", j.ID, j.Kind, j.State, errSuffix(j))
			}
		}
		return b.String(), nil
	})
}

// boundedRun submits fn to the shared registry and waits up to wait. On
// success it returns the result. If the work is still running it returns a
// structured pending payload the caller can poll. A failure is returned as-is.
func boundedRun(mgr *jobs.Manager, kind string, estimate int, wait time.Duration, fn func() (any, error)) (any, error) {
	job := mgr.Submit(kind, fn)
	done, finished := mgr.Await(job.ID, wait)
	if !finished {
		return formatJobPendingWithEstimate(done, estimate), nil
	}
	if done == nil {
		return nil, fmt.Errorf("job %s disappeared", job.ID)
	}
	if done.State == jobs.StateError {
		return nil, fmt.Errorf("%s", done.Error)
	}
	return done.Result, nil
}

// formatJobPending renders a pollable job handle. The estimate is a coarse
// prediction of the remaining wall time, refined over time.
func formatJobPendingWithEstimate(j *jobs.Job, estimate int) string {
	payload := map[string]any{
		"status":           "pending",
		"job_id":           j.ID,
		"kind":             j.Kind,
		"state":            j.State,
		"stage":            j.Stage,
		"estimate_seconds": estimate,
		"hint": fmt.Sprintf("Still running in the background. Call job_status with job_id=%q after about %d seconds (or do other work first). Do not resubmit — the work is already in progress.", j.ID, estimate),
	}
	b, _ := json.MarshalIndent(payload, "", "  ")
	return string(b)
}

func formatJobPending(j *jobs.Job) string {
	payload := map[string]any{
		"status": "pending",
		"job_id": j.ID,
		"kind":   j.Kind,
		"state":  j.State,
		"stage":  j.Stage,
	}
	if j.Kind == jobs.KindIndex {
		payload["done_chunks"] = j.DoneChunks
		payload["total_chunks"] = j.TotalChunks
	}
	b, _ := json.MarshalIndent(payload, "", "  ")
	return string(b)
}

func errSuffix(j *jobs.Job) string {
	if j.Error != "" {
		return ": " + j.Error
	}
	return ""
}

// estimateSearchSeconds is a coarse prediction of how long a search will take
// on the current compute path. Only used to tell the caller when to poll.
func estimateSearchSeconds(eng *engine.Engine) int {
	if eng.CPUComputeMode() {
		return 30
	}
	return 3
}

// estimateStoreSeconds is a coarse prediction for a store_memory call based on
// the text size and compute path.
func estimateStoreSeconds(eng *engine.Engine, textLen int) int {
	chunks := textLen/350 + 1 // effective chunk ~= size*(1-overlap) = 350
	if eng.CPUComputeMode() {
		if chunks <= 4 {
			return 15
		}
		return 15 + chunks*3
	}
	if chunks <= 32 {
		return 5
	}
	return 5 + chunks/4
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
// metadata.<key> matches (supported by both Qdrant and vectra).
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
