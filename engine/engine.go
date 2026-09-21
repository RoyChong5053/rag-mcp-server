package engine

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/chunking"
	"github.com/RoyChong5053/rag-mcp-server/registry"
)

// Engine is the core RAG engine
type Engine struct {
	qdrant    *QdrantClient
	embedding *EmbeddingClient
	rerank    *RerankClient
	config    *EngineConfig
	registry  *registry.Registry
}

// EngineConfig holds configuration for the engine
type EngineConfig struct {
	QdrantHost      string
	QdrantPort      int
	OneAPIBaseURL   string
	OneAPIBackupURL string
	EmbedModel      string
	RerankModel     string
	APIKey          string
	ChunkSize       int
	OverlapPercent  int
	RerankEnabled   bool
	RerankRecall    int
	QueryMaxChars   int
	DocMaxChars     int
	RegistryPath    string
}

// SearchResult represents a search result
type SearchResult struct {
	Text       string            `json:"text"`
	Score      float64           `json:"score"`
	Source     string            `json:"source,omitempty"`
	Collection string            `json:"collection,omitempty"`
	ChunkIndex int               `json:"chunk_index,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// IndexResult represents the result of indexing
type IndexResult struct {
	ChunksIndexed int    `json:"chunks_indexed"`
	Collection    string `json:"collection"`
}

// CollectionInfo represents collection information
type CollectionInfo struct {
	Name       string `json:"name"`
	ChunkCount int    `json:"chunk_count"`
}

// NewEngine creates a new RAG engine.
// A corrupt registry file is a loud error; a missing one starts empty.
func NewEngine(config *EngineConfig) (*Engine, error) {
	qdrant := NewQdrantClient(config.QdrantHost, config.QdrantPort)
	embedding := NewEmbeddingClient(config.OneAPIBaseURL, config.OneAPIBackupURL, config.EmbedModel, config.APIKey)
	rerank := NewRerankClient(config.OneAPIBaseURL, config.OneAPIBackupURL, config.RerankModel, config.APIKey, config.QueryMaxChars, config.DocMaxChars)

	regPath := config.RegistryPath
	if regPath == "" {
		regPath = "collections.json"
	}
	reg, err := registry.New(regPath)
	if err != nil {
		return nil, fmt.Errorf("load registry: %w", err)
	}

	return &Engine{
		qdrant:    qdrant,
		embedding: embedding,
		rerank:    rerank,
		config:    config,
		registry:  reg,
	}, nil
}

// Registry exposes the collection registry for management tools.
func (e *Engine) Registry() *registry.Registry {
	return e.registry
}

// Search performs a semantic search with optional reranking (single collection).
func (e *Engine) Search(query string, collectionID string, topK int, useRerank bool, threshold float64) ([]SearchResult, error) {
	return e.SearchMulti(query, []string{collectionID}, topK, useRerank, threshold)
}

// SearchMulti searches across several collections and merges the results.
// An empty collections list means "all enabled collections" (registry), or
// every Qdrant collection when the registry is empty (backward compatible).
// Disabled collections are always skipped. Explicitly requested collections
// that don't exist are a loud error; registry drift is logged and skipped.
func (e *Engine) SearchMulti(query string, collections []string, topK int, useRerank bool, threshold float64) ([]SearchResult, error) {
	targets, explicit := e.resolveTargets(collections)
	if len(targets) == 0 {
		return nil, fmt.Errorf("no collections to search (all disabled or none exist)")
	}

	// Embed the query once
	embeddings, err := e.embedding.CreateEmbeddings([]string{query})
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	queryVector := embeddings[0]

	// Determine recall count (more candidates if reranking)
	recallCount := topK
	if useRerank && e.config.RerankEnabled {
		recallCount = e.config.RerankRecall
		if recallCount < topK {
			recallCount = topK
		}
	}

	// Search each collection, tag results, merge
	var merged []SearchResult
	for _, name := range targets {
		if explicit {
			if en := e.registry.Get(name); en != nil && !en.Enabled {
				return nil, fmt.Errorf("collection '%s' is disabled", name)
			}
		}
		results, err := e.qdrant.Search(name, queryVector, recallCount, threshold)
		if err != nil {
			if explicit {
				return nil, fmt.Errorf("qdrant search failed on '%s': %w", name, err)
			}
			log.Printf("Search skipped collection '%s': %v", name, err)
			continue
		}
		for _, r := range results {
			merged = append(merged, qdrantToSearchResult(r, name))
		}
	}

	// Apply reranking if enabled
	if useRerank && e.config.RerankEnabled && len(merged) > 1 {
		reranked, err := e.applyRerank(query, merged, topK)
		if err != nil {
			log.Printf("Rerank failed, falling back to vector order: %v", err)
			return trimResults(merged, topK), nil
		}
		return reranked, nil
	}
	return trimResults(merged, topK), nil
}

// resolveTargets maps a requested collection list to searchable names.
// Returns the targets and whether the caller explicitly named them.
func (e *Engine) resolveTargets(collections []string) ([]string, bool) {
	var req []string
	for _, c := range collections {
		if c = strings.TrimSpace(c); c != "" {
			req = append(req, c)
		}
	}
	if len(req) > 0 {
		return req, true
	}
	// No explicit scope: all enabled registry entries...
	if enabled := e.registry.EnabledNames(); len(enabled) > 0 {
		var targets []string
		for _, name := range enabled {
			exists, err := e.qdrant.CollectionExists(name)
			if err != nil || !exists {
				log.Printf("Registry drift: '%s' enabled but missing in Qdrant, skipped", name)
				continue
			}
			targets = append(targets, name)
		}
		return targets, false
	}
	// ...or every Qdrant collection when the registry is empty.
	all, err := e.qdrant.ListCollections()
	if err != nil {
		log.Printf("ListCollections failed: %v", err)
		return nil, false
	}
	targets := make([]string, 0, len(all))
	for _, c := range all {
		targets = append(targets, c.Name)
	}
	return targets, false
}

func qdrantToSearchResult(r QdrantSearchResult, collection string) SearchResult {
	text, _ := r.Payload["text"].(string)
	source, _ := r.Payload["source"].(string)
	chunkIndex, _ := r.Payload["chunk_index"].(float64)
	metadata, _ := r.Payload["metadata"].(map[string]any)

	return SearchResult{
		Text:       text,
		Score:      r.Score,
		Source:     source,
		Collection: collection,
		ChunkIndex: int(chunkIndex),
		Metadata:   convertMetadata(metadata),
	}
}

func trimResults(results []SearchResult, topK int) []SearchResult {
	if len(results) > topK {
		return results[:topK]
	}
	return results
}

func (e *Engine) applyRerank(query string, results []SearchResult, topK int) ([]SearchResult, error) {
	// Extract texts for reranking
	texts := make([]string, len(results))
	for i, r := range results {
		texts[i] = r.Text
	}

	// Call rerank
	rerankResults, err := e.rerank.Rerank(query, texts, topK)
	if err != nil {
		return nil, err
	}

	// Reorder results based on rerank scores
	reranked := make([]SearchResult, 0, topK)
	for _, rr := range rerankResults {
		if rr.Index < len(results) {
			result := results[rr.Index]
			result.Score = rr.Score
			reranked = append(reranked, result)
		}
	}

	return reranked, nil
}

// IndexDocument indexes a file into a Qdrant collection
func (e *Engine) IndexDocument(path string, collectionID string, metadata map[string]string) (*IndexResult, error) {
	// Read file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	text := string(data)
	fileName := filepath.Base(path)
	if metadata == nil {
		metadata = make(map[string]string)
	}
	metadata["source"] = path

	res, err := e.indexTextInternal(text, collectionID, metadata, fileName, path)
	if err != nil {
		return nil, err
	}
	e.recordIndex(collectionID, path, res.ChunksIndexed)
	return res, nil
}

// IndexText indexes raw text into a Qdrant collection
func (e *Engine) IndexText(text string, collectionID string, metadata map[string]string) (*IndexResult, error) {
	res, err := e.indexTextInternal(text, collectionID, metadata, "direct_text", "direct_text")
	if err != nil {
		return nil, err
	}
	e.recordIndex(collectionID, "direct_text", res.ChunksIndexed)
	return res, nil
}

// recordIndex snapshots management metadata in the registry.
// Best-effort: a registry write failure is logged, indexing already succeeded.
func (e *Engine) recordIndex(collectionID, sourceFile string, chunks int) {
	err := e.registry.Update(collectionID, func(en *registry.Entry) {
		if en.SourceFile == "" {
			en.SourceFile = sourceFile
		}
		en.ChunkCount = chunks
		en.Chunk = registry.ChunkConfig{Size: e.config.ChunkSize, Overlap: e.config.ChunkSize * e.config.OverlapPercent / 100}
	})
	if err != nil {
		log.Printf("Registry update failed for '%s': %v", collectionID, err)
	}
}

// BackfillPayload tags every point in a collection with payload fields
// without re-embedding. Used to add source_file/indexed_at to chunks
// indexed before payload enrichment existed.
func (e *Engine) BackfillPayload(collectionID string, payload map[string]any) error {
	return e.qdrant.SetPayload(collectionID, payload, nil)
}

func (e *Engine) indexTextInternal(text string, collectionID string, metadata map[string]string, sourceName string, sourceFile string) (*IndexResult, error) {
	// Chunk the text
	delimiters := []string{"\n\n", "\n", " ", ""}
	overlapSize := e.config.ChunkSize * e.config.OverlapPercent / 100
	effectiveChunkSize := e.config.ChunkSize
	if overlapSize > 0 {
		effectiveChunkSize = e.config.ChunkSize - overlapSize
	}

	chunks := chunking.SplitRecursive(text, effectiveChunkSize, delimiters)
	if overlapSize > 0 {
		chunks = chunking.OverlapChunks(chunks, overlapSize)
	}

	// Validate chunks
	problems := chunking.ValidateChunks(chunks, sourceName)
	if len(problems) > 0 {
		log.Printf("Chunk validation warnings for %s: %d problems", sourceName, len(problems))
	}

	if len(chunks) == 0 {
		return &IndexResult{ChunksIndexed: 0, Collection: collectionID}, nil
	}

	// Embed all chunks
	embeddings, err := e.embedding.CreateEmbeddings(chunks)
	if err != nil {
		return nil, fmt.Errorf("failed to embed chunks: %w", err)
	}

	// Create Qdrant points with deterministic uint IDs:
	// high 32 bits = collection hash, low 32 bits = chunk hash.
	// Re-indexing the same file upserts the same IDs (idempotent).
	colHash := uint64(StringHash(collectionID)) << 32
	indexedAt := time.Now().UTC().Format(time.RFC3339)
	points := make([]Point, len(chunks))
	for i, chunk := range chunks {
		payload := map[string]any{
			"text":        chunk,
			"source":      sourceName,
			"source_file": sourceFile,
			"indexed_at":  indexedAt,
			"chunk_index": i,
			"hash":        StringHash(chunk),
		}
		if metadata != nil {
			payload["metadata"] = metadata
		}

		points[i] = Point{
			ID:      colHash | uint64(StringHash(chunk)),
			Payload: payload,
			Vector:  embeddings[i],
		}
	}

	// Ensure collection exists
	exists, err := e.qdrant.CollectionExists(collectionID)
	if err != nil {
		return nil, fmt.Errorf("failed to check collection: %w", err)
	}
	if !exists {
		if err := e.qdrant.CreateCollection(collectionID); err != nil {
			return nil, fmt.Errorf("failed to create collection: %w", err)
		}
	}

	// Upsert points in batches so large documents don't produce
	// a single oversized HTTP request to Qdrant
	const upsertBatchSize = 100
	for start := 0; start < len(points); start += upsertBatchSize {
		end := start + upsertBatchSize
		if end > len(points) {
			end = len(points)
		}
		if err := e.qdrant.UpsertPoints(collectionID, points[start:end]); err != nil {
			return nil, fmt.Errorf("failed to upsert points [%d:%d]: %w", start, end, err)
		}
	}
	log.Printf("Indexed %d chunks into collection '%s' (%d upsert batches)", len(chunks), collectionID, (len(points)+upsertBatchSize-1)/upsertBatchSize)

	return &IndexResult{
		ChunksIndexed: len(chunks),
		Collection:    collectionID,
	}, nil
}

// DeleteMemory deletes points from a collection
func (e *Engine) DeleteMemory(collectionID string, filter map[string]any) (int, error) {
	// Get count before delete
	info, err := e.qdrant.GetCollectionInfo(collectionID)
	if err != nil {
		return 0, fmt.Errorf("failed to get collection info: %w", err)
	}
	beforeCount := info.ChunkCount

	// Delete points
	if err := e.qdrant.DeletePoints(collectionID, filter); err != nil {
		return 0, fmt.Errorf("failed to delete points: %w", err)
	}

	// Get count after delete
	info, err = e.qdrant.GetCollectionInfo(collectionID)
	if err != nil {
		return 0, fmt.Errorf("failed to get collection info after delete: %w", err)
	}
	afterCount := info.ChunkCount

	return beforeCount - afterCount, nil
}

// CollectionDetail merges live Qdrant stats with registry metadata.
type CollectionDetail struct {
	Name         string   `json:"name"`
	ChunkCount   int      `json:"chunk_count"`
	Exists       bool     `json:"exists"`
	DisplayName  string   `json:"display_name,omitempty"`
	Description  string   `json:"description,omitempty"`
	Tags         []string `json:"tags,omitempty"`
	Enabled      bool     `json:"enabled"`
	Consumers    []string `json:"consumers,omitempty"`
	SourceFile   string   `json:"source_file,omitempty"`
	CreatedAt    string   `json:"created_at,omitempty"`
	UpdatedAt    string   `json:"updated_at,omitempty"`
	ChunkSize    int      `json:"chunk_size,omitempty"`
	ChunkOverlap int      `json:"chunk_overlap,omitempty"`
}

// DescribeCollections returns details for one collection, or all Qdrant
// collections merged with registry entries when name is empty.
// Registry-only entries (orphans: meta without vectors) are included
// with Exists=false so drift is visible instead of silent.
func (e *Engine) DescribeCollections(name string) ([]CollectionDetail, error) {
	all, err := e.qdrant.ListCollections()
	if err != nil {
		return nil, fmt.Errorf("failed to list collections: %w", err)
	}
	reg := e.registry.List()

	byName := make(map[string]*CollectionDetail)
	for _, c := range all {
		byName[c.Name] = &CollectionDetail{Name: c.Name, ChunkCount: c.ChunkCount, Exists: true, Enabled: true}
	}
	for n, en := range reg {
		d, ok := byName[n]
		if !ok {
			d = &CollectionDetail{Name: n, Exists: false, Enabled: en.Enabled}
			byName[n] = d
		}
		d.DisplayName = en.DisplayName
		d.Description = en.Description
		d.Tags = en.Tags
		d.Enabled = en.Enabled
		d.Consumers = en.Consumers
		d.SourceFile = en.SourceFile
		d.CreatedAt = en.CreatedAt
		d.UpdatedAt = en.UpdatedAt
		d.ChunkSize = en.Chunk.Size
		d.ChunkOverlap = en.Chunk.Overlap
	}

	if name != "" {
		d, ok := byName[name]
		if !ok {
			return nil, fmt.Errorf("collection '%s' not found in Qdrant or registry", name)
		}
		return []CollectionDetail{*d}, nil
	}
	out := make([]CollectionDetail, 0, len(byName))
	for _, d := range byName {
		out = append(out, *d)
	}
	return out, nil
}

// CollectionMetaUpdate holds optional metadata fields; nil means "leave unchanged".
type CollectionMetaUpdate struct {
	DisplayName *string
	Description *string
	Tags        *[]string
	Consumers   *[]string
	Enabled     *bool
}

// SetCollectionMeta updates registry metadata. Allowed for collections that
// don't exist in Qdrant yet (aspirational entry); callers are told.
func (e *Engine) SetCollectionMeta(name string, meta CollectionMetaUpdate) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("collection_id is required")
	}
	return e.registry.Update(name, func(en *registry.Entry) {
		if meta.DisplayName != nil {
			en.DisplayName = *meta.DisplayName
		}
		if meta.Description != nil {
			en.Description = *meta.Description
		}
		if meta.Tags != nil {
			en.Tags = *meta.Tags
		}
		if meta.Consumers != nil {
			en.Consumers = *meta.Consumers
		}
		if meta.Enabled != nil {
			en.Enabled = *meta.Enabled
		}
	})
}

// DeleteCollection drops the Qdrant collection and its registry entry.
// Refuses without confirm=true: this is irreversible.
func (e *Engine) DeleteCollection(name string, confirm bool) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("collection_id is required")
	}
	if !confirm {
		return fmt.Errorf("refused: pass confirm=true to permanently delete collection '%s'", name)
	}
	exists, err := e.qdrant.CollectionExists(name)
	if err != nil {
		return fmt.Errorf("failed to check collection: %w", err)
	}
	if !exists {
		return fmt.Errorf("collection '%s' does not exist in Qdrant", name)
	}
	if err := e.qdrant.DeleteCollection(name); err != nil {
		return fmt.Errorf("failed to delete collection: %w", err)
	}
	if err := e.registry.Delete(name); err != nil {
		log.Printf("Registry cleanup failed for deleted '%s': %v", name, err)
	}
	log.Printf("Deleted collection '%s'", name)
	return nil
}

// ListCollections lists all collections
func (e *Engine) ListCollections() ([]CollectionInfo, error) {
	collections, err := e.qdrant.ListCollections()
	if err != nil {
		return nil, fmt.Errorf("failed to list collections: %w", err)
	}

	result := make([]CollectionInfo, len(collections))
	for i, c := range collections {
		result[i] = CollectionInfo{
			Name:       c.Name,
			ChunkCount: c.ChunkCount,
		}
	}

	return result, nil
}

// HealthCheck checks all components
func (e *Engine) HealthCheck() map[string]string {
	status := make(map[string]string)

	// Check Qdrant
	if err := e.qdrant.Ping(); err != nil {
		status["qdrant"] = fmt.Sprintf("error: %v", err)
	} else {
		status["qdrant"] = "ok"
	}

	// Check embedding
	if err := e.embedding.Ping(); err != nil {
		status["embedding"] = fmt.Sprintf("error: %v", err)
	} else {
		status["embedding"] = "ok"
	}

	// Check rerank
	if err := e.rerank.Ping(); err != nil {
		status["rerank"] = fmt.Sprintf("error: %v", err)
	} else {
		status["rerank"] = "ok"
	}

	return status
}

// convertMetadata converts map[string]any to map[string]string
func convertMetadata(m map[string]any) map[string]string {
	if m == nil {
		return nil
	}
	result := make(map[string]string)
	for k, v := range m {
		result[k] = fmt.Sprintf("%v", v)
	}
	return result
}

// ensureCollectionPrefix adds "rag_" prefix if not already present
func ensureCollectionPrefix(name string) string {
	if !strings.HasPrefix(name, "rag_") {
		return "rag_" + name
	}
	return name
}
