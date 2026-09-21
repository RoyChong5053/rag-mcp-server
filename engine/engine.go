package engine

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/RoyChong5053/rag-mcp-server/chunking"
)

// Engine is the core RAG engine
type Engine struct {
	qdrant    *QdrantClient
	embedding *EmbeddingClient
	rerank    *RerankClient
	config    *EngineConfig
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
}

// SearchResult represents a search result
type SearchResult struct {
	Text       string            `json:"text"`
	Score      float64           `json:"score"`
	Source     string            `json:"source,omitempty"`
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

// NewEngine creates a new RAG engine
func NewEngine(config *EngineConfig) *Engine {
	qdrant := NewQdrantClient(config.QdrantHost, config.QdrantPort)
	embedding := NewEmbeddingClient(config.OneAPIBaseURL, config.OneAPIBackupURL, config.EmbedModel, config.APIKey)
	rerank := NewRerankClient(config.OneAPIBaseURL, config.OneAPIBackupURL, config.RerankModel, config.APIKey, config.QueryMaxChars, config.DocMaxChars)

	return &Engine{
		qdrant:    qdrant,
		embedding: embedding,
		rerank:    rerank,
		config:    config,
	}
}

// Search performs a semantic search with optional reranking
func (e *Engine) Search(query string, collectionID string, topK int, useRerank bool, threshold float64) ([]SearchResult, error) {
	if collectionID == "" {
		return nil, fmt.Errorf("collection_id is required")
	}

	// Embed the query
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

	// Search Qdrant
	results, err := e.qdrant.Search(collectionID, queryVector, recallCount, threshold)
	if err != nil {
		return nil, fmt.Errorf("qdrant search failed: %w", err)
	}

	// Convert to SearchResult
	searchResults := make([]SearchResult, len(results))
	for i, r := range results {
		text, _ := r.Payload["text"].(string)
		source, _ := r.Payload["source"].(string)
		chunkIndex, _ := r.Payload["chunk_index"].(float64)
		metadata, _ := r.Payload["metadata"].(map[string]any)

		searchResults[i] = SearchResult{
			Text:       text,
			Score:      r.Score,
			Source:     source,
			ChunkIndex: int(chunkIndex),
			Metadata:   convertMetadata(metadata),
		}
	}

	// Apply reranking if enabled
	if useRerank && e.config.RerankEnabled && len(searchResults) > 1 {
		reranked, err := e.applyRerank(query, searchResults, topK)
		if err != nil {
			log.Printf("Rerank failed, falling back to vector order: %v", err)
		} else {
			searchResults = reranked
		}
	} else {
		// Just take top K
		if len(searchResults) > topK {
			searchResults = searchResults[:topK]
		}
	}

	return searchResults, nil
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

	return e.indexTextInternal(text, collectionID, metadata, fileName)
}

// IndexText indexes raw text into a Qdrant collection
func (e *Engine) IndexText(text string, collectionID string, metadata map[string]string) (*IndexResult, error) {
	return e.indexTextInternal(text, collectionID, metadata, "direct_text")
}

func (e *Engine) indexTextInternal(text string, collectionID string, metadata map[string]string, sourceName string) (*IndexResult, error) {
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
	points := make([]Point, len(chunks))
	for i, chunk := range chunks {
		payload := map[string]any{
			"text":        chunk,
			"source":      sourceName,
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
