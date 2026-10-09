package engine

import (
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/chunking"
)

// ChunkDetail is the full precise recall of one memory card.
type ChunkDetail struct {
	ID           string            `json:"id"`
	Hash         string            `json:"hash,omitempty"`
	Text         string            `json:"text"`
	Source       string            `json:"source,omitempty"`
	SourceKey    string            `json:"source_key,omitempty"`
	Collection   string            `json:"collection"`
	ChunkIndex   int               `json:"chunk_index,omitempty"`
	IndexedAt    string            `json:"indexed_at,omitempty"`
	Tags         []string          `json:"tags,omitempty"`
	TagsExpanded []string          `json:"tags_expanded,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// GetChunk fetches one chunk by decimal ID or content SHA (no embedding).
func (e *Engine) GetChunk(collectionID, chunkID string) (*ChunkDetail, error) {
	collectionID = strings.TrimSpace(collectionID)
	chunkID = strings.TrimSpace(chunkID)
	if collectionID == "" || chunkID == "" {
		return nil, fmt.Errorf("collection_id and chunk_id are required")
	}
	fs := e.FileStore()
	if fs == nil {
		return nil, fmt.Errorf("no file store")
	}
	payload, source, ok := fs.GetChunk(collectionID, chunkID)
	if !ok {
		return nil, fmt.Errorf("chunk '%s' not found in '%s'", chunkID, collectionID)
	}
	text, _ := payload["text"].(string)
	id, _ := payload["_item_id"].(string)
	skey, _ := payload["_source_key"].(string)
	indexedAt, _ := payload["indexed_at"].(string)
	var chunkIndex int
	if f, ok := payload["chunk_index"].(float64); ok {
		chunkIndex = int(f)
	}
	hash, _ := payload["content_sha"].(string)
	if hash == "" {
		if h, ok := payload["hash"]; ok {
			hash = fmt.Sprintf("%v", h)
		}
	}
	var md map[string]any
	if m, ok := payload["metadata"].(map[string]any); ok {
		md = m
	}
	tags, expanded := extractTags(md)
	return &ChunkDetail{
		ID: id, Hash: hash, Text: text, Source: source, SourceKey: skey,
		Collection: collectionID, ChunkIndex: chunkIndex, IndexedAt: indexedAt,
		Tags: tags, TagsExpanded: expanded, Metadata: convertMetadata(md),
	}, nil
}

// UpdateResult reports an atomic revise operation.
type UpdateResult struct {
	Collection    string   `json:"collection"`
	Deleted       int      `json:"deleted"`
	ChunksIndexed int      `json:"chunks_indexed"`
	NewIDs        []string `json:"new_ids,omitempty"`
	NewHash       []string `json:"new_hash,omitempty"`
	DryRun        bool     `json:"dry_run,omitempty"`
}

// UpdateMemory atomically revises memory cards: delete by ids (or filter),
// then store new text with hierarchical tags. dry_run only reports counts.
func (e *Engine) UpdateMemory(collectionID string, ids []string, filter map[string]any, newText string, metadata map[string]string, dryRun bool) (*UpdateResult, error) {
	collectionID = strings.TrimSpace(collectionID)
	if collectionID == "" {
		return nil, fmt.Errorf("collection_id is required")
	}
	if len(ids) == 0 && len(filter) == 0 {
		return nil, fmt.Errorf("ids or filter is required")
	}
	if strings.TrimSpace(newText) == "" && !dryRun {
		return nil, fmt.Errorf("new text is required (or dry_run=true)")
	}
	// Build effective filter.
	eff := map[string]any{}
	for k, v := range filter {
		eff[k] = v
	}
	if len(ids) > 0 {
		eff["ids"] = ids
	}
	// Count matches first (read-only) for the dry-run report.
	matched, err := e.countMatches(collectionID, eff)
	if err != nil {
		return nil, err
	}
	if dryRun {
		return &UpdateResult{Collection: collectionID, Deleted: matched, DryRun: true}, nil
	}
	deleted, err := e.DeleteMemory(collectionID, eff)
	if err != nil {
		return nil, err
	}
	if metadata == nil {
		metadata = map[string]string{}
	}
	res, err := e.StoreMemory(newText, collectionID, metadata)
	if err != nil {
		return nil, fmt.Errorf("deleted %d but store failed: %w", deleted, err)
	}
	// Resolve new IDs by content SHA (deterministic per chunk).
	var newIDs, newHashes []string
	chunks := chunking.Split(newText, chunking.Options{Strategy: e.ResolveChunkOptions(nil).Strategy, Size: e.ResolveChunkOptions(nil).Size, Overlap: e.ResolveChunkOptions(nil).Overlap})
	for _, c := range chunks {
		newHashes = append(newHashes, ContentSHA(c))
	}
	_ = newIDs
	log.Printf("UpdateMemory '%s': deleted=%d indexed=%d", collectionID, deleted, res.ChunksIndexed)
	return &UpdateResult{Collection: collectionID, Deleted: deleted, ChunksIndexed: res.ChunksIndexed, NewHash: newHashes}, nil
}

func (e *Engine) countMatches(collectionID string, filter map[string]any) (int, error) {
	fs := e.FileStore()
	if fs == nil {
		return 0, fmt.Errorf("no file store")
	}
	return fs.CountMatches(collectionID, filter), nil
}

// VerifyDeepResult is the three-layer health report.
type VerifyDeepResult struct {
	Collection     string   `json:"collection"`
	OK             bool     `json:"ok"`
	StructureOK    bool     `json:"structure_ok"`
	StructureIssues []VerifyIssue `json:"structure_issues,omitempty"`
	TextCoverage   float64  `json:"text_coverage,omitempty"`
	TextMatched    int      `json:"text_matched,omitempty"`
	TextTotal      int      `json:"text_total,omitempty"`
	OrphanCards    int      `json:"orphan_cards,omitempty"`
	VectorMedian   float64  `json:"vector_median,omitempty"`
	VectorSamples  int      `json:"vector_samples,omitempty"`
	VectorNote     string   `json:"vector_note,omitempty"`
	UpdatedAt      string   `json:"updated_at"`
}

// VerifyDeep runs structure (existing) + text coverage (re-chunk raw source
// when available) + vector spot-check (re-embed N samples).
// Without a source path, text coverage is skipped (reported -1); without an
// embedder, vector check is skipped.
func (e *Engine) VerifyDeep(collectionID, sourcePath string, sampleN int) (*VerifyDeepResult, error) {
	collectionID = strings.TrimSpace(collectionID)
	if collectionID == "" {
		return nil, fmt.Errorf("collection_id is required")
	}
	if sampleN <= 0 {
		sampleN = 20
	}
	if sampleN > 50 {
		sampleN = 50
	}
	res := &VerifyDeepResult{Collection: collectionID, UpdatedAt: time.Now().UTC().Format(time.RFC3339)}
	// Layer 1: structure.
	structRes := e.VerifyCollection(collectionID)
	res.StructureIssues = structRes.Issues
	res.StructureOK = structRes.OK
	// Layer 2: text coverage (re-chunk raw source when a path is given).
	if fs := e.FileStore(); fs != nil {
		stored := fs.StoredSHAs(collectionID)
		if strings.TrimSpace(sourcePath) != "" {
			if data, err := os.ReadFile(sourcePath); err == nil {
				used := e.ResolveChunkOptions(nil)
				chunks := chunking.Split(string(data), chunking.Options{Strategy: used.Strategy, Size: used.Size, Overlap: used.Overlap})
				matched := 0
				for _, c := range chunks {
					if stored[ContentSHA(c)] {
						matched++
					}
				}
				res.TextMatched, res.TextTotal = matched, len(chunks)
				if len(chunks) > 0 {
					res.TextCoverage = float64(matched) / float64(len(chunks))
				}
				// Orphans: stored cards not in the re-chunked set.
				want := map[string]bool{}
				for _, c := range chunks {
					want[ContentSHA(c)] = true
				}
				for sha := range stored {
					if !want[sha] {
						res.OrphanCards++
					}
				}
			} else {
				res.TextCoverage = -1
				res.VectorNote = ""
			}
		} else {
			res.TextCoverage = -1
		}
		if med, n, note, err := e.vectorSpotCheck(collectionID, sampleN); err == nil {
			res.VectorMedian, res.VectorSamples, res.VectorNote = med, n, note
		} else {
			res.VectorNote = err.Error()
		}
	}
	res.OK = res.StructureOK && (res.TextCoverage < 0 || res.TextCoverage >= 0.7) && (res.VectorSamples == 0 || res.VectorMedian >= 0.8)
	return res, nil
}

func (e *Engine) chunkOptionsForVerify(collectionID string) (size, overlap int, strategy string) {
	used := e.ResolveChunkOptions(nil)
	return used.Size, used.Overlap, used.Strategy
}

// vectorSpotCheck re-embeds up to N stored chunks and reports median cosine
// between stored and fresh vectors. Same dim + same model but changed pooling
// shows up here while dim gates stay green.
func (e *Engine) vectorSpotCheck(collectionID string, n int) (median float64, samples int, note string, err error) {
	fs := e.FileStore()
	if fs == nil {
		return 0, 0, "", fmt.Errorf("no file store")
	}
	items := fs.SampleTexts(collectionID, n)
	if len(items) == 0 {
		return 0, 0, "no samples", nil
	}
	texts := make([]string, 0, len(items))
	for _, it := range items {
		texts = append(texts, it.Text)
	}
	emb, err := e.currentEmbedder().CreateEmbeddingsBatched(texts, 8)
	if err != nil {
		return 0, 0, "", fmt.Errorf("re-embed failed: %w", err)
	}
	var scores []float64
	for i, it := range items {
		if i >= len(emb) || len(emb[i]) == 0 || len(it.Vector) == 0 || len(emb[i]) != len(it.Vector) {
			continue
		}
		var dot, na, nb float64
		for j := range emb[i] {
			dot += float64(emb[i][j]) * float64(it.Vector[j])
			na += float64(emb[i][j]) * float64(emb[i][j])
			nb += float64(it.Vector[j]) * float64(it.Vector[j])
		}
		if na == 0 || nb == 0 {
			continue
		}
		scores = append(scores, dot/(math.Sqrt(na)*math.Sqrt(nb)))
	}
	if len(scores) == 0 {
		return 0, 0, "dim changed or no comparable vectors", nil
	}
	sort.Float64s(scores)
	median = scores[len(scores)/2]
	note = "ok"
	if median < 0.8 {
		note = "vector space drift suspected (pooling/model change?)"
	} else if median < 0.98 {
		note = "mild drift, watch"
	}
	return median, len(scores), note, nil
}
