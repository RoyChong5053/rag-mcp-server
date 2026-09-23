package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/chunking"
	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// Engine is the core RAG engine
type Engine struct {
	stores         map[string]VectorStore
	defaultBackend string
	embedding      *EmbeddingClient
	rerank         *RerankClient
	config         *EngineConfig
	registry       *registry.Registry
	settings       *settings.Store
	docsDir        string
	memoryDir      string

	// healthMu guards downUntil. A backend marked down is skipped until the
	// deadline so a dead qdrant doesn't cost a full timeout on every call.
	healthMu  sync.Mutex
	downUntil map[string]time.Time
}

// EngineConfig holds configuration for the engine
type EngineConfig struct {
	QdrantHost      string
	QdrantPort      int
	VectraDir       string
	Backend         string
	DocsDir         string
	MemoryDir       string
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
	Backend    string            `json:"backend,omitempty"`
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
// A nil settings store falls back to one seeded from the engine config.
func NewEngine(config *EngineConfig, st *settings.Store) (*Engine, error) {
	embedding := NewEmbeddingClient(config.OneAPIBaseURL, config.OneAPIBackupURL, config.EmbedModel, config.APIKey)
	rerank := NewRerankClient(config.OneAPIBaseURL, config.OneAPIBackupURL, config.RerankModel, config.APIKey, config.QueryMaxChars, config.DocMaxChars)

	stores := map[string]VectorStore{
		BackendQdrant: NewQdrantClient(config.QdrantHost, config.QdrantPort),
	}
	vectraDir := config.VectraDir
	if vectraDir == "" {
		vectraDir = "Vectra"
	}
	stores[BackendVectra] = NewFileStore(vectraDir, config.DocsDir)

	backend := config.Backend
	if backend == "" {
		backend = BackendQdrant
	}
	if _, ok := stores[backend]; !ok {
		return nil, fmt.Errorf("unknown storage backend %q (want %s or %s)", backend, BackendQdrant, BackendVectra)
	}

	regPath := config.RegistryPath
	if regPath == "" {
		regPath = "collections.json"
	}
	reg, err := registry.New(regPath)
	if err != nil {
		return nil, fmt.Errorf("load registry: %w", err)
	}

	if st == nil {
		st = settings.New("", settings.Settings{
			DefaultTopK:      10,
			DefaultThreshold: 0.25,
			RerankEnabled:    config.RerankEnabled,
			RerankRecall:     config.RerankRecall,
			QueryMaxChars:    config.QueryMaxChars,
			DocMaxChars:      config.DocMaxChars,
			FailoverEnabled:  true,
		})
	}

	e := &Engine{
		stores:         stores,
		defaultBackend: backend,
		embedding:      embedding,
		rerank:         rerank,
		config:         config,
		registry:       reg,
		settings:       st,
		docsDir:        config.DocsDir,
		memoryDir:      config.MemoryDir,
		downUntil:      make(map[string]time.Time),
	}
	// Rerank truncation follows runtime settings without a restart.
	rerank.SetLimitsProvider(func() (int, int) {
		s := st.Get()
		return s.QueryMaxChars, s.DocMaxChars
	})
	return e, nil
}

// Settings exposes the runtime settings store for management surfaces.
func (e *Engine) Settings() *settings.Store {
	return e.settings
}

// DefaultBackend returns the configured global storage backend.
func (e *Engine) DefaultBackend() string {
	return e.defaultBackend
}

// storeFor resolves the backend for a collection: an explicit per-collection
// registry `backend` wins, otherwise the global default. A name without a
// registry entry (new collection) uses the default backend.
func (e *Engine) storeFor(name string) VectorStore {
	if name != "" {
		if en := e.registry.Get(name); en != nil && en.Backend != "" {
			if s, ok := e.stores[en.Backend]; ok {
				return s
			}
		}
	}
	return e.stores[e.defaultBackend]
}

// storeForBackend returns the store for an explicit backend name, falling back
// to the configured default when the name is unknown.
func (e *Engine) storeForBackend(backend string) VectorStore {
	if s, ok := e.stores[backend]; ok {
		return s
	}
	return e.stores[e.defaultBackend]
}

// ActiveBackend reports the runtime-selected default backend (settings
// active_backend), or the configured storage.backend when unset/invalid.
func (e *Engine) ActiveBackend() string {
	b := strings.TrimSpace(e.settings.Get().ActiveBackend)
	if b == BackendQdrant || b == BackendVectra {
		return b
	}
	return e.defaultBackend
}

// backendDown reports whether backend was marked unreachable recently.
func (e *Engine) backendDown(backend string) bool {
	e.healthMu.Lock()
	defer e.healthMu.Unlock()
	t, ok := e.downUntil[backend]
	return ok && time.Now().Before(t)
}

func (e *Engine) markBackendDown(backend string) {
	e.healthMu.Lock()
	defer e.healthMu.Unlock()
	if e.downUntil == nil {
		e.downUntil = make(map[string]time.Time)
	}
	e.downUntil[backend] = time.Now().Add(15 * time.Second)
}

func (e *Engine) markBackendUp(backend string) {
	e.healthMu.Lock()
	defer e.healthMu.Unlock()
	delete(e.downUntil, backend)
}

// backendUnreachable pings a backend and reports whether it is unreachable.
// A successful ping clears a prior down mark.
func (e *Engine) backendUnreachable(backend string) bool {
	s, ok := e.stores[backend]
	if !ok {
		return true
	}
	if err := s.Ping(); err != nil {
		if IsUnavailable(err) {
			e.markBackendDown(backend)
			return true
		}
		return false
	}
	e.markBackendUp(backend)
	return false
}

// CollectionExistsOn checks existence in one explicit backend, used to validate
// the per-backend default collections in the dashboard.
func (e *Engine) CollectionExistsOn(backend, name string) (bool, error) {
	s, ok := e.stores[backend]
	if !ok {
		return false, fmt.Errorf("unknown backend %q", backend)
	}
	return s.CollectionExists(name)
}

// storeCollection is one listed collection plus the backend it lives in.
type storeCollection struct {
	Name       string
	ChunkCount int
	Backend    string
}

// listAllStoresDetailed lists collections across every backend, tagging each
// with the backend that holds it. A name present in both stores appears twice
// (qdrant before vectra), letting callers pick the authoritative one.
func (e *Engine) listAllStoresDetailed() []storeCollection {
	var out []storeCollection
	for _, backend := range []string{BackendQdrant, BackendVectra} {
		s, ok := e.stores[backend]
		if !ok {
			continue
		}
		cols, err := s.ListCollections()
		if err != nil {
			log.Printf("ListCollections on %s failed: %v", backend, err)
			continue
		}
		for _, c := range cols {
			out = append(out, storeCollection{Name: c.Name, ChunkCount: c.ChunkCount, Backend: backend})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Name == out[j].Name {
			return out[i].Backend < out[j].Backend
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// listAllStores merges collection listings to one entry per name.
func (e *Engine) listAllStores() ([]StoreCollectionInfo, error) {
	seen := map[string]bool{}
	var out []StoreCollectionInfo
	for _, c := range e.listAllStoresDetailed() {
		if seen[c.Name] {
			continue
		}
		seen[c.Name] = true
		out = append(out, StoreCollectionInfo{Name: c.Name, ChunkCount: c.ChunkCount})
	}
	return out, nil
}

// CollectionExists reports whether a collection is present in Qdrant.
func (e *Engine) CollectionExists(name string) (bool, error) {
	return e.storeFor(name).CollectionExists(name)
}

// Registry exposes the collection registry for management tools.
func (e *Engine) Registry() *registry.Registry {
	return e.registry
}

// Search performs a semantic search with optional reranking (single collection).
func (e *Engine) Search(query string, collectionID string, topK int, useRerank bool, threshold float64) ([]SearchResult, error) {
	return e.SearchMulti(query, []string{collectionID}, topK, useRerank, threshold)
}

// scopeTarget is one (backend, collection) search/write target.
type scopeTarget struct {
	backend    string
	collection string
}

// SearchDefault runs the settings-driven search used by the search_memory tool.
// Callers pass only what they care about; omitted values come from the runtime
// settings store (WebUI).
//
// Scope resolution:
//   - explicit collection_id: routed by registry backend; if that backend is
//     unreachable and a same-name collection exists on the other backend, that
//     one is used (same-name failover), otherwise a loud error.
//   - omitted: the active backend's default collection, with an automatic
//     fallback to the vectra default when qdrant is unreachable and failover is
//     enabled. With no defaults configured it searches all enabled collections.
//
// A missing primary default is a loud error: silently widening to a global scan
// dilutes recall and looks like "RAG mysticism" later.
func (e *Engine) SearchDefault(query, collectionID string, topK int, threshold *float64) ([]SearchResult, error) {
	st := e.settings.Get()

	if topK <= 0 {
		topK = st.DefaultTopK
	}
	if topK <= 0 {
		topK = 10
	}
	thr := st.DefaultThreshold
	if threshold != nil {
		thr = *threshold
	}

	if scope := strings.TrimSpace(collectionID); scope != "" {
		return e.searchResolved(query, scope, topK, st.RerankEnabled, thr)
	}

	targets := e.defaultSearchTargets()
	if len(targets) == 0 {
		// No defaults anywhere: search all enabled collections.
		return e.SearchMulti(query, nil, topK, st.RerankEnabled, thr)
	}

	var lastErr error
	for _, t := range targets {
		if t.backend == BackendQdrant && e.backendDown(BackendQdrant) {
			lastErr = fmt.Errorf("backend %s is unavailable", BackendQdrant)
			continue
		}
		res, err := e.searchCollection(t.backend, t.collection, query, topK, st.RerankEnabled, thr)
		if err == nil {
			e.markBackendUp(t.backend)
			return res, nil
		}
		if IsUnavailable(err) {
			e.markBackendDown(t.backend)
			lastErr = err
			log.Printf("Default search: %s/'%s' unavailable, trying next target: %v", t.backend, t.collection, err)
			continue
		}
		return nil, err
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("no searchable default collection configured")
	}
	return nil, lastErr
}

// defaultSearchTargets returns the default collection targets for a caller that
// omitted collection_id, in priority order.
func (e *Engine) defaultSearchTargets() []scopeTarget {
	st := e.settings.Get()
	if e.ActiveBackend() == BackendVectra {
		if n := strings.TrimSpace(st.DefaultCollectionVectra); n != "" {
			return []scopeTarget{{backend: BackendVectra, collection: n}}
		}
		return nil
	}
	var out []scopeTarget
	if n := strings.TrimSpace(st.DefaultCollectionQdrant); n != "" {
		out = append(out, scopeTarget{backend: BackendQdrant, collection: n})
	}
	if st.FailoverEnabled {
		if n := strings.TrimSpace(st.DefaultCollectionVectra); n != "" {
			out = append(out, scopeTarget{backend: BackendVectra, collection: n})
		}
	}
	return out
}

// searchResolved searches one explicitly named collection, applying same-name
// failover when its backend is unreachable.
func (e *Engine) searchResolved(query, scope string, topK int, useRerank bool, thr float64) ([]SearchResult, error) {
	if en := e.registry.Get(scope); en != nil && !en.Enabled {
		return nil, fmt.Errorf("collection '%s' is disabled", scope)
	}
	backend := e.backendOf(scope)
	res, err := e.searchCollection(backend, scope, query, topK, useRerank, thr)
	if err == nil {
		e.markBackendUp(backend)
		return res, nil
	}
	if !IsUnavailable(err) {
		if !e.collectionExistsOn(backend, scope) {
			return nil, fmt.Errorf("collection '%s' not found (explicit collection_id)", scope)
		}
		return nil, err
	}
	e.markBackendDown(backend)
	other := otherBackend(backend)
	if other == "" || !e.collectionExistsOn(other, scope) {
		return nil, fmt.Errorf("collection '%s' is on %s which is unavailable; no same-name collection on %s to fall back to: %w", scope, backend, other, err)
	}
	log.Printf("Collection '%s': %s unavailable, using same-name collection on %s", scope, backend, other)
	return e.searchCollection(other, scope, query, topK, useRerank, thr)
}

// collectionExistsOn is a best-effort existence check that swallows errors.
func (e *Engine) collectionExistsOn(backend, name string) bool {
	s, ok := e.stores[backend]
	if !ok {
		return false
	}
	exists, err := s.CollectionExists(name)
	return err == nil && exists
}

// embedQuery embeds a single query string once.
func (e *Engine) embedQuery(query string) ([]float32, error) {
	embeddings, err := e.embedding.CreateEmbeddings([]string{query})
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	return embeddings[0], nil
}

// searchCollection embeds the query once and searches one (backend, collection),
// optionally reranking and trimming to topK.
func (e *Engine) searchCollection(backend, name, query string, topK int, useRerank bool, threshold float64) ([]SearchResult, error) {
	st := e.settings.Get()
	rerankOn := useRerank && st.RerankEnabled
	queryVector, err := e.embedQuery(query)
	if err != nil {
		return nil, err
	}
	store, ok := e.stores[backend]
	if !ok {
		return nil, fmt.Errorf("unknown backend %q", backend)
	}
	recallCount := topK
	if rerankOn {
		recallCount = st.RerankRecall
		if recallCount < topK {
			recallCount = topK
		}
	}
	raw, err := store.Search(name, queryVector, recallCount, threshold)
	if err != nil {
		return nil, err
	}
	merged := make([]SearchResult, 0, len(raw))
	for _, r := range raw {
		sr := qdrantToSearchResult(r, name)
		sr.Backend = backend
		merged = append(merged, sr)
	}
	if rerankOn && len(merged) > 1 {
		if reranked, err := e.applyRerank(query, merged, topK); err == nil {
			return reranked, nil
		} else {
			log.Printf("Rerank failed, falling back to vector order: %v", err)
		}
	}
	return trimResults(merged, topK), nil
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
	queryVector, err := e.embedQuery(query)
	if err != nil {
		return nil, err
	}

	// Determine recall count (more candidates if reranking).
	// Runtime settings take precedence so WebUI edits apply live.
	st := e.settings.Get()
	rerankOn := useRerank && st.RerankEnabled
	recallCount := topK
	if rerankOn {
		recallCount = st.RerankRecall
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
		backend := e.backendOf(name)
		results, err := e.storeFor(name).Search(name, queryVector, recallCount, threshold)
		if err != nil {
			if IsUnavailable(err) {
				e.markBackendDown(backend)
			}
			if explicit {
				return nil, fmt.Errorf("search failed on '%s' (%s): %w", name, backend, err)
			}
			log.Printf("Search skipped collection '%s': %v", name, err)
			continue
		}
		e.markBackendUp(backend)
		for _, r := range results {
			sr := qdrantToSearchResult(r, name)
			sr.Backend = backend
			merged = append(merged, sr)
		}
	}

	// Apply reranking if enabled
	if rerankOn && len(merged) > 1 {
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
			exists, err := e.storeFor(name).CollectionExists(name)
			if err != nil || !exists {
				log.Printf("Registry drift: '%s' enabled but missing in its store, skipped", name)
				continue
			}
			targets = append(targets, name)
		}
		return targets, false
	}
	// ...or every stored collection when the registry is empty.
	all, err := e.listAllStores()
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

func qdrantToSearchResult(r StoreSearchResult, collection string) SearchResult {
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

// SearchDebugResult pairs kept results with candidates the threshold killed.
// Killed items are fetched with threshold=0 over the same recall window,
// so tuners see exactly what a higher threshold would have kept.
type SearchDebugResult struct {
	Results []SearchResult `json:"results"`
	Killed  []SearchResult `json:"killed"`
}

// SearchDebug runs a single-collection search and reports threshold kills.
// It embeds once and searches twice (threshold, then 0): no rerank on the
// killed set, they are shown in raw vector order.
func (e *Engine) SearchDebug(query string, collection string, topK int, useRerank bool, threshold float64) (*SearchDebugResult, error) {
	queryVector, err := e.embedQuery(query)
	if err != nil {
		return nil, err
	}

	// Use the configured over-fetch so the recall test mirrors what
	// search_memory will actually pull before reranking. The rerank toggle
	// stays explicit here so operators can compare with and without it.
	st := e.settings.Get()
	recallCount := topK
	if useRerank && st.RerankRecall > recallCount {
		recallCount = st.RerankRecall
	}

	kept, err := e.storeFor(collection).Search(collection, queryVector, recallCount, threshold)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}
	results := make([]SearchResult, 0, len(kept))
	keptIDs := make(map[any]bool, len(kept))
	for _, r := range kept {
		keptIDs[r.ID] = true
		results = append(results, qdrantToSearchResult(r, collection))
	}

	var killed []SearchResult
	if threshold > 0 {
		all, err := e.storeFor(collection).Search(collection, queryVector, recallCount, 0)
		if err == nil {
			for _, r := range all {
				if !keptIDs[r.ID] {
					killed = append(killed, qdrantToSearchResult(r, collection))
				}
			}
		}
	}

	if useRerank && len(results) > 1 {
		if reranked, err := e.applyRerank(query, results, topK); err == nil {
			results = reranked
		} else {
			log.Printf("Rerank failed, falling back to vector order: %v", err)
			results = trimResults(results, topK)
		}
	} else {
		results = trimResults(results, topK)
	}
	return &SearchDebugResult{Results: results, Killed: killed}, nil
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

// ChunkOptions overrides the global chunking config for one index job.
// Zero values fall back to global config. Delimiters are intentionally
// fixed (parity with the ST toolchain) and not exposed per job.
type ChunkOptions struct {
	Size           int `json:"size"`
	OverlapPercent int `json:"overlap_percent"`
}

// ResolvedChunkOptions is the effective chunking used by a job.
type ResolvedChunkOptions struct {
	Size    int
	Overlap int
}

// ProgressFunc reports upsert progress: doneChunks of totalChunks.
type ProgressFunc func(doneChunks, totalChunks int)

// ResolveChunkOptions fills zero values from global config.
// Zero means "use global" (the dashboard sends 0 for defaults);
// there is currently no way to request literal 0% overlap per job.
func (e *Engine) ResolveChunkOptions(opts *ChunkOptions) ResolvedChunkOptions {
	size := e.config.ChunkSize
	overlapPct := e.config.OverlapPercent
	if opts != nil {
		if opts.Size > 0 {
			size = opts.Size
		}
		if opts.OverlapPercent > 0 && opts.OverlapPercent < 100 {
			overlapPct = opts.OverlapPercent
		}
	}
	return ResolvedChunkOptions{Size: size, Overlap: size * overlapPct / 100}
}

// Provenance records how a collection's vectors were computed.
// Stored in the registry: "how it was built", never query policy.
type Provenance struct {
	SourceFile   string
	SourceSHA256 string
	ChunkSize    int
	ChunkOverlap int
	EmbedModel   string
	Chunks       int
}

// IndexDocument indexes a file into a Qdrant collection
func (e *Engine) IndexDocument(path string, collectionID string, metadata map[string]string) (*IndexResult, error) {
	return e.IndexDocumentWithOptions(path, collectionID, metadata, nil, nil)
}

// IndexDocumentWithOptions indexes a file with per-job chunking and progress.
func (e *Engine) IndexDocumentWithOptions(path string, collectionID string, metadata map[string]string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, error) {
	return e.indexDocumentInternal(e.backendOf(collectionID), path, collectionID, metadata, opts, prog)
}

// indexDocumentInternal indexes a file into one explicit backend, bypassing
// registry routing. Used by the default-path failover so a write can target the
// vectra fallback even when the collection name has no registry entry.
func (e *Engine) indexDocumentInternal(backend, path, collectionID string, metadata map[string]string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, error) {
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

	sum := sha256.Sum256(data)
	prov := Provenance{
		SourceFile:   path,
		SourceSHA256: hex.EncodeToString(sum[:]),
		EmbedModel:   e.config.EmbedModel,
	}

	res, used, err := e.indexTextInternalOn(backend, text, collectionID, metadata, fileName, path, opts, prog)
	if err != nil {
		return nil, err
	}
	prov.Chunks = res.ChunksIndexed
	prov.ChunkSize, prov.ChunkOverlap = used.Size, used.Overlap
	e.recordIndex(collectionID, prov)
	return res, nil
}

// ReindexDocument drops all vectors and rebuilds from file with new options.
// Required when chunking changes: different splits produce different point
// IDs, so plain upsert would leave stale chunks behind. Registry metadata
// (display name, tags, consumers) is preserved.
func (e *Engine) ReindexDocument(path string, collectionID string, metadata map[string]string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, error) {
	exists, err := e.storeFor(collectionID).CollectionExists(collectionID)
	if err != nil {
		return nil, fmt.Errorf("failed to check collection: %w", err)
	}
	if exists {
		if err := e.storeFor(collectionID).DeleteCollection(collectionID); err != nil {
			return nil, fmt.Errorf("failed to purge collection: %w", err)
		}
		log.Printf("Purged collection '%s' for re-index", collectionID)
	}
	return e.IndexDocumentWithOptions(path, collectionID, metadata, opts, prog)
}

// IndexText indexes raw text into a Qdrant collection
func (e *Engine) IndexText(text string, collectionID string, metadata map[string]string) (*IndexResult, error) {
	return e.IndexTextWithOptions(text, collectionID, metadata, nil, nil)
}

// IndexTextWithOptions indexes raw text with per-job chunking and progress.
func (e *Engine) IndexTextWithOptions(text string, collectionID string, metadata map[string]string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, error) {
	res, used, err := e.indexTextInternal(text, collectionID, metadata, "direct_text", "direct_text", opts, prog)
	if err != nil {
		return nil, err
	}
	e.recordIndex(collectionID, Provenance{
		SourceFile:   "direct_text",
		ChunkSize:    used.Size,
		ChunkOverlap: used.Overlap,
		EmbedModel:   e.config.EmbedModel,
		Chunks:       res.ChunksIndexed,
	})
	return res, nil
}

// StoreMemory indexes ad-hoc text for later semantic recall (the MCP
// store_memory tool). It is the write-side counterpart of SearchDefault: an
// omitted collection_id resolves to the active backend's default collection,
// with qdrant→vectra failover when qdrant is unreachable. Having no default at
// all is a loud error rather than a silent write to a random bucket.
//
// The raw text is first persisted under <memoryDir>/<YYYY-MM-DD>/ so the file
// backend has a source document and Qdrant deployments keep an on-disk copy;
// indexing then runs through the normal document path.
func (e *Engine) StoreMemory(text, collectionID string, metadata map[string]string) (*IndexResult, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text is required")
	}
	path, err := e.writeMemoryRaw(text)
	if err != nil {
		return nil, fmt.Errorf("failed to persist memory raw file: %w", err)
	}

	if metadata == nil {
		metadata = make(map[string]string)
	}
	metadata["kind"] = "store_memory"

	scope := strings.TrimSpace(collectionID)
	if scope == "" {
		target, err := e.defaultWriteTarget()
		if err != nil {
			return nil, err
		}
		if metadata["collection"] == "" {
			metadata["collection"] = target.collection
		}
		return e.indexDocumentInternal(target.backend, path, target.collection, metadata, nil, nil)
	}

	metadata["collection"] = scope
	backend := e.backendOf(scope)
	res, err := e.indexDocumentInternal(backend, path, scope, metadata, nil, nil)
	if err == nil {
		e.markBackendUp(backend)
		return res, nil
	}
	if !IsUnavailable(err) {
		return nil, err
	}
	e.markBackendDown(backend)
	other := otherBackend(backend)
	if other == "" || !e.collectionExistsOn(other, scope) {
		return nil, fmt.Errorf("collection '%s' is on %s which is unavailable; no same-name collection on %s to fall back to: %w", scope, backend, other, err)
	}
	log.Printf("store_memory: %s unavailable, using same-name collection '%s' on %s", backend, scope, other)
	return e.indexDocumentInternal(other, path, scope, metadata, nil, nil)
}

// defaultWriteTarget resolves where an omitted collection_id write goes: the
// active backend's default collection, with qdrant→vectra failover.
func (e *Engine) defaultWriteTarget() (scopeTarget, error) {
	st := e.settings.Get()
	if e.ActiveBackend() == BackendVectra {
		name := strings.TrimSpace(st.DefaultCollectionVectra)
		if name == "" {
			return scopeTarget{}, fmt.Errorf("no collection_id given and no default_collection_vectra configured; pass collection_id or set a default in the management dashboard")
		}
		return scopeTarget{backend: BackendVectra, collection: name}, nil
	}
	name := strings.TrimSpace(st.DefaultCollectionQdrant)
	if name == "" {
		return scopeTarget{}, fmt.Errorf("no collection_id given and no default_collection_qdrant configured; pass collection_id or set a default in the management dashboard")
	}
	if st.FailoverEnabled {
		fallback := strings.TrimSpace(st.DefaultCollectionVectra)
		if e.backendDown(BackendQdrant) || e.backendUnreachable(BackendQdrant) {
			if fallback == "" {
				return scopeTarget{}, fmt.Errorf("qdrant is unreachable and no default_collection_vectra is configured for failover; pass collection_id or configure a vectra default")
			}
			log.Printf("Default write: qdrant unavailable, using vectra default '%s'", fallback)
			return scopeTarget{backend: BackendVectra, collection: fallback}, nil
		}
	}
	return scopeTarget{backend: BackendQdrant, collection: name}, nil
}

// writeMemoryRaw persists ad-hoc memory text under <memoryDir>/<date>/.
// The content hash is part of the filename so two writes in the same second
// never clobber each other.
func (e *Engine) writeMemoryRaw(text string) (string, error) {
	base := e.memoryDir
	if base == "" {
		if e.docsDir != "" {
			base = filepath.Join(e.docsDir, "memory")
		} else {
			base = filepath.Join("docs", "memory")
		}
	}
	now := time.Now()
	sum := sha256.Sum256([]byte(text))
	dir := filepath.Join(base, now.Format("2006-01-02"))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	name := fmt.Sprintf("%s-%s.md", now.Format("150405"), hex.EncodeToString(sum[:])[:8])
	path := filepath.Join(dir, name)
	return path, os.WriteFile(path, []byte(text), 0o644)
}

// EnsureCollection creates a collection in Qdrant if it does not exist yet, so
// a fresh write target (e.g. a global memory bucket) can be set up from the
// dashboard before anything is indexed into it.
func (e *Engine) EnsureCollection(name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("collection name is required")
	}
	exists, err := e.storeFor(name).CollectionExists(name)
	if err != nil {
		return fmt.Errorf("failed to check collection '%s': %w", name, err)
	}
	if exists {
		return nil
	}
	if err := e.storeFor(name).CreateCollection(name); err != nil {
		return fmt.Errorf("failed to create collection '%s': %w", name, err)
	}
	return nil
}

// PreviewChunks splits text without embedding: zero-token cost tuning.
// Returns total count plus up to maxSamples leading chunks.
func PreviewChunks(text string, size, overlap int, maxSamples int) (int, []string) {
	if size <= 0 {
		size = 500
	}
	delimiters := []string{"\n\n", "\n", " ", ""}
	effective := size - overlap
	if effective <= 0 {
		effective = size
	}
	chunks := chunking.SplitRecursive(text, effective, delimiters)
	if overlap > 0 {
		chunks = chunking.OverlapChunks(chunks, overlap)
	}
	samples := chunks
	if maxSamples > 0 && len(chunks) > maxSamples {
		samples = chunks[:maxSamples]
	}
	return len(chunks), samples
}

// recordIndex snapshots provenance in the registry.
// Best-effort: a registry write failure is logged, indexing already succeeded.
func (e *Engine) recordIndex(collectionID string, prov Provenance) {
	err := e.registry.Update(collectionID, func(en *registry.Entry) {
		if en.SourceFile == "" {
			en.SourceFile = prov.SourceFile
		}
		if prov.SourceSHA256 != "" {
			en.SourceSHA256 = prov.SourceSHA256
		}
		if prov.EmbedModel != "" {
			en.EmbedModel = prov.EmbedModel
		}
		en.ChunkCount = prov.Chunks
		en.Chunk = registry.ChunkConfig{Size: prov.ChunkSize, Overlap: prov.ChunkOverlap}
	})
	if err != nil {
		log.Printf("Registry update failed for '%s': %v", collectionID, err)
	}
}

// BackfillPayload tags every point in a collection with payload fields
// without re-embedding. Used to add source_file/indexed_at to chunks
// indexed before payload enrichment existed.
func (e *Engine) BackfillPayload(collectionID string, payload map[string]any) error {
	return e.storeFor(collectionID).SetPayload(collectionID, payload, nil)
}

// indexTextInternal routes to the collection's registry/default backend.
func (e *Engine) indexTextInternal(text string, collectionID string, metadata map[string]string, sourceName string, sourceFile string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, ResolvedChunkOptions, error) {
	return e.indexTextInternalOn(e.backendOf(collectionID), text, collectionID, metadata, sourceName, sourceFile, opts, prog)
}

func (e *Engine) indexTextInternalOn(backend string, text string, collectionID string, metadata map[string]string, sourceName string, sourceFile string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, ResolvedChunkOptions, error) {
	// Chunk the text (per-job options fall back to global config)
	delimiters := []string{"\n\n", "\n", " ", ""}
	used := e.ResolveChunkOptions(opts)
	effectiveChunkSize := used.Size - used.Overlap
	if effectiveChunkSize <= 0 {
		effectiveChunkSize = used.Size
	}

	chunks := chunking.SplitRecursive(text, effectiveChunkSize, delimiters)
	if used.Overlap > 0 {
		chunks = chunking.OverlapChunks(chunks, used.Overlap)
	}

	// Validate chunks
	problems := chunking.ValidateChunks(chunks, sourceName)
	if len(problems) > 0 {
		log.Printf("Chunk validation warnings for %s: %d problems", sourceName, len(problems))
	}

	if len(chunks) == 0 {
		return &IndexResult{ChunksIndexed: 0, Collection: collectionID}, used, nil
	}

	// Embed all chunks
	embeddings, err := e.embedding.CreateEmbeddings(chunks)
	if err != nil {
		return nil, used, fmt.Errorf("failed to embed chunks: %w", err)
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
	store := e.storeForBackend(backend)
	exists, err := store.CollectionExists(collectionID)
	if err != nil {
		return nil, used, fmt.Errorf("failed to check collection: %w", err)
	}
	if !exists {
		if err := store.CreateCollection(collectionID); err != nil {
			return nil, used, fmt.Errorf("failed to create collection: %w", err)
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
		if err := store.UpsertPoints(collectionID, points[start:end]); err != nil {
			return &IndexResult{ChunksIndexed: start, Collection: collectionID}, used, fmt.Errorf("failed to upsert points [%d:%d]: %w", start, end, err)
		}
		if prog != nil {
			prog(end, len(points))
		}
	}
	log.Printf("Indexed %d chunks into collection '%s' (%d upsert batches)", len(chunks), collectionID, (len(points)+upsertBatchSize-1)/upsertBatchSize)

	return &IndexResult{
		ChunksIndexed: len(chunks),
		Collection:    collectionID,
	}, used, nil
}

// DeleteMemory deletes points from a collection
func (e *Engine) DeleteMemory(collectionID string, filter map[string]any) (int, error) {
	store := e.storeFor(collectionID)
	// Get count before delete
	info, err := store.GetCollectionInfo(collectionID)
	if err != nil {
		return 0, fmt.Errorf("failed to get collection info: %w", err)
	}
	beforeCount := info.ChunkCount

	// Delete points
	if err := store.DeletePoints(collectionID, filter); err != nil {
		return 0, fmt.Errorf("failed to delete points: %w", err)
	}

	// Get count after delete
	info, err = store.GetCollectionInfo(collectionID)
	if err != nil {
		return 0, fmt.Errorf("failed to get collection info after delete: %w", err)
	}
	afterCount := info.ChunkCount

	return beforeCount - afterCount, nil
}

// CollectionDetail merges live store stats with registry metadata.
type CollectionDetail struct {
	ID           int      `json:"id"`
	Name         string   `json:"name"`
	Backend      string   `json:"backend"`
	ChunkCount   int      `json:"chunk_count"`
	Exists       bool     `json:"exists"`
	DisplayName  string   `json:"display_name,omitempty"`
	Description  string   `json:"description,omitempty"`
	Tags         []string `json:"tags,omitempty"`
	Enabled      bool     `json:"enabled"`
	Consumers    []string `json:"consumers,omitempty"`
	SourceFile   string   `json:"source_file,omitempty"`
	SourceSHA256 string   `json:"source_sha256,omitempty"`
	EmbedModel   string   `json:"embed_model,omitempty"`
	CreatedAt    string   `json:"created_at,omitempty"`
	UpdatedAt    string   `json:"updated_at,omitempty"`
	ChunkSize    int      `json:"chunk_size,omitempty"`
	ChunkOverlap int      `json:"chunk_overlap,omitempty"`
}

// DescribeCollections returns details for one collection, or all collections
// across every backend merged with registry entries when name is empty.
// Registry-only entries (orphans: meta without vectors) are included
// with Exists=false so drift is visible instead of silent.
func (e *Engine) DescribeCollections(name string) ([]CollectionDetail, error) {
	detailed := e.listAllStoresDetailed()
	reg := e.registry.List()

	byName := make(map[string]*CollectionDetail)
	for _, c := range detailed {
		d, ok := byName[c.Name]
		if !ok {
			byName[c.Name] = &CollectionDetail{Name: c.Name, Backend: c.Backend, ChunkCount: c.ChunkCount, Exists: true, Enabled: true}
			continue
		}
		// Also present in another backend: an explicit registry backend wins,
		// otherwise the first (qdrant) listing stays authoritative.
		if en := reg[c.Name]; en != nil && en.Backend != "" && c.Backend == en.Backend {
			d.Backend = c.Backend
			d.ChunkCount = c.ChunkCount
		}
	}
	for n, en := range reg {
		d, ok := byName[n]
		if !ok {
			d = &CollectionDetail{Name: n, Backend: en.Backend, Exists: false, Enabled: en.Enabled}
			byName[n] = d
		}
		d.DisplayName = en.DisplayName
		d.Description = en.Description
		d.Tags = en.Tags
		d.Enabled = en.Enabled
		d.Consumers = en.Consumers
		d.SourceFile = en.SourceFile
		d.SourceSHA256 = en.SourceSHA256
		d.EmbedModel = en.EmbedModel
		d.CreatedAt = en.CreatedAt
		d.UpdatedAt = en.UpdatedAt
		d.ChunkSize = en.Chunk.Size
		d.ChunkOverlap = en.Chunk.Overlap
		if en.Backend != "" {
			d.Backend = en.Backend
		}
	}
	// Fill any unset backend with the effective default so the UI never shows
	// a blank cell.
	for _, d := range byName {
		if d.Backend == "" {
			d.Backend = e.defaultBackend
		}
	}

	if name != "" {
		d, ok := byName[name]
		if !ok {
			return nil, fmt.Errorf("collection '%s' not found in any store or registry", name)
		}
		d.ID = CollectionNumID(d.Name)
		return []CollectionDetail{*d}, nil
	}
	out := make([]CollectionDetail, 0, len(byName))
	for _, d := range byName {
		d.ID = CollectionNumID(d.Name)
		out = append(out, *d)
	}
	// Deterministic order: map iteration above is random, which made the
	// dashboard table reshuffle on every reload/manage click. Sort by name so
	// rows stay put.
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// backendOf reports the effective backend for a collection (registry override
// or global default), used for display.
func (e *Engine) backendOf(name string) string {
	if en := e.registry.Get(name); en != nil && en.Backend != "" {
		return en.Backend
	}
	return e.defaultBackend
}

// CollectionNumID derives a stable, human-friendly numeric id from a
// collection name. It is cosmetic (a row label in the dashboard) so it needs
// no registry migration; it is not guaranteed unique.
func CollectionNumID(name string) int {
	return int(StringHash(name) % 100000)
}

// CollectionMetaUpdate holds optional metadata fields; nil means "leave unchanged".
type CollectionMetaUpdate struct {
	DisplayName *string
	Description *string
	Tags        *[]string
	Consumers   *[]string
	Enabled     *bool
	Backend     *string
}

// SetCollectionMeta updates registry metadata. Allowed for collections that
// don't exist yet (aspirational entry); callers are told.
func (e *Engine) SetCollectionMeta(name string, meta CollectionMetaUpdate) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("collection_id is required")
	}
	if meta.Backend != nil {
		b := strings.TrimSpace(*meta.Backend)
		if b != "" && b != BackendQdrant && b != BackendVectra {
			return fmt.Errorf("unknown backend %q (want %s or %s)", b, BackendQdrant, BackendVectra)
		}
		meta.Backend = &b
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
		if meta.Backend != nil {
			en.Backend = *meta.Backend
		}
	})
}

// DeleteCollection drops the collection from whichever backend holds it and
// removes its registry entry. Refuses without confirm=true: irreversible.
func (e *Engine) DeleteCollection(name string, confirm bool) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("collection_id is required")
	}
	if !confirm {
		return fmt.Errorf("refused: pass confirm=true to permanently delete collection '%s'", name)
	}
	store := e.storeFor(name)
	exists, err := store.CollectionExists(name)
	if err != nil {
		return fmt.Errorf("failed to check collection: %w", err)
	}
	if !exists {
		return fmt.Errorf("collection '%s' does not exist in backend '%s'", name, e.backendOf(name))
	}
	if err := store.DeleteCollection(name); err != nil {
		return fmt.Errorf("failed to delete collection: %w", err)
	}
	if err := e.registry.Delete(name); err != nil {
		log.Printf("Registry cleanup failed for deleted '%s': %v", name, err)
	}
	log.Printf("Deleted collection '%s'", name)
	return nil
}

// ListCollections lists all collections across every backend.
func (e *Engine) ListCollections() ([]CollectionInfo, error) {
	collections, err := e.listAllStores()
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

// HealthCheck checks all components. Every configured backend is reported with
// its own key (qdrant/vectra) so a disabled backend is visible too.
func (e *Engine) HealthCheck() map[string]string {
	status := make(map[string]string)

	for _, backend := range []string{BackendQdrant, BackendVectra} {
		s, ok := e.stores[backend]
		if !ok {
			continue
		}
		if err := s.Ping(); err != nil {
			status[backend] = fmt.Sprintf("error: %v", err)
		} else {
			status[backend] = "ok"
		}
	}
	status["active_backend"] = e.ActiveBackend()

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
