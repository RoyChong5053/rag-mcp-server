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
	"sync/atomic"
	"time"
	"unicode/utf8"

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

	// observedEmbedDim records the vector length the active embedding model
	// actually returned on the last successful embed. It is the source of truth
	// for collection creation and provenance so a provider/model switch whose
	// dimension differs from the declared config self-heals instead of
	// hard-failing inside the store.
	observedEmbedDim atomic.Int32

	// embedMu/embedCache lazily build one EmbeddingClient per configured
	// provider so switching active_embed_provider does not rebuild the world.
	embedMu    sync.Mutex
	embedCache map[string]*EmbeddingClient
	docsDir    string
	memoryDir  string

	// healthMu guards downUntil. A backend marked down is skipped until the
	// deadline so a dead qdrant doesn't cost a full timeout on every call.
	healthMu  sync.Mutex
	downUntil map[string]time.Time
}

// EngineConfig holds configuration for the engine
type EngineConfig struct {
	VectraDir       string
	Backend         string
	DocsDir         string
	MemoryDir       string
	OneAPIBaseURL   string
	OneAPIBackupURL string
	EmbedModel      string
	EmbedProvider   string
	VectorDim       int
	VectorDistance  string
	RerankModel     string
	APIKey          string
	// EmbedTimeout/RerankTimeout are the total per-call budgets for one-api.
	// They must exceed one-api's own channel-failover window; rag-mcp never
	// cancels mid-failover. 0 = defaultOneAPITimeout (180s).
	EmbedTimeout  time.Duration
	RerankTimeout time.Duration
	// SearchBudget bounds one whole search (embed + stores + rerank) so the
	// server answers before the caller's outer fail-safe. Checked only at
	// stage boundaries. 0 = 170s.
	SearchBudget       time.Duration
	ChunkSize      int
	OverlapPercent int
	ChunkStrategy  string
	RerankEnabled  bool
	RerankRecall   int
	QueryMaxChars  int
	DocMaxChars    int
	RegistryPath   string
	// HistoryPath is the append-only JSONL audit of every index/reindex/
	// upload/delete so a raw file stays traceable to its collections even
	// after the collection itself is deleted. Empty disables history.
	HistoryPath string
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
	if config.EmbedProvider == "" {
		config.EmbedProvider = "one-api"
	}
	if config.VectorDim == 0 {
		config.VectorDim = 1024
	}
	if config.VectorDistance == "" {
		config.VectorDistance = "Cosine"
	}
	embedding := NewEmbeddingClient(config.OneAPIBaseURL, config.OneAPIBackupURL, config.EmbedModel, config.APIKey)
	rerank := NewRerankClient(config.OneAPIBaseURL, config.OneAPIBackupURL, config.RerankModel, config.APIKey, config.QueryMaxChars, config.DocMaxChars)
	embedding.SetTimeout(config.EmbedTimeout)
	rerank.SetTimeout(config.RerankTimeout)

	vectraDir := config.VectraDir
	if vectraDir == "" {
		vectraDir = "Vectra"
	}
	stores := map[string]VectorStore{
		BackendVectra: NewFileStore(vectraDir, config.DocsDir),
	}

	// Vectra-only: any configured backend (including legacy "qdrant") maps here.
	backend := BackendVectra

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
		embedCache:     map[string]*EmbeddingClient{},
		docsDir:        config.DocsDir,
		memoryDir:      config.MemoryDir,
		downUntil:      make(map[string]time.Time),
	}
	// Rerank truncation follows runtime settings without a restart.
	rerank.SetLimitsProvider(func() (int, int) {
		s := st.Get()
		return s.QueryMaxChars, s.DocMaxChars
	})
	// Embedding concurrency follows the compute mode: parallel on the GPU path,
	// serial on the CPU fallback so parallel searches/jobs cannot stampede it.
	embedding.SetConcurrencyProvider(func() int {
		if e.usesCPUCompute() {
			return embeddingConcurrencyCPU
		}
		return embeddingConcurrencyGPU
	})
	return e, nil
}

// PrewarmVectra loads file-backed indexes into memory so the first search after
// restart does not pay multi-second JSON parses. Safe to call in the background.
func (e *Engine) PrewarmVectra() {
	fs, ok := e.stores[BackendVectra].(*FileStore)
	if !ok {
		return
	}
	if err := fs.PrewarmAll(); err != nil {
		log.Printf("vectra prewarm: %v", err)
	}
}

// Settings exposes the runtime settings store for management surfaces.
func (e *Engine) Settings() *settings.Store {
	return e.settings
}

// DefaultBackend returns the configured global storage backend.
func (e *Engine) DefaultBackend() string {
	return e.defaultBackend
}

// storeFor resolves the store for a collection. The server is vectra-only:
// every name maps to the single file store regardless of legacy registry
// backend values.
func (e *Engine) storeFor(name string) VectorStore {
	return e.stores[BackendVectra]
}

// storeForBackend returns the single file store for any backend name
// (legacy callers may still pass "qdrant"; it is normalized to vectra).
func (e *Engine) storeForBackend(backend string) VectorStore {
	return e.stores[BackendVectra]
}

// FileStore exposes the underlying file backend for vault operations.
func (e *Engine) FileStore() *FileStore {
	if fs, ok := e.stores[BackendVectra].(*FileStore); ok {
		return fs
	}
	return nil
}

// ActiveBackend always reports vectra (legacy settings values map here).
func (e *Engine) ActiveBackend() string {
	return BackendVectra
}

// backendDownTTL is how long a failed backend stays marked down. Long enough
// that a dead host costs one probe instead of one full timeout per request,
// short enough that recovery is noticed promptly (the next request after
// expiry re-probes with the fast dial budget).
const backendDownTTL = 60 * time.Second

// Embedding compute-mode constants. The local-file backend means the
// GPU host is often off, so callers can opt into the CPU fallback batch cap.
// No health probe gates this: the CPU path is derived from settings.
const (
	embeddingConcurrencyGPU = 4
	embeddingConcurrencyCPU = 1
	embedBatchSizeGPU       = 32
	embedBatchSizeCPU       = 4
)

// usesCPUCompute is the cheap (no probe) compute-mode check used by the
// embedding limiter on every request. Vectra-only servers default to the CPU
// batch cap unless the operator raises embed_batch_size.
func (e *Engine) usesCPUCompute() bool {
	return true
}

// CPUComputeMode reports the compute class (true = CPU fallback).
func (e *Engine) CPUComputeMode() bool { return e.usesCPUCompute() }

// effectiveEmbedBatchSize resolves the index-time embedding batch size from the
// compute mode: the GPU path uses settings.embed_batch_size (0 => 32), while
// the CPU fallback is capped to embedBatchSizeCPU. This removes the manual bs
// flip when the LOQ powers off.
func (e *Engine) effectiveEmbedBatchSize(cpuMode bool) int {
	bs := e.settings.Get().EmbedBatchSize
	if bs <= 0 {
		bs = embedBatchSizeGPU
	}
	if cpuMode && bs > embedBatchSizeCPU {
		return embedBatchSizeCPU
	}
	return bs
}

// backendDown reports whether backend was marked down by a real failed call
// recently. It is BADGE STATE ONLY: the query path never uses it to refuse or
// skip a target, so a stale mark can never block a working backend.
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
	e.downUntil[backend] = time.Now().Add(backendDownTTL)
}

func (e *Engine) markBackendUp(backend string) {
	e.healthMu.Lock()
	defer e.healthMu.Unlock()
	delete(e.downUntil, backend)
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

// listAllStoresDetailed lists collections in the single file store, tagging
// each with BackendVectra. It never does a separate health Ping: the listing
// runs under a short budget so a wedged disk cannot stall the dashboard.
func (e *Engine) listAllStoresDetailed() []storeCollection {
	var out []storeCollection
	s, ok := e.stores[BackendVectra]
	if !ok {
		return nil
	}
	cols, err := listCollectionsWithTimeout(s, 2*time.Second)
	if err != nil {
		log.Printf("ListCollections on %s failed: %v", BackendVectra, err)
		return nil
	}
	for _, c := range cols {
		out = append(out, storeCollection{Name: c.Name, ChunkCount: c.ChunkCount, Backend: BackendVectra})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// listCollectionsWithTimeout bounds a store listing so a blackholed backend
// cannot block the caller for the full data timeout.
func listCollectionsWithTimeout(s VectorStore, d time.Duration) ([]StoreCollectionInfo, error) {
	type result struct {
		cols []StoreCollectionInfo
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		c, err := s.ListCollections()
		ch <- result{c, err}
	}()
	select {
	case r := <-ch:
		return r.cols, r.err
	case <-time.After(d):
		return nil, errProbeTimeout
	}
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

// CollectionExists reports whether a collection is present in the file store.
func (e *Engine) CollectionExists(name string) (bool, error) {
	return e.storeFor(name).CollectionExists(name)
}

// Registry exposes the collection registry for management tools.
func (e *Engine) Registry() *registry.Registry {
	return e.registry
}

// Search performs a semantic search with optional reranking (single collection).
func (e *Engine) Search(query string, collectionID string, topK int, useRerank bool, threshold float64, filter map[string]any) ([]SearchResult, error) {
	return e.SearchMulti(query, []string{collectionID}, topK, useRerank, threshold, filter)
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
// Scope resolution (vectra-only):
//   - explicit collection_id: searched directly, otherwise a loud error.
//   - omitted: the configured default collection; with no default configured
//     it searches all enabled collections.
func (e *Engine) SearchDefault(query, collectionID string, topK int, threshold *float64, filter map[string]any) ([]SearchResult, error) {
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
		return e.searchResolved(query, scope, topK, st.RerankEnabled, thr, filter, e.searchDeadline())
	}

	targets := e.defaultSearchTargets()
	if len(targets) == 0 {
		// No defaults anywhere: search all enabled collections.
		return e.SearchMulti(query, nil, topK, st.RerankEnabled, thr, filter)
	}

	deadline := e.searchDeadline()
	t := targets[0]
	if err := checkSearchBudget(deadline); err != nil {
		return nil, err
	}
	return e.searchCollection(t.backend, t.collection, query, topK, st.RerankEnabled, thr, filter, deadline)
}

// defaultCollection returns the configured default collection.
// Legacy default_collection_qdrant is honored when the vectra default is empty.
func (e *Engine) defaultCollection(backend string, st settings.Settings) string {
	if v := strings.TrimSpace(st.DefaultCollectionVectra); v != "" {
		return v
	}
	return strings.TrimSpace(st.DefaultCollectionQdrant)
}

// defaultSearchTargets returns the default collection target for a caller that
// omitted collection_id. Empty means "no default configured"; callers widen
// to all enabled collections.
func (e *Engine) defaultSearchTargets() []scopeTarget {
	st := e.settings.Get()
	if n := e.defaultCollection(BackendVectra, st); n != "" {
		return []scopeTarget{{backend: BackendVectra, collection: n}}
	}
	return nil
}

// searchDeadline returns the wall-clock bound for one whole search (embed +
// stores + rerank). The per-stage budgets alone can exceed the caller's outer
// fail-safe across retry attempts (each attempt re-embeds), so this
// makes the server answer first. Checked only at stage boundaries: an
// in-flight HTTP call is never cancelled, so one-api still finishes its own
// channel failover work.
func (e *Engine) searchDeadline() time.Time {
	d := e.config.SearchBudget
	if d <= 0 {
		d = 170 * time.Second
	}
	return time.Now().Add(d)
}

// checkSearchBudget errors once the overall search budget has passed.
func checkSearchBudget(deadline time.Time) error {
	if !deadline.IsZero() && time.Now().After(deadline) {
		return fmt.Errorf("search overall budget exceeded (the one-api pipeline did not finish in time; check channel health)")
	}
	return nil
}

// searchResolved searches one explicitly named collection.
// A not-found is reported with a case-insensitive "did you mean" hint.
func (e *Engine) searchResolved(query, scope string, topK int, useRerank bool, thr float64, filter map[string]any, deadline time.Time) ([]SearchResult, error) {
	if en := e.registry.Get(scope); en != nil && !en.Enabled {
		return nil, fmt.Errorf("collection '%s' is disabled", scope)
	}
	res, err := e.searchCollection(BackendVectra, scope, query, topK, useRerank, thr, filter, deadline)
	if err == nil {
		return res, nil
	}
	if !e.collectionExistsOn(BackendVectra, scope) {
		return nil, fmt.Errorf("collection '%s' not found on %s%s", scope, BackendVectra, e.collectionSuggestions(scope))
	}
	return nil, err
}

// collectionSuggestions returns a "did you mean" hint for a missing collection
// by case-insensitively matching registry names. It never auto-corrects: the
// caller must pass the exact id. Backends are shown so the operator sees which
// store holds the suggested name.
func (e *Engine) collectionSuggestions(missing string) string {
	var hints []string
	for name, en := range e.registry.List() {
		if !strings.EqualFold(name, missing) {
			continue
		}
		b := en.Backend
		if b == "" {
			b = e.defaultBackend
		}
		hints = append(hints, fmt.Sprintf("'%s' (backend: %s)", name, b))
		if len(hints) >= 3 {
			break
		}
	}
	if len(hints) == 0 {
		return ""
	}
	return fmt.Sprintf(" — did you mean %s? (collection ids are case-sensitive)", strings.Join(hints, ", "))
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

// boundQuery caps a query to the runtime query_max_chars limit (runes) before
// it is embedded or reranked. The head is kept, since callers put the actual
// question first and paste long source material after it. 0 = unlimited.
func (e *Engine) boundQuery(query string) string {
	limit := e.settings.Get().QueryMaxChars
	if limit <= 0 {
		return query
	}
	if n := utf8.RuneCountInString(query); n > limit {
		log.Printf("Query truncated before embedding: %d -> %d runes (query_max_chars)", n, limit)
	}
	return truncateRunes(query, limit)
}

// noteEmbedDim records the vector length the active model returned so later
// collection creation and provenance use the real shape, not the declared one.
func (e *Engine) noteEmbedDim(n int) {
	if n > 0 {
		e.observedEmbedDim.Store(int32(n))
	}
}

// activeEmbedProvenance reports which provider/model/vector space NEW
// embeddings (and their registry provenance) should be stamped with. An
// empty active_embed_provider falls back to the one-api config values.
// The dimension actually returned by the model wins over the declared config
// so a plain model swap behind the same alias self-heals.
func (e *Engine) activeEmbedProvenance() (provider, model string, dim int, distance string) {
	obs := int(e.observedEmbedDim.Load())
	st := e.settings.Get()
	if st.ActiveEmbedProvider != "" {
		for i := range st.EmbedProviders {
			p := st.EmbedProviders[i]
			if p.ID != st.ActiveEmbedProvider {
				continue
			}
			dim = p.Dim
			if dim == 0 {
				dim = e.config.VectorDim
			}
			if obs > 0 {
				dim = obs
			}
			dist := p.Distance
			if dist == "" {
				dist = e.config.VectorDistance
			}
			return p.ID, p.Model, dim, dist
		}
	}
	dim = e.config.VectorDim
	if obs > 0 {
		dim = obs
	}
	return e.config.EmbedProvider, e.config.EmbedModel, dim, e.config.VectorDistance
}

// ActiveEmbedProvenance exposes the current embedding labeling for audit
// logging (jobs, history). See activeEmbedProvenance.
func (e *Engine) ActiveEmbedProvenance() (provider, model string, dim int, distance string) {
	return e.activeEmbedProvenance()
}

// currentEmbedder returns the embedding client for the active provider,
// building and caching it on first use. Inactive providers are not touched.
func (e *Engine) currentEmbedder() *EmbeddingClient {
	st := e.settings.Get()
	if st.ActiveEmbedProvider == "" {
		return e.embedding
	}
	var prov *settings.EmbedProvider
	for i := range st.EmbedProviders {
		if st.EmbedProviders[i].ID == st.ActiveEmbedProvider {
			prov = &st.EmbedProviders[i]
			break
		}
	}
	if prov == nil || prov.BaseURL == "" {
		return e.embedding
	}
	e.embedMu.Lock()
	defer e.embedMu.Unlock()
	if c, ok := e.embedCache[prov.ID]; ok {
		return c
	}
	key := ""
	if prov.APIKeyEnv != "" {
		key = os.Getenv(prov.APIKeyEnv)
	}
	if key == "" {
		key = e.config.APIKey
	}
	c := NewEmbeddingClient(prov.BaseURL, prov.BackupURL, prov.Model, key)
	c.SetTimeout(e.config.EmbedTimeout)
	e.embedCache[prov.ID] = c
	return c
}

// embedQuery binds the current provider's client for one query embed. embeds a single query string once. The query is bounded first so
// a long pasted message never reaches the embedding model untruncated (the
// rerank-side bound only runs after embedding, too late to help).
func (e *Engine) embedQuery(query string) ([]float32, error) {
	embeddings, err := e.currentEmbedder().CreateEmbeddings([]string{e.boundQuery(query)})
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("embedding model returned no vectors")
	}
	e.noteEmbedDim(len(embeddings[0]))
	return embeddings[0], nil
}

// keywordSearcher is implemented by backends that can run BM25 (vectra +
// qdrant via in-memory scan). Stores that do not implement it silently skip
// the BM25 leg — keyword recall is an enhancement, not a requirement.
type keywordSearcher interface {
	KeywordSearch(collection string, query string, limit int) ([]StoreSearchResult, error)
}

// fuseWithBM25 merges a vector result list with a BM25 leg from the same
// (backend, collection) via RRF. Disabled or unsupported backends return the
// vector list unchanged.
func (e *Engine) fuseWithBM25(backend, name, query string, vectorList []SearchResult, recallCount int) []SearchResult {
	st := e.settings.Get()
	if !st.BM25Enabled {
		return vectorList
	}
	store, ok := e.stores[backend]
	if !ok {
		return vectorList
	}
	ks, ok := store.(keywordSearcher)
	if !ok {
		return vectorList
	}
	kw, err := ks.KeywordSearch(name, query, recallCount)
	if err != nil || len(kw) == 0 {
		return vectorList
	}
	bm25List := make([]SearchResult, 0, len(kw))
	for _, r := range kw {
		sr := storeToSearchResult(r, name)
		sr.Backend = backend
		bm25List = append(bm25List, sr)
	}
	return rrfFuse([][]SearchResult{vectorList, bm25List}, recallCount)
}

// searchCollection embeds the query once and searches one (backend, collection),
// optionally reranking and trimming to topK. deadline is the whole-search
// wall-clock bound; it is checked only before stages start.
func (e *Engine) searchCollection(backend, name, query string, topK int, useRerank bool, threshold float64, filter map[string]any, deadline time.Time) ([]SearchResult, error) {
	st := e.settings.Get()
	rerankOn := useRerank && st.RerankEnabled
	if err := checkSearchBudget(deadline); err != nil {
		return nil, err
	}
	queryVector, err := e.embedQuery(query)
	if err != nil {
		return nil, err
	}
	store, ok := e.stores[BackendVectra]
	if !ok {
		return nil, fmt.Errorf("vectra store not configured")
	}
	// Dimension gate: mixed vector spaces give garbage cosine scores, so a
	// stored dim that disagrees with the query dim is a loud error.
	if fs, ok := store.(*FileStore); ok && len(queryVector) > 0 {
		if have := fs.VectorDim(name); have > 0 && have != len(queryVector) {
			return nil, fmt.Errorf("collection '%s' stores %d-d vectors but the active embedding model returns %d-d: rebuild the collection or switch models", name, have, len(queryVector))
		}
	}
	recallCount := topK
	if rerankOn {
		recallCount = st.RerankRecall
		if recallCount < topK {
			recallCount = topK
		}
	}
	raw, err := store.Search(name, queryVector, recallCount, threshold, filter)
	if err != nil {
		return nil, err
	}
	merged := make([]SearchResult, 0, len(raw))
	for _, r := range raw {
		sr := storeToSearchResult(r, name)
		sr.Backend = backend
		merged = append(merged, sr)
	}
	merged = e.fuseWithBM25(backend, name, query, merged, recallCount)
	if rerankOn && len(merged) > 1 {
		// A rerank failure is an error: silently returning vector order would
		// make recall mysteriously "work differently" between runs.
		if err := checkSearchBudget(deadline); err != nil {
			return nil, err
		}
		reranked, err := e.applyRerank(query, merged, topK)
		if err != nil {
			return nil, fmt.Errorf("rerank failed for '%s': %w", name, err)
		}
		return reranked, nil
	}
	return trimResults(merged, topK), nil
}

// SearchMulti searches across several collections and merges the results.
// An empty collections list means "all enabled collections" (registry), or
// every stored collection when the registry is empty.
// Disabled collections are always skipped. A collection that fails on an
// explicit request (missing, down, dim mismatch) is skipped with a WARN log
// rather than aborting the whole batch; if every requested collection fails,
// that is a loud error. An optional payload filter is
// applied to every store.
func (e *Engine) SearchMulti(query string, collections []string, topK int, useRerank bool, threshold float64, filter map[string]any) ([]SearchResult, error) {
	targets, _ := e.resolveTargets(collections)
	if len(targets) == 0 {
		return nil, fmt.Errorf("no collections to search (all disabled or none exist)")
	}

	// Embed the query once
	if err := checkSearchBudget(e.searchDeadline()); err != nil {
		return nil, err
	}
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

	// Search each collection, tag results, merge. One bad library must not
	// silence the others: every skip is logged with its reason so the server
	// log is the report.
	deadline := e.searchDeadline()
	var merged []SearchResult
	var skipped []string
	for _, name := range targets {
		if err := checkSearchBudget(deadline); err != nil {
			return nil, err
		}
		if en := e.registry.Get(name); en != nil && !en.Enabled {
			log.Printf("SearchMulti skip '%s': collection is disabled", name)
			skipped = append(skipped, name+" (disabled)")
			continue
		}
		// Dim pre-check: a vector in the wrong space gives garbage cosine
		// scores, so skip with a clear reason instead of mixing spaces.
		if en := e.registry.Get(name); en != nil && en.VectorDim > 0 && len(queryVector) > 0 && en.VectorDim != len(queryVector) {
			log.Printf("SearchMulti skip '%s': dim mismatch (collection %d-d, query %d-d)", name, en.VectorDim, len(queryVector))
			skipped = append(skipped, name+" (dim mismatch)")
			continue
		}
		backend := BackendVectra
		results, err := e.storeFor(name).Search(name, queryVector, recallCount, threshold, filter)
		if err != nil {
			log.Printf("SearchMulti skip '%s' (%s): %v", name, backend, err)
			skipped = append(skipped, name)
			continue
		}
		var vecList []SearchResult
		for _, r := range results {
			sr := storeToSearchResult(r, name)
			sr.Backend = backend
			vecList = append(vecList, sr)
		}
		merged = append(merged, e.fuseWithBM25(backend, name, query, vecList, recallCount)...)
	}
	if len(skipped) == len(targets) {
		return nil, fmt.Errorf("all %d requested collections failed: %s", len(targets), strings.Join(skipped, "; "))
	}

	// Apply reranking if enabled. A rerank failure is an error, never a
	// silent fall back to vector order.
	if rerankOn && len(merged) > 1 {
		if err := checkSearchBudget(deadline); err != nil {
			return nil, err
		}
		reranked, err := e.applyRerank(query, merged, topK)
		if err != nil {
			return nil, err
		}
		return reranked, nil
	}
	return trimResults(merged, topK), nil
}

// SearchMultiDefault is SearchMulti with the settings-driven defaults that
// SearchDefault applies (topK, threshold, rerank flag), for callers that name
// an explicit list of collections.
func (e *Engine) SearchMultiDefault(query string, collectionIDs []string, topK int, threshold *float64, filter map[string]any) ([]SearchResult, error) {
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
	return e.SearchMulti(query, collectionIDs, topK, st.RerankEnabled, thr, filter)
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

func storeToSearchResult(r StoreSearchResult, collection string) SearchResult {
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
// Observability fields (query truncation, timings, score mode) let the RAG
// console diagnose recall metaphysics: long pastes silently truncated,
// CPU-slow embeds, jina-logit vs qwen-prob scales, saturated/weak reranks.
type SearchDebugResult struct {
	Results []SearchResult `json:"results"`
	Killed  []SearchResult `json:"killed"`
	// Query observability
	QueryOriginalLen  int     `json:"query_original_len"`
	QueryTruncatedLen int     `json:"query_truncated_len"`
	QueryPreview      string  `json:"query_preview"`
	QueryTruncated    bool    `json:"query_truncated"`
	Backend           string  `json:"backend"`
	Reranked          bool    `json:"reranked"`
	ScoreMode         string  `json:"score_mode"`
	TopScore          float64 `json:"top_score"`
	EmbedMs           int64   `json:"embed_ms"`
	RerankMs          int64   `json:"rerank_ms"`
	TookMs            int64   `json:"took_ms"`
}

// SearchDebug runs a single-collection search and reports threshold kills.
// It embeds once and searches twice (threshold, then 0): no rerank on the
// killed set, they are shown in raw vector order.
func (e *Engine) SearchDebug(query string, collection string, topK int, useRerank bool, threshold float64, filter map[string]any) (*SearchDebugResult, error) {
	t0 := time.Now()
	deadline := e.searchDeadline()
	// Truncation observability: boundQuery keeps the head; the console needs
	// both lengths plus a preview to tell "long paste silently cut" apart
	// from "genuinely no recall".
	origLen := utf8.RuneCountInString(query)
	bounded := e.boundQuery(query)
	truncLen := utf8.RuneCountInString(bounded)
	preview := bounded
	if utf8.RuneCountInString(preview) > 300 {
		preview = string([]rune(preview)[:300]) + "…"
	}

	if err := checkSearchBudget(deadline); err != nil {
		return nil, err
	}
	tEmbed := time.Now()
	queryVector, err := e.embedQuery(query)
	embedMs := time.Since(tEmbed).Milliseconds()
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

	backend := BackendVectra
	kept, err := e.storeFor(collection).Search(collection, queryVector, recallCount, threshold, filter)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}
	results := make([]SearchResult, 0, len(kept))
	keptIDs := make(map[any]bool, len(kept))
	for _, r := range kept {
		keptIDs[r.ID] = true
		results = append(results, storeToSearchResult(r, collection))
	}

	var killed []SearchResult
	if threshold > 0 {
		if s, ok := e.stores[backend]; ok {
			if all, err := s.Search(collection, queryVector, recallCount, 0, filter); err == nil {
				for _, r := range all {
					if !keptIDs[r.ID] {
						killed = append(killed, storeToSearchResult(r, collection))
					}
				}
			}
		}
	}

	results = e.fuseWithBM25(backend, collection, query, results, recallCount)

	reranked := false
	var rerankMs int64
	scoreMode := "vector-only"
	if useRerank && len(results) > 1 {
		// No fail-open here either: a rerank failure must be visible in the
		// console, not papered over with vector order. Operators can rerun
		// with the toggle off to see the raw vector results.
		if err := checkSearchBudget(deadline); err != nil {
			return nil, err
		}
		tRerank := time.Now()
		out, err := e.applyRerank(query, results, topK)
		if err != nil {
			return nil, fmt.Errorf("rerank failed: %w", err)
		}
		results = out
		reranked = true
		rerankMs = time.Since(tRerank).Milliseconds()
		scoreMode = e.rerank.ScoreModeName()
	} else {
		results = trimResults(results, topK)
	}
	var topScore float64
	if len(results) > 0 {
		topScore = results[0].Score
	}
	out := &SearchDebugResult{
		Results: results, Killed: killed,
		QueryOriginalLen: origLen, QueryTruncatedLen: truncLen,
		QueryPreview: preview, QueryTruncated: truncLen < origLen,
		Backend: backend, Reranked: reranked, ScoreMode: scoreMode,
		TopScore: topScore, EmbedMs: embedMs, RerankMs: rerankMs,
		TookMs: time.Since(t0).Milliseconds(),
	}
	// RAG-metaphysics warnings, distilled from ST Vector-Storage-5053:
	// saturated (all scores pinned near 0) vs weakly relevant (best < 0.5).
	if reranked && len(results) > 0 {
		if topScore < 0.01 {
			log.Printf("SearchDebug warn: reranker scores look saturated (best %.4f, mode %s) — check model scale", topScore, scoreMode)
		} else if topScore < 0.5 {
			log.Printf("SearchDebug warn: all reranked candidates weakly relevant (best %.3f, mode %s)", topScore, scoreMode)
		}
	}
	if out.QueryTruncated {
		log.Printf("SearchDebug: query truncated %d -> %d runes preview=%.100q", origLen, truncLen, preview)
	}
	return out, nil
}

// applyRerank reranks the merged candidates. Call sites propagate its error:
// a rerank failure must be visible to the caller, never silently replaced by
// vector order. There is no context deadline here — the rerank client owns its
// total budget and, like the embed call, never cancels one-api mid-failover.
func (e *Engine) applyRerank(query string, results []SearchResult, topK int) ([]SearchResult, error) {
	if len(results) <= 1 {
		return results, nil
	}

	texts := make([]string, len(results))
	for i, r := range results {
		texts[i] = r.Text
	}

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
	Size           int    `json:"size"`
	OverlapPercent int    `json:"overlap_percent"`
	Strategy       string `json:"strategy"`
}

// ResolvedChunkOptions is the effective chunking used by a job.
type ResolvedChunkOptions struct {
	Size     int
	Overlap  int
	Strategy string
}

// ProgressFunc reports upsert progress: doneChunks of totalChunks.
type ProgressFunc func(doneChunks, totalChunks int)

// ResolveChunkOptions fills zero values from global config.
// Zero means "use global" (the dashboard sends 0 for defaults);
// there is currently no way to request literal 0% overlap per job.
func (e *Engine) ResolveChunkOptions(opts *ChunkOptions) ResolvedChunkOptions {
	size := e.config.ChunkSize
	overlapPct := e.config.OverlapPercent
	strategy := e.config.ChunkStrategy
	if opts != nil {
		if opts.Size > 0 {
			size = opts.Size
		}
		if opts.OverlapPercent > 0 && opts.OverlapPercent < 100 {
			overlapPct = opts.OverlapPercent
		}
		if opts.Strategy != "" && chunking.IsValidStrategy(opts.Strategy) {
			strategy = opts.Strategy
		}
	}
	if strategy == "" {
		strategy = chunking.StrategyWindow
	}
	return ResolvedChunkOptions{Size: size, Overlap: size * overlapPct / 100, Strategy: strategy}
}

// Provenance records how a collection's vectors were computed.
// Stored in the registry: "how it was built", never query policy.
type Provenance struct {
	SourceFile     string
	SourceSHA256   string
	ChunkSize      int
	ChunkOverlap   int
	EmbedModel     string
	EmbedProvider  string
	VectorDim      int
	VectorDistance string
	ChunkStrategy  string
	Chunks         int
}

// IndexDocument indexes a file into the vectra file store.
func (e *Engine) IndexDocument(path string, collectionID string, metadata map[string]string) (*IndexResult, error) {
	return e.IndexDocumentWithOptions(path, collectionID, metadata, nil, nil)
}

// IndexDocumentWithOptions indexes a file with per-job chunking and progress.
func (e *Engine) IndexDocumentWithOptions(path string, collectionID string, metadata map[string]string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, error) {
	return e.indexDocumentInternal(BackendVectra, path, collectionID, metadata, opts, prog)
}

// IndexDocumentOn keeps the old (backend, path, ...) signature for the
// dashboard/jobs callers; the backend argument is normalized to vectra.
func (e *Engine) IndexDocumentOn(backend, path, collectionID string, metadata map[string]string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, error) {
	return e.indexDocumentInternal(BackendVectra, path, collectionID, metadata, opts, prog)
}

// indexDocumentInternal indexes a file into the vectra file store.
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
	embedProvider, embedModel, embedDim, embedDistance := e.activeEmbedProvenance()
	prov := Provenance{
		SourceFile:     path,
		SourceSHA256:   hex.EncodeToString(sum[:]),
		EmbedModel:     embedModel,
		EmbedProvider:  embedProvider,
		VectorDim:      embedDim,
		VectorDistance: embedDistance,
	}

	res, used, err := e.indexTextInternalOn(backend, text, collectionID, metadata, fileName, path, opts, prog)
	if err != nil {
		return nil, err
	}
	prov.Chunks = res.ChunksIndexed
	prov.ChunkSize, prov.ChunkOverlap, prov.ChunkStrategy = used.Size, used.Overlap, used.Strategy
	e.recordIndex(collectionID, prov, backend)
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

// ReindexDocumentOn purges and rebuilds a collection (backend arg kept for
// caller compatibility, always vectra). Registry display metadata is
// preserved; only the vectors are rebuilt.
func (e *Engine) ReindexDocumentOn(backend, path, collectionID string, metadata map[string]string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, error) {
	store := e.storeForBackend(BackendVectra)
	exists, err := store.CollectionExists(collectionID)
	if err != nil {
		return nil, fmt.Errorf("failed to check collection: %w", err)
	}
	if exists {
		if err := store.DeleteCollection(collectionID); err != nil {
			return nil, fmt.Errorf("failed to purge collection: %w", err)
		}
		log.Printf("Purged collection '%s' for re-index", collectionID)
	}
	return e.indexDocumentInternal(BackendVectra, path, collectionID, metadata, opts, prog)
}

// CollectionExistsOnOther is kept for API compatibility and always reports
// false: there is no second backend to collide with.
func (e *Engine) CollectionExistsOnOther(backend, name string) (bool, error) {
	return false, nil
}

// IndexText indexes raw text into the vectra file store
func (e *Engine) IndexText(text string, collectionID string, metadata map[string]string) (*IndexResult, error) {
	return e.IndexTextWithOptions(text, collectionID, metadata, nil, nil)
}

// IndexTextWithOptions indexes raw text with per-job chunking and progress.
func (e *Engine) IndexTextWithOptions(text string, collectionID string, metadata map[string]string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, error) {
	res, used, err := e.indexTextInternal(text, collectionID, metadata, "direct_text", "direct_text", opts, prog)
	if err != nil {
		return nil, err
	}
	embedProvider, embedModel, embedDim, embedDistance := e.activeEmbedProvenance()
	e.recordIndex(collectionID, Provenance{
		SourceFile:     "direct_text",
		ChunkSize:      used.Size,
		ChunkOverlap:   used.Overlap,
		ChunkStrategy:  used.Strategy,
		EmbedModel:     embedModel,
		EmbedProvider:  embedProvider,
		VectorDim:      embedDim,
		VectorDistance: embedDistance,
		Chunks:         res.ChunksIndexed,
	}, e.backendOf(collectionID))
	return res, nil
}

// StoreMemory indexes ad-hoc text for later semantic recall (the MCP
// store_memory tool). An omitted collection_id resolves to the configured
// default collection; having no default at all is a loud error.
//
// Raw text is appended to a daily journal file
// (<memoryDir>/agent/YYYY-MM-DD.md) so the data bank stays browsable without
// one file per write. Only the new text is embedded (the journal file is the
// provenance pointer, not re-embedded whole).
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
	metadata["visibility"] = "agent"

	scope := strings.TrimSpace(collectionID)
	if scope == "" {
		targets := e.defaultSearchTargets()
		if len(targets) == 0 {
			return nil, fmt.Errorf("no collection_id given and no default collection configured; pass collection_id or set a default in the management dashboard")
		}
		scope = targets[0].collection
	}
	metadata["collection"] = scope

	// Index only the new text; source_file points at the daily journal so
	// recall can open the human-readable log.
	sub := metadata
	res, used, err := e.indexTextInternal(text, scope, sub, filepath.Base(path), path, nil, nil)
	if err != nil {
		return nil, err
	}
	embedProvider, embedModel, embedDim, embedDistance := e.activeEmbedProvenance()
	sum := sha256.Sum256([]byte(text))
	e.recordIndex(scope, Provenance{
		SourceFile:     path,
		SourceSHA256:   hex.EncodeToString(sum[:]),
		ChunkSize:      used.Size,
		ChunkOverlap:   used.Overlap,
		ChunkStrategy:  used.Strategy,
		EmbedModel:     embedModel,
		EmbedProvider:  embedProvider,
		VectorDim:      embedDim,
		VectorDistance: embedDistance,
		Chunks:         res.ChunksIndexed,
	}, BackendVectra)
	e.writeVaultManifest(scope)
	return res, nil
}

// writeMemoryRaw appends ad-hoc memory text to the daily agent journal
// (<memoryDir>/agent/YYYY-MM-DD.md) with a timestamp header. It returns the
// journal path for provenance.
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
	dir := filepath.Join(base, "agent")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, now.Format("2006-01-02")+".md")
	entry := fmt.Sprintf("\n\n---\n## %s\n\n%s\n", now.Format("15:04:05"), strings.TrimSpace(text))
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if _, err := f.WriteString(entry); err != nil {
		return "", err
	}
	return path, nil
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
func PreviewChunks(text string, size, overlap int, strategy string, maxSamples int) (int, []string) {
	if size <= 0 {
		size = 500
	}
	chunks := chunking.Split(text, chunking.Options{Strategy: strategy, Size: size, Overlap: overlap})
	samples := chunks
	if maxSamples > 0 && len(chunks) > maxSamples {
		samples = chunks[:maxSamples]
	}
	return len(chunks), samples
}

// recordIndex snapshots provenance in the registry plus the vault catalog, so
// the folder stays self-describing for rclone backup + Verify/Import.
// Best-effort: a registry write failure is logged, indexing already succeeded.
func (e *Engine) recordIndex(collectionID string, prov Provenance, backend string) {
	backend = BackendVectra
	if fs := e.FileStore(); fs != nil && prov.SourceFile != "" && prov.SourceFile != "direct_text" {
		fs.UpdateCatalogMeta(collectionID, prov.SourceFile, prov.SourceSHA256, prov.ChunkSize, prov.ChunkOverlap, prov.ChunkStrategy, prov.EmbedModel)
	}
	e.writeVaultManifest(collectionID)
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
		if prov.EmbedProvider != "" {
			en.EmbedProvider = prov.EmbedProvider
		}
		if prov.VectorDim > 0 {
			en.VectorDim = prov.VectorDim
		}
		if prov.VectorDistance != "" {
			en.VectorDistance = prov.VectorDistance
		}
		if backend != "" {
			en.Backend = backend
		}
		en.ChunkCount = prov.Chunks
		en.Chunk = registry.ChunkConfig{Size: prov.ChunkSize, Overlap: prov.ChunkOverlap, Strategy: prov.ChunkStrategy}
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

// checkProvenanceConflict rejects an append whose current embedding setup
// would mix incompatible vectors into a collection that already has one.
// A nil registry entry (unregistered collection) or empty recorded fields
// (legacy entry) is treated as "unknown" and allowed through — recordIndex
// will stamp the current provenance, never silently overwrite a conflict.
func (e *Engine) checkProvenanceConflict(collectionID string) error {
	en := e.registry.Get(collectionID)
	if en == nil {
		return nil
	}
	curProvider, curModel, curDim, curDistance := e.activeEmbedProvenance()
	mismatch := []string{}
	if en.EmbedModel != "" && en.EmbedModel != curModel {
		mismatch = append(mismatch, fmt.Sprintf("embed_model: %q vs current %q", en.EmbedModel, curModel))
	}
	if en.EmbedProvider != "" && en.EmbedProvider != curProvider {
		mismatch = append(mismatch, fmt.Sprintf("embed_provider: %q vs current %q", en.EmbedProvider, curProvider))
	}
	if en.VectorDim > 0 && en.VectorDim != curDim {
		mismatch = append(mismatch, fmt.Sprintf("vector_dim: %d vs current %d", en.VectorDim, curDim))
	}
	if en.VectorDistance != "" && !strings.EqualFold(en.VectorDistance, curDistance) {
		mismatch = append(mismatch, fmt.Sprintf("vector_distance: %q vs current %q", en.VectorDistance, curDistance))
	}
	if len(mismatch) == 0 {
		return nil
	}
	return fmt.Errorf("collection '%s' provenance conflict (%s): refusing to mix vectors — rebuild the collection or use a new collection name", collectionID, strings.Join(mismatch, "; "))
}

// indexTextInternal routes to the vectra file store.
func (e *Engine) indexTextInternal(text string, collectionID string, metadata map[string]string, sourceName string, sourceFile string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, ResolvedChunkOptions, error) {
	return e.indexTextInternalOn(BackendVectra, text, collectionID, metadata, sourceName, sourceFile, opts, prog)
}

func (e *Engine) indexTextInternalOn(backend string, text string, collectionID string, metadata map[string]string, sourceName string, sourceFile string, opts *ChunkOptions, prog ProgressFunc) (*IndexResult, ResolvedChunkOptions, error) {
	// Chunk the text (per-job options fall back to global config)
	used := e.ResolveChunkOptions(opts)
	chunks := chunking.Split(text, chunking.Options{Strategy: used.Strategy, Size: used.Size, Overlap: used.Overlap})

	// Validate chunks
	problems := chunking.ValidateChunks(chunks, sourceName)
	if len(problems) > 0 {
		log.Printf("Chunk validation warnings for %s: %d problems", sourceName, len(problems))
	}

	if len(chunks) == 0 {
		return &IndexResult{ChunksIndexed: 0, Collection: collectionID}, used, nil
	}

	// Embed all chunks. Batch cap follows the compute mode: GPU uses the
	// configured embed_batch_size (0 => 32), the CPU fallback is capped at
	// embedBatchSizeCPU (4) so a 32-text batch never crashes the weak nodes.
	cpuMode := e.usesCPUCompute()
	bs := e.effectiveEmbedBatchSize(cpuMode)
	log.Printf("Index '%s': embedding %d chunks (batch_size=%d, cpu_mode=%v)", collectionID, len(chunks), bs, cpuMode)
	embeddings, err := e.currentEmbedder().CreateEmbeddingsBatched(chunks, bs)
	if err != nil {
		return nil, used, fmt.Errorf("failed to embed chunks: %w", err)
	}
	actualDim := 0
	if len(embeddings) > 0 && len(embeddings[0]) > 0 {
		actualDim = len(embeddings[0])
		e.noteEmbedDim(actualDim)
	}

	// Create deterministic uint IDs: high 32 bits = collection hash, low 32
	// bits = chunk hash. Re-indexing the same file upserts the same IDs.
	colHash := uint64(StringHash(collectionID)) << 32
	indexedAt := time.Now().UTC().Format(time.RFC3339)
	sourceRel := sourceFile
	if e.docsDir != "" {
		if abs, err := filepath.Abs(sourceFile); err == nil {
			if rel, err := filepath.Rel(e.docsDir, abs); err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
				sourceRel = filepath.ToSlash(rel)
			}
		}
	}
	points := make([]Point, len(chunks))
	for i, chunk := range chunks {
		payload := map[string]any{
			"text":        chunk,
			"source":      sourceName,
			"source_file": sourceFile,
			"source_rel":  sourceRel,
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

	// Ensure collection exists. Before appending to an existing collection,
	// refuse to mix provenance: if the recorded model/provider/vector space
	// differs from the current engine config, vectors would be incomparable.
	// Rebuilding drops the collection first, so it bypasses this gate.
	store := e.storeForBackend(BackendVectra)
	exists, err := store.CollectionExists(collectionID)
	if err != nil {
		return nil, used, fmt.Errorf("failed to check collection: %w", err)
	}
	if exists {
		if err := e.checkProvenanceConflict(collectionID); err != nil {
			return nil, used, err
		}
		if fs, ok := store.(*FileStore); ok && actualDim > 0 {
			if have := fs.VectorDim(collectionID); have > 0 && have != actualDim {
				return nil, used, fmt.Errorf("collection '%s' stores %d-d vectors but the active embedding model returns %d-d: rebuild the collection (purge + reindex) or switch to a %d-d model", collectionID, have, actualDim, have)
			}
		}
	}
	if !exists {
		if err := store.CreateCollection(collectionID); err != nil {
			return nil, used, fmt.Errorf("failed to create collection: %w", err)
		}
	}

	// Upsert points in batches so large documents stay in small files.
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
	ID             int      `json:"id"`
	Name           string   `json:"name"`
	Backend        string   `json:"backend"`
	ChunkCount     int      `json:"chunk_count"`
	Exists         bool     `json:"exists"`
	DisplayName    string   `json:"display_name,omitempty"`
	Description    string   `json:"description,omitempty"`
	Tags           []string `json:"tags,omitempty"`
	Enabled        bool     `json:"enabled"`
	Consumers      []string `json:"consumers,omitempty"`
	SourceFile     string   `json:"source_file,omitempty"`
	SourceSHA256   string   `json:"source_sha256,omitempty"`
	EmbedModel     string   `json:"embed_model,omitempty"`
	EmbedProvider  string   `json:"embed_provider,omitempty"`
	VectorDim      int      `json:"vector_dim,omitempty"`
	VectorDistance string   `json:"vector_distance,omitempty"`
	CreatedAt      string   `json:"created_at,omitempty"`
	UpdatedAt      string   `json:"updated_at,omitempty"`
	ChunkSize      int      `json:"chunk_size,omitempty"`
	ChunkOverlap   int      `json:"chunk_overlap,omitempty"`
	ChunkStrategy  string   `json:"chunk_strategy,omitempty"`
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
		byName[c.Name] = &CollectionDetail{Name: c.Name, Backend: BackendVectra, ChunkCount: c.ChunkCount, Exists: true, Enabled: true}
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
		d.EmbedProvider = en.EmbedProvider
		d.VectorDim = en.VectorDim
		d.VectorDistance = en.VectorDistance
		d.CreatedAt = en.CreatedAt
		d.UpdatedAt = en.UpdatedAt
		d.ChunkSize = en.Chunk.Size
		d.ChunkOverlap = en.Chunk.Overlap
		d.ChunkStrategy = en.Chunk.Strategy
		// Legacy registry backend values are normalized to vectra.
		d.Backend = BackendVectra
	}
	for _, d := range byName {
		if d.Backend == "" {
			d.Backend = BackendVectra
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

// backendOf always reports vectra (legacy per-collection overrides are
// normalized). Kept so callers don't need to change.
func (e *Engine) backendOf(name string) string {
	return BackendVectra
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
// don't exist yet (aspirational entry). The backend field is accepted for
// compatibility but always normalized to vectra (single-backend server).
func (e *Engine) SetCollectionMeta(name string, meta CollectionMetaUpdate) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("collection_id is required")
	}
	if meta.Backend != nil {
		b := strings.TrimSpace(*meta.Backend)
		if b != "" && b != BackendQdrant && b != BackendVectra {
			return fmt.Errorf("unknown backend %q (want %q)", b, BackendVectra)
		}
		norm := BackendVectra
		meta.Backend = &norm
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

// DeleteCollection drops the collection from the file store and removes its
// registry entry. Refuses without confirm=true: irreversible.
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
		// Registry-only ghost: still clean the registry so undeletable
		// entries can never linger.
		_ = e.registry.Delete(name)
		return fmt.Errorf("collection '%s' does not exist in vectra store (registry entry cleaned if present)", name)
	}
	if err := store.DeleteCollection(name); err != nil {
		return fmt.Errorf("failed to delete collection: %w", err)
	}
	if err := e.registry.Delete(name); err != nil {
		log.Printf("Registry cleanup failed for deleted '%s': %v", name, err)
	}
	if err := e.AppendHistory(HistoryEntry{Op: "delete_collection", Collection: name, Backend: BackendVectra, OK: true}); err != nil {
		log.Printf("history append failed: %v", err)
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

// HealthCheck probes every component concurrently. This is an ON-DEMAND probe
// (the dashboard Test button / health_check tool); the query path never calls
// it and never waits on it. Embedding/rerank are never cached as down —
// channel failover is one-api's job. Probes use short Ping budgets so the
// check never hangs.
func (e *Engine) HealthCheck() map[string]string {
	status := make(map[string]string)
	var mu sync.Mutex
	var wg sync.WaitGroup
	set := func(k, v string) { mu.Lock(); status[k] = v; mu.Unlock() }

	if s, ok := e.stores[BackendVectra]; ok {
		wg.Add(1)
		go func(store VectorStore) {
			defer wg.Done()
			switch err := probeWithTimeout(store.Ping, 4*time.Second); {
			case err == errProbeTimeout:
				set(BackendVectra, "timeout")
			case err != nil:
				set(BackendVectra, "error: "+err.Error())
			default:
				set(BackendVectra, "ok")
			}
		}(s)
	}
	set("active_backend", e.ActiveBackend())

	wg.Add(2)
	go func() {
		defer wg.Done()
		switch err := probeWithTimeout(e.currentEmbedder().Ping, 8*time.Second); {
		case err == errProbeTimeout:
			set("embedding", "slow")
		case err != nil:
			set("embedding", "error: "+err.Error())
		default:
			set("embedding", "ok")
		}
	}()
	go func() {
		defer wg.Done()
		switch err := probeWithTimeout(e.rerank.Ping, 8*time.Second); {
		case err == errProbeTimeout:
			set("rerank", "slow")
		case err != nil:
			set("rerank", "error: "+err.Error())
		default:
			set("rerank", "ok")
		}
	}()
	wg.Wait()
	return status
}

// errProbeTimeout is returned by probeWithTimeout when fn outlives the budget.
var errProbeTimeout = fmt.Errorf("probe timed out")

// probeWithTimeout bounds a blocking probe. The probe goroutine is abandoned on
// timeout (its result channel is buffered so it never leaks a blocked send).
func probeWithTimeout(fn func() error, d time.Duration) error {
	done := make(chan error, 1)
	go func() { done <- fn() }()
	select {
	case err := <-done:
		return err
	case <-time.After(d):
		return errProbeTimeout
	}
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
