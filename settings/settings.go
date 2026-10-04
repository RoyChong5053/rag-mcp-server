package settings

import (
	"encoding/json"
	"os"
	"sync"
)

// Settings holds runtime-adjustable search defaults. Persisted to
// settings.json and edited through the WebUI; every frontend (ST plugin,
// opencode, raw HTTP callers) reads these instead of hardcoding values.
type Settings struct {
	// ActiveBackend selects which backend's default collection search_memory
	// and store_memory use when the caller omits collection_id. Empty follows
	// the engine's configured storage.backend. It only affects the default
	// search/write scope, never where new named collections are created.
	ActiveBackend string `json:"active_backend"`
	// DefaultCollectionQdrant is the scope used when the active backend is
	// qdrant, and the primary target for an omitted collection_id. Empty means
	// "search all enabled collections".
	DefaultCollectionQdrant string `json:"default_collection_qdrant"`
	// DefaultCollectionVectra is the scope used when the active backend is
	// vectra, and the failover target when qdrant is unreachable. Empty means
	// no vectra default (failover disabled for that path).
	DefaultCollectionVectra string `json:"default_collection_vectra"`
	// FailoverEnabled lets a qdrant outage fall back to DefaultCollectionVectra.
	FailoverEnabled bool `json:"failover_enabled"`
	// DefaultTopK is how many results search_memory returns when top_k is omitted.
	DefaultTopK int `json:"default_top_k"`
	// DefaultThreshold is the minimum similarity when threshold is omitted.
	DefaultThreshold float64 `json:"default_threshold"`
	// RerankEnabled gates the rerank stage for search_memory.
	RerankEnabled bool `json:"rerank_enabled"`
	// RerankRecall is the vector over-fetch count fed to the reranker.
	RerankRecall int `json:"rerank_recall"`
	// QueryMaxChars caps the query text (runes) before it is embedded and
	// reranked; the head is kept (the actual question usually precedes pasted
	// source material). 0 = unlimited.
	QueryMaxChars int `json:"query_max_chars"`
	// DocMaxChars caps each document handed to the reranker (runes).
	// 0 = unlimited.
	DocMaxChars int `json:"doc_max_chars"`
	// EmbedBatchSize caps texts per embedding POST from index jobs when the
	// GPU path is active. 0 = auto (GPU 32, CPU fallback capped to 4). A
	// positive value overrides the GPU size; the CPU fallback still caps it.
	EmbedBatchSize int `json:"embed_batch_size"`
	// EmbedProviders lists every embed endpoint the server can talk to
	// (one-api fan-out, openrouter, ...). Switching providers only affects
	// new writes: existing collections keep the provenance recorded at
	// vectorize time and the engine refuses to mix vector spaces.
	EmbedProviders []EmbedProvider `json:"embed_providers,omitempty"`
	// ActiveEmbedProvider selects which EmbedProviders entry embedQuery and
	// index jobs use. Empty = the built-in one-api config (legacy default).
	ActiveEmbedProvider string `json:"active_embed_provider,omitempty"`
	// BM25Enabled adds a keyword-retrieval leg to searches; results merge
	// with vector hits via RRF before rerank. Off = pure vector+rerank.
	BM25Enabled bool `json:"bm25_enabled"`
	// FusionMethod picks how the vector and BM25 lists merge: "rrf" (default)
	// or "weighted".
	FusionMethod string `json:"fusion_method"`
}

// EmbedProvider describes one embedding endpoint. APIKeyEnv names an
// environment variable holding the key; empty falls back to the one-api key.
type EmbedProvider struct {
	ID        string `json:"id"`
	BaseURL   string `json:"base_url"`
	BackupURL string `json:"backup_url,omitempty"`
	Model     string `json:"model"`
	Dim       int    `json:"dim,omitempty"`
	Distance  string `json:"distance,omitempty"`
	APIKeyEnv string `json:"api_key_env,omitempty"`
}

// Patch carries optional settings updates; nil means "leave unchanged".
// Pointers distinguish an explicit zero from an omitted field.
type Patch struct {
	ActiveBackend           *string          `json:"active_backend"`
	DefaultCollectionQdrant *string          `json:"default_collection_qdrant"`
	DefaultCollectionVectra *string          `json:"default_collection_vectra"`
	FailoverEnabled         *bool            `json:"failover_enabled"`
	DefaultTopK             *int             `json:"default_top_k"`
	DefaultThreshold        *float64         `json:"default_threshold"`
	RerankEnabled           *bool            `json:"rerank_enabled"`
	RerankRecall            *int             `json:"rerank_recall"`
	QueryMaxChars           *int             `json:"query_max_chars"`
	DocMaxChars             *int             `json:"doc_max_chars"`
	EmbedBatchSize          *int             `json:"embed_batch_size"`
	EmbedProviders          *[]EmbedProvider `json:"embed_providers"`
	ActiveEmbedProvider     *string          `json:"active_embed_provider"`
	BM25Enabled             *bool            `json:"bm25_enabled"`
	FusionMethod            *string          `json:"fusion_method"`
}

// Store is a concurrency-safe runtime settings store backed by a JSON file.
type Store struct {
	mu   sync.RWMutex
	path string
	data Settings
}

// New builds a store seeded with base, then overlays path if it exists.
// Missing fields in the file keep their base value; a missing file is fine.
func New(path string, base Settings) *Store {
	s := &Store{path: path, data: base}
	s.load()
	return s
}

func (s *Store) load() {
	if s.path == "" {
		return
	}
	raw, err := os.ReadFile(s.path)
	if err != nil {
		return
	}
	// Pre-dual-backend files used a single "default_collection" key, which was
	// the qdrant scope. Migrate it to default_collection_qdrant so existing
	// settings.json files keep working without manual edits.
	var legacy struct {
		DefaultCollection *string `json:"default_collection"`
	}
	_ = json.Unmarshal(raw, &legacy)
	_ = json.Unmarshal(raw, &s.data)
	if s.data.DefaultCollectionQdrant == "" && legacy.DefaultCollection != nil {
		s.data.DefaultCollectionQdrant = *legacy.DefaultCollection
	}
}

// Get returns a snapshot of the current settings (safe for concurrent use).
func (s *Store) Get() Settings {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.data
}

// Update applies a partial patch and persists the result. A write failure is
// returned but the in-memory value is already updated, so the running service
// stays consistent with what the operator just saw.
func (s *Store) Update(p Patch) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if p.ActiveBackend != nil {
		s.data.ActiveBackend = *p.ActiveBackend
	}
	if p.DefaultCollectionQdrant != nil {
		s.data.DefaultCollectionQdrant = *p.DefaultCollectionQdrant
	}
	if p.DefaultCollectionVectra != nil {
		s.data.DefaultCollectionVectra = *p.DefaultCollectionVectra
	}
	if p.FailoverEnabled != nil {
		s.data.FailoverEnabled = *p.FailoverEnabled
	}
	if p.DefaultTopK != nil {
		s.data.DefaultTopK = *p.DefaultTopK
	}
	if p.DefaultThreshold != nil {
		s.data.DefaultThreshold = *p.DefaultThreshold
	}
	if p.RerankEnabled != nil {
		s.data.RerankEnabled = *p.RerankEnabled
	}
	if p.RerankRecall != nil {
		s.data.RerankRecall = *p.RerankRecall
	}
	if p.QueryMaxChars != nil {
		s.data.QueryMaxChars = *p.QueryMaxChars
	}
	if p.DocMaxChars != nil {
		s.data.DocMaxChars = *p.DocMaxChars
	}
	if p.EmbedBatchSize != nil {
		s.data.EmbedBatchSize = *p.EmbedBatchSize
	}
	if p.EmbedProviders != nil {
		s.data.EmbedProviders = *p.EmbedProviders
	}
	if p.ActiveEmbedProvider != nil {
		s.data.ActiveEmbedProvider = *p.ActiveEmbedProvider
	}
	if p.BM25Enabled != nil {
		s.data.BM25Enabled = *p.BM25Enabled
	}
	if p.FusionMethod != nil {
		s.data.FusionMethod = *p.FusionMethod
	}

	return s.save()
}

func (s *Store) save() error {
	if s.path == "" {
		return nil
	}
	data, err := json.MarshalIndent(s.data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, data, 0644)
}
