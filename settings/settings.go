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
	// QueryMaxChars / DocMaxChars bound rerank payload sizes (runes).
	QueryMaxChars int `json:"query_max_chars"`
	DocMaxChars   int `json:"doc_max_chars"`
}

// Patch carries optional settings updates; nil means "leave unchanged".
// Pointers distinguish an explicit zero from an omitted field.
type Patch struct {
	ActiveBackend           *string  `json:"active_backend"`
	DefaultCollectionQdrant *string  `json:"default_collection_qdrant"`
	DefaultCollectionVectra *string  `json:"default_collection_vectra"`
	FailoverEnabled         *bool    `json:"failover_enabled"`
	DefaultTopK             *int     `json:"default_top_k"`
	DefaultThreshold        *float64 `json:"default_threshold"`
	RerankEnabled           *bool    `json:"rerank_enabled"`
	RerankRecall            *int     `json:"rerank_recall"`
	QueryMaxChars           *int     `json:"query_max_chars"`
	DocMaxChars             *int     `json:"doc_max_chars"`
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
