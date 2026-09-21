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
	// DefaultCollection is the scope used when a caller omits collection_id.
	// Empty means "search all enabled collections".
	DefaultCollection string `json:"default_collection"`
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
	DefaultCollection *string  `json:"default_collection"`
	DefaultTopK       *int     `json:"default_top_k"`
	DefaultThreshold  *float64 `json:"default_threshold"`
	RerankEnabled     *bool    `json:"rerank_enabled"`
	RerankRecall      *int     `json:"rerank_recall"`
	QueryMaxChars     *int     `json:"query_max_chars"`
	DocMaxChars       *int     `json:"doc_max_chars"`
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
	_ = json.Unmarshal(raw, &s.data)
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

	if p.DefaultCollection != nil {
		s.data.DefaultCollection = *p.DefaultCollection
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
