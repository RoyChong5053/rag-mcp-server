package registry

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ChunkConfig snapshots the chunking used for a collection.
type ChunkConfig struct {
	Size    int `json:"size"`
	Overlap int `json:"overlap"`
}

// Entry is the management metadata for one Qdrant collection.
// It lives in collections.json on the server; Qdrant holds the vectors.
type Entry struct {
	DisplayName string      `json:"display_name"`
	Description string      `json:"description"`
	Tags        []string    `json:"tags"`
	Enabled     bool        `json:"enabled"`
	Consumers   []string    `json:"consumers"`
	SourceFile  string      `json:"source_file"`
	CreatedAt   string      `json:"created_at"`
	UpdatedAt   string      `json:"updated_at"`
	ChunkCount  int         `json:"chunk_count"`
	Chunk       ChunkConfig `json:"chunk"`
}

// Registry is a concurrency-safe file-backed collection registry.
// Missing file loads as empty; corrupt file is a loud error (never silently wiped).
type Registry struct {
	path    string
	mu      sync.RWMutex
	entries map[string]*Entry
}

// New loads the registry from path, or starts empty if the file is missing.
func New(path string) (*Registry, error) {
	r := &Registry{path: path, entries: make(map[string]*Entry)}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return r, nil
		}
		return nil, fmt.Errorf("read registry: %w", err)
	}
	var entries map[string]*Entry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("parse registry %s: %w", path, err)
	}
	if entries != nil {
		r.entries = entries
	}
	return r, nil
}

// Get returns a copy of the entry for name, or nil.
func (r *Registry) Get(name string) *Entry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.entries[name]
	if !ok {
		return nil
	}
	cp := *e
	return &cp
}

// List returns a copy of all entries keyed by collection name.
func (r *Registry) List() map[string]*Entry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make(map[string]*Entry, len(r.entries))
	for k, e := range r.entries {
		cp := *e
		out[k] = &cp
	}
	return out
}

// EnabledNames returns names with Enabled=true.
func (r *Registry) EnabledNames() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var out []string
	for k, e := range r.entries {
		if e.Enabled {
			out = append(out, k)
		}
	}
	return out
}

// Has reports whether name has a registry entry.
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.entries[name]
	return ok
}

// Update mutates (or creates) the entry for name, stamps UpdatedAt, and saves.
// The mutate function receives the entry to modify in place.
func (r *Registry) Update(name string, mutate func(*Entry)) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.entries[name]
	if !ok {
		e = &Entry{Enabled: true, CreatedAt: time.Now().UTC().Format(time.RFC3339)}
		r.entries[name] = e
	}
	mutate(e)
	e.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return r.saveLocked()
}

// Delete removes the entry for name. No-op if absent.
func (r *Registry) Delete(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.entries[name]; !ok {
		return nil
	}
	delete(r.entries, name)
	return r.saveLocked()
}

// saveLocked writes atomically (temp file + rename) so a crash
// never leaves a half-written registry.
func (r *Registry) saveLocked() error {
	data, err := json.MarshalIndent(r.entries, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal registry: %w", err)
	}
	if dir := filepath.Dir(r.path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("mkdir registry dir: %w", err)
		}
	}
	tmp := r.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write registry tmp: %w", err)
	}
	if err := os.Rename(tmp, r.path); err != nil {
		return fmt.Errorf("rename registry: %w", err)
	}
	return nil
}
