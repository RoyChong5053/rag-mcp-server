package engine

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// FileStore is a local, file-based vector backend whose on-disk shape follows
// Vectra/SillyTavern: one folder per collection containing a catalog plus one
// `index.json` per source document.
//
//	<root>/<collection>/catalog.json
//	<root>/<collection>/<source_key>/index.json
//
// index.json = {version, metadata_config, items:[{id, metadata, vector, norm}]}
// which is the same shape ST writes under data/<user>/vectors/<backend>/<src>/.
// We only ever write inside our own root; ST's data dir is never touched.
//
// It is intended for lightweight deployments as a fallback to Qdrant. Writes
// are serialized and land atomically (temp + rename), searches share a read
// lock. Item vectors are pre-normalized once (`norm`) so scoring is a dot
// product divided by the query norm.
type FileStore struct {
	root       string
	sourceRoot string // docs dir; stripped from source paths to key folders
	mu         sync.RWMutex
}

// vectraItem mirrors the Vectra/SillyTavern index item.
type vectraItem struct {
	ID       string         `json:"id"`
	Metadata map[string]any `json:"metadata"`
	Vector   []float32      `json:"vector"`
	Norm     float64        `json:"norm"`
}

// vectraIndex mirrors the Vectra/SillyTavern index.json.
type vectraIndex struct {
	Version        int            `json:"version"`
	MetadataConfig map[string]any `json:"metadata_config"`
	Items          []vectraItem   `json:"items"`
}

// vectraSourceMeta is catalog bookkeeping for one source document.
type vectraSourceMeta struct {
	Key        string `json:"key"`
	Chunks     int    `json:"chunks"`
	SHA256     string `json:"sha256,omitempty"`
	IndexedAt  string `json:"indexed_at,omitempty"`
	ChunkSize  int    `json:"chunk_size,omitempty"`
	Overlap    int    `json:"overlap,omitempty"`
	EmbedModel string `json:"embed_model,omitempty"`
}

// vectraCatalog maps source files to their index folders for fast listing.
type vectraCatalog struct {
	Collection string                      `json:"collection"`
	Backend    string                      `json:"backend"`
	Sources    map[string]vectraSourceMeta `json:"sources"`
}

// NewFileStore creates a file backend rooted at root. sourceRoot (the docs
// data-bank dir, may be empty) is stripped from source paths when deriving
// folder keys so the tree mirrors docs/.
func NewFileStore(root, sourceRoot string) *FileStore {
	if strings.TrimSpace(root) == "" {
		root = "Vectra"
	}
	if sourceRoot != "" {
		if abs, err := filepath.Abs(sourceRoot); err == nil {
			sourceRoot = abs
		}
	}
	return &FileStore{root: root, sourceRoot: sourceRoot}
}

var _ VectorStore = (*FileStore)(nil)

// --- paths ---------------------------------------------------------------

func (f *FileStore) collectionDir(name string) string {
	return filepath.Join(f.root, filepath.FromSlash(name))
}

func (f *FileStore) catalogPath(name string) string {
	return filepath.Join(f.collectionDir(name), "catalog.json")
}

func (f *FileStore) indexPath(name, key string) string {
	return filepath.Join(f.collectionDir(name), filepath.FromSlash(key), "index.json")
}

// --- key derivation ------------------------------------------------------

var unsafeSegment = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

func sanitizeSegment(p string) string {
	p = unsafeSegment.ReplaceAllString(p, "_")
	p = strings.Trim(p, "._ ")
	if p == "" {
		p = "_"
	}
	if len(p) > 80 {
		p = p[:80]
	}
	return p
}

// sourceKey turns a source path into a filesystem-safe, human-browsable folder
// key (mirrors docs/). If sanitizing alters the path, a short path hash is
// appended so distinct sources never collide.
func (f *FileStore) sourceKey(source string) string {
	s := strings.TrimSpace(source)
	if s == "" || s == "direct_text" {
		return "direct_text"
	}
	if f.sourceRoot != "" {
		if abs, err := filepath.Abs(s); err == nil {
			if rel, err := filepath.Rel(f.sourceRoot, abs); err == nil &&
				rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
				s = filepath.ToSlash(rel)
			}
		}
	}
	s = strings.TrimPrefix(filepath.ToSlash(s), "/")
	parts := strings.Split(s, "/")
	changed := false
	for i, p := range parts {
		q := sanitizeSegment(p)
		if q != p {
			changed = true
		}
		parts[i] = q
	}
	key := strings.Join(parts, "/")
	if changed {
		key += "__" + shortHash(s)
	}
	return key
}

func shortHash(s string) string {
	return fmt.Sprintf("%08x", StringHash(s))[:8]
}

// --- io ------------------------------------------------------------------

func writeFileAtomic(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (f *FileStore) loadCatalog(name string) (*vectraCatalog, error) {
	data, err := os.ReadFile(f.catalogPath(name))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read catalog: %w", err)
	}
	var cat vectraCatalog
	if err := json.Unmarshal(data, &cat); err != nil {
		return nil, fmt.Errorf("parse catalog for '%s': %w", name, err)
	}
	if cat.Sources == nil {
		cat.Sources = map[string]vectraSourceMeta{}
	}
	return &cat, nil
}

func (f *FileStore) saveCatalog(name string, cat *vectraCatalog) error {
	if cat.Sources == nil {
		cat.Sources = map[string]vectraSourceMeta{}
	}
	data, err := json.MarshalIndent(cat, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(f.catalogPath(name), data)
}

func (f *FileStore) loadIndex(name, key string) (*vectraIndex, error) {
	data, err := os.ReadFile(f.indexPath(name, key))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read index: %w", err)
	}
	var idx vectraIndex
	if err := json.Unmarshal(data, &idx); err != nil {
		return nil, fmt.Errorf("parse index %s/%s: %w", name, key, err)
	}
	return &idx, nil
}

func (f *FileStore) saveIndex(name, key string, idx *vectraIndex) error {
	if idx.MetadataConfig == nil {
		idx.MetadataConfig = map[string]any{}
	}
	if idx.Items == nil {
		idx.Items = []vectraItem{}
	}
	data, err := json.MarshalIndent(idx, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(f.indexPath(name, key), data)
}

// --- VectorStore ---------------------------------------------------------

// CreateCollection writes an empty catalog. Idempotent.
func (f *FileStore) CreateCollection(name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("collection name is required")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, err := os.Stat(f.catalogPath(name)); err == nil {
		return nil
	}
	return f.saveCatalog(name, &vectraCatalog{Collection: name, Backend: BackendVectra, Sources: map[string]vectraSourceMeta{}})
}

// CollectionExists reports whether the collection has a catalog.
func (f *FileStore) CollectionExists(name string) (bool, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	_, err := os.Stat(f.catalogPath(name))
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

// GetCollectionInfo sums catalog chunk counts.
func (f *FileStore) GetCollectionInfo(name string) (*StoreCollectionInfo, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	cat, err := f.loadCatalog(name)
	if err != nil {
		return nil, err
	}
	if cat == nil {
		return nil, fmt.Errorf("collection '%s' not found", name)
	}
	total := 0
	for _, m := range cat.Sources {
		total += m.Chunks
	}
	return &StoreCollectionInfo{Name: name, ChunkCount: total}, nil
}

// ListCollections returns every collection folder that has a catalog.
func (f *FileStore) ListCollections() ([]StoreCollectionInfo, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	entries, err := os.ReadDir(f.root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []StoreCollectionInfo
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		cat, err := f.loadCatalog(e.Name())
		if err != nil || cat == nil {
			continue
		}
		total := 0
		for _, m := range cat.Sources {
			total += m.Chunks
		}
		out = append(out, StoreCollectionInfo{Name: e.Name(), ChunkCount: total})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// UpsertPoints writes points grouped by source into their index files.
func (f *FileStore) UpsertPoints(collection string, points []Point) error {
	if len(points) == 0 {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()

	cat, err := f.loadCatalog(collection)
	if err != nil {
		return err
	}
	if cat == nil {
		cat = &vectraCatalog{Collection: collection, Backend: BackendVectra, Sources: map[string]vectraSourceMeta{}}
	}

	groups := map[string][]Point{}
	sourceOf := map[string]string{}
	for _, p := range points {
		src := payloadString(p.Payload, "source_file")
		if src == "" {
			src = payloadString(p.Payload, "source")
		}
		if src == "" {
			src = "direct_text"
		}
		key := f.sourceKey(src)
		groups[key] = append(groups[key], p)
		sourceOf[key] = src
	}

	for key, pts := range groups {
		idx, err := f.loadIndex(collection, key)
		if err != nil {
			return err
		}
		if idx == nil {
			idx = &vectraIndex{Version: 1, MetadataConfig: map[string]any{}}
		}
		byID := make(map[string]int, len(idx.Items))
		for i, it := range idx.Items {
			byID[it.ID] = i
		}
		for _, p := range pts {
			id := strconv.FormatUint(p.ID, 10)
			it := vectraItem{
				ID:       id,
				Metadata: cloneAnyMap(p.Payload),
				Vector:   p.Vector,
				Norm:     vectorNorm(p.Vector),
			}
			if i, ok := byID[id]; ok {
				idx.Items[i] = it
			} else {
				byID[id] = len(idx.Items)
				idx.Items = append(idx.Items, it)
			}
		}
		if err := f.saveIndex(collection, key, idx); err != nil {
			return err
		}
		meta := cat.Sources[sourceOf[key]]
		meta.Key = key
		meta.Chunks = len(idx.Items)
		if v := payloadString(pts[0].Payload, "indexed_at"); v != "" {
			meta.IndexedAt = v
		}
		cat.Sources[sourceOf[key]] = meta
	}
	return f.saveCatalog(collection, cat)
}

// Search loads every source index and returns the best cosine matches.
func (f *FileStore) Search(collection string, vector []float32, limit int, threshold float64) ([]StoreSearchResult, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	cat, err := f.loadCatalog(collection)
	if err != nil {
		return nil, err
	}
	if cat == nil {
		return nil, nil
	}
	qNorm := vectorNorm(vector)
	if qNorm == 0 {
		return nil, nil
	}

	var out []StoreSearchResult
	for source, meta := range cat.Sources {
		idx, err := f.loadIndex(collection, meta.Key)
		if err != nil || idx == nil {
			continue
		}
		for _, it := range idx.Items {
			if len(it.Vector) != len(vector) {
				continue
			}
			vNorm := it.Norm
			if vNorm == 0 {
				vNorm = vectorNorm(it.Vector)
			}
			score := cosine(vector, qNorm, it.Vector, vNorm)
			if threshold > 0 && score < threshold {
				continue
			}
			payload := cloneAnyMap(it.Metadata)
			if payload == nil {
				payload = map[string]any{}
			}
			if _, ok := payload["source_file"]; !ok {
				payload["source_file"] = source
			}
			out = append(out, StoreSearchResult{ID: parsePointID(it.ID), Score: score, Payload: payload})
		}
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// DeletePoints removes items matching the filter. An empty filter clears the
// whole collection. Only match.value clauses on payload keys are supported.
func (f *FileStore) DeletePoints(collection string, filter map[string]any) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	cat, err := f.loadCatalog(collection)
	if err != nil {
		return err
	}
	if cat == nil {
		return nil
	}
	matcher, err := newPayloadMatcher(filter)
	if err != nil {
		return err
	}
	for source, meta := range cat.Sources {
		idx, err := f.loadIndex(collection, meta.Key)
		if err != nil || idx == nil {
			continue
		}
		kept := make([]vectraItem, 0, len(idx.Items))
		for _, it := range idx.Items {
			if matcher(it.Metadata) {
				continue
			}
			kept = append(kept, it)
		}
		if len(kept) != len(idx.Items) {
			idx.Items = kept
			if err := f.saveIndex(collection, meta.Key, idx); err != nil {
				return err
			}
		}
		meta.Chunks = len(idx.Items)
		cat.Sources[source] = meta
	}
	return f.saveCatalog(collection, cat)
}

// DeleteCollection removes the collection directory.
func (f *FileStore) DeleteCollection(name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := os.RemoveAll(f.collectionDir(name)); err != nil {
		return err
	}
	return nil
}

// SetPayload backfills payload fields onto matching items.
func (f *FileStore) SetPayload(collection string, payload map[string]any, filter map[string]any) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	cat, err := f.loadCatalog(collection)
	if err != nil {
		return err
	}
	if cat == nil {
		return fmt.Errorf("collection '%s' not found", collection)
	}
	matcher, err := newPayloadMatcher(filter)
	if err != nil {
		return err
	}
	for _, meta := range cat.Sources {
		idx, err := f.loadIndex(collection, meta.Key)
		if err != nil || idx == nil {
			continue
		}
		changed := false
		for i := range idx.Items {
			if !matcher(idx.Items[i].Metadata) {
				continue
			}
			if idx.Items[i].Metadata == nil {
				idx.Items[i].Metadata = map[string]any{}
			}
			for k, v := range payload {
				idx.Items[i].Metadata[k] = v
			}
			changed = true
		}
		if changed {
			if err := f.saveIndex(collection, meta.Key, idx); err != nil {
				return err
			}
		}
	}
	return nil
}

// Ping verifies the root is present and writable.
func (f *FileStore) Ping() error {
	if err := os.MkdirAll(f.root, 0o755); err != nil {
		return fmt.Errorf("vectra root unavailable: %w", err)
	}
	probe := filepath.Join(f.root, ".ping")
	if err := os.WriteFile(probe, []byte("ok"), 0o644); err != nil {
		return fmt.Errorf("vectra root not writable: %w", err)
	}
	_ = os.Remove(probe)
	return nil
}

// --- helpers -------------------------------------------------------------

func vectorNorm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

func cosine(q []float32, qNorm float64, v []float32, vNorm float64) float64 {
	if qNorm == 0 || vNorm == 0 {
		return 0
	}
	var dot float64
	for i := range q {
		dot += float64(q[i]) * float64(v[i])
	}
	return dot / (qNorm * vNorm)
}

func payloadString(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	if s, ok := m[key].(string); ok {
		return s
	}
	return ""
}

func parsePointID(id string) any {
	if n, err := strconv.ParseUint(id, 10, 64); err == nil {
		return n
	}
	return id
}

func cloneAnyMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// newPayloadMatcher builds a predicate from a Qdrant-style filter. An empty
// filter matches everything. Supported: flat {key: value} and
// {"must":[{"key":..,"match":{"value":..}}]}.
func newPayloadMatcher(filter map[string]any) (func(map[string]any) bool, error) {
	if len(filter) == 0 {
		return func(map[string]any) bool { return true }, nil
	}
	var conds []func(map[string]any) bool
	if mustRaw, ok := filter["must"]; ok {
		must, ok := mustRaw.([]any)
		if !ok {
			return nil, fmt.Errorf("vectra backend: 'must' filter must be an array")
		}
		for _, m := range must {
			mm, ok := m.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("vectra backend: unsupported filter clause")
			}
			key, _ := mm["key"].(string)
			match, _ := mm["match"].(map[string]any)
			if key == "" || match == nil {
				return nil, fmt.Errorf("vectra backend: only {key, match:{value}} filters are supported")
			}
			val, present := match["value"]
			if !present {
				return nil, fmt.Errorf("vectra backend: only match.value filters are supported")
			}
			conds = append(conds, func(meta map[string]any) bool { return looseEqual(meta[key], val) })
		}
	} else {
		for k, v := range filter {
			conds = append(conds, func(meta map[string]any) bool { return looseEqual(meta[k], v) })
		}
	}
	return func(meta map[string]any) bool {
		for _, c := range conds {
			if !c(meta) {
				return false
			}
		}
		return true
	}, nil
}

func looseEqual(a, b any) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return fmt.Sprintf("%v", a) == fmt.Sprintf("%v", b)
}
