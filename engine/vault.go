package engine

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/registry"
)

// VaultFile is the self-describing marker that makes a folder a portable
// vault: rclone the whole folder and ImportVault rebuilds everything from it.
type VaultFile struct {
	VaultVersion int      `json:"vault_version"`
	Tool         string   `json:"tool"`
	Collections  []string `json:"collections,omitempty"`
	UpdatedAt    string   `json:"updated_at"`
}

// CollectionManifest is per-collection provenance inside the vault.
type CollectionManifest struct {
	Name           string `json:"name"`
	Backend        string `json:"backend"`
	EmbedModel     string `json:"embed_model,omitempty"`
	EmbedProvider  string `json:"embed_provider,omitempty"`
	VectorDim      int    `json:"vector_dim,omitempty"`
	VectorDistance string `json:"vector_distance,omitempty"`
	ChunkSize      int    `json:"chunk_size,omitempty"`
	ChunkOverlap   int    `json:"chunk_overlap,omitempty"`
	ChunkStrategy  string `json:"chunk_strategy,omitempty"`
	Chunks         int    `json:"chunks,omitempty"`
	UpdatedAt      string `json:"updated_at"`
}

func (e *Engine) vaultDir() string {
	if fs := e.FileStore(); fs != nil {
		return filepath.Join(fs.Root(), ".vectra")
	}
	return ".vectra"
}

func (e *Engine) manifestDir(collection string) string {
	if fs := e.FileStore(); fs != nil {
		return filepath.Join(fs.Root(), collection, ".vectra")
	}
	return filepath.Join("Vectra", collection, ".vectra")
}

// writeVaultManifest refreshes vault.json + the collection manifest after a
// successful write. Best-effort: failures are logged, indexing already won.
func (e *Engine) writeVaultManifest(collection string) {
	fs := e.FileStore()
	if fs == nil {
		return
	}
	provider, model, dim, distance := e.activeEmbedProvenance()
	chunks := 0
	if info, err := fs.GetCollectionInfo(collection); err == nil && info != nil {
		chunks = info.ChunkCount
	}
	var chunkSize, overlap int
	var strategy string
	if en := e.registry.Get(collection); en != nil {
		chunkSize, overlap, strategy = en.Chunk.Size, en.Chunk.Overlap, en.Chunk.Strategy
	}
	now := time.Now().UTC().Format(time.RFC3339)
	man := CollectionManifest{
		Name: collection, Backend: BackendVectra,
		EmbedModel: model, EmbedProvider: provider,
		VectorDim: dim, VectorDistance: distance,
		ChunkSize: chunkSize, ChunkOverlap: overlap, ChunkStrategy: strategy,
		Chunks: chunks, UpdatedAt: now,
	}
	if err := os.MkdirAll(e.manifestDir(collection), 0o755); err == nil {
		if data, err := json.MarshalIndent(man, "", "  "); err == nil {
			_ = os.WriteFile(filepath.Join(e.manifestDir(collection), "manifest.json"), data, 0o644)
		}
	}
	cols, _ := fs.ListCollections()
	names := make([]string, 0, len(cols))
	for _, c := range cols {
		names = append(names, c.Name)
	}
	vault := VaultFile{VaultVersion: 1, Tool: "rag-mcp-server", Collections: names, UpdatedAt: now}
	if err := os.MkdirAll(e.vaultDir(), 0o755); err == nil {
		if data, err := json.MarshalIndent(vault, "", "  "); err == nil {
			_ = os.WriteFile(filepath.Join(e.vaultDir(), "vault.json"), data, 0o644)
		}
	}
}

// VaultVerifyResult is the API/MCP shape for one collection check.
type VaultVerifyResult struct {
	Collection string        `json:"collection"`
	OK         bool          `json:"ok"`
	Issues     []VerifyIssue `json:"issues"`
}

// VerifyCollection runs the read-only self-check for one collection.
func (e *Engine) VerifyCollection(name string) VaultVerifyResult {
	fs := e.FileStore()
	if fs == nil {
		return VaultVerifyResult{Collection: name, Issues: []VerifyIssue{{Kind: "bad_index", Detail: "no file store"}}}
	}
	wantDim := 0
	if en := e.registry.Get(name); en != nil {
		wantDim = en.VectorDim
	}
	issues := fs.VerifyCollection(name, wantDim)
	if issues == nil {
		issues = []VerifyIssue{}
	}
	return VaultVerifyResult{Collection: name, OK: len(issues) == 0, Issues: issues}
}

// VerifyAll checks every stored collection (shallow: catalog/index shape +
// registry dim, no raw re-hash).
func (e *Engine) VerifyAll() []VaultVerifyResult {
	fs := e.FileStore()
	if fs == nil {
		return nil
	}
	cols, err := fs.ListCollections()
	if err != nil {
		return []VaultVerifyResult{{Collection: "*", Issues: []VerifyIssue{{Kind: "bad_index", Detail: err.Error()}}}}
	}
	out := make([]VaultVerifyResult, 0, len(cols))
	for _, c := range cols {
		out = append(out, e.VerifyCollection(c.Name))
	}
	return out
}

// PruneCollection removes empty sources from a collection and refreshes the
// manifest. Returns pruned source count.
func (e *Engine) PruneCollection(name string) (int, error) {
	fs := e.FileStore()
	if fs == nil {
		return 0, fmt.Errorf("no file store")
	}
	n, err := fs.PruneCollection(name)
	if err != nil {
		return n, err
	}
	// Refresh chunk count + manifest after prune.
	if info, err := fs.GetCollectionInfo(name); err == nil && info != nil {
		count := info.ChunkCount
		_ = e.registry.Update(name, func(en *registry.Entry) {
			en.ChunkCount = count
		})
	}
	e.writeVaultManifest(name)
	return n, nil
}

// ImportVault makes a vault folder live: it scans <root> (or <root>/.vectra
// layout) for collections, verifies them shallowly, registers missing registry
// entries from manifests, invalidates the cache and prewarms. It never
// deletes: broken collections are reported, not quarantined.
func (e *Engine) ImportVault(root string) ([]VaultVerifyResult, error) {
	fs := e.FileStore()
	if fs == nil {
		return nil, fmt.Errorf("no file store")
	}
	if root != "" {
		// Allow pointing at either the vault root or its .vectra dir.
		if _, err := os.Stat(filepath.Join(root, ".vectra", "vault.json")); err == nil {
			// root is already the vault root; nothing to remap (store root is
			// fixed at startup, this only validates the marker exists).
		} else if _, err := os.Stat(filepath.Join(root, "vault.json")); err == nil {
			// pointed directly at .vectra dir.
		} else {
			return nil, fmt.Errorf("not a vault: no .vectra/vault.json under %s", root)
		}
	}
	cols, err := fs.ListCollections()
	if err != nil {
		return nil, err
	}
	var out []VaultVerifyResult
	for _, c := range cols {
		res := e.VerifyCollection(c.Name)
		out = append(out, res)
		// Ensure a registry entry exists so the dashboard/search sees it.
		// Manifest provenance wins when present; otherwise fall back to the
		// active embedding setup (unknown dims stay 0 = unchecked).
		if e.registry.Get(c.Name) == nil {
			man := readManifest(fs.Root(), c.Name)
			provider, model, dim, dist := e.activeEmbedProvenance()
			if man != nil {
				if man.EmbedProvider != "" {
					provider = man.EmbedProvider
				}
				if man.EmbedModel != "" {
					model = man.EmbedModel
				}
				if man.VectorDim > 0 {
					dim = man.VectorDim
				}
				if man.VectorDistance != "" {
					dist = man.VectorDistance
				}
			}
			_ = e.registry.Update(c.Name, func(en *registry.Entry) {
				en.Backend = BackendVectra
				en.Enabled = true
				en.EmbedProvider = provider
				en.EmbedModel = model
				en.VectorDim = dim
				en.VectorDistance = dist
				en.ChunkCount = c.ChunkCount
			})
		}
	}
	fs.InvalidateAll()
	go func() {
		if err := fs.PrewarmAll(); err != nil {
			log.Printf("vault import prewarm: %v", err)
		}
	}()
	e.writeVaultManifestAll(cols)
	return out, nil
}

func (e *Engine) writeVaultManifestAll(cols []StoreCollectionInfo) {
	for _, c := range cols {
		e.writeVaultManifest(c.Name)
	}
}

// readManifest loads a collection manifest when present (import path).
func readManifest(root, collection string) *CollectionManifest {
	data, err := os.ReadFile(filepath.Join(root, collection, ".vectra", "manifest.json"))
	if err != nil {
		return nil
	}
	var man CollectionManifest
	if err := json.Unmarshal(data, &man); err != nil {
		return nil
	}
	return &man
}
