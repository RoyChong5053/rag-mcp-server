package webui

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/RoyChong5053/rag-mcp-server/chunking"
	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/jobs"
)

// Files larger than this skip hashing in listings (shown as status unknown-hash).
const maxHashBytes = 50 << 20

// Uploads cap at 100MB and always land in <docs>/staging/.
const maxUploadBytes = 100 << 20

var collectionNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`)

// FileEntry is one raw file under the docs root with its index status.
//
// The data bank is no longer browsed as a tree: the documents panel lists the
// vault through its ledger and searches it with ripgrep. Walking the whole
// tree on every request is what breaks at a million files.
type FileEntry struct {
	Path       string `json:"path"` // relative to docs root, slash-separated
	Size       int64  `json:"size"`
	Mtime      string `json:"mtime"`
	SHA256     string `json:"sha256,omitempty"`
	Status     string `json:"status"` // unindexed | indexed_fresh | indexed_stale | unknown
	Collection string `json:"collection,omitempty"`
}

// jail resolves rel under root and rejects escapes and absolute inputs.
func jail(root, rel string) (string, error) {
	if rel == "" || filepath.IsAbs(rel) {
		return "", fmt.Errorf("invalid path")
	}
	clean := filepath.Clean(filepath.Join(root, rel))
	if clean != root && !strings.HasPrefix(clean, root+string(os.PathSeparator)) {
		return "", fmt.Errorf("path escapes docs root")
	}
	return clean, nil
}

func fileSHA256(abs string, size int64) string {
	if size > maxHashBytes {
		return ""
	}
	f, err := os.Open(abs)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
}

// handleFileHistory returns the index/reindex/upload/delete audit trail for
// one raw file (?path= relative to the docs root), oldest first.
func (h *Handler) handleFileHistory(w http.ResponseWriter, r *http.Request) {
	rel := r.URL.Query().Get("path")
	abs, err := jail(h.docsRoot, rel)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	entries, err := h.eng.HistoryForFile(abs)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	if entries == nil {
		entries = []engine.HistoryEntry{}
	}
	writeJSON(w, http.StatusOK, entries)
}

// handleRecentUploads returns recent upload entries (newest first) so the
// dashboard can pin just-uploaded files to the top regardless of sort.
func (h *Handler) handleRecentUploads(w http.ResponseWriter, r *http.Request) {
	entries, err := h.eng.RecentUploads(50)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	if entries == nil {
		entries = []engine.HistoryEntry{}
	}
	writeJSON(w, http.StatusOK, entries)
}

func (h *Handler) handleFiles(w http.ResponseWriter, r *http.Request) {
	// The data bank is not browsed as a tree any more: the documents panel
	// lists the vault through its ledger and searches it with ripgrep. This
	// endpoint stays as an empty stub so old clients do not break loudly.
	writeJSON(w, http.StatusOK, []FileEntry{})
}

func (h *Handler) handleUpload(w http.ResponseWriter, r *http.Request) {
	// Uploads are filed into the documents vault now (see
	// handleDocumentUpload). This stub stays so old clients keep working.
	writeErr(w, http.StatusBadRequest, fmt.Errorf("uploads go to /api/documents/upload"))
}

type indexBody struct {
	Path           string `json:"path"`
	Collection     string `json:"collection"`
	Backend        string `json:"backend"` // accepted for compatibility, always vectra
	ChunkSize      int    `json:"chunk_size"`
	OverlapPercent int    `json:"overlap_percent"`
	Strategy       string `json:"strategy"`
	Mode           string `json:"mode"` // append (default) | rebuild
}

func (h *Handler) handleSubmitIndex(w http.ResponseWriter, r *http.Request) {
	var body indexBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	abs, err := jail(h.docsRoot, body.Path)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if st, err := os.Stat(abs); err != nil || st.IsDir() {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("file not found under docs/"))
		return
	}
	if !collectionNameRe.MatchString(body.Collection) {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("bad collection name (A-Za-z0-9_-, max 64, must start alnum)"))
		return
	}
	backend := engine.BackendVectra
	_ = strings.TrimSpace(body.Backend) // accepted for compatibility; always vectra
	if body.ChunkSize < 0 || body.ChunkSize > 20000 || body.OverlapPercent < 0 || body.OverlapPercent >= 100 {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("chunk_size 0-20000 (0=global), overlap_percent 0-99"))
		return
	}
	if body.Strategy != "" && !chunking.IsValidStrategy(body.Strategy) {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("unknown strategy %q (valid: %s)", body.Strategy, strings.Join(chunking.StrategyIDs(), ", ")))
		return
	}
	rebuild := body.Mode == "rebuild"
	if body.Mode != "" && body.Mode != "append" && body.Mode != "rebuild" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("mode must be append or rebuild"))
		return
	}
	opts := &engine.ChunkOptions{Size: body.ChunkSize, OverlapPercent: body.OverlapPercent, Strategy: body.Strategy}
	job := h.jobs.SubmitIndex(abs, body.Collection, backend, opts, rebuild)
	writeJSON(w, http.StatusAccepted, job)
}

func (h *Handler) handleJobs(w http.ResponseWriter, r *http.Request) {
	// The dashboard shows the background work that is actually running:
	// index jobs (an upload being vectorized) and web jobs (the pages a
	// web_search found, being filed while the user keeps chatting).
	writeJSON(w, http.StatusOK, h.jobs.ListKinds(jobs.KindIndex, jobs.KindWeb))
}

// handleClearJobs drops finished jobs from the sidebar; in-flight jobs stay.
func (h *Handler) handleClearJobs(w http.ResponseWriter, r *http.Request) {
	removed := h.jobs.Clear()
	writeJSON(w, http.StatusOK, map[string]any{"removed": removed})
}

func (h *Handler) handleJob(w http.ResponseWriter, r *http.Request) {
	job := h.jobs.Get(r.PathValue("id"))
	if job == nil {
		writeErr(w, http.StatusNotFound, fmt.Errorf("job not found"))
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func (h *Handler) handlePreview(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Path           string `json:"path"`
		ChunkSize      int    `json:"chunk_size"`
		OverlapPercent int    `json:"overlap_percent"`
		Strategy       string `json:"strategy"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	abs, err := jail(h.docsRoot, body.Path)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("read file: %w", err))
		return
	}
	size := body.ChunkSize
	overlapPct := body.OverlapPercent
	resolved := h.eng.ResolveChunkOptions(&engine.ChunkOptions{Size: size, OverlapPercent: overlapPct, Strategy: body.Strategy})
	count, samples := engine.PreviewChunks(string(data), resolved.Size, resolved.Overlap, resolved.Strategy, 5)
	writeJSON(w, http.StatusOK, map[string]any{
		"chunks":   count,
		"samples":  samples,
		"size":     resolved.Size,
		"overlap":  resolved.Overlap,
		"strategy": resolved.Strategy,
	})
}

func (h *Handler) handleBackup(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", `attachment; filename="collections.json"`)
	writeJSON(w, http.StatusOK, h.eng.Registry().List())
}
