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
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
)

// Files larger than this skip hashing in listings (shown as status unknown-hash).
const maxHashBytes = 50 << 20

// Uploads cap at 100MB and always land in <docs>/staging/.
const maxUploadBytes = 100 << 20

var collectionNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`)

// FileEntry is one raw file under the docs root with its index status.
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

// listFiles walks the docs root (cap 2000 entries) and matches registry provenance.
func (h *Handler) listFiles() ([]FileEntry, error) {
	bySource := make(map[string]string) // abs source_file -> collection
	for name, en := range h.eng.Registry().List() {
		if en.SourceFile != "" && en.SourceFile != "direct_text" {
			bySource[en.SourceFile] = name
		}
	}
	var out []FileEntry
	err := filepath.Walk(h.docsRoot, func(abs string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip unreadable, keep walking
		}
		if info.IsDir() {
			return nil
		}
		if len(out) >= 2000 {
			return filepath.SkipAll
		}
		rel, err := filepath.Rel(h.docsRoot, abs)
		if err != nil {
			return nil
		}
		fe := FileEntry{
			Path:   filepath.ToSlash(rel),
			Size:   info.Size(),
			Mtime:  info.ModTime().UTC().Format(time.RFC3339),
			Status: "unindexed",
		}
		if col, ok := bySource[abs]; ok {
			fe.Collection = col
			en := h.eng.Registry().Get(col)
			sum := fileSHA256(abs, info.Size())
			fe.SHA256 = sum
			switch {
			case sum == "" || en == nil || en.SourceSHA256 == "":
				fe.Status = "unknown"
			case sum == en.SourceSHA256:
				fe.Status = "indexed_fresh"
			default:
				fe.Status = "indexed_stale"
			}
		}
		out = append(out, fe)
		return nil
	})
	return out, err
}

func (h *Handler) handleFiles(w http.ResponseWriter, r *http.Request) {
	entries, err := h.listFiles()
	if err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	if entries == nil {
		entries = []FileEntry{}
	}
	writeJSON(w, http.StatusOK, entries)
}

func (h *Handler) handleUpload(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("parse upload: %w", err))
		return
	}
	f, hdr, err := r.FormFile("file")
	if err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("missing file field: %w", err))
		return
	}
	defer f.Close()
	if hdr.Size > maxUploadBytes {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("file too large (max 100MB)"))
		return
	}
	// Sanitize to a bare filename; subdir targeting is not allowed.
	name := filepath.Base(hdr.Filename)
	name = strings.TrimSpace(name)
	if name == "" || name == "." || strings.ContainsAny(name, `/\`) {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid filename"))
		return
	}
	staging := filepath.Join(h.docsRoot, "staging")
	if err := os.MkdirAll(staging, 0o755); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	abs := filepath.Join(staging, name)
	dst, err := os.Create(abs)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	written, err := io.Copy(dst, f)
	dst.Close()
	if err != nil {
		os.Remove(abs)
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	rel, _ := filepath.Rel(h.docsRoot, abs)
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"path":   filepath.ToSlash(rel),
		"size":   written,
	})
}

type indexBody struct {
	Path           string `json:"path"`
	Collection     string `json:"collection"`
	Backend        string `json:"backend"` // "" (default) | qdrant | vectra
	ChunkSize      int    `json:"chunk_size"`
	OverlapPercent int    `json:"overlap_percent"`
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
	backend := strings.TrimSpace(body.Backend)
	if backend != "" && backend != engine.BackendQdrant && backend != engine.BackendVectra {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("backend must be qdrant or vectra (empty = server default)"))
		return
	}
	// A collection's backend is fixed once chosen. Refuse to index into a
	// different backend than the one the registry says it lives in, and refuse
	// to create a same-name collection in the other store: either would leave
	// vectors split across backends and silently break recall.
	if backend != "" {
		if en := h.eng.Registry().Get(body.Collection); en != nil && en.Backend != "" && en.Backend != backend {
			writeErr(w, http.StatusConflict, fmt.Errorf("collection '%s' is fixed to %s; refusing to index into %s", body.Collection, en.Backend, backend))
			return
		}
		exists, err := h.eng.CollectionExistsOnOther(backend, body.Collection)
		if err != nil {
			writeErr(w, http.StatusBadGateway, err)
			return
		}
		if exists {
			writeErr(w, http.StatusConflict, fmt.Errorf("collection '%s' already exists in the other backend; delete or rename it before vectorizing into %s", body.Collection, backend))
			return
		}
	}
	if body.ChunkSize < 0 || body.ChunkSize > 20000 || body.OverlapPercent < 0 || body.OverlapPercent >= 100 {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("chunk_size 0-20000 (0=global), overlap_percent 0-99"))
		return
	}
	rebuild := body.Mode == "rebuild"
	if body.Mode != "" && body.Mode != "append" && body.Mode != "rebuild" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("mode must be append or rebuild"))
		return
	}
	opts := &engine.ChunkOptions{Size: body.ChunkSize, OverlapPercent: body.OverlapPercent}
	job := h.jobs.SubmitIndex(abs, body.Collection, backend, opts, rebuild)
	writeJSON(w, http.StatusAccepted, job)
}

func (h *Handler) handleJobs(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, h.jobs.List())
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
	resolved := h.eng.ResolveChunkOptions(&engine.ChunkOptions{Size: size, OverlapPercent: overlapPct})
	count, samples := engine.PreviewChunks(string(data), resolved.Size, resolved.Overlap, 5)
	writeJSON(w, http.StatusOK, map[string]any{
		"chunks":  count,
		"samples": samples,
		"size":    resolved.Size,
		"overlap": resolved.Overlap,
	})
}

func (h *Handler) handleBackup(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", `attachment; filename="collections.json"`)
	writeJSON(w, http.StatusOK, h.eng.Registry().List())
}
