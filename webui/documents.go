package webui

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/jobs"
)

// The documents panel is the other half of the console. The vector side is
// "what did I learn about X"; this side is "where is that exact string".
//
// Nothing here walks the corpus. The listing comes from the vault's ledger
// and the search comes from ripgrep (or the built-in scanner), so the
// dashboard stays usable when the corpus has millions of files.

// docRow is one document in the listing.
type docRow struct {
	engine.DocEntry
	// DocsPath is the document path relative to the data-bank root, so the
	// dashboard can hand it to the index/preview endpoints unchanged.
	DocsPath string `json:"docs_path"`
}

// handleDocuments lists the documents of the vault, newest first.
func (h *Handler) handleDocuments(w http.ResponseWriter, r *http.Request) {
	channel := strings.TrimSpace(r.URL.Query().Get("channel"))
	limit := 50
	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 1000 {
			limit = n
		}
	}
	entries := h.eng.ListDocuments(channel, limit)
	rows := make([]docRow, 0, len(entries))
	for _, e := range entries {
		rows = append(rows, docRow{DocEntry: e, DocsPath: h.eng.DocsPath(e.Path)})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"documents": rows,
		"channels":  h.eng.DocChannels(),
		"dir":       h.eng.DocumentsDir(),
		"dir_rel":   h.eng.DocumentsDirRel(),
		"count":     len(rows),
	})
}

// handleDocumentSearch searches the raw documents of the vault.
func (h *Handler) handleDocumentSearch(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	opts := engine.DocSearchOptions{
		Query:         q.Get("q"),
		Mode:          q.Get("mode"),
		Path:          q.Get("path"),
		Channel:       q.Get("channel"),
		Category:      q.Get("category"),
		Since:         q.Get("since"),
		Until:         q.Get("until"),
		Word:          boolParam(q.Get("word")),
		CaseSensitive: boolParam(q.Get("case_sensitive")),
		Context:       intParam(q.Get("context"), 0),
		Limit:         intParam(q.Get("limit"), 0),
		PerFile:       intParam(q.Get("per_file"), 0),
	}
	res, err := h.eng.SearchDocuments(opts)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

// handleDocumentRaw reads one document out of the vault.
func (h *Handler) handleDocumentRaw(w http.ResponseWriter, r *http.Request) {
	rel := strings.TrimSpace(r.URL.Query().Get("path"))
	if rel == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("path is required"))
		return
	}
	doc, err := h.eng.ReadDocument(rel)
	if err != nil {
		writeErr(w, http.StatusNotFound, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"path":      doc.Rel,
		"docs_path": h.eng.DocsPath(doc.Rel),
		"meta":      doc.Meta,
		"body":      doc.Body,
	})
}

// handleDocumentImport files a folder of raw material into the vault. It is
// a background job: a big folder takes minutes.
func (h *Handler) handleDocumentImport(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Path       string `json:"path"`
		Collection string `json:"collection"`
		Reindex    bool   `json:"reindex"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	path := strings.TrimSpace(body.Path)
	if path == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("path is required"))
		return
	}
	collection := strings.TrimSpace(body.Collection)
	reindex := body.Reindex || collection != ""
	job := h.jobs.Submit(jobs.KindImport, func() (any, error) {
		return h.eng.ImportDocuments(path, engine.DocChannelImport, collection, reindex, 0, nil)
	})
	writeJSON(w, http.StatusAccepted, job)
}

// handleDocumentUpload files an uploaded file into the vault. The document
// is filed by machine: dated, categorized and named after its content hash.
func (h *Handler) handleDocumentUpload(w http.ResponseWriter, r *http.Request) {
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
	name := filepath.Base(hdr.Filename)
	name = strings.TrimSpace(name)
	if name == "" || name == "." || strings.ContainsAny(name, `/\`) {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid filename"))
		return
	}
	data, err := readAllLimited(f, maxUploadBytes)
	if err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("read upload: %w", err))
		return
	}
	if looksBinaryData(data) {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("binary files are not filed"))
		return
	}
	ext := strings.ToLower(filepath.Ext(name))
	if ext == "" {
		ext = ".md"
	}
	file, err := h.eng.FileDocument(engine.Document{
		Channel: engine.DocChannelUpload,
		Title:   name,
		Text:    string(data),
		Ext:     ext,
	})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	if err := h.eng.AppendHistory(engine.HistoryEntry{Op: "upload", File: file.Path, OK: true}); err != nil {
		log.Printf("history append failed: %v", err)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":   "ok",
		"path":     file.Rel,
		"abs":      file.Path,
		"deduped":  file.Deduped,
		"docsPath": h.eng.DocsPath(file.Rel),
	})
}

// boolParam reads a boolean query parameter.
func boolParam(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// intParam reads an integer query parameter with a fallback.
func intParam(v string, def int) int {
	if strings.TrimSpace(v) == "" {
		return def
	}
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil {
		return def
	}
	return n
}

// readAllLimited reads at most limit bytes.
func readAllLimited(r io.Reader, limit int64) ([]byte, error) {
	return io.ReadAll(io.LimitReader(r, limit+1))
}

// looksBinaryData reports whether a buffer looks like a binary file.
func looksBinaryData(data []byte) bool {
	limit := len(data)
	if limit > 8000 {
		limit = 8000
	}
	for i := 0; i < limit; i++ {
		if data[i] == 0 {
			return true
		}
	}
	return false
}
