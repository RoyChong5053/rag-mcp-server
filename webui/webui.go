package webui

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

//go:embed index.html
var indexHTML string

// Handler serves the management dashboard and its JSON API.
// It is meant to bind localhost-only (see AdminConfig); reach it via ssh tunnel.
type Handler struct {
	eng       *engine.Engine
	auditPath string
	docsRoot  string
	jobs      *Manager
}

func New(eng *engine.Engine, auditPath, docsRoot string) *Handler {
	return &Handler{eng: eng, auditPath: auditPath, docsRoot: docsRoot, jobs: NewManager(eng, 2)}
}

func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", h.serveIndex)
	mux.HandleFunc("GET /api/health", h.health)
	mux.HandleFunc("GET /api/collections", h.listCollections)
	mux.HandleFunc("POST /api/collections", h.createCollection)
	mux.HandleFunc("GET /api/collections/{name}", h.getCollection)
	mux.HandleFunc("POST /api/collections/{name}/meta", h.setMeta)
	mux.HandleFunc("POST /api/collections/{name}/delete", h.deleteCollection)
	mux.HandleFunc("POST /api/collections/{name}/backfill", h.backfill)
	mux.HandleFunc("POST /api/collections/{name}/search", h.search)
	mux.HandleFunc("GET /api/files", h.handleFiles)
	mux.HandleFunc("POST /api/upload", h.handleUpload)
	mux.HandleFunc("POST /api/index", h.handleSubmitIndex)
	mux.HandleFunc("GET /api/jobs", h.handleJobs)
	mux.HandleFunc("POST /api/jobs/clear", h.handleClearJobs)
	mux.HandleFunc("GET /api/jobs/{id}", h.handleJob)
	mux.HandleFunc("POST /api/preview", h.handlePreview)
	mux.HandleFunc("GET /api/registry/backup", h.handleBackup)
	mux.HandleFunc("GET /api/audit", h.audit)
	mux.HandleFunc("GET /api/settings", h.getSettings)
	mux.HandleFunc("POST /api/settings", h.setSettings)
	return mux
}

func (h *Handler) serveIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, indexHTML)
}

func (h *Handler) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, h.eng.HealthCheck())
}

// getSettings returns the runtime search settings shared by all frontends.
func (h *Handler) getSettings(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, h.eng.Settings().Get())
}

// setSettings applies a partial settings patch. A non-empty default_collection
// must exist in Qdrant: a typo here would otherwise silently widen searches to
// a global scan, which is hard to notice and looks like flaky recall.
func (h *Handler) setSettings(w http.ResponseWriter, r *http.Request) {
	var patch settings.Patch
	if err := json.NewDecoder(r.Body).Decode(&patch); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	if patch.ActiveBackend != nil {
		b := strings.TrimSpace(*patch.ActiveBackend)
		if b != "" && b != engine.BackendQdrant && b != engine.BackendVectra {
			writeErr(w, http.StatusBadRequest, fmt.Errorf("active_backend must be %q or %q (empty = follow config)", engine.BackendQdrant, engine.BackendVectra))
			return
		}
		patch.ActiveBackend = &b
	}
	validate := func(field **string, backend string) error {
		if *field == nil {
			return nil
		}
		name := strings.TrimSpace(**field)
		*field = &name
		if name == "" {
			return nil
		}
		exists, err := h.eng.CollectionExistsOn(backend, name)
		if err != nil {
			return err
		}
		if !exists {
			return fmt.Errorf("collection '%s' not found in %s; refusing a default that cannot be searched", name, backend)
		}
		return nil
	}
	if err := validate(&patch.DefaultCollectionQdrant, engine.BackendQdrant); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if err := validate(&patch.DefaultCollectionVectra, engine.BackendVectra); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if err := h.eng.Settings().Update(patch); err != nil {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("save settings: %w", err))
		return
	}
	writeJSON(w, http.StatusOK, h.eng.Settings().Get())
}

func (h *Handler) listCollections(w http.ResponseWriter, r *http.Request) {
	details, err := h.eng.DescribeCollections("")
	if err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, details)
}

// createCollection makes an empty collection in Qdrant. Lets the dashboard
// set up a fresh write target (e.g. a global memory bucket) before anything is
// indexed into it — otherwise the settings default would reject the name as
// "not found".
func (h *Handler) createCollection(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	name := strings.TrimSpace(body.Name)
	if !collectionNameRe.MatchString(name) {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("bad collection name (A-Za-z0-9_-, max 64, must start alnum)"))
		return
	}
	if err := h.eng.EnsureCollection(name); err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "created", "name": name})
}

func (h *Handler) getCollection(w http.ResponseWriter, r *http.Request) {
	details, err := h.eng.DescribeCollections(r.PathValue("name"))
	if err != nil {
		writeErr(w, http.StatusNotFound, err)
		return
	}
	writeJSON(w, http.StatusOK, details[0])
}

type metaBody struct {
	DisplayName *string  `json:"display_name"`
	Description *string  `json:"description"`
	Tags        []string `json:"tags"`
	Consumers   []string `json:"consumers"`
	Enabled     *bool    `json:"enabled"`
	Backend     *string  `json:"backend"`
	// hasTags/hasConsumers distinguish "missing" from "empty array"
	HasTags      bool `json:"-"`
	HasConsumers bool `json:"-"`
}

func (h *Handler) setMeta(w http.ResponseWriter, r *http.Request) {
	var raw map[string]any
	if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	var body metaBody
	if err := mapToMeta(raw, &body); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	update := engine.CollectionMetaUpdate{
		DisplayName: body.DisplayName,
		Description: body.Description,
		Enabled:     body.Enabled,
		Backend:     body.Backend,
	}
	if body.HasTags {
		update.Tags = &body.Tags
	}
	if body.HasConsumers {
		update.Consumers = &body.Consumers
	}
	if err := h.eng.SetCollectionMeta(r.PathValue("name"), update); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func mapToMeta(raw map[string]any, out *metaBody) error {
	strPtr := func(key string) (*string, error) {
		v, ok := raw[key]
		if !ok || v == nil {
			return nil, nil
		}
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("%s must be a string", key)
		}
		return &s, nil
	}
	strSlice := func(key string) ([]string, bool, error) {
		v, ok := raw[key]
		if !ok || v == nil {
			return nil, false, nil
		}
		arr, ok := v.([]any)
		if !ok {
			return nil, false, fmt.Errorf("%s must be an array of strings", key)
		}
		out := make([]string, 0, len(arr))
		for _, item := range arr {
			s, ok := item.(string)
			if !ok {
				return nil, false, fmt.Errorf("%s must be an array of strings", key)
			}
			out = append(out, s)
		}
		return out, true, nil
	}
	var err error
	if out.DisplayName, err = strPtr("display_name"); err != nil {
		return err
	}
	if out.Description, err = strPtr("description"); err != nil {
		return err
	}
	if out.Backend, err = strPtr("backend"); err != nil {
		return err
	}
	if out.Tags, out.HasTags, err = strSlice("tags"); err != nil {
		return err
	}
	if out.Consumers, out.HasConsumers, err = strSlice("consumers"); err != nil {
		return err
	}
	if v, ok := raw["enabled"]; ok && v != nil {
		b, ok := v.(bool)
		if !ok {
			return fmt.Errorf("enabled must be a boolean")
		}
		out.Enabled = &b
	}
	return nil
}

func (h *Handler) deleteCollection(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Confirm bool `json:"confirm"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	if err := h.eng.DeleteCollection(r.PathValue("name"), body.Confirm); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

func (h *Handler) backfill(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var body struct {
		SourceFile string `json:"source_file"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body) // optional

	sourceFile := body.SourceFile
	if sourceFile == "" {
		if en := h.eng.Registry().Get(name); en != nil && en.SourceFile != "" {
			sourceFile = en.SourceFile
		}
	}
	if sourceFile == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("no source_file known; pass one explicitly"))
		return
	}
	payload := map[string]any{
		"source_file": sourceFile,
		"indexed_at":  time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.eng.BackfillPayload(name, payload); err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	// Adopt legacy collections: stamp source_file + sha + live chunk count
	// so the data bank can badge them fresh/stale without re-embedding.
	abs := sourceFile
	if !filepath.IsAbs(abs) {
		if j, err := jail(h.docsRoot, abs); err == nil {
			abs = j
		}
	}
	var sha string
	if st, err := os.Stat(abs); err == nil && !st.IsDir() {
		sha = fileSHA256(abs, st.Size())
	}
	chunkCount := 0
	if d, err := h.eng.DescribeCollections(name); err == nil && len(d) > 0 {
		chunkCount = d[0].ChunkCount
	}
	_ = h.eng.Registry().Update(name, func(en *registry.Entry) {
		if en.SourceFile == "" {
			en.SourceFile = sourceFile
		}
		if sha != "" && en.SourceSHA256 == "" {
			en.SourceSHA256 = sha
		}
		if chunkCount > 0 {
			en.ChunkCount = chunkCount
		}
	})
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "payload": payload})
}

func (h *Handler) search(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Query     string   `json:"query"`
		TopK      int      `json:"top_k"`
		Rerank    *bool    `json:"rerank"`
		Threshold *float64 `json:"threshold"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	if strings.TrimSpace(body.Query) == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("query is required"))
		return
	}
	if body.TopK <= 0 {
		body.TopK = 5
	}
	useRerank := true
	if body.Rerank != nil {
		useRerank = *body.Rerank
	}
	threshold := 0.0
	if body.Threshold != nil {
		threshold = *body.Threshold
	}
	// SearchDebug also reports candidates the threshold killed,
	// so recall tuning is evidence-based instead of guesswork.
	res, err := h.eng.SearchDebug(body.Query, r.PathValue("name"), body.TopK, useRerank, threshold)
	if err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (h *Handler) audit(w http.ResponseWriter, r *http.Request) {
	lines := 200
	if v := r.URL.Query().Get("lines"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 2000 {
			lines = n
		}
	}
	data, err := os.ReadFile(h.auditPath)
	if err != nil {
		writeErr(w, http.StatusNotFound, fmt.Errorf("audit log unavailable: %w", err))
		return
	}
	// Tail: walk back N newlines without loading line-split of a huge file twice
	text := string(data)
	cut := 0
	count := 0
	for i := len(text) - 1; i >= 0; i-- {
		if text[i] == '\n' {
			count++
			if count > lines {
				cut = i + 1
				break
			}
		}
	}
	writeJSON(w, http.StatusOK, map[string]string{"log": text[cut:]})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}
