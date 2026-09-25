package webui

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

//go:embed index.html
var indexHTML string

// BuildInfo is process metadata shown in the dashboard header.
type BuildInfo struct {
	Version   string
	Commit    string
	StartedAt time.Time
}

// Handler serves the management dashboard and its JSON API.
// It is meant to bind localhost-only (see AdminConfig); reach it via ssh tunnel.
type Handler struct {
	eng         *engine.Engine
	auditPath   string
	docsRoot    string
	jobs        *Manager
	build       BuildInfo
	sessions    *SessionManager
	adminUser   string
	adminPass   string // hex(sha256(password))
	sessionDays int
}

// AdminAuth carries the optional login gate (empty username = disabled).
type AdminAuth struct {
	Username       string
	PasswordSHA256 string
	SessionDays    int
	SessionFile    string
}

func New(eng *engine.Engine, auditPath, docsRoot string, info BuildInfo) *Handler {
	return NewWithAuth(eng, auditPath, docsRoot, info, AdminAuth{})
}

func NewWithAuth(eng *engine.Engine, auditPath, docsRoot string, info BuildInfo, auth AdminAuth) *Handler {
	if info.StartedAt.IsZero() {
		info.StartedAt = time.Now()
	}
	days := auth.SessionDays
	if days <= 0 {
		days = 30
	}
	return &Handler{
		eng: eng, auditPath: auditPath, docsRoot: docsRoot, jobs: NewManager(eng, 2), build: info,
		sessions:  NewSessionManager(auth.SessionFile),
		adminUser: strings.TrimSpace(auth.Username), adminPass: strings.TrimSpace(auth.PasswordSHA256),
		sessionDays: days,
	}
}

// authEnabled reports whether a login gate is configured.
func (h *Handler) authEnabled() bool { return h.adminUser != "" && h.adminPass != "" }

// requireAuth guards stateful/private APIs. /api/health, /api/info and
// /api/login stay public so probes and the login itself keep working.
func (h *Handler) requireAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !h.authEnabled() {
			next(w, r)
			return
		}
		if h.sessions.Valid(BearerToken(r.Header.Get("Authorization"))) {
			next(w, r)
			return
		}
		writeErr(w, http.StatusUnauthorized, fmt.Errorf("login required"))
	}
}

func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", h.serveIndex)
	mux.HandleFunc("GET /api/health", h.health)
	mux.HandleFunc("GET /api/info", h.info)
	mux.HandleFunc("POST /api/login", h.login)
	mux.HandleFunc("POST /api/logout", h.logout)
	mux.HandleFunc("GET /api/me", h.me)
	mux.HandleFunc("GET /api/collections", h.requireAuth(h.listCollections))
	mux.HandleFunc("POST /api/collections", h.requireAuth(h.createCollection))
	mux.HandleFunc("GET /api/collections/{name}", h.requireAuth(h.getCollection))
	mux.HandleFunc("POST /api/collections/{name}/meta", h.requireAuth(h.setMeta))
	mux.HandleFunc("POST /api/collections/{name}/delete", h.requireAuth(h.deleteCollection))
	mux.HandleFunc("POST /api/collections/{name}/backfill", h.requireAuth(h.backfill))
	mux.HandleFunc("POST /api/collections/{name}/search", h.requireAuth(h.search))
	mux.HandleFunc("GET /api/files", h.requireAuth(h.handleFiles))
	mux.HandleFunc("POST /api/upload", h.requireAuth(h.handleUpload))
	mux.HandleFunc("POST /api/index", h.requireAuth(h.handleSubmitIndex))
	mux.HandleFunc("GET /api/jobs", h.requireAuth(h.handleJobs))
	mux.HandleFunc("POST /api/jobs/clear", h.requireAuth(h.handleClearJobs))
	mux.HandleFunc("GET /api/jobs/{id}", h.requireAuth(h.handleJob))
	mux.HandleFunc("POST /api/preview", h.requireAuth(h.handlePreview))
	mux.HandleFunc("GET /api/registry/backup", h.requireAuth(h.handleBackup))
	mux.HandleFunc("GET /api/audit", h.requireAuth(h.audit))
	mux.HandleFunc("GET /api/settings", h.requireAuth(h.getSettings))
	mux.HandleFunc("POST /api/settings", h.requireAuth(h.setSettings))
	return mux
}

func (h *Handler) serveIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, indexHTML)
}

func (h *Handler) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, h.eng.HealthCheck())
}

// info reports build/uptime metadata for the dashboard header.
func (h *Handler) info(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"version":        h.build.Version,
		"commit":         h.build.Commit,
		"go_version":     runtime.Version(),
		"started_at":     h.build.StartedAt.UTC().Format(time.RFC3339),
		"uptime_seconds": int64(time.Since(h.build.StartedAt).Seconds()),
		"auth_enabled":   h.authEnabled(),
	})
}

// login verifies username/password and issues a Bearer token.
// Body: {username, password, remember?}. remember=true -> session_days TTL,
// otherwise 12h. Always 401 with the same message on any mismatch.
func (h *Handler) login(w http.ResponseWriter, r *http.Request) {
	if !h.authEnabled() {
		writeJSON(w, http.StatusOK, map[string]any{"auth": "disabled"})
		return
	}
	var body struct {
		Username string `json:"username"`
		Password string `json:"password"`
		Remember bool   `json:"remember"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	if strings.TrimSpace(body.Username) != h.adminUser || !VerifyPassword(h.adminPass, body.Password) {
		writeErr(w, http.StatusUnauthorized, fmt.Errorf("invalid credentials"))
		return
	}
	ttl := 12 * time.Hour
	if body.Remember {
		ttl = time.Duration(h.sessionDays) * 24 * time.Hour
	}
	tok, exp := h.sessions.Issue(ttl)
	writeJSON(w, http.StatusOK, map[string]any{"token": tok, "expires_at": exp.Format(time.RFC3339)})
}

func (h *Handler) logout(w http.ResponseWriter, r *http.Request) {
	h.sessions.Revoke(BearerToken(r.Header.Get("Authorization")))
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// me reports whether the request's token is valid (used by the UI boot).
func (h *Handler) me(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"auth_enabled": h.authEnabled(),
		"ok":           !h.authEnabled() || h.sessions.Valid(BearerToken(r.Header.Get("Authorization"))),
		"user":         h.adminUser,
	})
}

// getSettings returns the runtime search settings shared by all frontends.
func (h *Handler) getSettings(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, h.eng.Settings().Get())
}

// settingsResponse is the flat settings object plus non-blocking advisories,
// so the dashboard can persist an offline edit and still tell the operator
// which default collection could not be verified.
type settingsResponse struct {
	settings.Settings
	Warnings []string `json:"warnings,omitempty"`
}

// setSettings applies a partial settings patch. It never blocks on backend
// reachability: a default collection is only a routing hint, and refusing to
// save while qdrant is down would defeat the very failover the setting exists
// to configure. Names that cannot be verified are returned as warnings; a typo
// still fails loudly at the first search (it never silently widens).
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

	var warnings []string
	check := func(field **string, backend string) {
		if *field == nil {
			return
		}
		name := strings.TrimSpace(**field)
		*field = &name
		if name == "" {
			return
		}
		exists, err := h.eng.CollectionExistsOn(backend, name)
		switch {
		case err != nil && engine.IsUnavailable(err):
			warnings = append(warnings, fmt.Sprintf("%s unreachable; '%s' saved but not verified", backend, name))
		case err != nil:
			warnings = append(warnings, fmt.Sprintf("could not check '%s' on %s: %v", name, backend, err))
		case !exists:
			warnings = append(warnings, fmt.Sprintf("'%s' not found in %s; search/store will fail until it exists", name, backend))
		}
	}
	check(&patch.DefaultCollectionQdrant, engine.BackendQdrant)
	check(&patch.DefaultCollectionVectra, engine.BackendVectra)

	if err := h.eng.Settings().Update(patch); err != nil {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("save settings: %w", err))
		return
	}
	writeJSON(w, http.StatusOK, settingsResponse{Settings: h.eng.Settings().Get(), Warnings: warnings})
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
	res, err := h.eng.SearchDebug(body.Query, r.PathValue("name"), body.TopK, useRerank, threshold, nil)
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
	// Tail without loading the whole file: seek to the last 256KB at most,
	// then cut to the requested line count. Keeps the dashboard snappy even
	// when the audit log has grown to tens of MB between rotations.
	const maxTail = 256 << 10
	f, err := os.Open(h.auditPath)
	if err != nil {
		writeErr(w, http.StatusNotFound, fmt.Errorf("audit log unavailable: %w", err))
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		writeErr(w, http.StatusNotFound, fmt.Errorf("audit log unavailable: %w", err))
		return
	}
	start := int64(0)
	if st.Size() > maxTail {
		start = st.Size() - maxTail
	}
	if _, err := f.Seek(start, 0); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	buf := make([]byte, 0, 64<<10)
	chunk := make([]byte, 32<<10)
	for {
		n, rerr := f.Read(chunk)
		if n > 0 {
			buf = append(buf, chunk[:n]...)
		}
		if rerr != nil {
			break
		}
	}
	text := string(buf)
	// If we started mid-line, drop the first partial line.
	if start > 0 {
		if i := strings.IndexByte(text, '\n'); i >= 0 {
			text = text[i+1:]
		} else {
			text = ""
		}
	}
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
