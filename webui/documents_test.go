package webui

import (
	"encoding/json"
	"mime/multipart"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// newDocumentHandler builds a dashboard over an empty vault.
func newDocumentHandler(t *testing.T) (*Handler, *engine.Engine) {
	t.Helper()
	dir := t.TempDir()
	cfg := &engine.EngineConfig{
		VectraDir:    filepath.Join(dir, "Vectra"),
		Backend:      engine.BackendVectra,
		DocsDir:      dir,
		DocumentsDir: filepath.Join(dir, "documents"),
		RegistryPath: filepath.Join(dir, "collections.json"),
	}
	eng, err := engine.NewEngine(cfg, settings.New(filepath.Join(dir, "settings.json"), settings.Settings{}))
	if err != nil {
		t.Fatalf("engine: %v", err)
	}
	return New(eng, filepath.Join(dir, "audit.log"), dir, BuildInfo{}), eng
}

// The listing comes from the ledger, never from a walk of the corpus.
func TestDocumentsListingUsesTheLedger(t *testing.T) {
	h, eng := newDocumentHandler(t)
	if _, err := eng.FileDocument(engine.Document{
		Channel: "web", Title: "A page", Text: "the body of the page",
	}); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest("GET", "/api/documents?limit=10", nil)
	rec := httptest.NewRecorder()
	h.handleDocuments(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Count     int               `json:"count"`
		Documents []engine.DocEntry `json:"documents"`
		Channels  map[string]int    `json:"channels"`
		Dir       string            `json:"dir"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Count != 1 {
		t.Fatalf("expected 1 document, got %d", resp.Count)
	}
	if resp.Documents[0].Title != "A page" {
		t.Errorf("unexpected title: %q", resp.Documents[0].Title)
	}
	if resp.Channels["web"] != 1 {
		t.Errorf("expected 1 web document, got %v", resp.Channels)
	}
}

// The search returns the document's identity, not just a file name.
func TestDocumentSearchReturnsIdentity(t *testing.T) {
	h, eng := newDocumentHandler(t)
	if _, err := eng.FileDocument(engine.Document{
		Channel: "web", Title: "Hydroponics primer", URL: "https://example.com/hydro",
		Text: "nutrient solutions explained",
	}); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest("GET", "/api/documents/search?q=nutrient", nil)
	rec := httptest.NewRecorder()
	h.handleDocumentSearch(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Engine    string                `json:"engine"`
		Documents []engine.DocSearchHit `json:"documents"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Documents) != 1 {
		t.Fatalf("expected 1 hit, got %d", len(resp.Documents))
	}
	hit := resp.Documents[0]
	if hit.Title != "Hydroponics primer" {
		t.Errorf("the hit should carry the title, got %q", hit.Title)
	}
	if hit.URL != "https://example.com/hydro" {
		t.Errorf("the hit should carry the url, got %q", hit.URL)
	}
	if len(hit.Matches) != 1 || hit.Matches[0].Line != 1 {
		t.Errorf("unexpected matches: %+v", hit.Matches)
	}
}

// A path that leaves the vault must be refused.
func TestDocumentRawRefusesEscape(t *testing.T) {
	h, _ := newDocumentHandler(t)
	req := httptest.NewRequest("GET", "/api/documents/raw?path=../../etc/passwd", nil)
	rec := httptest.NewRecorder()
	h.handleDocumentRaw(rec, req)
	if rec.Code == 200 {
		t.Fatalf("a path that escapes the vault must be refused: %s", rec.Body.String())
	}
}

// An upload lands in the vault, named by machine.
func TestDocumentUploadFilesIntoTheVault(t *testing.T) {
	h, eng := newDocumentHandler(t)
	body := strings.NewReader("uploaded content")
	req := httptest.NewRequest("POST", "/api/documents/upload", body)
	req.Header.Set("Content-Type", "multipart/form-data")
	// A multipart body is fiddly to hand-roll; use the standard writer.
	var buf strings.Builder
	mw := multipart.NewWriter(&buf)
	fw, err := mw.CreateFormFile("file", "note.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fw.Write([]byte("uploaded content")); err != nil {
		t.Fatal(err)
	}
	mw.Close()
	req = httptest.NewRequest("POST", "/api/documents/upload", strings.NewReader(buf.String()))
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	h.handleDocumentUpload(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(resp.Path, "upload/") {
		t.Errorf("the upload should land in the upload channel: %q", resp.Path)
	}
	if _, err := os.Stat(filepath.Join(eng.DocumentsDir(), filepath.FromSlash(resp.Path))); err != nil {
		t.Fatalf("the document was not written: %v", err)
	}
}
