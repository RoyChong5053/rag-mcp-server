package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/registry"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// newVaultEngine builds an engine whose documents vault lives in a temp dir.
func newVaultEngine(t *testing.T) *Engine {
	t.Helper()
	root := t.TempDir()
	reg, err := registry.New(filepath.Join(root, "collections.json"))
	if err != nil {
		t.Fatal(err)
	}
	e := &Engine{
		stores:         map[string]VectorStore{BackendVectra: newFakeStore(true)},
		defaultBackend: BackendVectra,
		embedding:      NewEmbeddingClient("http://127.0.0.1:0", "", "embedding", ""),
		rerank:         NewRerankClient("http://127.0.0.1:0", "", "rerank", "", 0, 0),
		config: &EngineConfig{
			EmbedProvider: "one-api", EmbedModel: "embedding",
			VectorDim: 1024, VectorDistance: "Cosine",
			DocsDir:      root,
			DocumentsDir: filepath.Join(root, "documents"),
		},
		registry:  reg,
		settings:  settings.New("", settings.Settings{}),
		downUntil: map[string]time.Time{},
	}
	ledger, err := newDocLedger(e.documentsDir(), 30)
	if err != nil {
		t.Fatal(err)
	}
	e.ledger = ledger
	return e
}

func TestFileDocumentLayout(t *testing.T) {
	e := newVaultEngine(t)
	file, err := e.fileDocument(Document{
		Channel: "web",
		Title:   "How to embed vectors",
		URL:     "https://example.com/post",
		Text:    "the body of the page",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !file.Created {
		t.Fatalf("expected a new document, got deduped")
	}
	// <date>/<channel>/<category>/<time>-<sha8>-<slug>.md
	parts := strings.Split(filepath.ToSlash(file.Rel), "/")
	if len(parts) != 4 {
		t.Fatalf("expected 4 path segments, got %v", parts)
	}
	if !docDatePattern.MatchString(parts[0]) {
		t.Errorf("first segment should be the date, got %q", parts[0])
	}
	if parts[1] != "web" {
		t.Errorf("second segment should be the channel, got %q", parts[1])
	}
	if parts[2] != DocCategoryNone {
		t.Errorf("third segment should be the default category, got %q", parts[2])
	}
	name := parts[3]
	if !strings.Contains(name, file.SHA256[:8]) {
		t.Errorf("file name %q should carry the content hash %q", name, file.SHA256[:8])
	}
	if !strings.Contains(name, "how-to-embed-vectors") {
		t.Errorf("file name %q should carry the slug", name)
	}
	if _, err := os.Stat(file.Path); err != nil {
		t.Fatalf("document not written: %v", err)
	}
}

func TestFileDocumentDedupByContent(t *testing.T) {
	e := newVaultEngine(t)
	doc := Document{Channel: "web", Title: "Same page", URL: "https://example.com/post", Text: "identical body"}
	first, err := e.fileDocument(doc)
	if err != nil {
		t.Fatal(err)
	}
	second, err := e.fileDocument(Document{Channel: "web", Title: "Same page, other url",
		URL: "https://example.com/other", Text: "identical body"})
	if err != nil {
		t.Fatal(err)
	}
	if second.Path != first.Path {
		t.Fatalf("the same body must land on the same file: %s vs %s", first.Path, second.Path)
	}
	if !second.Deduped {
		t.Fatalf("the second write should be reported as a duplicate")
	}
	// The second URL is remembered, the first one is kept.
	doc2, err := e.ReadDocument(first.Rel)
	if err != nil {
		t.Fatal(err)
	}
	if len(doc2.Meta.URLs) != 2 {
		t.Errorf("expected both URLs to be remembered, got %v", doc2.Meta.URLs)
	}
}

func TestFileDocumentChangedBodyIsNewDocument(t *testing.T) {
	e := newVaultEngine(t)
	first, err := e.fileDocument(Document{Channel: "web", Title: "Page", Text: "body v1"})
	if err != nil {
		t.Fatal(err)
	}
	second, err := e.fileDocument(Document{Channel: "web", Title: "Page", Text: "body v2"})
	if err != nil {
		t.Fatal(err)
	}
	if second.Path == first.Path {
		t.Fatalf("a changed body must be a new document")
	}
	if !second.Created {
		t.Fatalf("a changed body must be reported as created")
	}
}

func TestFrontMatterRoundTrip(t *testing.T) {
	e := newVaultEngine(t)
	file, err := e.fileDocument(Document{
		Channel:  "web",
		Title:    "Title with: colon",
		URL:      "https://example.com/x?a=1&b=2",
		Text:     "body line one\nbody line two",
		Provider: "tavily",
		Tags:     []string{"web", "test"},
	})
	if err != nil {
		t.Fatal(err)
	}
	doc, err := e.ReadDocument(file.Rel)
	if err != nil {
		t.Fatal(err)
	}
	if doc.Meta.Title != "Title with: colon" {
		t.Errorf("title round-trip failed: %q", doc.Meta.Title)
	}
	if doc.Meta.URL != "https://example.com/x?a=1&b=2" {
		t.Errorf("url round-trip failed: %q", doc.Meta.URL)
	}
	if doc.Meta.Provider != "tavily" {
		t.Errorf("provider round-trip failed: %q", doc.Meta.Provider)
	}
	if len(doc.Meta.Tags) != 2 {
		t.Errorf("tags round-trip failed: %v", doc.Meta.Tags)
	}
	if !strings.Contains(doc.Body, "body line two") {
		t.Errorf("body round-trip failed: %q", doc.Body)
	}
	if doc.Meta.SHA256 == "" {
		t.Errorf("the front-matter should carry the content hash")
	}
}

func TestCategoryRules(t *testing.T) {
	e := newVaultEngine(t)
	re, err := compileCategoryRules([]CategoryRule{
		{Match: `^github\.com`, Category: "code"},
		{Match: `^arxiv\.org`, Category: "paper"},
	})
	if err != nil {
		t.Fatal(err)
	}
	e.docCategoryRules = re
	file, err := e.fileDocument(Document{
		Channel: "web",
		Title:   "Some repo",
		URL:     "https://github.com/some/repo",
		Text:    "body",
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := strings.Split(filepath.ToSlash(file.Rel), "/")
	if parts[2] != "code" {
		t.Errorf("expected the code category, got %q", parts[2])
	}
	// A document that matches no rule lands in "_".
	other, err := e.fileDocument(Document{Channel: "web", Title: "x", URL: "https://example.org", Text: "body2"})
	if err != nil {
		t.Fatal(err)
	}
	parts = strings.Split(filepath.ToSlash(other.Rel), "/")
	if parts[2] != DocCategoryNone {
		t.Errorf("expected the default category, got %q", parts[2])
	}
}

func TestReadDocumentRefusesEscape(t *testing.T) {
	e := newVaultEngine(t)
	if _, err := e.ReadDocument("../../etc/passwd"); err == nil {
		t.Fatalf("a path that escapes the vault must be refused")
	}
	if _, err := e.ReadDocument(""); err == nil {
		t.Fatalf("an empty path must be refused")
	}
}

func TestLedgerListsDocuments(t *testing.T) {
	e := newVaultEngine(t)
	for i := 0; i < 3; i++ {
		if _, err := e.fileDocument(Document{Channel: "agent", Text: "memory " + string(rune('a'+i))}); err != nil {
			t.Fatal(err)
		}
	}
	entries := e.ListDocuments("", 10)
	if len(entries) != 3 {
		t.Fatalf("expected 3 documents, got %d", len(entries))
	}
	// Newest first.
	if entries[0].UpdatedAt < entries[len(entries)-1].UpdatedAt {
		t.Errorf("the listing should be newest first")
	}
	agents := e.ListDocuments("agent", 10)
	if len(agents) != 3 {
		t.Errorf("expected 3 agent documents, got %d", len(agents))
	}
	web := e.ListDocuments("web", 10)
	if len(web) != 0 {
		t.Errorf("expected no web documents, got %d", len(web))
	}
	counts := e.DocChannels()
	if counts["agent"] != 3 {
		t.Errorf("expected 3 agent documents in the channel counts, got %d", counts["agent"])
	}
}

func TestLedgerPersistsAcrossReload(t *testing.T) {
	root := t.TempDir()
	l, err := newDocLedger(root, 30)
	if err != nil {
		t.Fatal(err)
	}
	l.Append(DocEntry{Path: "2026-10-22/web/_/a.md", DocMeta: DocMeta{
		SHA256: "abc123", Channel: "web", Category: "_", UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}})
	again, err := newDocLedger(root, 30)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := again.Lookup("abc123"); !ok {
		t.Fatalf("the ledger should survive a reload")
	}
}

func TestImportDocuments(t *testing.T) {
	e := newVaultEngine(t)
	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "one.md"), []byte("# One\n\nthe first document"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "two.txt"), []byte("the second document"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := e.ImportDocuments(src, "import", "", false, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Imported != 2 {
		t.Fatalf("expected 2 imported documents, got %d", res.Imported)
	}
	// The second import finds the same bodies and skips them.
	res2, err := e.ImportDocuments(src, "import", "", false, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res2.Imported != 0 || res2.Skipped != 2 {
		t.Fatalf("expected the second import to skip both, got imported=%d skipped=%d", res2.Imported, res2.Skipped)
	}
}
