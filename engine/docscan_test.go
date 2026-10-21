package engine

import (
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// writeVaultDoc files a document with a fixed date so the tests are stable.
func writeVaultDoc(t *testing.T, e *Engine, channel, title, body string) string {
	t.Helper()
	file, err := e.fileDocument(Document{Channel: channel, Title: title, Text: body})
	if err != nil {
		t.Fatal(err)
	}
	return file.Rel
}

func TestSearchDocumentsBuiltin(t *testing.T) {
	e := newVaultEngine(t)
	e.config.DocSearchBackend = "builtin"
	writeVaultDoc(t, e, "web", "First page", "the quick brown fox jumps over the lazy dog\nsecond line here")
	writeVaultDoc(t, e, "agent", "A memory", "remember the sqlite preference")

	res, err := e.SearchDocuments(DocSearchOptions{Query: "brown fox"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Engine != "builtin" {
		t.Fatalf("expected the built-in engine, got %q", res.Engine)
	}
	if len(res.Documents) != 1 {
		t.Fatalf("expected 1 matching document, got %d", len(res.Documents))
	}
	hit := res.Documents[0]
	if len(hit.Matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(hit.Matches))
	}
	if hit.Matches[0].Line != 1 {
		t.Errorf("expected the hit on line 1, got %d", hit.Matches[0].Line)
	}
	if hit.Title != "First page" {
		t.Errorf("the hit should carry the document title, got %q", hit.Title)
	}
	if hit.Channel != "web" {
		t.Errorf("the hit should carry the channel, got %q", hit.Channel)
	}
}

func TestSearchDocumentsIsCaseInsensitiveByDefault(t *testing.T) {
	e := newVaultEngine(t)
	e.config.DocSearchBackend = "builtin"
	writeVaultDoc(t, e, "web", "Case", "Something Happened Here")

	res, err := e.SearchDocuments(DocSearchOptions{Query: "something happened"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Documents) != 1 {
		t.Fatalf("a lower-case query should match regardless of case, got %d documents", len(res.Documents))
	}
	// An upper-case query is taken literally.
	res, err = e.SearchDocuments(DocSearchOptions{Query: "Something Happened"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Documents) != 1 {
		t.Fatalf("expected the exact case to match, got %d documents", len(res.Documents))
	}
}

func TestSearchDocumentsRegexMode(t *testing.T) {
	e := newVaultEngine(t)
	e.config.DocSearchBackend = "builtin"
	writeVaultDoc(t, e, "web", "Codes", "error code 404\nerror code 500")

	res, err := e.SearchDocuments(DocSearchOptions{Query: `code \d+`, Mode: "regex"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Documents) != 1 {
		t.Fatalf("expected the regex to match, got %d documents", len(res.Documents))
	}
	// One hit per line, like ripgrep.
	if res.TotalMatches != 2 {
		t.Errorf("expected 2 matches, got %d", res.TotalMatches)
	}
}

func TestSearchDocumentsWordBoundary(t *testing.T) {
	e := newVaultEngine(t)
	e.config.DocSearchBackend = "builtin"
	writeVaultDoc(t, e, "web", "Words", "catalog\nthe cat sat")

	res, err := e.SearchDocuments(DocSearchOptions{Query: "cat", Word: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.TotalMatches != 1 {
		t.Fatalf("expected 1 whole-word match, got %d", res.TotalMatches)
	}
	res, err = e.SearchDocuments(DocSearchOptions{Query: "cat"})
	if err != nil {
		t.Fatal(err)
	}
	// Three hits: the front-matter line "category: _" holds "cat" too. The
	// front-matter is part of the file, so it is searched - looking for a
	// title or a URL finds the document that carries them.
	if res.TotalMatches != 3 {
		t.Fatalf("expected 3 plain matches, got %d", res.TotalMatches)
	}
}

// TestSearchFindsFrontMatter pins that the identity of a document is
// searchable: searching for its title finds it even when the body does not
// mention the title.
func TestSearchFindsFrontMatter(t *testing.T) {
	e := newVaultEngine(t)
	e.config.DocSearchBackend = "builtin"
	writeVaultDoc(t, e, "web", "Hydroponics primer", "a body that never names the thing")
	res, err := e.SearchDocuments(DocSearchOptions{Query: "hydroponics"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Documents) != 1 {
		t.Fatalf("expected the document to be found by its title, got %d documents", len(res.Documents))
	}
}

func TestSearchDocumentsChannelScope(t *testing.T) {
	e := newVaultEngine(t)
	e.config.DocSearchBackend = "builtin"
	writeVaultDoc(t, e, "web", "Web page", "shared keyword here")
	writeVaultDoc(t, e, "agent", "Memory", "shared keyword here too")

	res, err := e.SearchDocuments(DocSearchOptions{Query: "shared", Channel: "agent"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Documents) != 1 {
		t.Fatalf("expected 1 document in the agent channel, got %d", len(res.Documents))
	}
	if res.Documents[0].Channel != "agent" {
		t.Errorf("expected the agent channel, got %q", res.Documents[0].Channel)
	}
}

// TestSearchEnginesAgree is the important one: the built-in scanner and
// ripgrep must return the same hits, otherwise the server answers differently
// depending on whether the host happens to have ripgrep.
func TestSearchEnginesAgree(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("ripgrep is not installed on this host")
	}
	e := newVaultEngine(t)
	writeVaultDoc(t, e, "web", "Alpha", "the alpha document mentions ripgrep and a version 1.2.3")
	writeVaultDoc(t, e, "web", "Beta", "the beta document mentions ripgrep too\nand a second line")
	writeVaultDoc(t, e, "agent", "Gamma", "an agent memory about ripgrep")

	collect := func(backend string) []string {
		e.config.DocSearchBackend = backend
		res, err := e.SearchDocuments(DocSearchOptions{Query: "ripgrep", PerFile: 10})
		if err != nil {
			t.Fatal(err)
		}
		var out []string
		for _, d := range res.Documents {
			for _, m := range d.Matches {
				out = append(out, d.Path+":"+itoa(m.Line))
			}
		}
		sort.Strings(out)
		return out
	}
	rgHits := collect("rg")
	builtinHits := collect("builtin")
	if len(rgHits) == 0 {
		t.Fatalf("ripgrep found nothing")
	}
	if len(rgHits) != len(builtinHits) {
		t.Fatalf("the engines disagree: rg=%v builtin=%v", rgHits, builtinHits)
	}
	for i := range rgHits {
		if rgHits[i] != builtinHits[i] {
			t.Fatalf("the engines disagree: rg=%v builtin=%v", rgHits, builtinHits)
		}
	}
}

func TestSearchDocumentsNoVault(t *testing.T) {
	e := newVaultEngine(t)
	e.config.DocSearchBackend = "builtin"
	res, err := e.SearchDocuments(DocSearchOptions{Query: "anything"})
	if err != nil {
		t.Fatalf("an empty vault must not be an error: %v", err)
	}
	if len(res.Documents) != 0 {
		t.Errorf("expected no documents, got %d", len(res.Documents))
	}
}

func TestSearchDocumentsRejectsBadInput(t *testing.T) {
	e := newVaultEngine(t)
	if _, err := e.SearchDocuments(DocSearchOptions{Query: "  "}); err == nil {
		t.Fatalf("an empty query must be refused")
	}
	if _, err := e.SearchDocuments(DocSearchOptions{Query: "x", Mode: "fuzzy"}); err == nil {
		t.Fatalf("an unknown mode must be refused")
	}
}

func TestDocHost(t *testing.T) {
	cases := map[string]string{
		"https://github.com/a/b":     "github.com",
		"http://www.example.com/x":   "example.com",
		"https://example.com:8443/x": "example.com",
		"":                           "",
	}
	for in, want := range cases {
		if got := docHost(in); got != want {
			t.Errorf("docHost(%q) = %q, want %q", in, got, want)
		}
	}
}

// itoa is a tiny helper so the test does not need strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// TestSearchFindsFilesWithoutFrontMatter makes sure a file that was not
// filed by the vault (dropped in by hand) is still searchable.
func TestSearchFindsFilesWithoutFrontMatter(t *testing.T) {
	e := newVaultEngine(t)
	e.config.DocSearchBackend = "builtin"
	day := filepath.Join(e.documentsDir(), "2026-10-22", "web", "_")
	if err := os.MkdirAll(day, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(day, "manual.md"), []byte("a hand written note about hydroponics"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := e.SearchDocuments(DocSearchOptions{Query: "hydroponics"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Documents) != 1 {
		t.Fatalf("expected the hand written file to be found, got %d documents", len(res.Documents))
	}
	if !strings.Contains(res.Documents[0].Path, "manual.md") {
		t.Errorf("unexpected path: %s", res.Documents[0].Path)
	}
}
