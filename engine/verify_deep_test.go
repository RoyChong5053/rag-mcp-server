package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/RoyChong5053/rag-mcp-server/chunking"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

func TestStripFrontMatter(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
		ok   bool
	}{
		{"plain", "# title\nbody\n", "", false},
		{"lf", "---\nid: x\nbytes: 1\n---\nbody\n", "body\n", true},
		{"crlf", "---\r\nid: x\r\n---\r\nbody\r\n", "body\r\n", true},
		{"bom", "\ufeff---\nid: x\n---\nbody\n", "body\n", true},
		{"unclosed", "---\nid: x\nbody\n", "", false},
		{"delimiter in body only", "text\n---\nmore\n", "", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := stripFrontMatter(c.in)
			if ok != c.ok {
				t.Fatalf("ok = %v, want %v", ok, c.ok)
			}
			if ok && got != c.want {
				t.Fatalf("body = %q, want %q", got, c.want)
			}
		})
	}
}

// A vault document carries a YAML front matter the index never saw: the index
// was built from the raw file. Re-chunking the vault file as-is reported zero
// coverage for text that is perfectly intact, so the check must read the body
// too.
func TestVerifyDeepMeasuresVaultBodyNotFrontMatter(t *testing.T) {
	root := t.TempDir()
	docs := filepath.Join(root, "docs")
	fs := NewFileStore(filepath.Join(root, "Vectra"), docs)
	e := newTestEngine(t, settings.Settings{}, fs)
	e.config.DocsDir = docs
	e.config.DocumentsDir = filepath.Join(docs, "documents")
	e.config.ChunkSize = 40
	e.config.OverlapPercent = 0

	body := strings.Join([]string{
		"静心是观照，不是集中。",
		"观照意味着你不选择，只是看着。",
		"集中是一种狭窄，观照是一种辽阔。",
		"当你看着一棵树，不要命名它。",
		"命名立刻把经验变成语言。",
		"语言属于过去，观照属于当下。",
	}, "\n")
	body = strings.Repeat(body, 8)
	used := e.ResolveChunkOptions(nil)
	chunks := chunking.Split(body, chunking.Options{Strategy: used.Strategy, Size: used.Size, Overlap: used.Overlap})

	src := filepath.Join(docs, "documents", "2026-10-22", "upload", "_", "012548-x-book.md")
	if err := os.MkdirAll(filepath.Dir(src), 0o755); err != nil {
		t.Fatal(err)
	}
	full := "---\nid: x\nbytes: 1\n---\n" + body
	if err := os.WriteFile(src, []byte(full), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := fs.CreateCollection("col"); err != nil {
		t.Fatal(err)
	}
	pts := make([]Point, 0, len(chunks))
	for i, c := range chunks {
		p := pt(uint64(i+1), src, []float32{1, 0, 0}, c)
		p.Payload["chunk_index"] = i
		pts = append(pts, p)
	}
	if err := fs.UpsertPoints("col", pts); err != nil {
		t.Fatal(err)
	}

	res, err := e.VerifyDeep("col", src, 1)
	if err != nil {
		t.Fatal(err)
	}
	if res.TextCoverage != 1 {
		t.Fatalf("coverage = %v (matched %d/%d), want 1", res.TextCoverage, res.TextMatched, res.TextTotal)
	}
	if res.OrphanCards != 0 {
		t.Fatalf("orphan cards = %d, want 0", res.OrphanCards)
	}
}
