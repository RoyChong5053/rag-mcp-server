package registry

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestNewMissingFile(t *testing.T) {
	r, err := New(filepath.Join(t.TempDir(), "nope.json"))
	if err != nil {
		t.Fatalf("missing file should load empty: %v", err)
	}
	if len(r.List()) != 0 {
		t.Fatalf("expected empty registry")
	}
}

func TestNewCorruptFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(p, []byte("{oops"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := New(p); err == nil {
		t.Fatalf("corrupt file must be a loud error")
	}
}

func TestUpdateGetDelete(t *testing.T) {
	p := filepath.Join(t.TempDir(), "reg.json")
	r, err := New(p)
	if err != nil {
		t.Fatal(err)
	}

	if err := r.Update("col_a", func(e *Entry) {
		e.DisplayName = "A"
		e.Tags = []string{"chat"}
		e.ChunkCount = 42
	}); err != nil {
		t.Fatal(err)
	}

	e := r.Get("col_a")
	if e == nil || e.DisplayName != "A" || e.ChunkCount != 42 {
		t.Fatalf("unexpected entry: %+v", e)
	}
	if !e.Enabled {
		t.Fatalf("new entries default enabled")
	}
	if e.CreatedAt == "" || e.UpdatedAt == "" {
		t.Fatalf("timestamps must be stamped")
	}

	// Reload from disk: must survive round-trip
	r2, err := New(p)
	if err != nil {
		t.Fatal(err)
	}
	if got := r2.Get("col_a"); got == nil || got.DisplayName != "A" {
		t.Fatalf("round-trip failed: %+v", got)
	}

	if err := r2.Delete("col_a"); err != nil {
		t.Fatal(err)
	}
	if r2.Get("col_a") != nil {
		t.Fatalf("delete failed")
	}
	if err := r2.Delete("col_a"); err != nil {
		t.Fatalf("double delete must be no-op: %v", err)
	}
}

func TestConcurrentUpdate(t *testing.T) {
	p := filepath.Join(t.TempDir(), "reg.json")
	r, err := New(p)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = r.Update("col", func(e *Entry) { e.ChunkCount++ })
		}()
	}
	wg.Wait()
	if got := r.Get("col").ChunkCount; got != 20 {
		t.Fatalf("lost updates: got %d want 20", got)
	}
}
