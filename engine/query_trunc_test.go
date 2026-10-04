package engine

import (
	"testing"

	"github.com/RoyChong5053/rag-mcp-server/settings"
)

func TestBoundQueryKeepsHeadWithinLimit(t *testing.T) {
	e := &Engine{settings: settings.New("", settings.Settings{QueryMaxChars: 5})}
	got := e.boundQuery("你好世界再见朋友")
	want := "你好世界再…"
	if got != want {
		t.Fatalf("boundQuery = %q, want %q", got, want)
	}
}

func TestBoundQueryUnlimitedWhenZero(t *testing.T) {
	e := &Engine{settings: settings.New("", settings.Settings{QueryMaxChars: 0})}
	long := "abcdefghij"
	if got := e.boundQuery(long); got != long {
		t.Fatalf("boundQuery with 0 limit = %q, want unchanged", got)
	}
}

func TestBoundQueryRuneSafe(t *testing.T) {
	e := &Engine{settings: settings.New("", settings.Settings{QueryMaxChars: 2})}
	if got := e.boundQuery("😀😀😀😀"); got != "😀😀…" {
		t.Fatalf("boundQuery = %q, want emoji kept whole", got)
	}
}
