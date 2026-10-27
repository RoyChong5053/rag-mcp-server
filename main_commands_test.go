package main

import (
	"flag"
	"testing"
)

// Flags used to be silently ignored when written after the collection name,
// even though the usage text advertises that order: the stdlib flag parser
// stops at the first non-flag token. Order must not matter.
func TestParseMaintenanceArgsAllowsFlagsAfterPositional(t *testing.T) {
	fs := flag.NewFlagSet("verify-deep", flag.ContinueOnError)
	source := fs.String("source", "", "")
	samples := fs.Int("samples", 20, "")
	apply := fs.Bool("apply", false, "")

	pos := parseMaintenanceArgs(fs, []string{"osho-book", "-source", "book.md", "-samples", "5", "-apply"})
	if len(pos) != 1 || pos[0] != "osho-book" {
		t.Fatalf("positional = %v, want [osho-book]", pos)
	}
	if *source != "book.md" || *samples != 5 || !*apply {
		t.Fatalf("flags not parsed: source=%q samples=%d apply=%v", *source, *samples, *apply)
	}
}

func TestParseMaintenanceArgsFlagsFirstStillWork(t *testing.T) {
	fs := flag.NewFlagSet("docs-search", flag.ContinueOnError)
	query := fs.String("query", "", "")
	mode := fs.String("mode", "literal", "")

	pos := parseMaintenanceArgs(fs, []string{"-query=静心", "-mode", "regex", "leftover"})
	if len(pos) != 1 || pos[0] != "leftover" {
		t.Fatalf("positional = %v, want [leftover]", pos)
	}
	if *query != "静心" || *mode != "regex" {
		t.Fatalf("flags not parsed: query=%q mode=%q", *query, *mode)
	}
}

func TestParseMaintenanceArgsDoubleDash(t *testing.T) {
	fs := flag.NewFlagSet("docs-search", flag.ContinueOnError)
	fs.String("query", "", "")

	pos := parseMaintenanceArgs(fs, []string{"-query", "x", "--", "-not-a-flag", "name"})
	if len(pos) != 2 || pos[0] != "-not-a-flag" || pos[1] != "name" {
		t.Fatalf("positional = %v", pos)
	}
}
