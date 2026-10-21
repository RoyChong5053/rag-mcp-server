package engine

import (
	"os"
	"path/filepath"
	"testing"
)

// PruneLedger is the only way to drop a ledger row, and it is the way to
// close the gap left by a document that was removed from the vault by hand:
// the row stays, and every later verification reports the vault as broken.
func TestPruneLedgerDropsRowsForMissingFiles(t *testing.T) {
	e := newVaultEngine(t)
	alpha, err := e.fileDocument(Document{Channel: "agent", Title: "Alpha", Text: "the alpha body"})
	if err != nil {
		t.Fatal(err)
	}
	beta, err := e.fileDocument(Document{Channel: "agent", Title: "Beta", Text: "the beta body"})
	if err != nil {
		t.Fatal(err)
	}

	// Remove one document the way an operator would: from the disk.
	if err := os.Remove(alpha.Path); err != nil {
		t.Fatal(err)
	}

	v := e.VerifyLedger()
	if v.OK {
		t.Fatal("the ledger should be inconsistent while a file is missing")
	}
	if len(v.Missing) != 1 || v.Missing[0] != alpha.Rel {
		t.Fatalf("missing = %v, want [%s]", v.Missing, alpha.Rel)
	}

	// A dry run reports and changes nothing.
	n, err := e.PruneLedger(true)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("dry run pruned %d, want 1", n)
	}
	if got := len(e.ledger.Entries()); got != 2 {
		t.Fatalf("dry run changed the ledger: %d rows, want 2", got)
	}

	// The real thing drops the row and leaves the other one alone.
	n, err = e.PruneLedger(false)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("pruned %d, want 1", n)
	}
	entries := e.ledger.Entries()
	if len(entries) != 1 {
		t.Fatalf("ledger has %d rows, want 1", len(entries))
	}
	if entries[0].Path != beta.Rel {
		t.Fatalf("the surviving row is %+v, want the beta document", entries[0])
	}
	if _, err := os.Stat(beta.Path); err != nil {
		t.Fatalf("the surviving document was removed: %v", err)
	}

	after := e.VerifyLedger()
	if !after.OK {
		t.Fatalf("the ledger should be consistent now: %+v", after)
	}
}

// The monthly ledger file a row is rewritten into must be the one the
// document is filed under. A timestamp in UTC can name a different month
// than the path: a document filed at 2026-11-01T02:00+08:00 is in the
// 2026-11 folder, but in UTC it is still October.
func TestLedgerMonthFollowsThePathNotTheClock(t *testing.T) {
	cases := []struct {
		name string
		en   DocEntry
		want string
	}{
		{
			name: "path wins over a timestamp that falls in the previous month",
			en: DocEntry{
				Path:      "2026-11-01/agent/_/0210-abcdef01-thing.md",
				UpdatedAt: "2026-11-01T02:00:00+08:00",
			},
			want: "2026-11",
		},
		{
			name: "a path without a date falls back to the timestamp",
			en: DocEntry{
				Path:      "loose/thing.md",
				UpdatedAt: "2026-10-22T03:19:15+08:00",
			},
			want: "2026-10",
		},
		{
			name: "nothing to go on",
			en:   DocEntry{Path: "loose/thing.md"},
			want: "unknown",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := ledgerMonth(c.en); got != c.want {
				t.Fatalf("ledgerMonth = %q, want %q", got, c.want)
			}
		})
	}
}

// The ledger is written per month, and the file it is written to has to be
// the file the rows came from.
func TestPruneLedgerKeepsTheMonthlyFileOfItsRows(t *testing.T) {
	e := newVaultEngine(t)
	doc, err := e.fileDocument(Document{Channel: "agent", Title: "Kept", Text: "the kept body"})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(doc.Path); err != nil {
		t.Fatal(err)
	}
	if _, err := e.PruneLedger(false); err != nil {
		t.Fatal(err)
	}
	// The monthly file for the path's month must exist and be empty of the
	// dropped row.
	month := ledgerMonth(DocEntry{Path: doc.Rel})
	path := filepath.Join(e.ledgerDir(), month+".jsonl")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 0 {
		t.Fatalf("%s still holds %d bytes after pruning", path, len(data))
	}
}
