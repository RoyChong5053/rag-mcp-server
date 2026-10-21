package engine

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// The ledger is the vault's index. It is a set of append-only JSONL files,
// one per month, next to the documents:
//
//	<documents_dir>/.ledger/2026-10.jsonl
//
// One line per filed document. It answers three questions without touching
// the corpus: which documents exist, where they are, and which body hash is
// already stored (so the same text is not filed twice).
//
// The ledger is deliberately not a database. It is text, so ripgrep can query
// it when the in-memory window does not reach back far enough, and it can be
// rsynced or deleted without a migration.

const (
	// ledgerDirName is the ledger directory inside the vault.
	ledgerDirName = ".ledger"
	// defaultLedgerWindowDays is how far back the in-memory index reaches.
	defaultLedgerWindowDays = 30
	// ledgerTailBytes is how much of a monthly file is read when the file is
	// larger than this. The ledger is append-only, so the tail holds the
	// newest entries.
	ledgerTailBytes = 32 << 20
	// maxLedgerEntries bounds the in-memory window. Beyond it the oldest
	// rows are dropped; the monthly files still hold them.
	maxLedgerEntries = 200000
)

// ledgerLog reports a ledger write problem without breaking the caller: the
// document itself is already on disk, so a ledger hiccup must not fail the
// write that produced it.
func ledgerLog(err error) {
	log.Printf("documents ledger: %v", err)
}

// docLedger is the append-only index of the documents vault.
type docLedger struct {
	mu     sync.RWMutex
	dir    string
	window time.Duration
	// bySHA indexes the body hash: the vault's identity.
	bySHA map[string]DocEntry
	// byRaw indexes the hash of the text as it was handed over. An index
	// built from a file records that one, so the repair needs it to find
	// the document again.
	byRaw map[string]DocEntry
	// byID indexes the short id of a document. The id is the first bytes of
	// the body hash and it is part of the file name, so a path that points
	// at a document that has been renamed can still be resolved.
	byID    map[string]DocEntry
	ordered []DocEntry
}

// newDocLedger opens (and creates) the ledger of a vault and loads the
// recent window into memory.
func newDocLedger(root string, windowDays int) (*docLedger, error) {
	if windowDays <= 0 {
		windowDays = defaultLedgerWindowDays
	}
	dir := filepath.Join(root, ledgerDirName)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create ledger dir: %w", err)
	}
	l := &docLedger{
		dir:    dir,
		window: time.Duration(windowDays) * 24 * time.Hour,
		bySHA:  map[string]DocEntry{},
		byRaw:  map[string]DocEntry{},
		byID:   map[string]DocEntry{},
	}
	l.load()
	return l, nil
}

// ledgerFile is the monthly file a timestamp belongs to.
func (l *docLedger) ledgerFile(when time.Time) string {
	return filepath.Join(l.dir, when.Local().Format("2006-01")+".jsonl")
}

// Append records one document: it is written to the monthly file and kept in
// the in-memory window. The file is append-only, so the newest line wins.
func (l *docLedger) Append(entry DocEntry) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.persist(entry); err != nil {
		ledgerLog(err)
	}
	if entry.SHA256 != "" {
		l.bySHA[entry.SHA256] = entry
	}
	if entry.RawSHA256 != "" {
		l.byRaw[entry.RawSHA256] = entry
	}
	if entry.ID != "" {
		l.byID[entry.ID] = entry
	}
	l.ordered = append(l.ordered, entry)
	l.trimLocked()
}

// trimLocked keeps the in-memory window bounded. The corpus may hold millions
// of documents; the index only has to cover the recent window, and the rest
// is still reachable through the ledger files on disk.
func (l *docLedger) trimLocked() {
	if len(l.ordered) <= maxLedgerEntries {
		return
	}
	drop := len(l.ordered) - maxLedgerEntries
	l.ordered = append(l.ordered[:0], l.ordered[drop:]...)
	l.bySHA = make(map[string]DocEntry, len(l.ordered))
	l.byRaw = make(map[string]DocEntry, len(l.ordered))
	l.byID = make(map[string]DocEntry, len(l.ordered))
	for _, en := range l.ordered {
		if en.SHA256 != "" {
			l.bySHA[en.SHA256] = en
		}
		if en.RawSHA256 != "" {
			l.byRaw[en.RawSHA256] = en
		}
		if en.ID != "" {
			l.byID[en.ID] = en
		}
	}
}

// Lookup finds a document by its body hash.
func (l *docLedger) Lookup(sha string) (DocEntry, bool) {
	if l == nil || sha == "" {
		return DocEntry{}, false
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	e, ok := l.bySHA[sha]
	return e, ok
}

// LookupRaw finds a document by the hash of the text as it was handed over
// (before it was trimmed). An index built from a file records that one.
func (l *docLedger) LookupRaw(sha string) (DocEntry, bool) {
	if l == nil || sha == "" {
		return DocEntry{}, false
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	e, ok := l.byRaw[sha]
	return e, ok
}

// LookupID finds a document by its short id (the first bytes of the body
// hash, and part of its file name).
func (l *docLedger) LookupID(id string) (DocEntry, bool) {
	if l == nil || id == "" {
		return DocEntry{}, false
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	e, ok := l.byID[id]
	return e, ok
}

// Entries returns the known documents, newest first.
func (l *docLedger) Entries() []DocEntry {
	if l == nil {
		return nil
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	out := make([]DocEntry, len(l.ordered))
	copy(out, l.ordered)
	// Newest first. The append order is already chronological, so this is
	// just a reversal; sorting by timestamp keeps it right when entries are
	// loaded from disk.
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].UpdatedAt > out[j].UpdatedAt
	})
	return out
}

// reload rebuilds the in-memory window from the ledger files.
func (l *docLedger) reload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.bySHA = map[string]DocEntry{}
	l.byRaw = map[string]DocEntry{}
	l.byID = map[string]DocEntry{}
	l.ordered = nil
	l.loadLocked()
	return nil
}

// load reads the recent window of the ledger into memory.
func (l *docLedger) load() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.loadLocked()
}

// loadLocked reads the recent window of the ledger into memory. Only the tail
// of a monthly file is read when it grew large: the file is append-only, so
// the newest entries are at the end.
func (l *docLedger) loadLocked() {
	cutoff := time.Now().Add(-l.window)
	files, err := os.ReadDir(l.dir)
	if err != nil {
		return
	}
	names := make([]string, 0, len(files))
	for _, f := range files {
		if f.IsDir() || !strings.HasSuffix(f.Name(), ".jsonl") {
			continue
		}
		names = append(names, f.Name())
	}
	sort.Strings(names)
	for _, name := range names {
		l.loadFile(filepath.Join(l.dir, name), cutoff)
	}
}

// loadFile reads one monthly ledger file into the in-memory window.
func (l *docLedger) loadFile(path string, cutoff time.Time) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return
	}
	if st.Size() > ledgerTailBytes {
		if _, err := f.Seek(st.Size()-ledgerTailBytes, 0); err != nil {
			return
		}
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 8*1024*1024)
	first := true
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if first {
			// The first line of a tail read may be cut in half.
			first = false
			if !strings.HasPrefix(line, "{") {
				continue
			}
		}
		var entry DocEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			continue
		}
		if entry.SHA256 == "" || entry.Path == "" {
			continue
		}
		ts := entry.UpdatedAt
		if ts == "" {
			ts = entry.FetchedAt
		}
		if t, err := time.Parse(time.RFC3339, ts); err == nil && t.Before(cutoff) {
			continue
		}
		l.bySHA[entry.SHA256] = entry
		if entry.RawSHA256 != "" {
			l.byRaw[entry.RawSHA256] = entry
		}
		if entry.ID != "" {
			l.byID[entry.ID] = entry
		}
		l.ordered = append(l.ordered, entry)
	}
	l.trimLocked()
}

// persist appends one entry to the monthly file. The caller holds no lock:
// the ledger's own mutex guards the in-memory index, the file append is a
// single O_APPEND write.
func (l *docLedger) persist(entry DocEntry) error {
	if l == nil {
		return nil
	}
	ts := entry.UpdatedAt
	if ts == "" {
		ts = entry.FetchedAt
	}
	when := time.Now().UTC()
	if t, err := time.Parse(time.RFC3339, ts); err == nil {
		when = t
	}
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(l.ledgerFile(when), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open ledger: %w", err)
	}
	defer f.Close()
	if _, err := f.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("append ledger: %w", err)
	}
	return nil
}

// LedgerVerifyResult reports one inconsistency between the ledger and the
// files on disk.
type LedgerVerifyResult struct {
	OK      bool     `json:"ok"`
	Known   int      `json:"known"`
	Missing []string `json:"missing,omitempty"`
	Orphans []string `json:"orphans,omitempty"`
	Note    string   `json:"note,omitempty"`
}

// VerifyLedger checks the ledger against the files on disk: every ledger row
// must still have its file, and every file in the vault must have a ledger
// row. It walks the vault, so it is a maintenance command, not a request
// handler.
func (e *Engine) VerifyLedger() LedgerVerifyResult {
	res := LedgerVerifyResult{OK: true}
	root := e.documentsDir()
	known := map[string]bool{}
	if e.ledger != nil {
		for _, en := range e.ledger.Entries() {
			known[en.Path] = true
			res.Known++
			abs := filepath.Join(root, filepath.FromSlash(en.Path))
			if _, err := os.Stat(abs); err != nil {
				res.Missing = append(res.Missing, en.Path)
				res.OK = false
			}
		}
	}
	// Files on disk that the ledger does not know about.
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if d.Name() == ledgerDirName {
				return filepath.SkipDir
			}
			return nil
		}
		rel, rerr := filepath.Rel(root, path)
		if rerr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if !known[rel] {
			res.Orphans = append(res.Orphans, rel)
			res.OK = false
		}
		return nil
	})
	if len(res.Missing) > 50 {
		res.Note = fmt.Sprintf("%d missing files (first 50 shown)", len(res.Missing))
		res.Missing = res.Missing[:50]
	}
	if len(res.Orphans) > 50 {
		if res.Note != "" {
			res.Note += "; "
		}
		res.Note += fmt.Sprintf("%d orphan files (first 50 shown)", len(res.Orphans))
		res.Orphans = res.Orphans[:50]
	}
	return res
}

// ledgerMonth is the monthly ledger file a document belongs to.
//
// The path carries the filing date (<yyyy-mm-dd>/<channel>/...), so it is the
// truth about where the file lives. The timestamp is only a fallback: taken
// in UTC it can name a different month than the document is filed under, and
// a row rewritten into the wrong monthly file is a row that has left the file
// that held it.
func ledgerMonth(en DocEntry) string {
	// The path starts with the filing date: <yyyy-mm-dd>/<channel>/...
	if len(en.Path) >= 11 && en.Path[4] == '-' && en.Path[7] == '-' && en.Path[10] == '/' {
		return en.Path[:7]
	}
	if t, err := time.Parse(time.RFC3339, en.UpdatedAt); err == nil {
		return t.UTC().Format("2006-01")
	}
	return "unknown"
}

// PruneLedger drops the ledger rows whose file is gone.
func (e *Engine) PruneLedger(dryRun bool) (int, error) {
	if e.ledger == nil {
		return 0, nil
	}
	root := e.documentsDir()
	entries := e.ledger.Entries()
	var gone []DocEntry
	for _, en := range entries {
		abs := filepath.Join(root, filepath.FromSlash(en.Path))
		if _, err := os.Stat(abs); err != nil {
			gone = append(gone, en)
		}
	}
	if dryRun || len(gone) == 0 {
		return len(gone), nil
	}
	// Rewrite the affected monthly files without the dead rows.
	byMonth := map[string][]DocEntry{}
	for _, en := range entries {
		month := ledgerMonth(en)
		byMonth[month] = append(byMonth[month], en)
	}
	for month, list := range byMonth {
		keep := list[:0]
		for _, en := range list {
			abs := filepath.Join(root, filepath.FromSlash(en.Path))
			if _, err := os.Stat(abs); err == nil {
				keep = append(keep, en)
			}
		}
		path := filepath.Join(e.ledgerDir(), month+".jsonl")
		if err := rewriteLedgerFile(path, keep); err != nil {
			return 0, err
		}
	}
	if err := e.ledger.reload(); err != nil {
		return 0, err
	}
	return len(gone), nil
}

// ledgerDir is the ledger directory of the vault.
func (e *Engine) ledgerDir() string {
	if e.ledger != nil {
		return e.ledger.dir
	}
	return filepath.Join(e.documentsDir(), ledgerDirName)
}

// rewriteLedgerFile replaces a monthly ledger file with the given rows.
func rewriteLedgerFile(path string, entries []DocEntry) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("rewrite ledger: %w", err)
	}
	w := bufio.NewWriter(f)
	for _, en := range entries {
		data, err := json.Marshal(en)
		if err != nil {
			f.Close()
			return err
		}
		_, _ = w.Write(append(data, '\n'))
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
