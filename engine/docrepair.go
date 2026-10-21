package engine

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// A file can move. The vector index remembers the path a chunk came from, so
// when the raw file moves the pointer goes stale: a recall still returns the
// text, but it points at a file that is not there any more.
//
// The repair finds the new home by content. Every collection records the
// hash of the text it indexed, and the vault's ledger records the hash of
// every document it holds. Two hashes that match are the same text, so the
// pointer can be moved to the document that now holds it.
//
// This is what a half-finished migration needs: the files are filed, but the
// index was not re-pointed yet.

// RepairReport reports what a repair did.
type RepairReport struct {
	Sources         int      `json:"sources"`
	Missing         int      `json:"missing"`
	Repaired        int      `json:"repaired"`
	ChunksRepointed int      `json:"chunks_repointed"`
	Unresolved      []string `json:"unresolved,omitempty"`
	Errors          []string `json:"errors,omitempty"`
}

// sourceFileExists reports whether the file a collection was indexed from is
// still there.
//
// The recorded path can be absolute, relative to the working directory (the
// way the index used to record it) or relative to the data bank. All three
// are tried, because a path that cannot be resolved must not be reported as
// a missing file.
func (e *Engine) sourceFileExists(name string) bool {
	if strings.TrimSpace(name) == "" {
		return false
	}
	var candidates []string
	if filepath.IsAbs(name) {
		candidates = append(candidates, name)
	} else {
		if abs, err := filepath.Abs(name); err == nil {
			candidates = append(candidates, abs)
		}
		if root := e.docsRoot(); root != "" {
			candidates = append(candidates, filepath.Join(root, filepath.FromSlash(name)))
		}
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return true
		}
	}
	return false
}

// RepairSources re-points the collections at the documents that now hold
// their text.
//
// A source whose file is still where the index says is left alone. A source
// whose file is gone is matched by content hash against the vault; if the
// hash is unknown or the document is not in the vault, the source is
// reported and left as it is.
func (e *Engine) RepairSources() (*RepairReport, error) {
	rep := &RepairReport{}
	fs := e.FileStore()
	if fs == nil {
		return rep, nil
	}
	cols, err := fs.ListCollections()
	if err != nil {
		return rep, err
	}
	for _, c := range cols {
		srcs, err := fs.ListSources(c.Name)
		if err != nil {
			rep.Errors = append(rep.Errors, fmt.Sprintf("%s: %v", c.Name, err))
			continue
		}
		moves := map[string]string{}
		for _, s := range srcs {
			rep.Sources++
			if e.sourceFileExists(s.Name) {
				continue // the file is still where the index says
			}
			rep.Missing++
			if s.SHA256 == "" {
				rep.Unresolved = append(rep.Unresolved, c.Name+": "+s.Name+" (no hash)")
				continue
			}
			// The index recorded the hash of the text it indexed: the
			// trimmed body for text, the raw bytes for a file. The vault
			// knows both.
			entry, ok := e.ledger.Lookup(s.SHA256)
			if !ok || entry.Path == "" {
				entry, ok = e.ledger.LookupRaw(s.SHA256)
			}
			if !ok || entry.Path == "" {
				rep.Unresolved = append(rep.Unresolved, c.Name+": "+s.Name+" (no document with that hash)")
				continue
			}
			moves[s.Name] = filepath.Join(e.documentsDir(), filepath.FromSlash(entry.Path))
		}
		if len(moves) == 0 {
			continue
		}
		n, err := fs.RenameSources(c.Name, moves)
		if err != nil {
			rep.Errors = append(rep.Errors, fmt.Sprintf("%s: %v", c.Name, err))
			continue
		}
		rep.Repaired += len(moves)
		rep.ChunksRepointed += n
	}
	return rep, nil
}
