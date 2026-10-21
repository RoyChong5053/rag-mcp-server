package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/registry"
)

// The migration moves the raw files of the old layout into the documents
// vault:
//
//	docs/memory/web/<hash>.md   ->  documents/<date>/web/<category>/<time>-<hash>-<slug>.md
//	docs/staging/<name>         ->  documents/<date>/upload/<category>/<time>-<hash>-<name>
//	docs/chat_history/<name>    ->  documents/<date>/chat/<category>/<time>-<hash>-<name>
//
// The agent journals (docs/memory/agent/YYYY-MM-DD.md) stay where they are:
// they are the human-readable log of the day, and they are still written.
//
// Moving a file breaks provenance: the chunks remember the path they came
// from. So the migration rewrites the vector index as well - the catalog
// entry, the index folder and the source_file of every chunk - and the
// registry entry of each collection. Without that step a recall would point
// at a file that no longer exists.
//
// The old files are moved into docs/.migrated/ instead of being deleted, so
// a mistake can be undone by hand.

// MigrateReport reports what one migration did.
type MigrateReport struct {
	DryRun            bool     `json:"dry_run"`
	Scanned           int      `json:"scanned"`
	Migrated          int      `json:"migrated"`
	Skipped           int      `json:"skipped"`
	Failed            int      `json:"failed"`
	ChunksRepointed   int      `json:"chunks_repointed"`
	RegistryUpdated   int      `json:"registry_updated"`
	BackupDir         string   `json:"backup_dir,omitempty"`
	Documents         []string `json:"documents,omitempty"`
	Errors            []string `json:"errors,omitempty"`
	LegacyDirectories []string `json:"legacy_directories,omitempty"`
}

// legacySource is one folder of legacy raw files and how it is filed now.
type legacySource struct {
	dir     string
	channel string
}

// legacyWebHeader is the header the old web raw files carried:
//
//	# Title
//
//	- source: https://...
//	- provider: tavily
//	- fetched_at: 2026-10-22T14:35:12Z
//
//	body...
type legacyWebHeader struct {
	Title     string
	URL       string
	Provider  string
	FetchedAt string
}

// parseLegacyWebHeader reads the header of an old web raw file and returns
// the body.
//
// The header is a fixed block:
//
//	# Title
//	<blank>
//	- source: https://...
//	- provider: tavily
//	- fetched_at: 2026-10-22T14:35:12Z
//	<blank>
//	<body>
//
// Only the known field names are treated as header. A body that starts with
// a heading or a list item is body, not header: eating the first line of
// the text would change its hash, and the index would no longer find the
// document again.
func parseLegacyWebHeader(data []byte) (legacyWebHeader, string) {
	text := strings.ReplaceAll(string(data), "\r\n", "\n")
	var h legacyWebHeader
	lines := strings.Split(text, "\n")
	i := 0
	// The title is the first line, if it is a heading.
	if i < len(lines) && strings.HasPrefix(strings.TrimSpace(lines[i]), "# ") {
		h.Title = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(lines[i]), "# "))
		i++
	}
	// The known field lines follow, separated by blank lines.
header:
	for i < len(lines) {
		trimmed := strings.TrimSpace(lines[i])
		if trimmed == "" {
			i++
			continue
		}
		lower := strings.ToLower(trimmed)
		switch {
		case strings.HasPrefix(lower, "- source:"):
			h.URL = strings.TrimSpace(trimmed[len("- source:"):])
		case strings.HasPrefix(lower, "- provider:"):
			h.Provider = strings.TrimSpace(trimmed[len("- provider:"):])
		case strings.HasPrefix(lower, "- fetched_at:"):
			h.FetchedAt = strings.TrimSpace(trimmed[len("- fetched_at:"):])
		default:
			// Not a header field: the body starts here.
			break header
		}
		i++
	}
	body := strings.Join(lines[i:], "\n")
	return h, strings.TrimSpace(body)
}

// MigrateDocuments moves the legacy raw files into the documents vault.
//
// dryRun reports what would happen without touching anything. The vector
// index is only rewritten when the file actually moved.
func (e *Engine) MigrateDocuments(dryRun bool) (*MigrateReport, error) {
	docsRoot := e.docsRoot()
	if docsRoot == "" {
		docsRoot = "docs"
	}
	if abs, err := filepath.Abs(docsRoot); err == nil {
		docsRoot = abs
	}
	report := &MigrateReport{DryRun: dryRun}

	sources := []legacySource{
		{dir: filepath.Join(docsRoot, "memory", "web"), channel: DocChannelWeb},
		{dir: filepath.Join(docsRoot, "staging"), channel: DocChannelUpload},
		{dir: filepath.Join(docsRoot, "chat_history"), channel: DocChannelChat},
	}
	backupDir := filepath.Join(docsRoot, ".migrated", time.Now().Format("2006-01-02"))
	report.BackupDir = backupDir

	// old -> new, so the vector index can be re-pointed.
	moves := map[string]string{}

	for _, src := range sources {
		st, err := os.Stat(src.dir)
		if err != nil || !st.IsDir() {
			continue
		}
		report.LegacyDirectories = append(report.LegacyDirectories, src.dir)
		entries, err := os.ReadDir(src.dir)
		if err != nil {
			continue
		}
		for _, ent := range entries {
			if ent.IsDir() {
				continue
			}
			oldPath := filepath.Join(src.dir, ent.Name())
			report.Scanned++

			doc := Document{
				Channel: src.channel,
				Title:   ent.Name(),
				Text:    "",
			}
			ext := strings.ToLower(filepath.Ext(ent.Name()))
			if ext == "" {
				ext = ".md"
			}
			doc.Ext = ext

			data, err := os.ReadFile(oldPath)
			if err != nil {
				report.Failed++
				report.Errors = append(report.Errors, fmt.Sprintf("%s: %v", ent.Name(), err))
				continue
			}
			// The hash of the file exactly as it is. An index that was built
			// from this file recorded that hash, so the repair can find the
			// document again.
			fileSum := sha256.Sum256(data)
			doc.RawSHA256 = hex.EncodeToString(fileSum[:])
			if src.channel == DocChannelWeb {
				h, body := parseLegacyWebHeader(data)
				doc.Title = h.Title
				doc.URL = h.URL
				doc.Provider = h.Provider
				doc.Text = body
				if doc.Title == "" {
					doc.Title = ent.Name()
				}
			} else {
				doc.Text = string(data)
			}
			if strings.TrimSpace(doc.Text) == "" {
				report.Skipped++
				continue
			}

			if dryRun {
				report.Migrated++
				continue
			}

			file, err := e.fileDocument(doc)
			if err != nil {
				report.Failed++
				report.Errors = append(report.Errors, fmt.Sprintf("%s: %v", ent.Name(), err))
				continue
			}
			moves[oldPath] = file.Path
			report.Migrated++
			if len(report.Documents) < 500 {
				report.Documents = append(report.Documents, file.Rel)
			}

			// Move the old file into the backup folder.
			if err := moveAside(oldPath, backupDir); err != nil {
				report.Errors = append(report.Errors, fmt.Sprintf("backup %s: %v", ent.Name(), err))
			}
		}
	}

	if dryRun || len(moves) == 0 {
		return report, nil
	}

	// Re-point the vector index at the new files. One pass per collection:
	// a collection with a big index would otherwise be loaded and saved
	// once per moved file.
	fs := e.FileStore()
	if fs != nil {
		cols, err := fs.ListCollections()
		if err == nil {
			for _, c := range cols {
				n, err := fs.RenameSources(c.Name, moves)
				if err != nil {
					report.Errors = append(report.Errors,
						fmt.Sprintf("%s: re-point sources: %v", c.Name, err))
					continue
				}
				report.ChunksRepointed += n
			}
		}
	}

	// The registry remembers one source per collection: point it at the
	// file that still exists.
	for name, en := range e.registry.List() {
		if en.SourceFile == "" {
			continue
		}
		abs := en.SourceFile
		if !filepath.IsAbs(abs) {
			if a, err := filepath.Abs(abs); err == nil {
				abs = a
			}
		}
		newPath, ok := moves[abs]
		if !ok {
			continue
		}
		if err := e.registry.Update(name, func(entry *registry.Entry) {
			entry.SourceFile = newPath
		}); err != nil {
			report.Errors = append(report.Errors, fmt.Sprintf("%s: registry: %v", name, err))
			continue
		}
		report.RegistryUpdated++
	}

	return report, nil
}

// moveAside moves a file into the backup folder, keeping its name.
func moveAside(path, backupDir string) error {
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		return err
	}
	dst := filepath.Join(backupDir, filepath.Base(path))
	if _, err := os.Stat(dst); err == nil {
		// Keep both: the backup of a same-named file gets a suffix.
		dst = fmt.Sprintf("%s.%d%s", strings.TrimSuffix(dst, filepath.Ext(dst)), time.Now().UnixNano(), filepath.Ext(dst))
	}
	return os.Rename(path, dst)
}
