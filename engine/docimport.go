package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ImportResult reports what one import did.
type ImportResult struct {
	// Root is the folder that was imported.
	Root string `json:"root"`
	// Scanned is how many files were looked at.
	Scanned int `json:"scanned"`
	// Imported is how many documents were filed.
	Imported int `json:"imported"`
	// Skipped is how many were already in the vault (same body hash).
	Skipped int `json:"skipped"`
	// Failed is how many could not be filed.
	Failed int `json:"failed"`
	// Indexed is how many documents were vectorized.
	Indexed int `json:"indexed"`
	// Collection is the collection the documents were vectorized into.
	Collection string `json:"collection,omitempty"`
	// Errors holds the first few failures.
	Errors []string `json:"errors,omitempty"`
	// Documents lists what was filed.
	Documents []string `json:"documents,omitempty"`
}

// ImportProgress reports the progress of an import.
type ImportProgress func(done, total int, name string)

// importSkipDirs are directories that never hold documents worth filing.
var importSkipDirs = map[string]bool{
	".git":         true,
	".svn":         true,
	"node_modules": true,
	"__pycache__":  true,
	".venv":        true,
	"venv":         true,
	".migrated":    true,
}

// ImportDocuments files every text file of a folder into the documents vault.
//
// It is the bulk path: a folder of raw material arrives, everything in it is
// filed by machine (dated, categorized, named after its content hash) and
// optionally vectorized into a collection. Files that are already in the
// vault (same body hash) are skipped instead of copied twice.
//
// maxFiles bounds the work of one call; 0 means no bound.
func (e *Engine) ImportDocuments(root string, channel string, collection string, reindex bool, maxFiles int, prog ImportProgress) (*ImportResult, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, fmt.Errorf("path is required")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	st, err := os.Stat(abs)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", root, err)
	}
	if !st.IsDir() {
		return nil, fmt.Errorf("%s is not a directory", root)
	}
	if channel == "" {
		channel = DocChannelImport
	}
	res := &ImportResult{Root: abs, Collection: collection}

	// Collect the files first so the total is known for the progress report.
	var files []string
	err = filepath.WalkDir(abs, func(path string, d os.DirEntry, werr error) error {
		if werr != nil {
			return nil
		}
		if d.IsDir() {
			if path != abs && importSkipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasPrefix(d.Name(), ".") {
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		return nil, err
	}
	total := len(files)
	if maxFiles > 0 && total > maxFiles {
		files = files[:maxFiles]
		total = maxFiles
	}

	for i, path := range files {
		res.Scanned++
		if prog != nil {
			prog(i, total, filepath.Base(path))
		}
		name, err := e.importOneFile(path, channel, collection, reindex, res)
		if err != nil {
			res.Failed++
			if len(res.Errors) < 10 {
				res.Errors = append(res.Errors, fmt.Sprintf("%s: %v", filepath.Base(path), err))
			}
			continue
		}
		if name == "" {
			res.Skipped++
			continue
		}
		res.Imported++
		if len(res.Documents) < 500 {
			res.Documents = append(res.Documents, name)
		}
	}
	return res, nil
}

// importOneFile files one file into the vault. It returns the vault-relative
// path, or "" when the file was already there.
func (e *Engine) importOneFile(path string, channel string, collection string, reindex bool, res *ImportResult) (string, error) {
	st, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if st.Size() > maxDocBytes {
		return "", fmt.Errorf("file is too large (%d bytes)", st.Size())
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if looksBinary(data) {
		return "", fmt.Errorf("binary file")
	}
	text := string(data)
	title := strings.TrimSpace(filepath.Base(path))
	if ext := filepath.Ext(title); ext != "" {
		title = strings.TrimSuffix(title, ext)
	}
	ext := strings.ToLower(filepath.Ext(path))
	if ext == "" {
		ext = ".md"
	}
	// The hash of the file exactly as it is: an index built from this file
	// recorded that hash.
	fileSum := sha256.Sum256(data)
	file, err := e.fileDocument(Document{
		Channel:    channel,
		Title:      title,
		Text:       text,
		Collection: collection,
		Ext:        ext,
		RawSHA256:  hex.EncodeToString(fileSum[:]),
	})
	if err != nil {
		return "", err
	}
	if file.Deduped {
		return "", nil
	}
	if reindex && collection != "" {
		meta := map[string]string{
			"source_type": "document",
			"kind":        "import",
			"visibility":  "agent",
			"collection":  collection,
		}
		if _, err := e.IndexDocument(file.Path, collection, meta); err != nil {
			return "", fmt.Errorf("filed but not vectorized: %w", err)
		}
		res.Indexed++
	}
	return file.Rel, nil
}

// looksBinary reports whether a buffer looks like a binary file.
func looksBinary(data []byte) bool {
	limit := len(data)
	if limit > 8000 {
		limit = 8000
	}
	for i := 0; i < limit; i++ {
		if data[i] == 0 {
			return true
		}
	}
	return false
}
