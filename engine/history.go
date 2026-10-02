package engine

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// HistoryEntry is one audit row in logs/index_history.jsonl. It answers
// "which collection did this raw file get vectorized into, with what model,
// and when" — even after the collection is gone.
type HistoryEntry struct {
	TS            string `json:"ts"`
	Op            string `json:"op"` // index | reindex | upload | delete_collection
	File          string `json:"file,omitempty"`
	SHA256        string `json:"sha256,omitempty"`
	Collection    string `json:"collection,omitempty"`
	Backend       string `json:"backend,omitempty"`
	Mode          string `json:"mode,omitempty"` // append | rebuild
	Chunks        int    `json:"chunks,omitempty"`
	EmbedProvider string `json:"embed_provider,omitempty"`
	EmbedModel    string `json:"embed_model,omitempty"`
	VectorDim     int    `json:"vector_dim,omitempty"`
	OK            bool   `json:"ok"`
	Error         string `json:"error,omitempty"`
}

var historyMu sync.Mutex

// AppendHistory appends one JSONL row. Best-effort by contract: callers log
// the error but never fail the user-facing operation because of it.
func (e *Engine) AppendHistory(en HistoryEntry) error {
	if e.config.HistoryPath == "" {
		return nil
	}
	if en.TS == "" {
		en.TS = time.Now().UTC().Format(time.RFC3339)
	}
	data, err := json.Marshal(en)
	if err != nil {
		return err
	}
	historyMu.Lock()
	defer historyMu.Unlock()
	if dir := filepath.Dir(e.config.HistoryPath); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(e.config.HistoryPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(data, '\n'))
	return err
}

// HistoryForFile returns every entry whose File matches absPath, oldest first.
func (e *Engine) HistoryForFile(absPath string) ([]HistoryEntry, error) {
	var out []HistoryEntry
	f, err := os.Open(e.config.HistoryPath)
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	buf := make([]byte, 1024*1024)
	sc.Buffer(buf, len(buf))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var en HistoryEntry
		if json.Unmarshal([]byte(line), &en) != nil {
			continue
		}
		if en.File == absPath {
			out = append(out, en)
		}
	}
	return out, sc.Err()
}

// RecentHistory returns entries with Op=="upload" (most recent first), used
// by the Data Bank "recent uploads first" toggle.
func (e *Engine) RecentUploads(limit int) ([]HistoryEntry, error) {
	var out []HistoryEntry
	f, err := os.Open(e.config.HistoryPath)
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	buf := make([]byte, 1024*1024)
	sc.Buffer(buf, len(buf))
	for sc.Scan() {
		var en HistoryEntry
		if json.Unmarshal(sc.Bytes(), &en) != nil || en.Op != "upload" {
			continue
		}
		out = append(out, en)
	}
	// reverse to most-recent-first
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}
