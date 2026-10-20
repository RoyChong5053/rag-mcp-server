package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// WebDocument is one web page ready to be turned into vectors.
type WebDocument struct {
	// URL is the address the page was reached at.
	URL string
	// CanonicalURL is the tracking-parameter-free form of URL. It is the
	// document's identity: the same article reached through different
	// links maps to the same stored document.
	CanonicalURL string
	Title        string
	SiteName     string
	// Text is the page body (markdown). It is what gets chunked.
	Text string
	// Provider names the service that produced the document.
	Provider string
	// PublishedAt is the article's publication time when known.
	PublishedAt time.Time
	// Authority is the provider's source-quality tier (0-4), or -1.
	Authority int
	// Score is the provider's relevance score for the hit.
	Score float64
	Lang  string
	// Tags are hierarchical tags attached to every chunk.
	Tags []string
}

// WebIngestOptions tunes one ingest call.
type WebIngestOptions struct {
	// MaxChars caps how much of the page body is stored.
	MaxChars int
}

// WebIngestResult reports what happened to one document.
type WebIngestResult struct {
	Collection    string
	ChunksIndexed int
	// Replaced is the number of chunks that were dropped from a previous
	// version of the same URL before the new ones were written.
	Replaced int
}

// IngestWebDocument stores a web page into a collection.
//
// Everything that was searched is stored. There is no relevance gate here:
// deciding what is worth keeping is the caller's job, and a server that
// silently drops material makes the knowledge base impossible to reason
// about. Two things are kept, because they are identity rather than
// judgement:
//
//   - One page per canonical URL. The raw body lives in a file named after
//     the URL, so re-searching the same article replaces it instead of
//     piling a second copy on top.
//   - Chunk IDs are derived from the chunk text, so re-indexing the same
//     text overwrites the same points.
//
// The publication date travels with every chunk (published_at) as well as
// the moment it was fetched (fetched_at), because a reader has to be able
// to tell yesterday's news from today's.
func (e *Engine) IngestWebDocument(collectionID string, doc WebDocument, opts WebIngestOptions) (*WebIngestResult, error) {
	collectionID = strings.TrimSpace(collectionID)
	if collectionID == "" {
		return nil, fmt.Errorf("collection_id is required")
	}
	text := strings.TrimSpace(doc.Text)
	if text == "" {
		return nil, fmt.Errorf("document has no text")
	}
	if opts.MaxChars > 0 && len(text) > opts.MaxChars {
		text = text[:opts.MaxChars]
	}
	if doc.CanonicalURL == "" {
		doc.CanonicalURL = doc.URL
	}

	path, sha, err := e.writeWebRaw(doc, text)
	if err != nil {
		return nil, err
	}

	// Drop the previous chunks of this URL so a changed article does not
	// leave its old text behind.
	replaced := 0
	if err := e.deleteWebSource(collectionID, path); err != nil {
		log.Printf("web source cleanup failed for %s: %v", path, err)
	}

	metadata := e.webMetadata(doc)
	res, used, err := e.indexTextInternal(text, collectionID, metadata, doc.URL, path, nil, nil)
	if err != nil {
		return nil, err
	}
	embedProvider, embedModel, embedDim, embedDistance := e.activeEmbedProvenance()
	e.recordIndex(collectionID, Provenance{
		SourceFile:     path,
		SourceSHA256:   sha,
		ChunkSize:      used.Size,
		ChunkOverlap:   used.Overlap,
		ChunkStrategy:  used.Strategy,
		EmbedModel:     embedModel,
		EmbedProvider:  embedProvider,
		VectorDim:      embedDim,
		VectorDistance: embedDistance,
		Chunks:         res.ChunksIndexed,
	}, BackendVectra)
	e.writeVaultManifest(collectionID)
	return &WebIngestResult{
		Collection:    collectionID,
		ChunksIndexed: res.ChunksIndexed,
		Replaced:      replaced,
	}, nil
}

// webMetadata builds the metadata map attached to every chunk of a web page.
// The keys are flat strings because that is what the file store's payload
// matcher can filter on.
//
// published_at and fetched_at are both kept on purpose: published_at says
// how old the news is, fetched_at says when it entered the library. A reader
// that only has one of them can still tell yesterday from today.
func (e *Engine) webMetadata(doc WebDocument) map[string]string {
	md := map[string]string{
		"source_type":   "web",
		"kind":          "web_search",
		"visibility":    "agent",
		"collection":    "",
		"provider":      doc.Provider,
		"url":           doc.URL,
		"canonical_url": doc.CanonicalURL,
		"title":         doc.Title,
		"site_name":     doc.SiteName,
		"fetched_at":    time.Now().UTC().Format(time.RFC3339),
	}
	if doc.PublishedAt.IsZero() {
		md["published_at"] = ""
	} else {
		md["published_at"] = doc.PublishedAt.UTC().Format(time.RFC3339)
	}
	if doc.Authority >= 0 {
		md["authority"] = strconv.Itoa(doc.Authority)
	}
	if doc.Score != 0 {
		md["score"] = strconv.FormatFloat(doc.Score, 'f', 4, 64)
	}
	if doc.Lang != "" {
		md["lang"] = doc.Lang
	}
	if len(doc.Tags) > 0 {
		md["tags"] = strings.Join(doc.Tags, ",")
	}
	return md
}

// writeWebRaw persists the page body next to the other data-bank files and
// returns the file path plus its content hash. The path is derived from the
// canonical URL, so the same article always lands on the same file and a
// re-fetch replaces it instead of piling up.
func (e *Engine) writeWebRaw(doc WebDocument, text string) (string, string, error) {
	base := e.memoryDir
	if base == "" {
		if e.docsDir != "" {
			base = filepath.Join(e.docsDir, "memory")
		} else {
			base = filepath.Join("docs", "memory")
		}
	}
	dir := filepath.Join(base, "web")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", "", fmt.Errorf("create web dir: %w", err)
	}

	sum := sha256.Sum256([]byte(text))
	sha := hex.EncodeToString(sum[:])
	key := sha256.Sum256([]byte(doc.CanonicalURL))
	name := hex.EncodeToString(key[:])[:16] + ".md"
	path := filepath.Join(dir, name)

	header := fmt.Sprintf("# %s\n\n- source: %s\n- provider: %s\n- fetched_at: %s\n\n",
		doc.Title, doc.CanonicalURL, doc.Provider, time.Now().UTC().Format(time.RFC3339))
	if err := os.WriteFile(path, []byte(header+text), 0o644); err != nil {
		return "", "", fmt.Errorf("write web file: %w", err)
	}
	return path, sha, nil
}

// deleteWebSource removes the chunks previously stored for one source file so
// a re-fetched page replaces its old text.
func (e *Engine) deleteWebSource(collectionID, sourceFile string) error {
	store := e.storeFor(collectionID)
	exists, err := store.CollectionExists(collectionID)
	if err != nil || !exists {
		return err
	}
	filter := map[string]any{"source_file": sourceFile}
	return store.DeletePoints(collectionID, filter)
}

// WebDocumentFresh reports whether a stored web document is still within the
// given freshness window. An unknown timestamp is treated as stale so the
// caller can tell that it does not know how old the material is.
func WebDocumentFresh(fetchedAt string, days int) bool {
	if days <= 0 {
		return true
	}
	fetchedAt = strings.TrimSpace(fetchedAt)
	if fetchedAt == "" {
		return false
	}
	t, err := time.Parse(time.RFC3339, fetchedAt)
	if err != nil {
		return false
	}
	return time.Since(t) <= time.Duration(days)*24*time.Hour
}

// WebDocumentAge renders how old a stored document is, for display next to
// a recalled chunk ("3 days ago", "unknown").
func WebDocumentAge(timestamp string, now time.Time) string {
	timestamp = strings.TrimSpace(timestamp)
	if timestamp == "" {
		return "unknown"
	}
	t, err := time.Parse(time.RFC3339, timestamp)
	if err != nil {
		return "unknown"
	}
	d := now.Sub(t)
	if d < 0 {
		return "just now"
	}
	switch {
	case d < time.Hour:
		return fmt.Sprintf("%d minutes ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%d hours ago", int(d.Hours()))
	case d < 30*24*time.Hour:
		return fmt.Sprintf("%d days ago", int(d.Hours()/24))
	case d < 365*24*time.Hour:
		return fmt.Sprintf("%d months ago", int(d.Hours()/24/30))
	default:
		return fmt.Sprintf("%d years ago", int(d.Hours()/24/365))
	}
}
