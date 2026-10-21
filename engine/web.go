package engine

import (
	"fmt"
	"log"
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
	// Unchanged reports that this exact content is already indexed for this
	// URL, so nothing had to be embedded again. Re-searching the same article
	// is common; embedding it a second time costs a full pass for nothing.
	Unchanged bool
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
//
// The old version of the page is dropped only AFTER its replacement has been
// embedded and the provenance/dimension gates passed, so a slow or failing
// embed never leaves the reader with neither version. Content that is already
// indexed (same content hash) is not embedded again at all.
func (e *Engine) IngestWebDocument(collectionID string, doc WebDocument, opts WebIngestOptions) (*WebIngestResult, error) {
	collectionID = strings.TrimSpace(collectionID)
	if collectionID == "" {
		return nil, fmt.Errorf("collection_id is required")
	}
	text := strings.TrimSpace(doc.Text)
	if text == "" {
		return nil, fmt.Errorf("document has no text")
	}
	if opts.MaxChars > 0 {
		// Cut on a rune boundary: a byte cut can split a multi-byte
		// character in half and leave a replacement rune in the chunk.
		if r := []rune(text); len(r) > opts.MaxChars {
			text = string(r[:opts.MaxChars])
		}
	}
	if doc.CanonicalURL == "" {
		doc.CanonicalURL = doc.URL
	}

	path, sha, err := e.writeWebRaw(collectionID, doc, text)
	if err != nil {
		return nil, err
	}

	// Already indexed with the same content? Nothing to embed. The page was
	// just verified though, so its fetched_at is refreshed: otherwise a
	// re-verified page would be flagged as stale.
	if stored := e.storedSourceSHA(collectionID, path); stored != "" && stored == sha {
		if err := e.touchWebSource(collectionID, path, doc); err != nil {
			log.Printf("web ingest: touch %s failed: %v", path, err)
		}
		log.Printf("web ingest: '%s' is unchanged, skipping embedding", doc.CanonicalURL)
		return &WebIngestResult{Collection: collectionID, ChunksIndexed: 0, Unchanged: true}, nil
	}

	metadata := e.webMetadata(doc)
	// replaceSource: the previous version of this URL is dropped inside the
	// index, after the embedding succeeded and the gates passed.
	res, used, err := e.indexTextInternal(text, collectionID, metadata, doc.URL, path, nil, nil, true)
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
		Replaced:      res.Replaced,
	}, nil
}

// storedSourceSHA returns the content hash the collection recorded for one
// source file, or "" when the source is unknown. It is what makes a repeated
// search of the same article free.
func (e *Engine) storedSourceSHA(collectionID, sourceFile string) string {
	store := e.storeFor(collectionID)
	fs, ok := store.(*FileStore)
	if !ok {
		return ""
	}
	return fs.SourceSHA(collectionID, sourceFile)
}

// touchWebSource refreshes the metadata of a stored page whose content did
// not change: the vectors stay, but the page was just seen again.
func (e *Engine) touchWebSource(collectionID, sourceFile string, doc WebDocument) error {
	store := e.storeFor(collectionID)
	fs, ok := store.(*FileStore)
	if !ok {
		return nil
	}
	md := e.webMetadata(doc)
	fields := map[string]string{"fetched_at": md["fetched_at"]}
	return fs.TouchSource(collectionID, sourceFile, fields)
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

// writeWebRaw files the page body in the documents vault and returns the file
// path plus its content hash.
//
// The document is filed by machine: the date folder shards the corpus, the
// channel says where it came from, the category comes from the rule table and
// the file name carries the time and the body hash. The same page fetched
// again lands on the same file (the body hash is the identity), so the corpus
// does not fill up with copies of the same article.
func (e *Engine) writeWebRaw(collectionID string, doc WebDocument, text string) (string, string, error) {
	file, err := e.fileDocument(Document{
		Channel:     DocChannelWeb,
		Title:       doc.Title,
		URL:         doc.CanonicalURL,
		Text:        text,
		Provider:    doc.Provider,
		PublishedAt: doc.PublishedAt,
		Tags:        doc.Tags,
		Collection:  collectionID,
	})
	if err != nil {
		return "", "", err
	}
	return file.Path, file.SHA256, nil
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
