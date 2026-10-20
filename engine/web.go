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
	// DedupThreshold is the cosine similarity at or above which an
	// existing document counts as the same page and is skipped.
	// 0 disables the near-duplicate gate.
	DedupThreshold float64
	// MaxChars caps how much of the page body is stored.
	MaxChars int
}

// WebIngestResult reports what happened to one document.
type WebIngestResult struct {
	Collection     string
	ChunksIndexed  int
	Duplicate      bool
	DuplicateOf    string
	DuplicateScore float64
	Reason         string
}

// IngestWebDocument stores a web page into a collection.
//
// Two gates protect the collection from bloat:
//
//  1. An exact re-fetch of an unchanged page (same canonical URL, same
//     content hash) is a no-op.
//  2. A page that is semantically near-identical to something already
//     stored (the same story from another outlet) is skipped, judged by
//     the cosine similarity of a page-level fingerprint embedding.
//
// The near-duplicate check costs one embedding per page, not per chunk.
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

	// Gate 1: has this exact page already been stored?
	path, sha, unchanged, err := e.writeWebRaw(doc, text)
	if err != nil {
		return nil, err
	}
	if unchanged {
		return &WebIngestResult{
			Collection: collectionID,
			Duplicate:  true,
			Reason:     "identical content already stored for this URL",
		}, nil
	}
	_ = sha

	// Gate 2: is this page a near-duplicate of something already stored?
	if opts.DedupThreshold > 0 {
		dupSource, dupScore, err := e.findWebDuplicate(collectionID, doc, text, opts.DedupThreshold)
		if err != nil {
			log.Printf("web dedup check failed for %s: %v", doc.CanonicalURL, err)
		} else if dupSource != "" {
			return &WebIngestResult{
				Collection:     collectionID,
				Duplicate:      true,
				DuplicateOf:    dupSource,
				DuplicateScore: dupScore,
				Reason:         "near-duplicate of an already stored page",
			}, nil
		}
	}

	// Replacing a page: drop the previous chunks of this URL first so a
	// changed article does not leave its old text behind.
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
	}, nil
}

// webMetadata builds the metadata map attached to every chunk of a web page.
// The keys are flat strings because that is what the file store's payload
// matcher can filter on.
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
//
// The second return value reports whether the file already held exactly this
// content (an unchanged re-fetch).
func (e *Engine) writeWebRaw(doc WebDocument, text string) (string, string, bool, error) {
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
		return "", "", false, fmt.Errorf("create web dir: %w", err)
	}

	sum := sha256.Sum256([]byte(text))
	sha := hex.EncodeToString(sum[:])
	key := sha256.Sum256([]byte(doc.CanonicalURL))
	name := hex.EncodeToString(key[:])[:16] + ".md"
	path := filepath.Join(dir, name)

	if prev, err := os.ReadFile(path); err == nil {
		prevSum := sha256.Sum256(prev)
		if hex.EncodeToString(prevSum[:]) == sha {
			return path, sha, true, nil
		}
	}

	header := fmt.Sprintf("# %s\n\n- source: %s\n- provider: %s\n- fetched_at: %s\n\n",
		doc.Title, doc.CanonicalURL, doc.Provider, time.Now().UTC().Format(time.RFC3339))
	if err := os.WriteFile(path, []byte(header+text), 0o644); err != nil {
		return "", "", false, fmt.Errorf("write web file: %w", err)
	}
	return path, sha, false, nil
}

// findWebDuplicate looks for a stored page that is semantically the same as
// the candidate.
//
// The comparison is judged by the reranker, not by cosine. With this server's
// embedding model, cosine similarity sits in a narrow band (0.13-0.25)
// whatever the query is, so a cosine gate would either never fire or always
// fire. The reranker separates a matching page from an unrelated one by an
// order of magnitude, so the page fingerprint is used as the query and the
// stored chunks as the documents.
//
// It returns the matching source URL and its rerank score, or empty strings
// when nothing close enough is stored.
func (e *Engine) findWebDuplicate(collectionID string, doc WebDocument, text string, threshold float64) (string, float64, error) {
	store := e.storeFor(collectionID)
	exists, err := store.CollectionExists(collectionID)
	if err != nil {
		return "", 0, err
	}
	if !exists {
		return "", 0, nil
	}
	fingerprint := webFingerprint(doc.Title, text)
	filter := map[string]any{"metadata.source_type": "web"}
	hits, err := e.Search(fingerprint, collectionID, 5, true, 0, filter)
	if err != nil {
		return "", 0, err
	}
	for _, h := range hits {
		if h.Score < threshold {
			continue
		}
		source := doc.CanonicalURL
		if h.Metadata != nil {
			if u := h.Metadata["canonical_url"]; u != "" {
				source = u
			} else if u := h.Metadata["url"]; u != "" {
				source = u
			} else if h.Source != "" {
				source = h.Source
			}
		}
		// A page is never a duplicate of itself when the content changed.
		if source != "" && source == doc.CanonicalURL {
			continue
		}
		return source, h.Score, nil
	}
	return "", 0, nil
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

// webFingerprint builds the text used for the page-level duplicate check: the
// title plus the opening of the body, which is where two write-ups of the
// same story overlap most.
func webFingerprint(title, text string) string {
	body := text
	if len(body) > 800 {
		body = body[:800]
	}
	if strings.TrimSpace(title) == "" {
		return body
	}
	return title + "\n" + body
}

// WebDocumentFresh reports whether a stored web document is still within the
// given freshness window. An unknown timestamp is treated as stale so the
// caller re-fetches rather than trusting undated content.
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

// webStopwords are function words that carry no topical signal. They are
// dropped before the term-overlap gate so a shared "the" or "and" cannot make
// an unrelated page look like a match. Topic words (news, latest, new) are
// deliberately kept: they do carry signal.
var webStopwords = map[string]bool{
	"a": true, "an": true, "the": true, "of": true, "to": true, "in": true,
	"on": true, "for": true, "and": true, "or": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "by": true,
	"with": true, "from": true, "as": true, "at": true, "it": true,
	"its": true, "this": true, "that": true, "these": true, "those": true,
	"about": true, "how": true, "what": true, "when": true, "where": true,
	"which": true, "who": true, "why": true, "me": true, "my": true,
	"we": true, "our": true, "you": true, "your": true, "they": true,
	"them": true, "their": true, "he": true, "she": true, "his": true,
	"her": true, "do": true, "does": true, "did": true, "have": true,
	"has": true, "had": true, "can": true, "will": true, "would": true,
	"should": true, "could": true, "am": true, "i": true, "if": true,
	"so": true, "but": true, "not": true, "no": true, "yes": true,
	"up": true, "down": true, "out": true, "over": true, "under": true,
	"again": true, "very": true, "just": true, "only": true, "also": true,
	"more": true, "most": true, "some": true, "any": true, "all": true,
	"please": true, "tell": true, "give": true, "show": true, "find": true,
	"的": true, "了": true, "和": true, "是": true, "在": true, "有": true,
	"我": true, "你": true, "他": true, "她": true, "它": true, "们": true,
	"这": true, "那": true, "什么": true, "怎么": true, "如何": true,
	"请": true, "告诉": true, "一个": true, "可以": true, "需要": true,
}

// webContentTerms tokenizes text and keeps the terms that carry signal:
// stopwords and one-character tokens are dropped.
func webContentTerms(s string) []string {
	raw := tokenize(s)
	out := make([]string, 0, len(raw))
	seen := make(map[string]bool, len(raw))
	for _, t := range raw {
		if len(t) < 2 || webStopwords[t] || seen[t] {
			continue
		}
		seen[t] = true
		out = append(out, t)
	}
	return out
}

// webTermOverlap returns the fraction of the query's content terms that also
// occur in the document.
//
// Matching is prefix-based so "release" still matches "released" and "model"
// matches "models"; plain equality would make the gate far too brittle.
//
// It is the lexical half of the "do we already know this?" gate. The
// reranker alone is not enough: a short query that happens to share one word
// with an unrelated page ("fare" against a paragraph about "off-peak prices")
// scores 0.81 on the reranker, which would wrongly short-circuit a live
// search. Requiring the query's own terms to actually appear in the stored
// text keeps that case out while a genuine repeat still passes.
func webTermOverlap(query, doc string) float64 {
	qTerms := webContentTerms(query)
	if len(qTerms) == 0 {
		// Nothing to compare (e.g. a query of only stopwords). Fall back to
		// the semantic signal alone.
		return 1
	}
	docTerms := webContentTerms(doc)
	if len(docTerms) == 0 {
		return 0
	}
	hit := 0
	for _, q := range qTerms {
		if webTermInList(q, docTerms) {
			hit++
		}
	}
	return float64(hit) / float64(len(qTerms))
}

// webTermInList reports whether term appears in terms, allowing a prefix
// match in either direction. Short terms are compared exactly: a 2-3 letter
// prefix is too noisy ("ai" would match "aircraft").
func webTermInList(term string, terms []string) bool {
	for _, t := range terms {
		if t == term {
			return true
		}
		if len(t) >= 4 && len(term) >= 4 && (strings.HasPrefix(t, term) || strings.HasPrefix(term, t)) {
			return true
		}
	}
	return false
}

// WebCacheHit is a stored web document that already covers a query.
type WebCacheHit struct {
	Result  SearchResult
	Fetched string
	Stale   bool
}

// WebCacheHits finds stored web documents that already answer the query.
//
// Two signals have to agree before a stored page counts as "already known":
//
//   - the rerank score, because with this embedding model the raw cosine
//     sits in a narrow band (0.13-0.25) for a matching query AND for a
//     completely unrelated one, so it cannot decide anything;
//   - the term overlap, because the reranker on its own can score a
//     superficially similar page at 0.81 (a query about train fares against
//     a paragraph about off-peak prices).
//
// Together they separate "we already covered this" from "the word 'fare'
// appears somewhere".
func (e *Engine) WebCacheHits(query, collectionID string, limit int, minScore float64, minOverlap float64, maxAgeDays int) ([]WebCacheHit, error) {
	collectionID = strings.TrimSpace(collectionID)
	if collectionID == "" {
		return nil, nil
	}
	store := e.storeFor(collectionID)
	exists, err := store.CollectionExists(collectionID)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, nil
	}
	filter := map[string]any{"metadata.source_type": "web"}
	results, err := e.Search(query, collectionID, limit, true, 0, filter)
	if err != nil {
		return nil, err
	}
	out := make([]WebCacheHit, 0, len(results))
	for _, res := range results {
		if res.Score < minScore {
			continue
		}
		if webTermOverlap(query, res.Text) < minOverlap {
			continue
		}
		fetched := res.Metadata["fetched_at"]
		out = append(out, WebCacheHit{
			Result:  res,
			Fetched: fetched,
			Stale:   !WebDocumentFresh(fetched, maxAgeDays),
		})
	}
	return out, nil
}
