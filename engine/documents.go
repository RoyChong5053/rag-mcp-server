package engine

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode"

	"gopkg.in/yaml.v3"
)

// The documents vault is the raw-text half of the system. The vector side
// answers "what did I learn about X"; the vault answers "where is that exact
// string". The corpus is append-only and is never curated by hand: every
// document is filed by machine under
//
//	<documents_dir>/<yyyy-mm-dd>/<channel>/<category>/<HHMMSS>-<sha8>-<slug><ext>
//
// The date comes first on purpose. A date range then maps to a list of
// top-level directories, so "search the last week" touches seven directories
// instead of walking the whole corpus, and `ls` is chronological without any
// bookkeeping.
//
// Channel says where the document came from (web / agent / upload / chat /
// import). Category says what it is about and comes from a rule table; a
// document that matches no rule lands in "_". Neither is curated by a human.

const (
	// DocChannelWeb is a page filed by a web search or a direct fetch.
	DocChannelWeb = "web"
	// DocChannelAgent is a memory stored by an agent.
	DocChannelAgent = "agent"
	// DocChannelUpload is a file uploaded through the dashboard.
	DocChannelUpload = "upload"
	// DocChannelChat is an imported conversation log.
	DocChannelChat = "chat"
	// DocChannelImport is a file brought in from an external folder.
	DocChannelImport = "import"

	// DocCategoryNone is the category of a document no rule matched.
	DocCategoryNone = "_"

	// maxDocBytes is the largest document body the vault stores. A book or
	// a long chat log is a legitimate document, so the limit is generous;
	// it only guards against a runaway file being copied in.
	maxDocBytes = 100 << 20
)

// CategoryRule files a document into a category folder when Match matches the
// document subject (title + url + tags).
type CategoryRule struct {
	Match    string `yaml:"match"`
	Category string `yaml:"category"`
}

// Document is one raw document to file into the vault.
type Document struct {
	// Channel is the source channel. Empty becomes "web".
	Channel string
	// Category overrides the rule table when set.
	Category string
	// Title is the human-readable name of the document.
	Title string
	// URL is where the document came from.
	URL string
	// Text is the body. It is what gets chunked, embedded and searched.
	Text string
	// Provider names the service that produced the document.
	Provider string
	// PublishedAt is the document's own publication time when known.
	PublishedAt time.Time
	// Tags are attached to the document and to every chunk of it.
	Tags []string
	// Collection is the collection the document was vectorized into.
	Collection string
	// Ext is the file extension, including the dot. Empty becomes ".md".
	Ext string
}

// DocMeta is the front-matter of a vaulted document. It is the document's
// identity: the search returns it next to every hit so the reader knows which
// document it is looking at, not just which file.
type DocMeta struct {
	ID          string   `yaml:"id" json:"id"`
	Channel     string   `yaml:"channel" json:"channel"`
	Category    string   `yaml:"category" json:"category"`
	Title       string   `yaml:"title,omitempty" json:"title,omitempty"`
	URL         string   `yaml:"url,omitempty" json:"url,omitempty"`
	URLs        []string `yaml:"urls,omitempty" json:"urls,omitempty"`
	SHA256      string   `yaml:"sha256" json:"sha256"`
	Provider    string   `yaml:"provider,omitempty" json:"provider,omitempty"`
	PublishedAt string   `yaml:"published_at,omitempty" json:"published_at,omitempty"`
	FetchedAt   string   `yaml:"fetched_at" json:"fetched_at"`
	UpdatedAt   string   `yaml:"updated_at" json:"updated_at"`
	Bytes       int      `yaml:"bytes" json:"bytes"`
	Collection  string   `yaml:"collection,omitempty" json:"collection,omitempty"`
	Tags        []string `yaml:"tags,omitempty" json:"tags,omitempty"`
}

// DocEntry is one ledger row: a document plus where it lives.
type DocEntry struct {
	Path string `yaml:"path" json:"path"`
	DocMeta
}

// DocFile is the outcome of filing one document.
type DocFile struct {
	// Path is the absolute path of the document on disk.
	Path string `json:"path"`
	// Rel is the path relative to the documents dir (slash separated).
	Rel string `json:"rel"`
	// SHA256 is the hash of the body.
	SHA256 string `json:"sha256"`
	// Deduped reports that the body was already in the vault, so the
	// existing document was refreshed instead of a second copy being made.
	Deduped bool `json:"deduped"`
	// Created reports that this call wrote a new file.
	Created bool `json:"created"`
}

// docSegmentRe strips everything that is not safe in a path segment.
var docSegmentRe = regexp.MustCompile(`[^a-z0-9._-]+`)

// sanitizeDocSegment turns an arbitrary string into a safe path segment.
// CJK and other non-ASCII letters are transliterated away rather than kept:
// a segment that survives a copy to another filesystem is worth more than a
// pretty one.
func sanitizeDocSegment(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = docSegmentRe.ReplaceAllString(s, "-")
	s = strings.Trim(s, "-.")
	if len(s) > 40 {
		s = s[:40]
		s = strings.Trim(s, "-.")
	}
	return s
}

// docSlug builds the readable part of a file name from the title, falling back
// to the last segment of the URL.
func docSlug(title, url string) string {
	base := strings.TrimSpace(title)
	if base == "" {
		if u := strings.TrimSpace(url); u != "" {
			clean := u
			if i := strings.Index(clean, "://"); i >= 0 {
				clean = clean[i+3:]
			}
			if i := strings.IndexAny(clean, "?#"); i >= 0 {
				clean = clean[:i]
			}
			parts := strings.Split(strings.Trim(clean, "/"), "/")
			base = parts[len(parts)-1]
		}
	}
	if base == "" {
		base = "document"
	}
	// Keep letters, digits and CJK; everything else becomes a dash.
	var b strings.Builder
	for _, r := range base {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(unicode.ToLower(r))
		case r > 0x3000: // CJK, punctuation and anything above
			b.WriteRune(r)
		default:
			b.WriteRune('-')
		}
	}
	slug := b.String()
	slug = strings.Trim(slug, "-.")
	if slug == "" {
		slug = "document"
	}
	if len([]rune(slug)) > 48 {
		slug = string([]rune(slug)[:48])
		slug = strings.Trim(slug, "-.")
	}
	return slug
}

// docsRoot is the data-bank root. The engine field wins; the config is the
// fallback for an engine that was built without it.
func (e *Engine) docsRoot() string {
	if e.docsDir != "" {
		return e.docsDir
	}
	if e.config != nil {
		return e.config.DocsDir
	}
	return ""
}

// documentsDir is the vault root. It defaults to <docs_dir>/documents so the
// vault travels with the data bank, and it is resolved to an absolute path so
// every path the vault reports can be compared with the paths the dashboard
// walks.
func (e *Engine) documentsDir() string {
	dir := ""
	if e.config != nil {
		dir = e.config.DocumentsDir
	}
	if dir == "" {
		if base := e.docsRoot(); base != "" {
			dir = filepath.Join(base, "documents")
		} else {
			dir = filepath.Join("docs", "documents")
		}
	}
	if abs, err := filepath.Abs(dir); err == nil {
		return abs
	}
	return dir
}

// DocumentsDir exposes the vault root (absolute) for the management surfaces.
func (e *Engine) DocumentsDir() string { return e.documentsDir() }

// DocumentsDirRel is the vault root relative to the data-bank root, so a
// browser can turn a vault-relative document path into a data-bank path.
func (e *Engine) DocumentsDirRel() string {
	base := e.docsRoot()
	if base == "" {
		base = "docs"
	}
	if abs, err := filepath.Abs(base); err == nil {
		base = abs
	}
	if rel, err := filepath.Rel(base, e.documentsDir()); err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return filepath.ToSlash(rel)
	}
	return ""
}

// docRelPath builds the vault-relative path of a document. The date is the
// ingest date: it is the shard the filesystem cares about, not a claim about
// when the document was written.
func (e *Engine) docRelPath(when time.Time, channel, category, sha8, slug, ext string) string {
	parts := []string{
		when.Format("2006-01-02"),
		sanitizeDocSegment(channel),
		sanitizeDocSegment(category),
		when.Format("150405") + "-" + sha8 + "-" + slug + ext,
	}
	return filepath.Join(parts...)
}

// FileDocument files one document into the vault. It is the exported door
// into the vault for surfaces outside the engine (uploads, imports).
func (e *Engine) FileDocument(doc Document) (*DocFile, error) {
	return e.fileDocument(doc)
}

// DocsPath returns a document path relative to the data-bank root, so a
// dashboard can hand it to the index endpoints unchanged.
func (e *Engine) DocsPath(rel string) string {
	base := e.DocumentsDirRel()
	if base == "" {
		return filepath.ToSlash(rel)
	}
	return filepath.ToSlash(filepath.Join(base, filepath.FromSlash(rel)))
}

// compileCategoryRules compiles the category rule table. An invalid rule is
// reported and skipped: one bad regex must not take the vault down.
func compileCategoryRules(rules []CategoryRule) ([]compiledCategoryRule, error) {
	out := make([]compiledCategoryRule, 0, len(rules))
	for _, rule := range rules {
		pattern := strings.TrimSpace(rule.Match)
		if pattern == "" {
			continue
		}
		re, err := regexp.Compile(pattern)
		if err != nil {
			return out, fmt.Errorf("category rule %q: %w", pattern, err)
		}
		out = append(out, compiledCategoryRule{re: re, Category: rule.Category})
	}
	return out, nil
}

// docCategory resolves the category folder of a document: the explicit
// category wins, otherwise the first matching rule decides, otherwise "_".
//
// A rule is tested against the URL host, the title and the tags separately.
// Testing the host on its own is what makes a rule like "^github\.com" do
// what it looks like it does.
func (e *Engine) docCategory(doc Document) string {
	if c := sanitizeDocSegment(doc.Category); c != "" {
		return c
	}
	host := docHost(doc.URL)
	title := strings.ToLower(strings.TrimSpace(doc.Title))
	tags := make([]string, 0, len(doc.Tags))
	for _, t := range doc.Tags {
		tags = append(tags, strings.ToLower(strings.TrimSpace(t)))
	}
	for _, rule := range e.docCategoryRules {
		if rule.re == nil {
			continue
		}
		if host != "" && rule.re.MatchString(host) {
			return e.docRuleCategory(rule)
		}
		if title != "" && rule.re.MatchString(title) {
			return e.docRuleCategory(rule)
		}
		for _, t := range tags {
			if t != "" && rule.re.MatchString(t) {
				return e.docRuleCategory(rule)
			}
		}
	}
	return DocCategoryNone
}

// docRuleCategory is the category folder of a matched rule.
func (e *Engine) docRuleCategory(rule compiledCategoryRule) string {
	if c := sanitizeDocSegment(rule.Category); c != "" {
		return c
	}
	return DocCategoryNone
}

// docHost is the host of a URL, without the www prefix.
func docHost(raw string) string {
	u := strings.TrimSpace(raw)
	if u == "" {
		return ""
	}
	if i := strings.Index(u, "://"); i >= 0 {
		u = u[i+3:]
	}
	if i := strings.IndexAny(u, "/?#"); i >= 0 {
		u = u[:i]
	}
	if i := strings.LastIndex(u, "@"); i >= 0 {
		u = u[i+1:]
	}
	if i := strings.Index(u, ":"); i >= 0 {
		u = u[:i]
	}
	return strings.ToLower(strings.TrimPrefix(u, "www."))
}

// fileDocument files one document into the vault.
//
// The body hash is the identity: a body that is already in the vault refreshes
// that document instead of adding a second copy. The file name keeps the
// first ingest's timestamp, so the corpus keeps one row per body and the
// "same article, changed text" case becomes a new row instead of a silent
// overwrite.
func (e *Engine) fileDocument(doc Document) (*DocFile, error) {
	body := strings.TrimSpace(doc.Text)
	if body == "" {
		return nil, fmt.Errorf("document has no text")
	}
	if len(body) > maxDocBytes {
		return nil, fmt.Errorf("document is too large (%d bytes, max %d)", len(body), maxDocBytes)
	}
	channel := sanitizeDocSegment(doc.Channel)
	if channel == "" {
		channel = DocChannelWeb
	}
	category := e.docCategory(doc)
	// The day folder follows the server's local clock: "yesterday" means
	// yesterday where the server lives, not yesterday in UTC.
	now := time.Now()

	sum := sha256.Sum256([]byte(body))
	sha := hex.EncodeToString(sum[:])
	sha8 := sha[:8]

	ext := doc.Ext
	if ext == "" {
		ext = ".md"
	}
	if !strings.HasPrefix(ext, ".") {
		ext = "." + ext
	}

	title := strings.TrimSpace(doc.Title)
	slug := docSlug(title, doc.URL)

	// Already in the vault? Refresh that document instead of copying it.
	if e.ledger != nil {
		if prev, ok := e.ledger.Lookup(sha); ok && prev.Path != "" {
			abs := filepath.Join(e.documentsDir(), filepath.FromSlash(prev.Path))
			meta := prev.DocMeta
			meta.SHA256 = sha
			meta.UpdatedAt = now.Format(time.RFC3339)
			meta.FetchedAt = now.Format(time.RFC3339)
			meta.Bytes = len(body)
			if doc.Collection != "" {
				meta.Collection = doc.Collection
			}
			if doc.Provider != "" {
				meta.Provider = doc.Provider
			}
			if !doc.PublishedAt.IsZero() {
				meta.PublishedAt = doc.PublishedAt.UTC().Format(time.RFC3339)
			}
			meta.URLs = mergeURLs(meta.URLs, doc.URL, meta.URL)
			if err := writeDocFile(abs, meta, body); err != nil {
				return nil, err
			}
			e.ledger.Append(DocEntry{Path: prev.Path, DocMeta: meta})
			return &DocFile{Path: abs, Rel: prev.Path, SHA256: sha, Deduped: true}, nil
		}
	}

	rel := e.docRelPath(now, channel, category, sha8, slug, ext)
	abs := filepath.Join(e.documentsDir(), filepath.FromSlash(rel))

	meta := DocMeta{
		ID:         sha8,
		Channel:    channel,
		Category:   category,
		Title:      title,
		URL:        doc.URL,
		SHA256:     sha,
		Provider:   doc.Provider,
		FetchedAt:  now.Format(time.RFC3339),
		UpdatedAt:  now.Format(time.RFC3339),
		Bytes:      len(body),
		Collection: doc.Collection,
		Tags:       doc.Tags,
	}
	if !doc.PublishedAt.IsZero() {
		meta.PublishedAt = doc.PublishedAt.UTC().Format(time.RFC3339)
	}
	if doc.URL != "" {
		meta.URLs = []string{doc.URL}
	}

	if err := writeDocFile(abs, meta, body); err != nil {
		return nil, err
	}
	if e.ledger != nil {
		e.ledger.Append(DocEntry{Path: filepath.ToSlash(rel), DocMeta: meta})
	}
	return &DocFile{Path: abs, Rel: filepath.ToSlash(rel), SHA256: sha, Created: true}, nil
}

// mergeURLs adds a URL to the list without repeating it.
func mergeURLs(existing []string, url, primary string) []string {
	out := make([]string, 0, len(existing)+1)
	seen := map[string]bool{}
	add := func(u string) {
		u = strings.TrimSpace(u)
		if u == "" || seen[u] {
			return
		}
		seen[u] = true
		out = append(out, u)
	}
	for _, u := range existing {
		add(u)
	}
	add(primary)
	add(url)
	return out
}

// writeDocFile writes front-matter + body, atomically.
func writeDocFile(abs string, meta DocMeta, body string) error {
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return fmt.Errorf("create document dir: %w", err)
	}
	data := renderDocFile(meta, body)
	tmp, err := os.CreateTemp(filepath.Dir(abs), ".doc-*")
	if err != nil {
		return fmt.Errorf("create temp document: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write document: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("write document: %w", err)
	}
	if err := os.Rename(tmpName, abs); err != nil {
		return fmt.Errorf("rename document: %w", err)
	}
	return nil
}

// renderDocFile builds the on-disk form of a document.
func renderDocFile(meta DocMeta, body string) []byte {
	head, err := yaml.Marshal(meta)
	if err != nil {
		head = []byte("id: " + meta.ID + "\n")
	}
	var buf bytes.Buffer
	buf.WriteString("---\n")
	buf.Write(head)
	buf.WriteString("---\n\n")
	buf.WriteString(body)
	if !strings.HasSuffix(body, "\n") {
		buf.WriteString("\n")
	}
	return buf.Bytes()
}

// parseDocFrontMatter splits a document into its front-matter and its body.
func parseDocFrontMatter(data []byte) (DocMeta, string, error) {
	meta, body, _, err := parseDocFrontMatterOffset(data)
	return meta, body, err
}

// parseDocFrontMatterOffset splits a document into its front-matter and its
// body, and reports how many lines come before the body.
//
// The offset matters: the search reports the line a hit is on, and the
// reader sees the body without the front-matter. Reporting body lines keeps
// the two in step.
func parseDocFrontMatterOffset(data []byte) (DocMeta, string, int, error) {
	text := string(data)
	if !strings.HasPrefix(text, "---\n") {
		return DocMeta{}, text, 0, fmt.Errorf("no front-matter")
	}
	rest := text[4:]
	lines := strings.Split(rest, "\n")
	for i, ln := range lines {
		if strings.TrimRight(ln, " \t\r") != "---" {
			continue
		}
		var meta DocMeta
		yamlPart := strings.Join(lines[:i], "\n")
		if err := yaml.Unmarshal([]byte(yamlPart), &meta); err != nil {
			return meta, strings.Join(lines[i+1:], "\n"), 0, fmt.Errorf("bad front-matter: %w", err)
		}
		// The body starts after the closing fence. The blank line that
		// separates the two is not part of the body.
		bodyStart := i + 1
		offset := i + 2
		if bodyStart < len(lines) && strings.TrimSpace(lines[bodyStart]) == "" {
			bodyStart++
			offset = i + 3
		}
		body := strings.Join(lines[bodyStart:], "\n")
		return meta, body, offset, nil
	}
	return DocMeta{}, text, 0, fmt.Errorf("unterminated front-matter")
}

// DocContent is one document read back from the vault.
type DocContent struct {
	Path string   `json:"path"`
	Rel  string   `json:"rel"`
	Meta DocMeta  `json:"meta"`
	Body string   `json:"body"`
	Tags []string `json:"tags,omitempty"`
}

// ReadDocument reads one document out of the vault. rel is relative to the
// vault root; anything that escapes it is refused.
func (e *Engine) ReadDocument(rel string) (*DocContent, error) {
	abs, err := e.docAbsolutePath(rel)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return nil, fmt.Errorf("read document: %w", err)
	}
	meta, body, ferr := parseDocFrontMatter(data)
	if ferr != nil {
		// A file without front-matter is still readable; it just has no
		// identity beyond its path.
		meta = DocMeta{}
		body = string(data)
	}
	return &DocContent{Path: abs, Rel: filepath.ToSlash(rel), Meta: meta, Body: body}, nil
}

// docAbsolutePath resolves a vault-relative path and refuses to leave the
// vault.
func (e *Engine) docAbsolutePath(rel string) (string, error) {
	rel = strings.TrimSpace(rel)
	if rel == "" {
		return "", fmt.Errorf("path is required")
	}
	if filepath.IsAbs(rel) {
		return "", fmt.Errorf("path must be relative to the documents dir")
	}
	root := e.documentsDir()
	clean := filepath.Clean(filepath.Join(root, filepath.FromSlash(rel)))
	if clean != root && !strings.HasPrefix(clean, root+string(os.PathSeparator)) {
		return "", fmt.Errorf("path escapes the documents dir")
	}
	if st, err := os.Stat(clean); err != nil || st.IsDir() {
		return "", fmt.Errorf("document not found: %s", rel)
	}
	return clean, nil
}

// UpdateDocumentMeta rewrites the front-matter of one document. It is used
// when a document is vectorized into a collection after it was filed.
func (e *Engine) UpdateDocumentMeta(rel string, fn func(*DocMeta)) error {
	abs, err := e.docAbsolutePath(rel)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return err
	}
	meta, body, ferr := parseDocFrontMatter(data)
	if ferr != nil {
		return ferr
	}
	fn(&meta)
	meta.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	if err := writeDocFile(abs, meta, body); err != nil {
		return err
	}
	if e.ledger != nil {
		e.ledger.Append(DocEntry{Path: filepath.ToSlash(rel), DocMeta: meta})
	}
	return nil
}

// ListDocuments returns the documents the ledger knows about, newest first.
// It never walks the disk, which is what keeps the dashboard usable when the
// corpus has millions of files.
func (e *Engine) ListDocuments(channel string, limit int) []DocEntry {
	if e.ledger == nil {
		return []DocEntry{}
	}
	entries := e.ledger.Entries()
	out := make([]DocEntry, 0, len(entries))
	for _, e := range entries {
		if channel != "" && e.Channel != channel {
			continue
		}
		out = append(out, e)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

// DocChannels lists the channels the vault currently holds, with counts.
func (e *Engine) DocChannels() map[string]int {
	out := map[string]int{}
	if e.ledger == nil {
		return out
	}
	for _, en := range e.ledger.Entries() {
		out[en.Channel]++
	}
	return out
}
