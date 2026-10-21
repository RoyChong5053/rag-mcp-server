package engine

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
)

// The document search is the other half of the system. The vector side
// answers "what did I learn about X"; this side answers "where is that exact
// string". Embeddings fuzz over URLs, error codes, version numbers and names;
// a literal search does not.
//
// Two engines produce the same results:
//
//   - ripgrep, when the host has it: multi-threaded, mmap, SIMD, PCRE2.
//   - a built-in scanner, when it does not: WalkDir + worker pool + RE2.
//
// ripgrep is never bundled into the binary. It is a normal program that
// normal systems have (it is installed on the server this was written for),
// and a 5MB binary in the repository would cost an architecture matrix and
// CVE tracking for no gain. The built-in scanner is what makes the feature
// independent of the host: it is always there.

// DocSearchOptions is one document search.
type DocSearchOptions struct {
	// Query is what to look for.
	Query string
	// Mode is "literal" (default) or "regex".
	Mode string
	// Path narrows inside the matched folders.
	Path string
	// Channel narrows to one source channel.
	Channel string
	// Category narrows to one category folder.
	Category string
	// Since / Until bound the ingest date (yyyy-mm-dd). Empty = no bound.
	Since string
	Until string
	// Word requires the match to sit on a word boundary.
	Word bool
	// CaseSensitive forces case sensitivity; SmartCase (the default) makes
	// a lower-case query case-insensitive and an upper-case one sensitive.
	CaseSensitive bool
	SmartCase     bool
	// Context is how many lines of context each hit carries.
	Context int
	// Limit caps the total number of returned matches.
	Limit int
	// PerFile caps the matches reported per document.
	PerFile int
}

// DocMatch is one hit inside one document.
type DocMatch struct {
	Path    string   `json:"path"`
	Line    int      `json:"line"`
	Text    string   `json:"text"`
	Context []string `json:"context,omitempty"`
}

// DocSearchHit is one document that matched, with its identity.
type DocSearchHit struct {
	Path        string     `json:"path"`
	DocsPath    string     `json:"docs_path,omitempty"`
	Title       string     `json:"title,omitempty"`
	URL         string     `json:"url,omitempty"`
	Channel     string     `json:"channel,omitempty"`
	Category    string     `json:"category,omitempty"`
	PublishedAt string     `json:"published_at,omitempty"`
	FetchedAt   string     `json:"fetched_at,omitempty"`
	Matches     []DocMatch `json:"matches"`
}

// DocSearchResult is the answer to one document search.
type DocSearchResult struct {
	Engine       string         `json:"engine"`
	ScannedFiles int            `json:"scanned_files"`
	ElapsedMS    int64          `json:"elapsed_ms"`
	TotalMatches int            `json:"total_matches"`
	Truncated    bool           `json:"truncated"`
	Documents    []DocSearchHit `json:"documents"`
}

// SearchDocuments searches the raw documents of the vault.
func (e *Engine) SearchDocuments(opts DocSearchOptions) (*DocSearchResult, error) {
	query := strings.TrimSpace(opts.Query)
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}
	if opts.Mode == "" {
		opts.Mode = "literal"
	}
	if opts.Mode != "literal" && opts.Mode != "regex" {
		return nil, fmt.Errorf("mode must be literal or regex")
	}
	if opts.Limit <= 0 {
		opts.Limit = e.config.DocMaxMatches
	}
	if opts.Limit <= 0 {
		opts.Limit = 200
	}
	if opts.PerFile <= 0 {
		opts.PerFile = e.config.DocPerFile
	}
	if opts.PerFile <= 0 {
		opts.PerFile = 5
	}
	if !opts.CaseSensitive && !opts.SmartCase {
		// Explicitly case-insensitive.
		opts.SmartCase = false
	} else if !opts.CaseSensitive {
		opts.SmartCase = true
	}

	roots, err := e.docScanRoots(opts)
	if err != nil {
		return nil, err
	}
	start := time.Now()
	res := &DocSearchResult{Documents: []DocSearchHit{}}

	engine, bin, err := e.docSearchEngine()
	if err != nil {
		return nil, err
	}
	res.Engine = engine
	if len(roots) == 0 {
		res.ElapsedMS = time.Since(start).Milliseconds()
		return res, nil
	}

	if engine == "rg" {
		if err := e.searchWithRipgrep(bin, opts, roots, res); err != nil {
			return nil, err
		}
	} else {
		if err := e.searchWithBuiltin(opts, roots, res); err != nil {
			return nil, err
		}
	}
	res.ElapsedMS = time.Since(start).Milliseconds()
	return res, nil
}

// docSearchEngine picks the engine to use. "auto" prefers ripgrep and falls
// back to the built-in scanner; "rg" insists on ripgrep; "builtin" always
// uses the built-in scanner.
func (e *Engine) docSearchEngine() (string, string, error) {
	backend := strings.ToLower(strings.TrimSpace(e.config.DocSearchBackend))
	if backend == "" {
		backend = "auto"
	}
	switch backend {
	case "builtin":
		return "builtin", "", nil
	case "rg":
		bin, err := e.ripgrepPath()
		if err != nil {
			return "", "", err
		}
		return "rg", bin, nil
	case "auto":
		if bin, err := e.ripgrepPath(); err == nil {
			return "rg", bin, nil
		}
		return "builtin", "", nil
	default:
		return "", "", fmt.Errorf("unknown documents.search_backend %q (auto, rg, builtin)", backend)
	}
}

// ripgrepPath locates the ripgrep binary: the configured path first, then
// PATH.
func (e *Engine) ripgrepPath() (string, error) {
	if p := strings.TrimSpace(e.config.DocRGPath); p != "" {
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("configured rg_path %q is not usable: %w", p, err)
		}
		return p, nil
	}
	bin, err := exec.LookPath("rg")
	if err != nil {
		return "", fmt.Errorf("ripgrep not found: %w", err)
	}
	return bin, nil
}

// docDatePattern matches the day folders of the vault.
var docDatePattern = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)

// docScanRoots turns the filters into the list of directories to search.
//
// The date is the first level of the layout, so the scan always starts from
// the day folders: a date range is a list of directories instead of a
// full-tree walk, and "search the last week" touches seven of them no
// matter how large the corpus is.
//
// The path filter is a sub-path of the vault root. When it points at a day
// folder itself, that folder is the base; otherwise the day folders inside
// it are.
func (e *Engine) docScanRoots(opts DocSearchOptions) ([]string, error) {
	root := e.documentsDir()
	if st, err := os.Stat(root); err != nil || !st.IsDir() {
		return nil, nil
	}
	scanRoot := root
	if p := strings.TrimSpace(opts.Path); p != "" {
		clean := filepath.Clean(filepath.Join(root, filepath.FromSlash(p)))
		if clean != root && !strings.HasPrefix(clean, root+string(os.PathSeparator)) {
			return nil, nil
		}
		scanRoot = clean
	}

	var bases []string
	if day := filepath.Base(scanRoot); docDatePattern.MatchString(day) {
		// The path points at a day folder.
		if (opts.Since == "" || day >= opts.Since) && (opts.Until == "" || day <= opts.Until) {
			bases = []string{scanRoot}
		}
	} else {
		entries, err := os.ReadDir(scanRoot)
		if err != nil {
			return nil, nil
		}
		for _, ent := range entries {
			if !ent.IsDir() || !docDatePattern.MatchString(ent.Name()) {
				continue
			}
			if opts.Since != "" && ent.Name() < opts.Since {
				continue
			}
			if opts.Until != "" && ent.Name() > opts.Until {
				continue
			}
			bases = append(bases, filepath.Join(scanRoot, ent.Name()))
		}
	}

	var out []string
	for _, base := range bases {
		dir := base
		if c := sanitizeDocSegment(opts.Channel); c != "" {
			dir = filepath.Join(dir, c)
		}
		if c := sanitizeDocSegment(opts.Category); c != "" {
			dir = filepath.Join(dir, c)
		}
		// Never let the scope escape the vault.
		clean := filepath.Clean(dir)
		if clean != root && !strings.HasPrefix(clean, root+string(os.PathSeparator)) {
			continue
		}
		if st, err := os.Stat(clean); err != nil || !st.IsDir() {
			continue
		}
		out = append(out, clean)
	}
	return out, nil
}

// rgPathField is ripgrep's path/line payload, which is a plain string in old
// builds and an object in new ones.
type rgPathField struct {
	Text  string
	Bytes string
}

// UnmarshalJSON accepts both shapes.
func (p *rgPathField) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		p.Text = s
		return nil
	}
	var obj struct {
		Text  string `json:"text"`
		Bytes string `json:"bytes"`
	}
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	p.Text, p.Bytes = obj.Text, obj.Bytes
	return nil
}

// value returns the field as text, decoding the base64 form if needed.
func (p rgPathField) value() string {
	if p.Text != "" {
		return p.Text
	}
	if p.Bytes != "" {
		if b, err := base64.StdEncoding.DecodeString(p.Bytes); err == nil {
			return string(b)
		}
	}
	return ""
}

// rgEvent is one line of ripgrep's --json output.
type rgEvent struct {
	Type string `json:"type"`
	Data struct {
		Path       rgPathField `json:"path"`
		Lines      rgPathField `json:"lines"`
		LineNumber int         `json:"line_number"`
		Submatches []struct {
			Start int `json:"start"`
			End   int `json:"end"`
		} `json:"submatches"`
	} `json:"data"`
}

// searchWithRipgrep runs ripgrep over the roots and fills the result.
func (e *Engine) searchWithRipgrep(bin string, opts DocSearchOptions, roots []string, res *DocSearchResult) error {
	args := []string{"--json", "--no-messages", "--no-ignore-vcs", "--hidden"}
	// The ledger is an index, not a document: it must never show up as a hit.
	args = append(args, "--glob", "!.ledger/**", "--glob", "!**/.ledger/**", "--glob", "!*.jsonl")
	if opts.PerFile > 0 {
		args = append(args, "-m", strconv.Itoa(opts.PerFile))
	}
	if opts.Context > 0 {
		args = append(args, "-C", strconv.Itoa(opts.Context))
	}
	if opts.Mode == "literal" {
		args = append(args, "-F")
	}
	switch {
	case opts.CaseSensitive:
		args = append(args, "--case-sensitive")
	case opts.SmartCase:
		args = append(args, "--smart-case")
	default:
		args = append(args, "-i")
	}
	args = append(args, "--max-columns", "400")
	args = append(args, "--", opts.Query)
	args = append(args, roots...)

	ctx, cancel := context.WithTimeout(context.Background(), e.docSearchTimeout())
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()

	// Exit code 1 means "no matches", which is not an error.
	if runErr != nil {
		if ctx.Err() != nil {
			return fmt.Errorf("document search timed out after %s", e.docSearchTimeout())
		}
		var ee *exec.ExitError
		if !errors.As(runErr, &ee) || ee.ExitCode() > 1 {
			msg := strings.TrimSpace(stderr.String())
			if msg != "" {
				return fmt.Errorf("ripgrep failed: %s", msg)
			}
			return fmt.Errorf("ripgrep failed: %w", runErr)
		}
	}

	root := e.documentsDir()
	sc := bufio.NewScanner(bytes.NewReader(stdout.Bytes()))
	sc.Buffer(make([]byte, 1024*1024), 8*1024*1024)
	byPath := map[string]*DocSearchHit{}
	var order []string
	for sc.Scan() {
		var ev rgEvent
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			continue
		}
		switch ev.Type {
		case "begin":
			res.ScannedFiles++
		case "match", "context":
			raw := ev.Data.Path.value()
			if raw == "" {
				continue
			}
			abs := raw
			if !filepath.IsAbs(abs) {
				abs = filepath.Join(root, raw)
			}
			rel, err := filepath.Rel(root, abs)
			if err != nil {
				continue
			}
			rel = filepath.ToSlash(rel)
			line := strings.TrimRight(ev.Data.Lines.value(), "\r\n")
			if ev.Type == "match" && opts.Word && !wordBoundaryOK(line, ev.Data.Submatches) {
				continue
			}
			hit, ok := byPath[rel]
			if !ok {
				hit = &DocSearchHit{Path: rel}
				byPath[rel] = hit
				order = append(order, rel)
			}
			if ev.Type == "match" {
				if len(hit.Matches) >= opts.PerFile {
					continue
				}
				hit.Matches = append(hit.Matches, DocMatch{Path: rel, Line: ev.Data.LineNumber, Text: line})
				res.TotalMatches++
			} else {
				if n := len(hit.Matches); n > 0 {
					hit.Matches[n-1].Context = append(hit.Matches[n-1].Context, line)
				}
			}
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("read ripgrep output: %w", err)
	}
	e.finishDocSearch(res, byPath, order, opts)
	return nil
}

// searchWithBuiltin scans the roots with the built-in scanner.
func (e *Engine) searchWithBuiltin(opts DocSearchOptions, roots []string, res *DocSearchResult) error {
	re, err := compileDocPattern(opts)
	if err != nil {
		return err
	}
	type candidate struct {
		path string
		rel  string
	}
	files := make(chan candidate, 1024)
	var (
		mu      sync.Mutex
		byPath  = map[string]*DocSearchHit{}
		order   []string
		scanned int
	)
	workers := runtime.NumCPU()
	if workers < 1 {
		workers = 1
	}
	if workers > 16 {
		workers = 16
	}
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range files {
				lines, err := readDocLines(c.path)
				if err != nil {
					continue
				}
				var matches []DocMatch
				for idx, line := range lines {
					loc := firstDocMatch(re, line, opts.Word)
					if loc == nil {
						continue
					}
					m := DocMatch{Path: c.rel, Line: idx + 1, Text: line}
					if opts.Context > 0 {
						lo := idx - opts.Context
						if lo < 0 {
							lo = 0
						}
						hi := idx + opts.Context + 1
						if hi > len(lines) {
							hi = len(lines)
						}
						for k := lo; k < hi; k++ {
							if k == idx {
								continue
							}
							m.Context = append(m.Context, lines[k])
						}
					}
					matches = append(matches, m)
					if len(matches) >= opts.PerFile {
						break
					}
				}
				if len(matches) == 0 {
					continue
				}
				mu.Lock()
				scanned++
				hit := &DocSearchHit{Path: c.rel, Matches: matches}
				byPath[c.rel] = hit
				order = append(order, c.rel)
				res.TotalMatches += len(matches)
				mu.Unlock()
			}
		}()
	}

	root := e.documentsDir()
	ctx, cancel := context.WithTimeout(context.Background(), e.docSearchTimeout())
	defer cancel()
	for _, r := range roots {
		_ = filepath.WalkDir(r, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if ctx.Err() != nil {
				return filepath.SkipAll
			}
			if d.IsDir() {
				name := d.Name()
				if name == ledgerDirName || (strings.HasPrefix(name, ".") && path != r) {
					return filepath.SkipDir
				}
				return nil
			}
			name := d.Name()
			if strings.HasSuffix(name, ".jsonl") {
				return nil
			}
			rel, rerr := filepath.Rel(root, path)
			if rerr != nil {
				return nil
			}
			select {
			case files <- candidate{path: path, rel: filepath.ToSlash(rel)}:
			case <-ctx.Done():
				return filepath.SkipAll
			}
			return nil
		})
	}
	close(files)
	wg.Wait()
	if ctx.Err() != nil {
		return fmt.Errorf("document search timed out after %s", e.docSearchTimeout())
	}
	res.ScannedFiles = scanned
	e.finishDocSearch(res, byPath, order, opts)
	return nil
}

// finishDocSearch fills the document identity in, shifts the line numbers
// onto the body and applies the global cap.
func (e *Engine) finishDocSearch(res *DocSearchResult, byPath map[string]*DocSearchHit, order []string, opts DocSearchOptions) {
	root := e.documentsDir()
	for _, rel := range order {
		hit := byPath[rel]
		if hit == nil {
			continue
		}
		if len(res.Documents) >= opts.Limit {
			res.Truncated = true
			break
		}
		if data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel))); err == nil {
			if meta, _, offset, ferr := parseDocFrontMatterOffset(data); ferr == nil {
				hit.Title = meta.Title
				hit.URL = meta.URL
				hit.Channel = meta.Channel
				hit.Category = meta.Category
				hit.PublishedAt = meta.PublishedAt
				hit.FetchedAt = meta.FetchedAt
				// The reader sees the body, so the line numbers do too.
				for i := range hit.Matches {
					if hit.Matches[i].Line > offset {
						hit.Matches[i].Line -= offset
					} else {
						hit.Matches[i].Line = 1
					}
				}
			}
		}
		hit.DocsPath = e.DocsPath(rel)
		res.Documents = append(res.Documents, *hit)
	}
}

// compileDocPattern builds the regexp of the built-in scanner.
func compileDocPattern(opts DocSearchOptions) (*regexp.Regexp, error) {
	pattern := opts.Query
	if opts.Mode != "regex" {
		pattern = regexp.QuoteMeta(pattern)
	}
	switch {
	case opts.CaseSensitive:
		// as is
	case opts.SmartCase:
		if hasUpper(opts.Query) {
			// An upper-case letter in the query means the writer was
			// deliberate: match exactly.
		} else {
			pattern = "(?i)" + pattern
		}
	default:
		pattern = "(?i)" + pattern
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("bad pattern: %w", err)
	}
	return re, nil
}

// hasUpper reports whether the query contains an upper-case letter.
func hasUpper(s string) bool {
	for _, r := range s {
		if unicode.IsUpper(r) {
			return true
		}
	}
	return false
}

// firstDocMatch returns the first match of a line that satisfies the word
// rule. A line that only contains the word inside another word ("catalog"
// for "cat") does not count when whole words were asked for.
func firstDocMatch(re *regexp.Regexp, line string, word bool) []int {
	locs := re.FindAllStringIndex(line, -1)
	if len(locs) == 0 {
		return nil
	}
	if !word {
		return locs[0]
	}
	for _, loc := range locs {
		if wordBoundaryAt(line, loc[0], loc[1]) {
			return loc
		}
	}
	return nil
}

// wordBoundaryAt reports whether the match at [start,end) sits on a word
// boundary. Unlike \b this works for CJK text, where every character counts
// as a word character.
func wordBoundaryAt(line string, start, end int) bool {
	if start > 0 {
		r := []rune(line[:start])
		if len(r) > 0 && isWordRune(r[len(r)-1]) {
			return false
		}
	}
	if end < len(line) {
		r := []rune(line[end:])
		if len(r) > 0 && isWordRune(r[0]) {
			return false
		}
	}
	return true
}

// wordBoundaryOK applies the word rule to ripgrep's submatch offsets. Any
// submatch that sits on a boundary is enough: the line contains the word.
func wordBoundaryOK(line string, subs []struct {
	Start int `json:"start"`
	End   int `json:"end"`
}) bool {
	if len(subs) == 0 {
		return true
	}
	for _, s := range subs {
		if wordBoundaryAt(line, s.Start, s.End) {
			return true
		}
	}
	return false
}

// isWordRune reports whether a rune counts as part of a word.
func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_'
}

// readDocLines reads a document into its lines.
func readDocLines(path string) ([]string, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if st.Size() > maxDocBytes {
		return nil, fmt.Errorf("file too large")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	text := strings.ReplaceAll(string(data), "\r\n", "\n")
	return strings.Split(text, "\n"), nil
}

// docSearchTimeout is the budget of one document search.
func (e *Engine) docSearchTimeout() time.Duration {
	if e.config.DocSearchTimeout > 0 {
		return e.config.DocSearchTimeout
	}
	return 60 * time.Second
}
