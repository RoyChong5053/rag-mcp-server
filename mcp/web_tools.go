package mcp

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/jobs"
	"github.com/RoyChong5053/rag-mcp-server/search"
)

// WebOptions configures the web-search tools.
type WebOptions struct {
	// Collection is where fetched pages are vectorized.
	Collection string
	// Count is the default number of results to request.
	Count int
	// IngestTopN is how many ranked results get fetched and stored.
	// 0 = every result.
	IngestTopN int
	// FreshnessDays is the default age limit on search results. 0 = no limit.
	FreshnessDays int
	// StaleAfterDays is the age after which a recalled web page is flagged
	// as stale. 0 = never flag.
	StaleAfterDays int
	// MaxContentChars caps how much of a page body is stored.
	MaxContentChars int
	// Timeout bounds one provider request.
	Timeout time.Duration
	// Topic is the default search topic.
	Topic string
	// Lang is the default language bias.
	Lang string
	// Tags are attached to every ingested chunk.
	Tags []string
}

// WebService wires the external search providers to the RAG engine.
//
// The design rule is that the agent never has to know which provider is
// configured: it calls web_search, and the server picks a provider (the
// one the operator made the default, or the failover chain).
//
// The filing of what was found is NOT part of the answer. A search that
// returns ten pages used to spend an hour vectorizing them before the tool
// replied, which meant the caller's timeout expired long before the result
// arrived and the fresh material was lost even though the server had it.
// Now the search answers with the provider's hits, and the pages are filed in
// the background, one worker at a time, while the user keeps talking.
type WebService struct {
	eng   *engine.Engine
	chain *search.Chain
	opts  WebOptions
	jobs  *jobs.Manager

	// inflight holds the canonical URLs that are being filed right now, so a
	// repeated search does not queue the same page twice.
	mu       sync.Mutex
	inflight map[string]bool
}

// NewWebService builds the service.
func NewWebService(eng *engine.Engine, chain *search.Chain, opts WebOptions) *WebService {
	if opts.Collection == "" {
		opts.Collection = "web_cache"
	}
	if opts.Count <= 0 {
		opts.Count = 10
	}
	if opts.IngestTopN < 0 {
		opts.IngestTopN = 0
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 45 * time.Second
	}
	return &WebService{eng: eng, chain: chain, opts: opts, inflight: map[string]bool{}}
}

// WithJobs wires the background job registry, so the pages a search found
// show up in the dashboard's Jobs panel while they are being filed.
func (s *WebService) WithJobs(mgr *jobs.Manager) *WebService {
	s.jobs = mgr
	return s
}

// Providers lists the configured provider chain in order.
func (s *WebService) Providers() []string {
	if s == nil || s.chain == nil {
		return nil
	}
	return s.chain.Names()
}

// Register adds the web tools to an MCP server.
func (s *WebService) Register(server *Server) {
	server.RegisterTool(Tool{
		Name: "web_search",
		Description: "Search the live web and return what the search provider found. " +
			"This always queries the live web: it never answers from stored material, so the results are current. " +
			"The pages it finds are also vectorized into the web collection in the BACKGROUND, so later turns can recall them with search_memory instead of searching again. " +
			"Every hit carries its publication date (or 'unknown'), because a reader has to be able to tell yesterday's news from today's. " +
			"Returns a ranked list with title, url, publication date and snippet; use fetch_url for the full text of one page. " +
			"The call returns as soon as the provider answered: the filing continues afterwards, so a later recall may still be incomplete for a while.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "The search query. Supports site:/after:/lang: style operators, which are forwarded to the provider.",
				},
				"count": map[string]any{
					"type":        "integer",
					"description": "Optional: how many results to return. Defaults to the server setting.",
				},
				"store": map[string]any{
					"type":        "boolean",
					"description": "Optional: also vectorize the pages into the web collection so later turns can recall them. Default true. Set false to search without filing anything.",
				},
				"ingest": map[string]any{
					"type":        "integer",
					"description": "Optional: how many of the results to fetch and vectorize into the web collection. 0 = store all of them. Ignored when store=false.",
				},
				"freshness_days": map[string]any{
					"type":        "integer",
					"description": "Optional: only return results published in the last N days. 0 = no limit.",
				},
				"provider": map[string]any{
					"type":        "string",
					"description": "Optional: force one provider instead of the failover chain.",
				},
				"topic": map[string]any{
					"type":        "string",
					"description": "Optional: 'general' or 'news'.",
				},
				"lang": map[string]any{
					"type":        "string",
					"description": "Optional: ISO 639-1 language code to bias results by language.",
				},
				"domains": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Optional: restrict results to these hostnames.",
				},
			},
			"required": []string{"query"},
		},
	}, s.handleWebSearch)

	server.RegisterTool(Tool{
		Name: "fetch_url",
		Description: "Fetch one web page and return it as markdown. " +
			"With save=true the page is also vectorized into the web collection so it can be recalled later. " +
			"Use this when you already know the URL and need its actual content.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"url": map[string]any{
					"type":        "string",
					"description": "The page URL (http or https).",
				},
				"save": map[string]any{
					"type":        "boolean",
					"description": "Optional: also vectorize the page into the web collection. Default true.",
				},
				"max_chars": map[string]any{
					"type":        "integer",
					"description": "Optional: cap the returned text length.",
				},
			},
			"required": []string{"url"},
		},
	}, s.handleFetchURL)
}

// handleWebSearch is the web_search tool handler.
func (s *WebService) handleWebSearch(args map[string]any) (any, error) {
	query := strings.TrimSpace(getStringArg(args, "query", ""))
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}
	st := s.eng.Settings().Get()
	countDefault := s.opts.Count
	if st.DefaultSearchCount > 0 {
		countDefault = st.DefaultSearchCount
	}
	count := getIntArg(args, "count", countDefault)
	if count <= 0 {
		count = countDefault
	}
	// store=false searches without filing anything. ingest is how many of
	// the results get filed; 0 = all of them.
	store := getBoolArg(args, "store", true)
	ingest := getIntArg(args, "ingest", s.opts.IngestTopN)
	if ingest < 0 {
		ingest = 0
	}
	freshness := getIntArg(args, "freshness_days", s.opts.FreshnessDays)
	if freshness < 0 {
		freshness = 0
	}
	topic := strings.TrimSpace(getStringArg(args, "topic", ""))
	if topic == "" {
		topic = s.opts.Topic
	}
	lang := strings.TrimSpace(getStringArg(args, "lang", ""))
	if lang == "" {
		lang = s.opts.Lang
	}
	domains := getStringSliceArg(args, "domains")
	providerName, err := s.resolveProvider(strings.TrimSpace(getStringArg(args, "provider", "")))
	if err != nil {
		return nil, err
	}

	collection := s.opts.Collection

	// web_search always goes to the live web and reports what the provider
	// returned. It never answers from the library: a tool named web_search
	// that silently serves stored material lies to its caller, and yesterday's
	// news is not today's news. Recalling what was searched before is
	// search_memory's job, and the caller decides when to ask for fresh truth.
	q := search.Query{
		Text:          query,
		Count:         count,
		NeedContent:   store,
		FreshnessDays: freshness,
		Domains:       domains,
		Lang:          lang,
		Topic:         topic,
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.opts.Timeout)
	defer cancel()

	var results []search.Result
	var usedProvider string
	if providerName != "" {
		p := s.chain.Get(providerName)
		if p == nil {
			return nil, fmt.Errorf("unknown provider '%s' (configured: %s)", providerName, strings.Join(s.Providers(), ", "))
		}
		if !p.Available() {
			return nil, fmt.Errorf("provider '%s' is not configured", providerName)
		}
		res, err := p.Search(ctx, q)
		if err != nil {
			return nil, fmt.Errorf("provider '%s' failed: %w", providerName, err)
		}
		results = res
		usedProvider = p.Name()
	} else {
		res, err := s.chain.Search(ctx, q)
		if err != nil {
			return nil, err
		}
		results = res
		if len(results) > 0 {
			usedProvider = results[0].Provider
		}
	}

	if len(results) == 0 {
		return fmt.Sprintf("No live web results for %q (provider: %s).", query, orDefault(usedProvider, "none")), nil
	}

	// File what was found so the next turn can recall it. The filing happens
	// in the background: the caller gets the hits now, and the library fills
	// up while the conversation continues. A search that vectorizes before it
	// answers used to outlast every caller timeout, and the fresh material
	// was lost even though the server had it.
	var jobID string
	var queued int
	if store {
		jobID, queued = s.startFiling(query, collection, results, ingest)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Live web results for %q (provider: %s, %d hits, fetched %s)\n\n",
		query, orDefault(usedProvider, "none"), len(results), time.Now().UTC().Format("2006-01-02 15:04 UTC"))
	for i, r := range results {
		fmt.Fprintf(&b, "%d. %s\n", i+1, orDefault(r.Title, r.URL))
		fmt.Fprintf(&b, "   url: %s\n", r.URL)
		// The date matters: a reader has to be able to tell yesterday's
		// news from today's, so it is always shown, never omitted.
		if r.PublishedAt.IsZero() {
			fmt.Fprintf(&b, "   published: unknown\n")
		} else {
			fmt.Fprintf(&b, "   published: %s (%s)\n",
				r.PublishedAt.Format("2006-01-02"),
				engine.WebDocumentAge(r.PublishedAt.UTC().Format(time.RFC3339), time.Now()))
		}
		// Authority is a DeepSeek tier; the other providers do not rate
		// their sources. The line is always there so the shape of a hit
		// does not change from one provider to the next.
		if r.Authority >= 0 {
			fmt.Fprintf(&b, "   authority: %d/4\n", r.Authority)
		} else {
			fmt.Fprintf(&b, "   authority: n/a\n")
		}
		summary := r.Summary
		if summary == "" {
			summary = r.Snippet
		}
		if summary != "" {
			fmt.Fprintf(&b, "   %s\n", truncateText(summary, 400))
		}
		if r.Content != "" {
			fmt.Fprintf(&b, "   [full text available — use fetch_url]\n")
		}
		b.WriteString("\n")
	}
	if store {
		switch {
		case queued == 0:
			b.WriteString("Nothing to file: every result is already being filed by another search.")
		case jobID != "":
			fmt.Fprintf(&b, "Filing %d page(s) into '%s' in the background (job %s).\n", queued, collection, jobID)
			b.WriteString("They become recallable with search_memory as they land; the hits above are already in context. Do not search the same material again while the filing is running.")
		default:
			fmt.Fprintf(&b, "Filing %d page(s) into '%s' in the background.\n", queued, collection)
			b.WriteString("They become recallable with search_memory as they land; the hits above are already in context.")
		}
	} else {
		b.WriteString("Not stored (store=false): these pages were not vectorized, so they cannot be recalled later.")
	}
	return b.String(), nil
}

// resolveProvider picks the provider for one call. An explicit argument
// always wins and is validated by the caller. Otherwise the operator's
// default provider is used, and only that one: when it is configured but
// not reachable the call fails with an explanation instead of silently
// falling through to another provider. A silent switch would make the
// output look like a provider it never used, and the operator would have
// no way to tell which backend actually answered.
//
// An empty default means "failover chain". That is an explicit choice in
// the dashboard, not a fallback for a broken configuration.
func (s *WebService) resolveProvider(explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	def := strings.TrimSpace(s.eng.Settings().Get().DefaultSearchProvider)
	if def == "" {
		return "", nil
	}
	p := s.chain.Get(def)
	if p == nil {
		return "", fmt.Errorf("default web search provider %q is not in the configured chain (configured: %s)", def, strings.Join(s.Providers(), ", "))
	}
	if !p.Available() {
		return "", fmt.Errorf("default web search provider %q is not configured: set its base_url/API key in config.yaml, or pick another provider in the dashboard", def)
	}
	return def, nil
}

// startFiling queues the results of a search for background filing and
// returns the job id (empty when no job registry is wired) and how many
// pages were queued.
//
// Pages that are already being filed are dropped: the same article found by
// two searches does not need to be embedded twice.
func (s *WebService) startFiling(query, collection string, results []search.Result, ingest int) (string, int) {
	if ingest <= 0 {
		ingest = len(results)
	}
	if ingest > len(results) {
		ingest = len(results)
	}
	pending := make([]search.Result, 0, ingest)
	keys := make([]string, 0, ingest)
	s.mu.Lock()
	for i := 0; i < ingest; i++ {
		key := filingKey(results[i].URL)
		if key == "" || s.inflight[key] {
			continue
		}
		s.inflight[key] = true
		pending = append(pending, results[i])
		keys = append(keys, key)
	}
	s.mu.Unlock()
	if len(pending) == 0 {
		return "", 0
	}

	label := query
	run := func(prog func(jobs.WebProgress)) (any, error) {
		defer s.releaseKeys(keys)
		return s.fileResults(collection, pending, prog)
	}
	return s.submitFiling(label, collection, run), len(pending)
}

// startFilingDoc queues one already-fetched page for background filing.
func (s *WebService) startFilingDoc(label, collection string, doc engine.WebDocument) (string, int) {
	key := filingKey(doc.URL)
	s.mu.Lock()
	if key != "" && s.inflight[key] {
		s.mu.Unlock()
		return "", 0
	}
	if key != "" {
		s.inflight[key] = true
	}
	s.mu.Unlock()

	run := func(prog func(jobs.WebProgress)) (any, error) {
		defer s.releaseKeys([]string{key})
		return s.fileDoc(collection, doc, prog)
	}
	return s.submitFiling(label, collection, run), 1
}

// submitFiling hands the work to the job registry (so the dashboard shows
// it) or, without a registry, to a plain goroutine. Either way the caller
// does not wait.
func (s *WebService) submitFiling(label, collection string, run func(prog func(jobs.WebProgress)) (any, error)) string {
	if s.jobs != nil {
		job := s.jobs.SubmitWeb(label, collection, run)
		return job.ID
	}
	go func() {
		if _, err := run(func(jobs.WebProgress) {}); err != nil {
			log.Printf("web filing failed: %v", err)
		}
	}()
	return ""
}

// releaseKeys frees the in-flight marks of a finished filing.
func (s *WebService) releaseKeys(keys []string) {
	s.mu.Lock()
	for _, k := range keys {
		if k != "" {
			delete(s.inflight, k)
		}
	}
	s.mu.Unlock()
}

// filingKey is the identity of a page for the in-flight check.
func filingKey(rawURL string) string {
	if k := search.CanonicalizeURL(rawURL); k != "" {
		return k
	}
	return strings.TrimSpace(rawURL)
}

// fileResults fetches (when the provider gave only a snippet), chunks and
// embeds the pages, in rank order: the most relevant material lands first,
// so an early recall already finds the good stuff.
func (s *WebService) fileResults(collection string, results []search.Result, prog func(jobs.WebProgress)) (any, error) {
	pages, chunks, skipped, unchanged := 0, 0, 0, 0
	for i, r := range results {
		content := r.Content
		if strings.TrimSpace(content) == "" {
			// The provider returned a snippet only: fetch the page itself.
			// The fetch runs on its own context: the caller's request is
			// long gone and its deadline must not kill the filing.
			ctx, cancel := context.WithTimeout(context.Background(), s.opts.Timeout)
			page, err := s.chain.Fetch(ctx, r.URL)
			cancel()
			if err != nil {
				log.Printf("web filing: fetch %s failed: %v", r.URL, err)
				skipped++
				prog(jobs.WebProgress{PagesDone: i + 1, PagesTotal: len(results), Chunks: chunks})
				continue
			}
			content = page.Content
			if strings.TrimSpace(page.Title) != "" && strings.TrimSpace(r.Title) == "" {
				r.Title = page.Title
			}
		}
		if strings.TrimSpace(content) == "" {
			skipped++
			prog(jobs.WebProgress{PagesDone: i + 1, PagesTotal: len(results), Chunks: chunks})
			continue
		}
		doc := engine.WebDocument{
			URL:          r.URL,
			CanonicalURL: search.CanonicalizeURL(r.URL),
			Title:        r.Title,
			SiteName:     siteNameOf(r.URL),
			Text:         content,
			Provider:     r.Provider,
			PublishedAt:  r.PublishedAt,
			Authority:    r.Authority,
			Score:        r.Score,
			Lang:         s.opts.Lang,
			Tags:         s.opts.Tags,
		}
		res, err := s.eng.IngestWebDocument(collection, doc, engine.WebIngestOptions{
			MaxChars: s.opts.MaxContentChars,
		})
		if err != nil {
			log.Printf("web filing: ingest %s failed: %v", r.URL, err)
			skipped++
			prog(jobs.WebProgress{PagesDone: i + 1, PagesTotal: len(results), Chunks: chunks})
			continue
		}
		pages++
		if res.Unchanged {
			unchanged++
		}
		chunks += res.ChunksIndexed
		prog(jobs.WebProgress{PagesDone: i + 1, PagesTotal: len(results), Chunks: chunks})
	}
	log.Printf("web filing done: %d page(s), %d chunk(s), %d unchanged, %d skipped", pages, chunks, unchanged, skipped)
	return map[string]any{"pages": pages, "chunks": chunks, "unchanged": unchanged, "skipped": skipped}, nil
}

// fileDoc files one already-fetched page.
func (s *WebService) fileDoc(collection string, doc engine.WebDocument, prog func(jobs.WebProgress)) (any, error) {
	res, err := s.eng.IngestWebDocument(collection, doc, engine.WebIngestOptions{
		MaxChars: s.opts.MaxContentChars,
	})
	if err != nil {
		return nil, err
	}
	prog(jobs.WebProgress{PagesDone: 1, PagesTotal: 1, Chunks: res.ChunksIndexed})
	return map[string]any{"pages": 1, "chunks": res.ChunksIndexed, "unchanged": res.Unchanged}, nil
}

// handleFetchURL is the fetch_url tool handler.
func (s *WebService) handleFetchURL(args map[string]any) (any, error) {
	url := strings.TrimSpace(getStringArg(args, "url", ""))
	if url == "" {
		return nil, fmt.Errorf("url is required")
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return nil, fmt.Errorf("url must start with http:// or https://")
	}
	save := getBoolArg(args, "save", true)
	maxChars := getIntArg(args, "max_chars", 0)

	ctx, cancel := context.WithTimeout(context.Background(), s.opts.Timeout)
	defer cancel()

	page, err := s.chain.Fetch(ctx, url)
	if err != nil {
		return nil, err
	}
	content := page.Content
	if maxChars > 0 && len(content) > maxChars {
		content = content[:maxChars]
	}
	if save {
		doc := engine.WebDocument{
			URL:          orDefault(page.URL, url),
			CanonicalURL: search.CanonicalizeURL(orDefault(page.URL, url)),
			Title:        page.Title,
			SiteName:     siteNameOf(orDefault(page.URL, url)),
			Text:         page.Content,
			Provider:     "fetch",
			Tags:         s.opts.Tags,
		}
		// The vectorization happens in the background: the caller asked for
		// the page, not for a wait. The text is returned right away.
		jobID, _ := s.startFilingDoc(page.Title, s.opts.Collection, doc)
		if jobID != "" {
			return fmt.Sprintf("%s\n\n[filing in '%s' in the background (job %s)]\n\n%s", page.Title, s.opts.Collection, jobID, content), nil
		}
		return fmt.Sprintf("%s\n\n[filing in '%s' in the background]\n\n%s", page.Title, s.opts.Collection, content), nil
	}
	return fmt.Sprintf("%s\n\n%s", page.Title, content), nil
}

// siteNameOf returns the hostname of a URL for display purposes.
func siteNameOf(raw string) string {
	s := raw
	if i := strings.Index(s, "://"); i >= 0 {
		s = s[i+3:]
	}
	if i := strings.IndexAny(s, "/?#"); i >= 0 {
		s = s[:i]
	}
	return s
}

// truncateText cuts s to at most n runes.
func truncateText(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

// orDefault returns value when it is not empty, otherwise fallback.
func orDefault(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}
