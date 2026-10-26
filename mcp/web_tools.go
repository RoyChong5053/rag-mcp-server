package mcp

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/jobs"
	"github.com/RoyChong5053/rag-mcp-server/search"
)

// WebOptions configures the web-search tool. Everything behavioural (count,
// freshness, store, ingest, provider options, ...) lives in the WebUI
// settings and is read live on every call; only values that cannot change at
// runtime are here.
type WebOptions struct {
	// Collection is where returned pages are vectorized.
	Collection string
	// Timeout bounds one provider request.
	Timeout time.Duration
	// Tags are attached to every ingested chunk.
	Tags []string
}

// WebService wires the external search provider modules to the RAG engine.
//
// There is no failover chain. web_search resolves exactly the provider
// selected in the dashboard, calls it, and reports its answer or its error.
// A silent switch would make the output look like a provider that never ran.
//
// Filing is not part of the answer: the tool returns the provider's hits as
// soon as they arrive, and the text is chunked and embedded in the
// background.
type WebService struct {
	eng      *engine.Engine
	registry *search.Registry
	opts     WebOptions
	jobs     *jobs.Manager

	// inflight holds the canonical URLs that are being filed right now, so a
	// repeated search does not queue the same page twice.
	mu       sync.Mutex
	inflight map[string]bool
}

// NewWebService builds the service.
func NewWebService(eng *engine.Engine, registry *search.Registry, opts WebOptions) *WebService {
	if opts.Collection == "" {
		opts.Collection = "web_cache"
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 45 * time.Second
	}
	return &WebService{eng: eng, registry: registry, opts: opts, inflight: map[string]bool{}}
}

// WithJobs wires the background job registry, so the pages a search found
// show up in the dashboard's Jobs panel while they are being filed.
func (s *WebService) WithJobs(mgr *jobs.Manager) *WebService {
	s.jobs = mgr
	return s
}

// Providers lists the provider modules this server was built with.
func (s *WebService) Providers() []string {
	if s == nil || s.registry == nil {
		return nil
	}
	return s.registry.Names()
}

// Register adds web_search to an MCP server.
func (s *WebService) Register(server *Server) {
	server.RegisterTool(Tool{
		Name: "web_search",
		Description: "Search the live web and return what the selected provider found. " +
			"This always queries the live web: it never answers from stored material, so the results are current. " +
			"The provider is fixed by the operator in the dashboard; web_search does not choose one and never falls back to another provider. " +
			"By default the returned pages are also vectorized into the web collection in the BACKGROUND, so later turns can recall them with search_memory instead of searching again. " +
			"Every hit carries its publication date (or 'unknown'), because a reader has to be able to tell yesterday's news from today's. " +
			"The call returns as soon as the provider answered: filing continues afterwards, so a later recall may still be incomplete for a while.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "The search query. Supports site:/after:/lang: style operators, which are forwarded to the provider.",
				},
			},
			"required":             []string{"query"},
			"additionalProperties": false,
		},
	}, s.handleWebSearch)
}

// handleWebSearch is the web_search tool handler.
func (s *WebService) handleWebSearch(args map[string]any) (any, error) {
	if err := rejectUnknownArgs(args, "query"); err != nil {
		return nil, err
	}
	query := strings.TrimSpace(getStringArg(args, "query", ""))
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}

	// Every default is read from the live settings, so the dashboard is
	// authoritative and a change takes effect on the next call.
	st := s.eng.Settings().Get()
	count := st.DefaultSearchCount
	if count <= 0 {
		count = 10
	}
	freshness := st.DefaultSearchFreshnessDays
	if freshness < 0 {
		freshness = 0
	}
	topic := strings.TrimSpace(st.DefaultSearchTopic)
	lang := strings.TrimSpace(st.DefaultSearchLang)
	store := st.DefaultSearchStore
	ingest := st.DefaultSearchIngest
	if ingest < 0 {
		ingest = 0
	}
	maxChars := st.DefaultSearchMaxChars
	if maxChars < 0 {
		maxChars = 0
	}

	// Strict selection: the selected provider is the called provider. An
	// empty selection is an operator error, not an invitation to guess.
	providerName := strings.TrimSpace(st.DefaultSearchProvider)
	available := s.Providers()
	if providerName == "" {
		return nil, fmt.Errorf("no web search provider is selected: choose one in the dashboard (available: %s)", orDefault(strings.Join(available, ", "), "none"))
	}
	p := s.registry.Get(providerName)
	if p == nil {
		return nil, fmt.Errorf("selected web search provider %q is not built on this server (available: %s)", providerName, orDefault(strings.Join(available, ", "), "none"))
	}
	if !p.Available() {
		return nil, fmt.Errorf("selected web search provider %q is unavailable: set its base_url/API key in config.yaml, or choose another provider in the dashboard", providerName)
	}

	q := search.Query{
		Text:          query,
		Count:         count,
		NeedContent:   store,
		FreshnessDays: freshness,
		Lang:          lang,
		Topic:         topic,
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.opts.Timeout)
	defer cancel()

	results, err := p.Search(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("web search failed with provider %q: %w", p.Name(), err)
	}

	collection := s.opts.Collection
	if len(results) == 0 {
		return fmt.Sprintf("No live web results for %q (provider: %s).", query, p.Name()), nil
	}

	// File what was found so the next turn can recall it. The filing happens
	// in the background: the caller gets the hits now, and the library fills
	// up while the conversation continues.
	var jobID string
	var queued int
	if store {
		jobID, queued = s.startFiling(query, collection, results, ingest, maxChars, lang)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Live web results for %q (provider: %s, %d hits, searched %s)\n\n",
		query, p.Name(), len(results), time.Now().UTC().Format("2006-01-02 15:04 UTC"))
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
		b.WriteString("\n")
	}
	if store {
		switch {
		case queued == 0:
			b.WriteString("Nothing to file: every result is already being filed by another search, or the provider returned no text.")
		case jobID != "":
			fmt.Fprintf(&b, "Filing %d result(s) into '%s' in the background (job %s).\n", queued, collection, jobID)
			b.WriteString("They become recallable with search_memory as they land; the hits above are already in context. Do not search the same material again while the filing is running.")
		default:
			fmt.Fprintf(&b, "Filing %d result(s) into '%s' in the background.\n", queued, collection)
			b.WriteString("They become recallable with search_memory as they land; the hits above are already in context.")
		}
	} else {
		b.WriteString("Not stored (store is off): these results were not vectorized, so they cannot be recalled later.")
	}
	return b.String(), nil
}

// startFiling queues the results of a search for background filing and
// returns the job id (empty when no job registry is wired) and how many
// results were queued.
//
// Pages that are already being filed are dropped: the same article found by
// two searches does not need to be embedded twice.
func (s *WebService) startFiling(query, collection string, results []search.Result, ingest, maxChars int, lang string) (string, int) {
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
		return s.fileResults(collection, pending, maxChars, lang, prog)
	}
	return s.submitFiling(label, collection, run), len(pending)
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

// fileResults chunks and embeds what the provider returned, in rank order:
// the most relevant material lands first, so an early recall already finds
// the good stuff.
//
// Nothing is fetched here. A provider that only returned a snippet files the
// snippet; a provider that returned full text files that; a provider that
// returned nothing is skipped. web_search has no second network call, so the
// selected provider remains the only backend that was used.
func (s *WebService) fileResults(collection string, results []search.Result, maxChars int, lang string, prog func(jobs.WebProgress)) (any, error) {
	pages, chunks, skipped, unchanged := 0, 0, 0, 0
	for i, r := range results {
		content := strings.TrimSpace(r.Content)
		if content == "" {
			content = strings.TrimSpace(r.Summary)
		}
		if content == "" {
			content = strings.TrimSpace(r.Snippet)
		}
		if content == "" {
			log.Printf("web filing: no text for %s (provider %s returned neither content, summary nor snippet)", r.URL, r.Provider)
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
			Lang:         lang,
			Tags:         s.opts.Tags,
		}
		res, err := s.eng.IngestWebDocument(collection, doc, engine.WebIngestOptions{
			MaxChars: maxChars,
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

// rejectUnknownArgs enforces the tool's schema. Older clients used to send
// count/store/freshness/provider and friends; silently ignoring them would
// make the caller believe a knob worked when it did not.
func rejectUnknownArgs(args map[string]any, allowed ...string) error {
	if len(args) == 0 {
		return nil
	}
	ok := make(map[string]bool, len(allowed))
	for _, k := range allowed {
		ok[k] = true
	}
	unknown := make([]string, 0, len(args))
	for k := range args {
		if !ok[k] {
			unknown = append(unknown, k)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	return fmt.Errorf("unknown argument(s): %s. web_search accepts only %s; configure the rest in the dashboard",
		strings.Join(unknown, ", "), strings.Join(allowed, ", "))
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
