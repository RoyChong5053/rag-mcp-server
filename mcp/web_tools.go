package mcp

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
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
}

// NewWebService builds the service.
func NewWebService(eng *engine.Engine, registry *search.Registry, opts WebOptions) *WebService {
	if opts.Collection == "" {
		opts.Collection = "web_cache"
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 45 * time.Second
	}
	return &WebService{eng: eng, registry: registry, opts: opts}
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
		jobID, queued = s.startFiling(query, collection, results, ingest, lang)
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
			fmt.Fprintf(&b, "   %s\n", summary)
		}
		b.WriteString("\n")
	}
	if store {
		switch {
		case queued == 0:
			b.WriteString("Nothing to file: the provider returned no text.")
		case jobID != "":
			fmt.Fprintf(&b, "Filing %d result(s) into '%s' as one document in the background (job %s).\n", queued, collection, jobID)
			b.WriteString("They become recallable with search_memory as they land; the hits above are already in context. Do not search the same material again while the filing is running.")
		default:
			fmt.Fprintf(&b, "Filing %d result(s) into '%s' as one document in the background.\n", queued, collection)
			b.WriteString("They become recallable with search_memory as they land; the hits above are already in context.")
		}
	} else {
		b.WriteString("Not stored (store is off): these results were not vectorized, so they cannot be recalled later.")
	}
	return b.String(), nil
}

// startFiling queues one search for background filing and returns the job id
// (empty when no job registry is wired) and how many results went into the
// bundle.
//
// There is no cross-search dedup. Every call becomes its own dated document,
// even when a different provider finds the same keyword: the body carries the
// search timestamp, which is what keeps the documents distinct.
func (s *WebService) startFiling(query, collection string, results []search.Result, ingest int, lang string) (string, int) {
	if ingest <= 0 || ingest > len(results) {
		ingest = len(results)
	}
	pending := results[:ingest]
	if len(pending) == 0 {
		return "", 0
	}
	label := query
	run := func(prog func(jobs.WebProgress)) (any, error) {
		return s.fileResults(collection, query, pending, lang, prog)
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

// buildWebBundle renders everything one search returned as a single markdown
// document: the provider's summaries at the top, then each hit's raw text.
//
// Nothing is transformed or truncated here. Whatever the provider returned is
// written verbatim, because it was paid for and the chunker downstream is what
// makes it retrievable.
func buildWebBundle(query string, results []search.Result, now time.Time) (title, url, body string) {
	provider := ""
	if len(results) > 0 {
		provider = results[0].Provider
	}
	title = "web_search: " + query
	// The synthetic URL only names the vault file; the timestamp keeps every
	// call its own document.
	url = fmt.Sprintf("web_search://%s/%s/%s",
		orDefault(provider, "unknown"), now.Format("20060102T150405"), strings.TrimSpace(query))

	var b strings.Builder
	fmt.Fprintf(&b, "# web_search: %s\n\n", query)
	fmt.Fprintf(&b, "- provider: %s\n", orDefault(provider, "unknown"))
	fmt.Fprintf(&b, "- searched_at: %s\n", now.Format(time.RFC3339))
	fmt.Fprintf(&b, "- results: %d\n\n", len(results))

	// Summaries first: the model-readable gist of every hit, together.
	b.WriteString("## Summary\n\n")
	for i, r := range results {
		summary := strings.TrimSpace(r.Summary)
		if summary == "" {
			summary = strings.TrimSpace(r.Snippet)
		}
		fmt.Fprintf(&b, "%d. %s\n", i+1, orDefault(r.Title, r.URL))
		if r.URL != "" {
			fmt.Fprintf(&b, "   %s\n", r.URL)
		}
		if summary != "" {
			fmt.Fprintf(&b, "   %s\n", summary)
		}
		b.WriteString("\n")
	}

	// Then the raw text, per hit, in rank order.
	b.WriteString("## Results\n\n")
	for i, r := range results {
		fmt.Fprintf(&b, "### %d. %s\n\n", i+1, orDefault(r.Title, r.URL))
		if r.URL != "" {
			fmt.Fprintf(&b, "- url: %s\n", r.URL)
		}
		if r.PublishedAt.IsZero() {
			b.WriteString("- published: unknown\n")
		} else {
			fmt.Fprintf(&b, "- published: %s\n", r.PublishedAt.UTC().Format(time.RFC3339))
		}
		if r.Authority >= 0 {
			fmt.Fprintf(&b, "- authority: %d/4\n", r.Authority)
		}
		if r.Score != 0 {
			fmt.Fprintf(&b, "- score: %.4f\n", r.Score)
		}
		if r.Provider != "" {
			fmt.Fprintf(&b, "- provider: %s\n", r.Provider)
		}
		b.WriteString("\n")
		content := strings.TrimSpace(r.Content)
		if content == "" {
			content = strings.TrimSpace(r.Summary)
		}
		if content == "" {
			content = strings.TrimSpace(r.Snippet)
		}
		if content == "" {
			b.WriteString("_(the provider returned no text for this hit)_\n\n")
			continue
		}
		b.WriteString(content)
		b.WriteString("\n\n")
	}
	return title, url, b.String()
}

// fileResults stores one search as one document. The whole result set is
// embedded in a single call, so the vault grows by one file per search instead
// of one file per hit.
func (s *WebService) fileResults(collection, query string, results []search.Result, lang string, prog func(jobs.WebProgress)) (any, error) {
	provider := ""
	if len(results) > 0 {
		provider = results[0].Provider
	}
	title, url, body := buildWebBundle(query, results, time.Now())
	doc := engine.WebDocument{
		URL:          url,
		CanonicalURL: url,
		Title:        title,
		Text:         body,
		Provider:     provider,
		Authority:    -1,
		Lang:         lang,
		Tags:         s.opts.Tags,
	}
	res, err := s.eng.IngestWebDocument(collection, doc, engine.WebIngestOptions{})
	if err != nil {
		log.Printf("web filing: ingest bundle for %q failed: %v", query, err)
		prog(jobs.WebProgress{PagesDone: 1, PagesTotal: 1})
		return map[string]any{"pages": 0, "chunks": 0, "unchanged": 0, "skipped": 1}, nil
	}
	unchanged := 0
	if res.Unchanged {
		unchanged = 1
	}
	prog(jobs.WebProgress{PagesDone: 1, PagesTotal: 1, Chunks: res.ChunksIndexed})
	log.Printf("web filing done: 1 document (%d result(s)), %d chunk(s), unchanged=%d", len(results), res.ChunksIndexed, unchanged)
	return map[string]any{"pages": 1, "chunks": res.ChunksIndexed, "unchanged": unchanged, "skipped": 0}, nil
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

// orDefault returns value when it is not empty, otherwise fallback.
func orDefault(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}
