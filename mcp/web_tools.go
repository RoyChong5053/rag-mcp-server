package mcp

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
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
type WebService struct {
	eng   *engine.Engine
	chain *search.Chain
	opts  WebOptions
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
	return &WebService{eng: eng, chain: chain, opts: opts}
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
			"The pages it finds are also vectorized into the web collection, so later turns can recall them with search_memory instead of searching again. " +
			"Every hit carries its publication date (or 'unknown'), because a reader has to be able to tell yesterday's news from today's. " +
			"Returns a ranked list with title, url, publication date and snippet; use fetch_url for the full text of one page.",
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
	providerName := s.resolveProvider(strings.TrimSpace(getStringArg(args, "provider", "")))

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

	// File what was found so the next turn can recall it.
	var pages, chunks, skipped int
	if store {
		pages, chunks, skipped = s.ingestResults(ctx, collection, results, ingest)
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
		fmt.Fprintf(&b, "Stored: %d page(s) (%d chunks) vectorized into '%s'%s.\n", pages, chunks, collection, ingestNote(skipped))
		b.WriteString("These pages are now recallable with search_memory. Use web_search again when you need newer information than the dates above.")
	} else {
		b.WriteString("Not stored (store=false): these pages were not vectorized, so they cannot be recalled later.")
	}
	return b.String(), nil
}

// resolveProvider picks the provider for one call. An explicit argument
// always wins and is validated by the caller; otherwise the operator's
// default provider is used when it is reachable, and the failover chain
// covers everything else. The default is a preference, never a requirement:
// a provider without a key must not take the whole search down.
func (s *WebService) resolveProvider(explicit string) string {
	if explicit != "" {
		return explicit
	}
	def := strings.TrimSpace(s.eng.Settings().Get().DefaultSearchProvider)
	if def == "" {
		return ""
	}
	if p := s.chain.Get(def); p != nil && p.Available() {
		return def
	}
	log.Printf("web_search: default provider %q is not available, using the failover chain", def)
	return ""
}

// ingestNote describes pages that could not be filed, if any. The reason is
// not always the fetch: a page that could be fetched but failed to embed is
// also skipped, so the wording stays deliberately vague.
func ingestNote(skipped int) string {
	if skipped <= 0 {
		return ""
	}
	return fmt.Sprintf(", %d could not be stored", skipped)
}

// ingestResults fetches and stores the results. ingest == 0 means every
// result: whatever was searched goes into the collection. It returns how many
// pages were stored, how many chunks they produced, and how many could not be
// fetched. Pages and chunks are counted apart on purpose: a page is a web
// document, a chunk is a slice of it, and "65 pages" would be a lie.
func (s *WebService) ingestResults(ctx context.Context, collection string, results []search.Result, ingest int) (pages, chunks, skipped int) {
	if ingest <= 0 {
		ingest = len(results)
	}
	if ingest > len(results) {
		ingest = len(results)
	}
	for i := 0; i < ingest; i++ {
		r := results[i]
		content := r.Content
		if strings.TrimSpace(content) == "" {
			page, err := s.chain.Fetch(ctx, r.URL)
			if err != nil {
				log.Printf("web_search: fetch %s failed: %v", r.URL, err)
				skipped++
				continue
			}
			content = page.Content
			if strings.TrimSpace(page.Title) != "" && strings.TrimSpace(r.Title) == "" {
				r.Title = page.Title
			}
		}
		if strings.TrimSpace(content) == "" {
			skipped++
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
			log.Printf("web_search: ingest %s failed: %v", r.URL, err)
			skipped++
			continue
		}
		pages++
		chunks += res.ChunksIndexed
	}
	return pages, chunks, skipped
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
		res, err := s.eng.IngestWebDocument(s.opts.Collection, doc, engine.WebIngestOptions{
			MaxChars: s.opts.MaxContentChars,
		})
		if err != nil {
			log.Printf("fetch_url: ingest failed: %v", err)
		} else {
			return fmt.Sprintf("%s\n\n[stored in '%s': %d chunks]\n\n%s", page.Title, s.opts.Collection, res.ChunksIndexed, content), nil
		}
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
