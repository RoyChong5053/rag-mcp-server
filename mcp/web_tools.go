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
	IngestTopN int
	// FreshnessDays bounds how old a stored page may be before it counts
	// as stale. 0 = never stale.
	FreshnessDays int
	// CacheScoreThreshold is the rerank score a stored hit needs for the
	// server to answer from the library instead of searching again.
	CacheScoreThreshold float64
	// CacheMinOverlap is the fraction of the query's content terms that
	// must also appear in the stored text for the library hit to count.
	CacheMinOverlap float64
	// DedupThreshold is the near-duplicate gate for ingestion.
	DedupThreshold float64
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
// configured: it calls web_search, the server checks its own library first,
// and only reaches out when the library has nothing fresh enough.
type WebService struct {
	eng   *engine.Engine
	chain *search.Chain
	opts  WebOptions
}

// NewWebService builds the service.
func NewWebService(eng *engine.Engine, chain *search.Chain, opts WebOptions) *WebService {
	if opts.Collection == "" {
		opts.Collection = "web_kb"
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
		Description: "Search the live web and file what you find into the shared knowledge base. " +
			"Before hitting the network the server checks its own library: if the query is already covered by a stored page that is still fresh, it answers from the library and makes no external request. " +
			"Otherwise it queries the configured providers (in order, first one that answers wins), returns the ranked hits, and vectorizes the top results into the web collection so later turns can recall them instead of searching again. " +
			"Pass force_refresh=true to skip the library check and always search. " +
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
				"ingest": map[string]any{
					"type":        "integer",
					"description": "Optional: how many of the top results to fetch and vectorize into the web collection. 0 = do not store anything.",
				},
				"freshness_days": map[string]any{
					"type":        "integer",
					"description": "Optional: only results from the last N days, and stored pages older than N days count as stale. 0 = no limit.",
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
				"force_refresh": map[string]any{
					"type":        "boolean",
					"description": "Optional: skip the library check and always search the live web.",
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
	count := getIntArg(args, "count", s.opts.Count)
	if count <= 0 {
		count = s.opts.Count
	}
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
	forceRefresh := getBoolArg(args, "force_refresh", false)
	providerName := strings.TrimSpace(getStringArg(args, "provider", ""))

	collection := s.opts.Collection

	// Step 1: the library first. A fresh hit means no external call at all.
	if !forceRefresh {
		hits, err := s.eng.WebCacheHits(query, collection, count, s.opts.CacheScoreThreshold, s.opts.CacheMinOverlap, freshness)
		if err != nil {
			log.Printf("web_search: library lookup failed: %v", err)
		} else if len(hits) > 0 {
			return s.formatCacheAnswer(query, collection, hits, freshness), nil
		}
	}

	// Step 2: query the providers.
	q := search.Query{
		Text:          query,
		Count:         count,
		NeedContent:   ingest > 0,
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
		return fmt.Sprintf("No web results for %q (provider: %s). The library has no fresh copy either.", query, orDefault(usedProvider, "none")), nil
	}

	// Step 3: store the top hits so the next turn can recall them.
	stored, skipped := s.ingestResults(ctx, collection, results, ingest)

	var b strings.Builder
	fmt.Fprintf(&b, "Web results for %q (provider: %s, %d hits)\n\n", query, orDefault(usedProvider, "none"), len(results))
	for i, r := range results {
		fmt.Fprintf(&b, "%d. %s\n", i+1, orDefault(r.Title, r.URL))
		fmt.Fprintf(&b, "   %s\n", r.URL)
		if !r.PublishedAt.IsZero() {
			fmt.Fprintf(&b, "   published: %s\n", r.PublishedAt.Format("2006-01-02"))
		}
		if r.Authority >= 0 {
			fmt.Fprintf(&b, "   authority: %d/4\n", r.Authority)
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
	fmt.Fprintf(&b, "Library: %d page(s) vectorized into '%s', %d skipped (duplicate or no content).\n", stored, collection, skipped)
	b.WriteString("Later turns can recall these with search_memory; no need to search again unless the topic moved.")
	return b.String(), nil
}

// ingestResults fetches and stores the top N results. It returns how many
// pages were stored and how many were skipped.
func (s *WebService) ingestResults(ctx context.Context, collection string, results []search.Result, ingest int) (int, int) {
	if ingest <= 0 {
		return 0, 0
	}
	if ingest > len(results) {
		ingest = len(results)
	}
	stored, skipped := 0, 0
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
			DedupThreshold: s.opts.DedupThreshold,
			MaxChars:       s.opts.MaxContentChars,
		})
		if err != nil {
			log.Printf("web_search: ingest %s failed: %v", r.URL, err)
			skipped++
			continue
		}
		if res.Duplicate {
			skipped++
			continue
		}
		stored++
	}
	return stored, skipped
}

// formatCacheAnswer renders the "already in the library" reply.
func (s *WebService) formatCacheAnswer(query, collection string, hits []engine.WebCacheHit, freshness int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Already in the library — no external search was made for %q.\n\n", query)
	for i, h := range hits {
		md := h.Result.Metadata
		title := md["title"]
		if title == "" {
			title = h.Result.Source
		}
		fmt.Fprintf(&b, "%d. %s\n", i+1, title)
		if u := md["url"]; u != "" {
			fmt.Fprintf(&b, "   %s\n", u)
		}
		if md["published_at"] != "" {
			fmt.Fprintf(&b, "   published: %s\n", md["published_at"])
		}
		fmt.Fprintf(&b, "   fetched: %s (similarity %.2f)\n", orDefault(h.Fetched, "unknown"), h.Result.Score)
		if h.Stale {
			fmt.Fprintf(&b, "   [stale: older than %d days — call web_search with force_refresh=true to refresh]\n", freshness)
		}
		if h.Result.Text != "" {
			fmt.Fprintf(&b, "   %s\n", truncateText(h.Result.Text, 400))
		}
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "Collection: '%s'. Use search_memory to pull more of this material in one call.", collection)
	return b.String()
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
			DedupThreshold: s.opts.DedupThreshold,
			MaxChars:       s.opts.MaxContentChars,
		})
		if err != nil {
			log.Printf("fetch_url: ingest failed: %v", err)
		} else if res.Duplicate {
			return fmt.Sprintf("%s\n\n[already stored in '%s' — skipped]\n\n%s", page.Title, s.opts.Collection, content), nil
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
