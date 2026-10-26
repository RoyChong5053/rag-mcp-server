// Package tavily implements Tavily's search API as a web_search module.
//
// The request body, response shape, URL and parameters are owned here so a
// change on Tavily's side never reaches the other providers.
package tavily

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/search"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// Config is the connection information that comes from config.yaml/env.
type Config struct {
	BaseURL     string
	APIKey      string
	Timeout     time.Duration
	SearchDepth string
	Country     string
}

// Provider talks to Tavily's search API (https://docs.tavily.com/).
type Provider struct {
	baseURL  string
	apiKey   string
	client   *http.Client
	defaults Options
	st       *settings.Store
}

// Options are the WebUI-tunable, non-secret parameters for this provider.
// SearchDepth is "basic" (1 credit), "advanced" (2 credits) or "fast".
type Options struct {
	SearchDepth string
	Country     string
}

// New builds a Tavily provider.
func New(cfg Config, st *settings.Store) *Provider {
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if baseURL == "" {
		baseURL = "https://api.tavily.com"
	}
	depth := normalizeDepth(cfg.SearchDepth)
	return &Provider{
		baseURL:  baseURL,
		apiKey:   strings.TrimSpace(cfg.APIKey),
		client:   search.NewHTTPClient(cfg.Timeout),
		defaults: Options{SearchDepth: depth, Country: strings.ToUpper(strings.TrimSpace(cfg.Country))},
		st:       st,
	}
}

// Name implements search.Provider.
func (p *Provider) Name() string { return "tavily" }

// Available implements search.Provider.
func (p *Provider) Available() bool { return p != nil && p.apiKey != "" }

// options reads the live settings on every call, so WebUI edits take effect
// without rebuilding the provider or restarting the server.
func (p *Provider) options() Options {
	if p.st == nil {
		return p.defaults
	}
	s := p.st.Get()
	return Options{
		SearchDepth: normalizeDepth(s.TavilySearchDepth),
		Country:     strings.ToUpper(strings.TrimSpace(s.TavilyCountry)),
	}
}

func normalizeDepth(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "advanced":
		return "advanced"
	case "fast":
		return "fast"
	default:
		return "basic"
	}
}

type searchRequest struct {
	Query             string `json:"query"`
	SearchDepth       string `json:"search_depth,omitempty"`
	ChunksPerSource   int    `json:"chunks_per_source,omitempty"`
	MaxResults        int    `json:"max_results,omitempty"`
	Topic             string `json:"topic,omitempty"`
	TimeRange         string `json:"time_range,omitempty"`
	IncludeRawContent string `json:"include_raw_content,omitempty"`
	Country           string `json:"country,omitempty"`
	Language          string `json:"language,omitempty"`
	IncludeAnswer     bool   `json:"include_answer,omitempty"`
	FilterByPublished bool   `json:"filter_by_published_date,omitempty"`
	IncludePublished  bool   `json:"include_published_date,omitempty"`
}

type searchResponse struct {
	Query   string `json:"query"`
	Answer  string `json:"answer"`
	Results []struct {
		Title         string   `json:"title"`
		URL           string   `json:"url"`
		Content       string   `json:"content"`
		RawContent    string   `json:"raw_content"`
		Score         float64  `json:"score"`
		PublishedDate string   `json:"published_date"`
		Chunks        []string `json:"chunks"`
	} `json:"results"`
	ResponseTime float64 `json:"response_time"`
}

// Search implements search.Provider.
func (p *Provider) Search(ctx context.Context, q search.Query) ([]search.Result, error) {
	if !p.Available() {
		return nil, fmt.Errorf("tavily: no api key configured")
	}
	opts := p.options()
	maxResults := q.Count
	if maxResults <= 0 {
		maxResults = 10
	}
	if maxResults > 20 {
		maxResults = 20
	}
	topic := strings.TrimSpace(q.Topic)
	if topic == "" {
		topic = "general"
	}
	req := searchRequest{
		Query:            q.Text,
		SearchDepth:      opts.SearchDepth,
		MaxResults:       maxResults,
		Topic:            topic,
		Language:         q.Lang,
		IncludePublished: true,
		ChunksPerSource:  3,
	}
	// Tavily only accepts country for the general topic; sending it with
	// news is rejected by the API.
	if topic == "general" {
		req.Country = opts.Country
	}
	if q.NeedContent {
		req.IncludeRawContent = "markdown"
	}
	if q.FreshnessDays > 0 {
		req.TimeRange = timeRange(q.FreshnessDays)
	}

	var resp searchResponse
	if err := search.PostJSON(ctx, p.client, p.baseURL+"/search",
		map[string]string{"Authorization": "Bearer " + p.apiKey}, req, &resp); err != nil {
		return nil, search.ProviderError(p.Name(), err)
	}
	if strings.TrimSpace(resp.Answer) != "" {
		// The LLM answer is provider-side synthesis; it is not a search
		// hit and is deliberately not surfaced as a Result.
		_ = resp.Answer
	}

	out := make([]search.Result, 0, len(resp.Results))
	for _, r := range resp.Results {
		url := strings.TrimSpace(r.URL)
		if url == "" {
			continue
		}
		published := search.ParseDate(r.PublishedDate)
		if published.IsZero() && r.RawContent != "" {
			// Tavily did not date this hit. The raw page often does.
			published = search.ParseDateFromText(r.RawContent)
		}
		out = append(out, search.Result{
			Title:       r.Title,
			URL:         url,
			Snippet:     r.Content,
			Content:     r.RawContent,
			Chunks:      r.Chunks,
			PublishedAt: published,
			Authority:   -1,
			Score:       r.Score,
			Provider:    p.Name(),
		})
	}
	return out, nil
}

// timeRange maps a day count onto Tavily's coarse time_range buckets.
func timeRange(days int) string {
	switch {
	case days <= 1:
		return "day"
	case days <= 7:
		return "week"
	case days <= 31:
		return "month"
	default:
		return "year"
	}
}
