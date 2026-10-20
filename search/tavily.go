package search

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// TavilyProvider talks to Tavily's search API (https://docs.tavily.com/).
//
// Its distinguishing feature is chunks_per_source: the API returns passages
// cut from the source pages rather than one blob of HTML, which is already
// close to the granularity the RAG chunker wants.
type TavilyProvider struct {
	BaseURL string
	APIKey  string
	Client  *http.Client
	// SearchDepth is "basic" (1 credit), "advanced" (2 credits) or "fast".
	SearchDepth string
}

// NewTavily builds a Tavily provider.
func NewTavily(baseURL, apiKey string, timeout time.Duration) *TavilyProvider {
	if baseURL == "" {
		baseURL = "https://api.tavily.com"
	}
	return &TavilyProvider{
		BaseURL:     strings.TrimRight(baseURL, "/"),
		APIKey:      strings.TrimSpace(apiKey),
		Client:      httpClient(timeout),
		SearchDepth: "basic",
	}
}

// Name implements Provider.
func (p *TavilyProvider) Name() string { return "tavily" }

// Available implements Provider.
func (p *TavilyProvider) Available() bool { return p != nil && p.APIKey != "" }

type tavilySearchRequest struct {
	Query             string   `json:"query"`
	SearchDepth       string   `json:"search_depth,omitempty"`
	ChunksPerSource   int      `json:"chunks_per_source,omitempty"`
	MaxResults        int      `json:"max_results,omitempty"`
	Topic             string   `json:"topic,omitempty"`
	TimeRange         string   `json:"time_range,omitempty"`
	IncludeRawContent string   `json:"include_raw_content,omitempty"`
	IncludeDomains    []string `json:"include_domains,omitempty"`
	Country           string   `json:"country,omitempty"`
	Language          string   `json:"language,omitempty"`
	IncludeAnswer     bool     `json:"include_answer,omitempty"`
	FilterByPublished bool     `json:"filter_by_published_date,omitempty"`
	IncludePublished  bool     `json:"include_published_date,omitempty"`
}

type tavilySearchResponse struct {
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

// Search implements Provider.
func (p *TavilyProvider) Search(ctx context.Context, q Query) ([]Result, error) {
	if !p.Available() {
		return nil, fmt.Errorf("tavily: no api key configured")
	}
	maxResults := q.Count
	if maxResults <= 0 {
		maxResults = 10
	}
	if maxResults > 20 {
		maxResults = 20
	}
	req := tavilySearchRequest{
		Query:            q.Text,
		SearchDepth:      p.SearchDepth,
		MaxResults:       maxResults,
		Topic:            q.Topic,
		IncludeDomains:   q.Domains,
		Language:         q.Lang,
		IncludePublished: true,
		ChunksPerSource:  3,
	}
	if q.NeedContent {
		req.IncludeRawContent = "markdown"
	}
	if q.FreshnessDays > 0 {
		req.TimeRange = tavilyTimeRange(q.FreshnessDays)
	}
	// A topic of "news" is only valid for Tavily's general/news buckets;
	// an empty topic keeps the API default.
	if req.Topic == "" {
		req.Topic = "general"
	}

	var resp tavilySearchResponse
	if err := postJSON(ctx, p.Client, p.BaseURL+"/search",
		map[string]string{"Authorization": "Bearer " + p.APIKey}, req, &resp); err != nil {
		return nil, errProvider(p.Name(), err)
	}
	if strings.TrimSpace(resp.Answer) != "" {
		// The LLM answer is provider-side synthesis; it is not a search
		// hit and is deliberately not surfaced as a Result.
		_ = resp.Answer
	}

	out := make([]Result, 0, len(resp.Results))
	for _, r := range resp.Results {
		url := strings.TrimSpace(r.URL)
		if url == "" {
			continue
		}
		out = append(out, Result{
			Title:       r.Title,
			URL:         url,
			Snippet:     r.Content,
			Content:     r.RawContent,
			Chunks:      r.Chunks,
			PublishedAt: parseDate(r.PublishedDate),
			Authority:   -1,
			Score:       r.Score,
			Provider:    p.Name(),
		})
	}
	return out, nil
}

// Fetch implements Provider. Tavily's extract API lives on a different path
// and a different product, so this provider does not implement it.
func (p *TavilyProvider) Fetch(ctx context.Context, url string) (Page, error) {
	return Page{}, ErrUnsupported
}

// tavilyTimeRange maps a day count onto Tavily's coarse time_range buckets.
func tavilyTimeRange(days int) string {
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
