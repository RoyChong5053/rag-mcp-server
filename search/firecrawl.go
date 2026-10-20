package search

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// FirecrawlProvider talks to Firecrawl's v2 search API
// (https://docs.firecrawl.dev/api-reference/endpoint/search).
//
// Its edge over the other providers is scraping robustness: the returned
// markdown comes from Firecrawl's proxy layer, so pages that block a plain
// HTTP fetch still resolve. It also has dedicated news results and fine
// time-range filters.
type FirecrawlProvider struct {
	BaseURL string
	APIKey  string
	Client  *http.Client
	// Country is the ISO 3166-1 alpha-2 code used for geo-targeting.
	Country string
	// Location is a free-form city/region string ("Kuala Lumpur").
	Location string
}

// NewFirecrawl builds a Firecrawl provider.
func NewFirecrawl(baseURL, apiKey string, timeout time.Duration) *FirecrawlProvider {
	if baseURL == "" {
		baseURL = "https://api.firecrawl.dev/v2"
	}
	return &FirecrawlProvider{
		BaseURL:  strings.TrimRight(baseURL, "/"),
		APIKey:   strings.TrimSpace(apiKey),
		Client:   httpClient(timeout),
		Country:  "MY",
		Location: "",
	}
}

// Name implements Provider.
func (p *FirecrawlProvider) Name() string { return "firecrawl" }

// Available implements Provider.
func (p *FirecrawlProvider) Available() bool { return p != nil && p.APIKey != "" }

type firecrawlFormat struct {
	Type string `json:"type"`
}

type firecrawlScrapeOptions struct {
	Formats         []firecrawlFormat `json:"formats"`
	OnlyMainContent bool              `json:"onlyMainContent"`
	MaxAge          int               `json:"maxAge,omitempty"`
}

type firecrawlSearchRequest struct {
	Query         string                  `json:"query"`
	Limit         int                     `json:"limit,omitempty"`
	Sources       []string                `json:"sources,omitempty"`
	ScrapeOptions *firecrawlScrapeOptions `json:"scrapeOptions,omitempty"`
	TBS           string                  `json:"tbs,omitempty"`
	Country       string                  `json:"country,omitempty"`
	Location      string                  `json:"location,omitempty"`
	IncludeDomain []string                `json:"includeDomains,omitempty"`
	Timeout       int                     `json:"timeout,omitempty"`
}

type firecrawlSearchResponse struct {
	Success bool `json:"success"`
	Data    struct {
		Web []struct {
			URL         string `json:"url"`
			Title       string `json:"title"`
			Description string `json:"description"`
			Markdown    string `json:"markdown"`
			Metadata    struct {
				Title     string `json:"title"`
				SourceURL string `json:"sourceURL"`
				URL       string `json:"url"`
				Status    int    `json:"statusCode"`
			} `json:"metadata"`
		} `json:"web"`
		News []struct {
			URL      string `json:"url"`
			Title    string `json:"title"`
			Snippet  string `json:"snippet"`
			Markdown string `json:"markdown"`
			Date     string `json:"date"`
			Metadata struct {
				Title     string `json:"title"`
				SourceURL string `json:"sourceURL"`
			} `json:"metadata"`
		} `json:"news"`
	} `json:"data"`
	CreditsUsed int    `json:"creditsUsed"`
	Error       string `json:"error"`
}

type firecrawlScrapeRequest struct {
	URL             string            `json:"url"`
	Formats         []firecrawlFormat `json:"formats"`
	OnlyMainContent bool              `json:"onlyMainContent"`
	Timeout         int               `json:"timeout,omitempty"`
}

type firecrawlScrapeResponse struct {
	Success bool `json:"success"`
	Data    struct {
		Markdown string `json:"markdown"`
		Metadata struct {
			Title     string `json:"title"`
			SourceURL string `json:"sourceURL"`
			URL       string `json:"url"`
			Status    int    `json:"statusCode"`
		} `json:"metadata"`
	} `json:"data"`
	Error string `json:"error"`
}

// Search implements Provider.
func (p *FirecrawlProvider) Search(ctx context.Context, q Query) ([]Result, error) {
	if !p.Available() {
		return nil, fmt.Errorf("firecrawl: no api key configured")
	}
	limit := q.Count
	if limit <= 0 {
		limit = 10
	}
	if limit > 50 {
		limit = 50
	}
	req := firecrawlSearchRequest{
		Query:         q.Text,
		Limit:         limit,
		Sources:       []string{"web"},
		Country:       p.Country,
		Location:      p.Location,
		IncludeDomain: q.Domains,
	}
	if q.Topic == "news" {
		req.Sources = []string{"news"}
	}
	if q.NeedContent {
		req.ScrapeOptions = &firecrawlScrapeOptions{
			Formats:         []firecrawlFormat{{Type: "markdown"}},
			OnlyMainContent: true,
		}
	}
	if q.FreshnessDays > 0 {
		req.TBS = firecrawlTBS(q.FreshnessDays)
	}

	var resp firecrawlSearchResponse
	if err := postJSON(ctx, p.Client, p.BaseURL+"/search",
		map[string]string{"Authorization": "Bearer " + p.APIKey}, req, &resp); err != nil {
		return nil, errProvider(p.Name(), err)
	}
	if !resp.Success && resp.Error != "" {
		return nil, errProvider(p.Name(), fmt.Errorf("%s", resp.Error))
	}

	out := make([]Result, 0, len(resp.Data.Web)+len(resp.Data.News))
	for _, r := range resp.Data.Web {
		url := strings.TrimSpace(r.URL)
		if url == "" {
			continue
		}
		out = append(out, Result{
			Title:       joinTitle(r.Title, r.Metadata.Title),
			URL:         url,
			Snippet:     r.Description,
			Content:     r.Markdown,
			PublishedAt: time.Time{},
			Authority:   -1,
			Provider:    p.Name(),
		})
	}
	for _, r := range resp.Data.News {
		url := strings.TrimSpace(r.URL)
		if url == "" {
			continue
		}
		out = append(out, Result{
			Title:       r.Title,
			URL:         url,
			Snippet:     r.Snippet,
			Content:     r.Markdown,
			PublishedAt: parseDate(r.Date),
			Authority:   -1,
			Provider:    p.Name(),
		})
	}
	return out, nil
}

// Fetch implements Provider.
func (p *FirecrawlProvider) Fetch(ctx context.Context, url string) (Page, error) {
	if !p.Available() {
		return Page{}, fmt.Errorf("firecrawl: no api key configured")
	}
	url = strings.TrimSpace(url)
	if url == "" {
		return Page{}, fmt.Errorf("firecrawl: url is required")
	}
	var resp firecrawlScrapeResponse
	if err := postJSON(ctx, p.Client, p.BaseURL+"/scrape",
		map[string]string{"Authorization": "Bearer " + p.APIKey},
		firecrawlScrapeRequest{
			URL:             url,
			Formats:         []firecrawlFormat{{Type: "markdown"}},
			OnlyMainContent: true,
		}, &resp); err != nil {
		return Page{}, errProvider(p.Name(), err)
	}
	if !resp.Success {
		if resp.Error != "" {
			return Page{}, errProvider(p.Name(), fmt.Errorf("%s", resp.Error))
		}
		return Page{}, errProvider(p.Name(), fmt.Errorf("scrape failed"))
	}
	final := resp.Data.Metadata.URL
	if final == "" {
		final = resp.Data.Metadata.SourceURL
	}
	if final == "" {
		final = url
	}
	return Page{Title: resp.Data.Metadata.Title, URL: final, Content: resp.Data.Markdown}, nil
}

// firecrawlTBS maps a day count onto Firecrawl's time-range syntax.
func firecrawlTBS(days int) string {
	switch {
	case days <= 1:
		return "qdr:d"
	case days <= 7:
		return "qdr:w"
	case days <= 31:
		return "qdr:m"
	case days <= 365:
		return "qdr:y"
	default:
		return ""
	}
}
