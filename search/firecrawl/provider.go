// Package firecrawl implements Firecrawl's v2 search API as a web_search
// module.
//
// The request body, response shape, URL and parameters are owned here so a
// change on Firecrawl's side never reaches the other providers.
package firecrawl

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
	BaseURL  string
	APIKey   string
	Timeout  time.Duration
	Country  string
	Location string
}

// Provider talks to Firecrawl's v2 search API
// (https://docs.firecrawl.dev/api-reference/endpoint/search).
type Provider struct {
	baseURL  string
	apiKey   string
	client   *http.Client
	defaults Options
	st       *settings.Store
}

// Options are the WebUI-tunable, non-secret parameters for this provider.
type Options struct {
	Country  string
	Location string
}

// New builds a Firecrawl provider.
func New(cfg Config, st *settings.Store) *Provider {
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if baseURL == "" {
		baseURL = "https://api.firecrawl.dev/v2"
	}
	country := strings.ToUpper(strings.TrimSpace(cfg.Country))
	if country == "" {
		country = "MY"
	}
	return &Provider{
		baseURL:  baseURL,
		apiKey:   strings.TrimSpace(cfg.APIKey),
		client:   search.NewHTTPClient(cfg.Timeout),
		defaults: Options{Country: country, Location: strings.TrimSpace(cfg.Location)},
		st:       st,
	}
}

// Name implements search.Provider.
func (p *Provider) Name() string { return "firecrawl" }

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
		Country:  strings.ToUpper(strings.TrimSpace(s.FirecrawlCountry)),
		Location: strings.TrimSpace(s.FirecrawlLocation),
	}
}

type format struct {
	Type string `json:"type"`
}

type scrapeOptions struct {
	Formats         []format `json:"formats"`
	OnlyMainContent bool     `json:"onlyMainContent"`
	MaxAge          int      `json:"maxAge,omitempty"`
}

type searchRequest struct {
	Query         string         `json:"query"`
	Limit         int            `json:"limit,omitempty"`
	Sources       []string       `json:"sources,omitempty"`
	ScrapeOptions *scrapeOptions `json:"scrapeOptions,omitempty"`
	TBS           string         `json:"tbs,omitempty"`
	Country       string         `json:"country,omitempty"`
	Location      string         `json:"location,omitempty"`
	Timeout       int            `json:"timeout,omitempty"`
}

type searchResponse struct {
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

// Search implements search.Provider.
func (p *Provider) Search(ctx context.Context, q search.Query) ([]search.Result, error) {
	if !p.Available() {
		return nil, fmt.Errorf("firecrawl: no api key configured")
	}
	opts := p.options()
	limit := q.Count
	if limit <= 0 {
		limit = 10
	}
	if limit > 50 {
		limit = 50
	}
	req := searchRequest{
		Query:    q.Text,
		Limit:    limit,
		Sources:  []string{"web"},
		Country:  opts.Country,
		Location: opts.Location,
	}
	if q.Topic == "news" {
		req.Sources = []string{"news"}
	}
	if q.NeedContent {
		req.ScrapeOptions = &scrapeOptions{
			Formats:         []format{{Type: "markdown"}},
			OnlyMainContent: true,
		}
	}
	if q.FreshnessDays > 0 {
		req.TBS = tbs(q.FreshnessDays)
	}

	var resp searchResponse
	if err := search.PostJSON(ctx, p.client, p.baseURL+"/search",
		map[string]string{"Authorization": "Bearer " + p.apiKey}, req, &resp); err != nil {
		return nil, search.ProviderError(p.Name(), err)
	}
	if !resp.Success && resp.Error != "" {
		return nil, search.ProviderError(p.Name(), fmt.Errorf("%s", resp.Error))
	}

	out := make([]search.Result, 0, len(resp.Data.Web)+len(resp.Data.News))
	for _, r := range resp.Data.Web {
		url := strings.TrimSpace(r.URL)
		if url == "" {
			continue
		}
		// Firecrawl's web source carries no date at all. The page usually
		// does, near the top, so it is worth looking — but only best
		// effort: a hit that stays undated is honest, a wrong date is not.
		published := time.Time{}
		if t := search.ParseDateFromText(r.Markdown); !t.IsZero() {
			published = t
		} else if t := search.ParseDateFromText(r.Description); !t.IsZero() {
			published = t
		}
		out = append(out, search.Result{
			Title:       search.JoinTitle(r.Title, r.Metadata.Title),
			URL:         url,
			Snippet:     r.Description,
			Content:     r.Markdown,
			PublishedAt: published,
			Authority:   -1,
			Provider:    p.Name(),
		})
	}
	for _, r := range resp.Data.News {
		url := strings.TrimSpace(r.URL)
		if url == "" {
			continue
		}
		out = append(out, search.Result{
			Title:       r.Title,
			URL:         url,
			Snippet:     r.Snippet,
			Content:     r.Markdown,
			PublishedAt: search.ParseDate(r.Date),
			Authority:   -1,
			Provider:    p.Name(),
		})
	}
	return out, nil
}

// tbs maps a day count onto Firecrawl's time-range syntax.
func tbs(days int) string {
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
