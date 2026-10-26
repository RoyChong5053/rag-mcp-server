// Package searxng implements a self-hosted SearXNG instance as a web_search
// module.
//
// The query parameters, response shape and URL are owned here so a change on
// SearXNG's side never reaches the other providers.
package searxng

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/search"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// Config is the connection information that comes from config.yaml/env.
type Config struct {
	BaseURL    string
	Timeout    time.Duration
	Language   string
	Categories string
	SafeSearch int
	Engines    []string
}

// Provider queries a self-hosted SearXNG instance
// (https://docs.searxng.org/dev/search_api.html).
type Provider struct {
	baseURL  string
	client   *http.Client
	defaults Options
	st       *settings.Store
}

// Options are the WebUI-tunable, non-secret parameters for this provider.
// SafeSearch is 0 (off), 1 (moderate) or 2 (strict).
type Options struct {
	Language   string
	Categories string
	SafeSearch int
	Engines    []string
}

// New builds a SearXNG provider. The instance must be reachable and must have
// the json format enabled in settings.yml.
func New(cfg Config, st *settings.Store) *Provider {
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	categories := strings.TrimSpace(cfg.Categories)
	if categories == "" {
		categories = "general"
	}
	return &Provider{
		baseURL: baseURL,
		client:  search.NewHTTPClient(cfg.Timeout),
		defaults: Options{
			Language:   strings.TrimSpace(cfg.Language),
			Categories: categories,
			SafeSearch: cfg.SafeSearch,
			Engines:    cfg.Engines,
		},
		st: st,
	}
}

// Name implements search.Provider.
func (p *Provider) Name() string { return "searxng" }

// Available implements search.Provider. A SearXNG instance needs no key, so
// only a configured base URL counts.
func (p *Provider) Available() bool { return p != nil && p.baseURL != "" }

// options reads the live settings on every call, so WebUI edits take effect
// without rebuilding the provider or restarting the server.
func (p *Provider) options() Options {
	if p.st == nil {
		return p.defaults
	}
	s := p.st.Get()
	return Options{
		Language:   strings.TrimSpace(s.SearXNGLanguage),
		Categories: strings.TrimSpace(s.SearXNGCategories),
		SafeSearch: s.SearXNGSafeSearch,
		Engines:    s.SearXNGEngines,
	}
}

type response struct {
	Query    string `json:"query"`
	NumberOf int    `json:"number_of_results"`
	Results  []struct {
		URL           string  `json:"url"`
		Title         string  `json:"title"`
		Content       string  `json:"content"`
		Engine        string  `json:"engine"`
		Score         float64 `json:"score"`
		Category      string  `json:"category"`
		PublishedDate string  `json:"publishedDate"`
	} `json:"results"`
	// Answers is a list of objects ({"answer": ..., "url": ...}), not plain
	// strings. Declaring []string here broke the whole decode as soon as an
	// answerer (calculator, time, unit conversion, ...) contributed one.
	Answers []struct {
		Answer string `json:"answer"`
		URL    string `json:"url"`
	} `json:"answers"`
	// Unresponsive is a list of [engine, error] pairs and is non-empty
	// whenever any upstream engine failed, which is routine. It must not
	// break the decode of the results that did come back.
	Unresponsive [][]string `json:"unresponsive_engines"`
}

// Search implements search.Provider.
func (p *Provider) Search(ctx context.Context, q search.Query) ([]search.Result, error) {
	if !p.Available() {
		return nil, fmt.Errorf("searxng: no base url configured")
	}
	opts := p.options()
	count := q.Count
	if count <= 0 {
		count = 10
	}
	values := url.Values{}
	values.Set("q", q.Text)
	values.Set("format", "json")
	if opts.Language != "" {
		values.Set("language", opts.Language)
	} else if q.Lang != "" {
		values.Set("language", q.Lang)
	}
	if opts.Categories != "" {
		values.Set("categories", opts.Categories)
	}
	if opts.SafeSearch > 0 {
		values.Set("safesearch", strconv.Itoa(clampSafeSearch(opts.SafeSearch)))
	}
	if len(opts.Engines) > 0 {
		values.Set("engines", strings.Join(opts.Engines, ","))
	}
	if q.FreshnessDays > 0 {
		switch {
		case q.FreshnessDays <= 1:
			values.Set("time_range", "day")
		case q.FreshnessDays <= 31:
			values.Set("time_range", "month")
		case q.FreshnessDays <= 365:
			values.Set("time_range", "year")
		}
	}

	var resp response
	endpoint := p.baseURL + "/search?" + values.Encode()
	if err := search.GetJSON(ctx, p.client, endpoint, nil, &resp); err != nil {
		return nil, search.ProviderError(p.Name(), err)
	}
	// Engines fail all the time (captchas, rate limits, markup changes).
	// Surface that for the operator without failing the call: the results
	// below are still good.
	if len(resp.Unresponsive) > 0 {
		log.Printf("searxng: unresponsive engines: %v", resp.Unresponsive)
	}
	out := make([]search.Result, 0, len(resp.Results))
	for i, r := range resp.Results {
		link := strings.TrimSpace(r.URL)
		if link == "" {
			continue
		}
		// SearXNG does not rank the way the other providers do; fall back
		// to a positional score so results stay orderable.
		score := r.Score
		if score == 0 {
			score = 1.0 / float64(i+1)
		}
		out = append(out, search.Result{
			Title:       r.Title,
			URL:         link,
			Snippet:     r.Content,
			PublishedAt: search.ParseDate(r.PublishedDate),
			Authority:   -1,
			Score:       score,
			Provider:    p.Name(),
		})
	}
	if len(out) > count {
		out = out[:count]
	}
	return out, nil
}

func clampSafeSearch(v int) int {
	if v < 0 {
		return 0
	}
	if v > 2 {
		return 2
	}
	return v
}
