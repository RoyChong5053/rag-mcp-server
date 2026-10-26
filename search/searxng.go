package search

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// SearXNGProvider queries a self-hosted SearXNG instance
// (https://docs.searxng.org/dev/search_api.html).
//
// SearXNG is a metasearch proxy: it fans the query out to many upstream
// engines and merges the results. It has no API key and no fetch endpoint,
// which makes it a cheap always-on fallback rather than a primary source.
type SearXNGProvider struct {
	BaseURL string
	Client  *http.Client
	// Language is the SearXNG language code (e.g. "zh-CN").
	Language string
	// Categories restricts which engine groups run (e.g. "general").
	Categories string
}

// NewSearXNG builds a SearXNG provider. The instance must be reachable and
// must have the json format enabled in settings.yml.
func NewSearXNG(baseURL string, timeout time.Duration) *SearXNGProvider {
	return &SearXNGProvider{
		BaseURL:    strings.TrimRight(strings.TrimSpace(baseURL), "/"),
		Client:     httpClient(timeout),
		Language:   "",
		Categories: "general",
	}
}

// Name implements Provider.
func (p *SearXNGProvider) Name() string { return "searxng" }

// Available implements Provider. A SearXNG instance needs no key, so only a
// configured base URL counts.
func (p *SearXNGProvider) Available() bool { return p != nil && p.BaseURL != "" }

type searxngResponse struct {
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

// Search implements Provider.
func (p *SearXNGProvider) Search(ctx context.Context, q Query) ([]Result, error) {
	if !p.Available() {
		return nil, fmt.Errorf("searxng: no base url configured")
	}
	count := q.Count
	if count <= 0 {
		count = 10
	}
	values := url.Values{}
	values.Set("q", q.Text)
	values.Set("format", "json")
	if p.Language != "" {
		values.Set("language", p.Language)
	} else if q.Lang != "" {
		values.Set("language", q.Lang)
	}
	if p.Categories != "" {
		values.Set("categories", p.Categories)
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

	var resp searxngResponse
	endpoint := p.BaseURL + "/search?" + values.Encode()
	if err := getJSON(ctx, p.Client, endpoint, nil, &resp); err != nil {
		return nil, errProvider(p.Name(), err)
	}
	// Engines fail all the time (captchas, rate limits, markup changes).
	// Surface that for the operator without failing the call: the results
	// below are still good.
	if len(resp.Unresponsive) > 0 {
		log.Printf("searxng: unresponsive engines: %v", resp.Unresponsive)
	}
	out := make([]Result, 0, len(resp.Results))
	for i, r := range resp.Results {
		link := strings.TrimSpace(r.URL)
		if link == "" {
			continue
		}
		// SearXNG does not rank the way the other providers do; fall back
		// to a positional score so the chain can still order results.
		score := r.Score
		if score == 0 {
			score = 1.0 / float64(i+1)
		}
		out = append(out, Result{
			Title:       r.Title,
			URL:         link,
			Snippet:     r.Content,
			PublishedAt: parseDate(r.PublishedDate),
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

// Fetch implements Provider. SearXNG only aggregates search results; it does
// not fetch page bodies.
func (p *SearXNGProvider) Fetch(ctx context.Context, url string) (Page, error) {
	return Page{}, ErrUnsupported
}
