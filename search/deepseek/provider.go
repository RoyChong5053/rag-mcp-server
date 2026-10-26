// Package deepseek implements DeepSeek's Search API as a web_search module.
//
// The request body, response shape, URL and parameters are owned here so a
// change on DeepSeek's side never reaches the other providers.
package deepseek

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
	BaseURL string
	APIKey  string
	Timeout time.Duration
	// MaxCount seeds the per-request cap (the API allows 1-50).
	MaxCount int
}

// Provider talks to DeepSeek's Search API
// (https://api-docs.deepseek.com/guides/search).
type Provider struct {
	baseURL  string
	apiKey   string
	client   *http.Client
	defaults Options
	st       *settings.Store
}

// Options are the WebUI-tunable, non-secret parameters for this provider.
type Options struct {
	MaxCount int
}

// New builds a DeepSeek provider. An empty key leaves it unavailable.
func New(cfg Config, st *settings.Store) *Provider {
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if baseURL == "" {
		baseURL = "https://api.deepseek.com"
	}
	max := cfg.MaxCount
	if max <= 0 {
		max = 50
	}
	return &Provider{
		baseURL:  baseURL,
		apiKey:   strings.TrimSpace(cfg.APIKey),
		client:   search.NewHTTPClient(cfg.Timeout),
		defaults: Options{MaxCount: max},
		st:       st,
	}
}

// Name implements search.Provider.
func (p *Provider) Name() string { return "deepseek" }

// Available implements search.Provider.
func (p *Provider) Available() bool { return p != nil && p.apiKey != "" }

// options reads the live settings on every call, so WebUI edits take effect
// without rebuilding the provider or restarting the server.
func (p *Provider) options() Options {
	if p.st == nil {
		return p.defaults
	}
	o := Options{MaxCount: p.st.Get().DeepSeekMaxCount}
	if o.MaxCount <= 0 {
		o.MaxCount = p.defaults.MaxCount
	}
	if o.MaxCount <= 0 {
		o.MaxCount = 50
	}
	if o.MaxCount > 50 {
		o.MaxCount = 50
	}
	return o
}

type searchRequest struct {
	Query        string `json:"query"`
	Count        int    `json:"count"`
	OriQuestion  string `json:"ori_question,omitempty"`
	Localization string `json:"localization,omitempty"`
	NeedContent  bool   `json:"need_content"`
}

type searchResponse struct {
	Results []struct {
		RerankScore float64 `json:"rerank_score"`
		Title       string  `json:"title"`
		URL         string  `json:"url"`
		Snippet     string  `json:"snippet"`
		Summary     string  `json:"summary"`
		Content     string  `json:"content"`
		SiteName    string  `json:"site_name"`
		SiteIcon    string  `json:"site_icon"`
		Date        string  `json:"date"`
		Authority   *int    `json:"authority"`
	} `json:"results"`
}

// Search implements search.Provider.
func (p *Provider) Search(ctx context.Context, q search.Query) ([]search.Result, error) {
	if !p.Available() {
		return nil, fmt.Errorf("deepseek: no api key configured")
	}
	count := q.Count
	if count <= 0 {
		count = 10
	}
	if max := p.options().MaxCount; max > 0 && count > max {
		count = max
	}
	text := q.Text
	if q.FreshnessDays > 0 && !hasTimeOperator(text) {
		// DeepSeek forwards the query with its search syntax intact, so a
		// date bound is applied server-side instead of being filtered
		// locally after the fact.
		since := time.Now().AddDate(0, 0, -q.FreshnessDays).Format("2006-01-02")
		text = fmt.Sprintf("%s after:%s", text, since)
	}

	req := searchRequest{
		Query:       text,
		Count:       count,
		NeedContent: q.NeedContent,
	}
	var resp searchResponse
	if err := search.PostJSON(ctx, p.client, p.baseURL+"/v1/tools/search",
		map[string]string{"Authorization": "Bearer " + p.apiKey}, req, &resp); err != nil {
		return nil, search.ProviderError(p.Name(), err)
	}

	out := make([]search.Result, 0, len(resp.Results))
	for _, r := range resp.Results {
		url := strings.TrimSpace(r.URL)
		if url == "" {
			continue
		}
		auth := -1
		if r.Authority != nil {
			auth = *r.Authority
		}
		out = append(out, search.Result{
			Title:       search.JoinTitle(r.Title, r.SiteName),
			URL:         url,
			Snippet:     r.Snippet,
			Summary:     r.Summary,
			Content:     r.Content,
			PublishedAt: search.ParseDate(r.Date),
			Authority:   auth,
			Score:       r.RerankScore,
			Provider:    p.Name(),
		})
	}
	return out, nil
}

// hasTimeOperator reports whether the query already carries a date bound, in
// which case the provider must not add a second one.
func hasTimeOperator(q string) bool {
	lq := strings.ToLower(q)
	return strings.Contains(lq, "after:") || strings.Contains(lq, "before:")
}
