package search

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// DeepSeekProvider talks to DeepSeek's Search & Fetch API
// (https://api-docs.deepseek.com/guides/search).
//
// It is the primary provider because a single request can return both the
// ranked hits and the full page text (need_content), and because the response
// carries publication date and a source-authority tier, which the ingest path
// needs for freshness and trust decisions.
type DeepSeekProvider struct {
	BaseURL string
	APIKey  string
	Client  *http.Client
	// MaxCount caps the requested result count (the API allows 1-50).
	MaxCount int
}

// NewDeepSeek builds a DeepSeek provider. An empty key makes it unavailable,
// which lets the chain skip it without a config error.
func NewDeepSeek(baseURL, apiKey string, timeout time.Duration) *DeepSeekProvider {
	if baseURL == "" {
		baseURL = "https://api.deepseek.com"
	}
	return &DeepSeekProvider{
		BaseURL:  strings.TrimRight(baseURL, "/"),
		APIKey:   strings.TrimSpace(apiKey),
		Client:   httpClient(timeout),
		MaxCount: 50,
	}
}

// Name implements Provider.
func (p *DeepSeekProvider) Name() string { return "deepseek" }

// Available implements Provider.
func (p *DeepSeekProvider) Available() bool { return p != nil && p.APIKey != "" }

type deepSeekSearchRequest struct {
	Query        string `json:"query"`
	Count        int    `json:"count"`
	OriQuestion  string `json:"ori_question,omitempty"`
	Localization string `json:"localization,omitempty"`
	NeedContent  bool   `json:"need_content"`
}

type deepSeekSearchResponse struct {
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

type deepSeekFetchRequest struct {
	URL string `json:"url"`
}

type deepSeekFetchResponse struct {
	Title   string `json:"title"`
	Content string `json:"content"`
}

// Search implements Provider.
func (p *DeepSeekProvider) Search(ctx context.Context, q Query) ([]Result, error) {
	if !p.Available() {
		return nil, fmt.Errorf("deepseek: no api key configured")
	}
	count := q.Count
	if count <= 0 {
		count = 10
	}
	if count > p.MaxCount {
		count = p.MaxCount
	}
	text := q.Text
	if q.FreshnessDays > 0 && !hasTimeOperator(text) {
		// DeepSeek forwards the query with its search syntax intact, so a
		// date bound is applied server-side instead of being filtered
		// locally after the fact.
		since := time.Now().AddDate(0, 0, -q.FreshnessDays).Format("2006-01-02")
		text = fmt.Sprintf("%s after:%s", text, since)
	}

	req := deepSeekSearchRequest{
		Query:       text,
		Count:       count,
		NeedContent: q.NeedContent,
	}
	var resp deepSeekSearchResponse
	if err := postJSON(ctx, p.Client, p.BaseURL+"/v1/tools/search",
		map[string]string{"Authorization": "Bearer " + p.APIKey}, req, &resp); err != nil {
		return nil, errProvider(p.Name(), err)
	}

	out := make([]Result, 0, len(resp.Results))
	for _, r := range resp.Results {
		url := strings.TrimSpace(r.URL)
		if url == "" {
			continue
		}
		auth := -1
		if r.Authority != nil {
			auth = *r.Authority
		}
		out = append(out, Result{
			Title:       joinTitle(r.Title, r.SiteName),
			URL:         url,
			Snippet:     r.Snippet,
			Summary:     r.Summary,
			Content:     r.Content,
			PublishedAt: parseDate(r.Date),
			Authority:   auth,
			Score:       r.RerankScore,
			Provider:    p.Name(),
		})
	}
	return out, nil
}

// Fetch implements Provider.
func (p *DeepSeekProvider) Fetch(ctx context.Context, url string) (Page, error) {
	if !p.Available() {
		return Page{}, fmt.Errorf("deepseek: no api key configured")
	}
	url = strings.TrimSpace(url)
	if url == "" {
		return Page{}, fmt.Errorf("deepseek: url is required")
	}
	var resp deepSeekFetchResponse
	if err := postJSON(ctx, p.Client, p.BaseURL+"/v1/tools/fetch",
		map[string]string{"Authorization": "Bearer " + p.APIKey},
		deepSeekFetchRequest{URL: url}, &resp); err != nil {
		return Page{}, errProvider(p.Name(), err)
	}
	return Page{Title: resp.Title, URL: url, Content: resp.Content}, nil
}

// hasTimeOperator reports whether the query already carries a date bound, in
// which case the provider must not add a second one.
func hasTimeOperator(q string) bool {
	lq := strings.ToLower(q)
	return strings.Contains(lq, "after:") || strings.Contains(lq, "before:")
}
