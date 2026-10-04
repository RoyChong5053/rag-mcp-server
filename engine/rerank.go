package engine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

// RerankClient calls one-api's rerank endpoint
type RerankClient struct {
	baseURL       string
	backupURL     string
	model         string
	apiKey        string
	queryMaxChars int
	docMaxChars   int
	// limits, when set, supplies runtime truncation limits (query, doc) so
	// WebUI edits take effect without a restart. It overrides the static values.
	limits     func() (int, int)
	httpClient *http.Client

	// timeout is the TOTAL budget for one logical rerank across retries. Like
	// the embedding client it stays above one-api's failover window so rag-mcp
	// never cancels mid-failover.
	timeout time.Duration
}

// NewRerankClient creates a new rerank client. The default budget is
// defaultOneAPITimeout (180s): rerank is fail-open, so a slow/dead reranker
// costs ranking quality (or a bounded wait), never correctness. Channel
// fallback is one-api's job.
func NewRerankClient(baseURL, backupURL, model, apiKey string, queryMaxChars, docMaxChars int) *RerankClient {
	return &RerankClient{
		baseURL:       baseURL,
		backupURL:     backupURL,
		model:         model,
		apiKey:        apiKey,
		queryMaxChars: queryMaxChars,
		docMaxChars:   docMaxChars,
		httpClient: &http.Client{
			Timeout: defaultOneAPITimeout,
		},
		timeout: defaultOneAPITimeout,
	}
}

// SetTimeout sets the total per-call budget. 0 keeps the default.
func (c *RerankClient) SetTimeout(d time.Duration) {
	if d <= 0 {
		return
	}
	if d > 15*time.Minute {
		d = 15 * time.Minute
	}
	c.timeout = d
	c.httpClient.Timeout = d
}

func (c *RerankClient) budget() time.Duration {
	if c.timeout <= 0 {
		return defaultOneAPITimeout
	}
	return c.timeout
}

// SetLimitsProvider wires a callback for runtime truncation limits.
func (c *RerankClient) SetLimitsProvider(f func() (int, int)) {
	c.limits = f
}

func (c *RerankClient) effectiveLimits() (int, int) {
	if c.limits != nil {
		return c.limits()
	}
	return c.queryMaxChars, c.docMaxChars
}

// RerankRequest represents the request to one-api's rerank endpoint
type RerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n"`
}

// RerankResponse represents the response from one-api's rerank endpoint
type RerankResponse struct {
	Results []RerankResult `json:"results"`
}

// RerankResult represents a single rerank result
type RerankResult struct {
	Index    int     `json:"index"`
	Score    float64 `json:"relevance_score"`
	RawScore float64 `json:"score,omitempty"`
}

// RerankScoreMode indicates the score scale
type RerankScoreMode int

const (
	ScoreModeAuto        RerankScoreMode = iota
	ScoreModeProbability                 // scores in [0, 1]
	ScoreModeLogit                       // unbounded, needs sigmoid
)

// Rerank reranks documents against a query. Failure is fail-open at the call
// site (the search keeps vector order), so this only reports the error. The
// client never cancels one-api mid-failover; one-api owns channel selection and
// penalizes bad channels only when it — not rag-mcp — returns the error.
func (c *RerankClient) Rerank(query string, documents []string, topN int) ([]RerankResult, error) {
	if len(documents) == 0 {
		return nil, nil
	}

	queryMax, docMax := c.effectiveLimits()
	query = truncateRunes(query, queryMax)

	boundedDocs := make([]string, len(documents))
	for i, doc := range documents {
		boundedDocs[i] = truncateRunes(doc, docMax)
	}

	req := RerankRequest{
		Model:     c.model,
		Query:     query,
		Documents: boundedDocs,
		TopN:      topN,
	}

	results, err := c.callWithRetry(c.baseURL, req)
	if err != nil && c.backupURL != "" {
		results, err = c.callWithRetry(c.backupURL, req)
	}
	return results, err
}

// callWithRetry runs one logical rerank within a single total budget, retrying
// only on a genuine one-api error (5xx/429/transport) and only while time
// remains. A client timeout is not retried (the budget is already spent).
func (c *RerankClient) callWithRetry(baseURL string, req RerankRequest) ([]RerankResult, error) {
	budget := c.budget()
	deadline := time.Now().Add(budget)
	backoff := time.Second
	const maxAttempts = 2
	var err error
	var results []RerankResult
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			break
		}
		results, err = c.callEndpointWithClient(&http.Client{Timeout: remaining}, baseURL, req)
		if err == nil {
			return results, nil
		}
		if !rerankRetryable(err) || attempt == maxAttempts {
			return results, err
		}
		if time.Until(deadline) <= backoff {
			break
		}
		time.Sleep(backoff)
	}
	if err == nil {
		err = fmt.Errorf("rerank unavailable: budget of %.0fs exhausted", budget.Seconds())
	}
	return results, err
}

// rerankRetryable reports whether a rerank error deserves another attempt:
// one-api server errors and transport failures yes, an exhausted client
// timeout no (there is no budget left to retry into).
func rerankRetryable(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	if strings.Contains(msg, "Client.Timeout exceeded") || strings.Contains(msg, "context deadline exceeded") {
		return false
	}
	for _, code := range []string{" 429", " 500", " 502", " 503", " 504", "request failed"} {
		if strings.Contains(msg, code) {
			return true
		}
	}
	return false
}

func (c *RerankClient) callEndpointWithClient(client *http.Client, baseURL string, req RerankRequest) ([]RerankResult, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal rerank request: %w", err)
	}

	url := baseURL + "/v1/rerank"
	httpReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("rerank request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("rerank API error %d: %s", resp.StatusCode, string(body))
	}

	var rerankResp RerankResponse
	if err := json.Unmarshal(body, &rerankResp); err != nil {
		return nil, fmt.Errorf("failed to parse rerank response: %w", err)
	}

	// Detect score mode and apply sigmoid if needed
	mode := c.detectScoreMode(rerankResp.Results)
	if mode == ScoreModeLogit {
		for i := range rerankResp.Results {
			rerankResp.Results[i].Score = sigmoid(rerankResp.Results[i].Score)
		}
	}

	// Sort by score descending
	sortResults(rerankResp.Results)

	return rerankResp.Results, nil
}

// ScoreModeName reports the configured model scale for observability:
// qwen3/probability models emit [0,1] directly; jina/bge/logit models emit
// unbounded logits converted via sigmoid. Distilled from ST Vector-Storage.
func (c *RerankClient) ScoreModeName() string {
	modelLower := strings.ToLower(c.model)
	if strings.Contains(modelLower, "qwen3") || strings.Contains(modelLower, "minicpm") || strings.Contains(modelLower, "probability") {
		return "probability"
	}
	if strings.Contains(modelLower, "jina") || strings.Contains(modelLower, "bge") || strings.Contains(modelLower, "gte") || strings.Contains(modelLower, "logit") {
		return "logit"
	}
	return "auto"
}

// detectScoreMode determines if scores are probability or logit
func (c *RerankClient) detectScoreMode(results []RerankResult) RerankScoreMode {
	if len(results) == 0 {
		return ScoreModeAuto
	}

	// Check model name hints
	modelLower := strings.ToLower(c.model)
	if strings.Contains(modelLower, "qwen3") || strings.Contains(modelLower, "probability") {
		return ScoreModeProbability
	}
	if strings.Contains(modelLower, "jina") || strings.Contains(modelLower, "bge") || strings.Contains(modelLower, "logit") {
		return ScoreModeLogit
	}

	// Heuristic: if all scores are in [0, 1], assume probability
	allInUnit := true
	for _, r := range results {
		if r.Score < 0 || r.Score > 1 {
			allInUnit = false
			break
		}
	}
	if allInUnit {
		return ScoreModeProbability
	}

	return ScoreModeLogit
}

// sigmoid applies sigmoid function to convert logit to probability
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// truncateRunes cuts s to at most max runes (not bytes), appending an ellipsis
// when clipped. Bytes would split multi-byte CJK runes into invalid UTF-8.
func truncateRunes(s string, max int) string {
	if max <= 0 || utf8.RuneCountInString(s) <= max {
		return s
	}
	return string([]rune(s)[:max]) + "…"
}

// sortResults sorts results by score descending
func sortResults(results []RerankResult) {
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
}

// Ping tests connectivity to the rerank endpoint with a short 8s budget and
// no retry, so health checks stay fast and never hold a probe hostage.
func (c *RerankClient) Ping() error {
	probe := &http.Client{Timeout: 8 * time.Second}
	_, err := c.callEndpointWithClient(probe, c.baseURL, RerankRequest{
		Model:     c.model,
		Query:     "test",
		Documents: []string{"test document"},
		TopN:      1,
	})
	if err != nil && c.backupURL != "" {
		_, err = c.callEndpointWithClient(probe, c.backupURL, RerankRequest{
			Model:     c.model,
			Query:     "test",
			Documents: []string{"test document"},
			TopN:      1,
		})
	}
	return err
}
