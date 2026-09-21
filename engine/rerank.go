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
}

// NewRerankClient creates a new rerank client
func NewRerankClient(baseURL, backupURL, model, apiKey string, queryMaxChars, docMaxChars int) *RerankClient {
	return &RerankClient{
		baseURL:       baseURL,
		backupURL:     backupURL,
		model:         model,
		apiKey:        apiKey,
		queryMaxChars: queryMaxChars,
		docMaxChars:   docMaxChars,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
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
	Index       int     `json:"index"`
	Score       float64 `json:"relevance_score"`
	RawScore    float64 `json:"score,omitempty"`
}

// RerankScoreMode indicates the score scale
type RerankScoreMode int

const (
	ScoreModeAuto       RerankScoreMode = iota
	ScoreModeProbability                 // scores in [0, 1]
	ScoreModeLogit                       // unbounded, needs sigmoid
)

// Rerank reranks documents against a query
func (c *RerankClient) Rerank(query string, documents []string, topN int) ([]RerankResult, error) {
	if len(documents) == 0 {
		return nil, nil
	}

	// Truncate query
	queryMax, docMax := c.effectiveLimits()
	query = truncateRunes(query, queryMax)

	// Truncate documents
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

	// Try primary endpoint first
	results, err := c.callEndpoint(c.baseURL, req)
	if err != nil && c.backupURL != "" {
		// Fallback to backup endpoint
		results, err = c.callEndpoint(c.backupURL, req)
	}

	return results, err
}

func (c *RerankClient) callEndpoint(baseURL string, req RerankRequest) ([]RerankResult, error) {
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

	resp, err := c.httpClient.Do(httpReq)
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

// Ping tests connectivity to the rerank endpoint
func (c *RerankClient) Ping() error {
	_, err := c.Rerank("test", []string{"test document"}, 1)
	return err
}
