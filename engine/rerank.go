package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/chunking"
)

// rerankDownTTL is how long the rerank circuit breaker stays open after a
// failure. A dead/absent reranker (LOQ off) is skipped for this window so
// searches degrade to vector order instantly instead of paying the timeout
// again; the first successful call clears it.
const rerankDownTTL = 2 * time.Minute

// RerankClient calls one-api's rerank endpoint
type RerankClient struct {
	baseURL       string
	backupURL     string
	model         string
	apiKey        string
	queryMaxTokens int
	docMaxTokens   int
	// limits, when set, supplies runtime truncation limits (query, doc) so
	// WebUI edits take effect without a restart. It overrides the static values.
	limits     func() (int, int)
	httpClient *http.Client

	// healthMu guards downUntil, the rerank circuit breaker. After a failure
	// the reranker is skipped for rerankDownTTL so a search degrades to vector
	// order instantly instead of paying the timeout again.
	healthMu  sync.Mutex
	downUntil time.Time
}

// NewRerankClient creates a new rerank client.
// Timeout is 45s: a 30-doc fan-out through one-api takes ~36s when healthy.
// Rerank is fail-open (failure degrades to vector order), so a tight budget
// here only costs ranking quality, never availability. Channel fallback is
// one-api's job; this client retries at most once.
func NewRerankClient(baseURL, backupURL, model, apiKey string, queryMaxTokens, docMaxTokens int) *RerankClient {
	return &RerankClient{
		baseURL:       baseURL,
		backupURL:     backupURL,
		model:         model,
		apiKey:        apiKey,
		queryMaxTokens: queryMaxTokens,
		docMaxTokens:   docMaxTokens,
		httpClient: &http.Client{
			Timeout: 45 * time.Second,
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
	return c.queryMaxTokens, c.docMaxTokens
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

// Rerank reranks documents against a query.
//
// wait is the caller's total deadline (embed+reranker budget). It is threaded
// down as a context so a tight budget fails open at `wait` instead of eating
// the full 45s HTTP timeout: on timeout we fall back to vector order and do
// NOT latch the breaker, keeping a transient stall from becoming a persistent
// outage. See RerankClient for the breaker policy (fatal only).
func (c *RerankClient) Rerank(ctx context.Context, query string, documents []string, topN int, wait time.Duration) ([]RerankResult, error) {
	if len(documents) == 0 {
		return nil, nil
	}

	// Truncate query
	queryMax, docMax := c.effectiveLimits()
	query = truncateTokens(query, queryMax)

	// Circuit breaker: a recent failure skips rerank entirely so a dead
	// upstream never costs the full timeout on every search.
	if c.isDown() {
		return nil, fmt.Errorf("rerank temporarily disabled after a recent failure")
	}

	// Truncate documents
	boundedDocs := make([]string, len(documents))
	for i, doc := range documents {
		boundedDocs[i] = truncateTokens(doc, docMax)
	}

	req := RerankRequest{
		Model:     c.model,
		Query:     query,
		Documents: boundedDocs,
		TopN:      topN,
	}

	// Primary endpoint first (one retry per callWithRetry attempt is ctx-bounded);
	// fall back to the backup once so a bad primary doesn't cost the whole budget.
	var results []RerankResult
	var err error
	results, err = c.callWithRetry(ctx, c.baseURL, req)
	if err != nil && ctx.Err() == nil && c.backupURL != "" {
		results, err = c.callWithRetry(ctx, c.backupURL, req)
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		log.Printf("rerank timed out at %.0fs — fail-open: keeping vector order", wait.Seconds())
		return results, fmt.Errorf("rerank deadline exceeded (%.0fs): %w", wait.Seconds(), ctxErr)
	}
	if err != nil {
		switch {
		case fatalRerankErr(err):
			log.Printf("rerank FATAL error: %v — breaker OPEN (fatal misconfig, not retryable)", err)
			c.markDown()
			return nil, err
		default:
			if rerankRetryable(err) {
				c.markDown()
			} else {
				c.markUp()
			}
			return nil, err
		}
	}
	c.markUp()
	return results, nil
}

// isDown reports whether the rerank circuit breaker is open.
func (c *RerankClient) isDown() bool {
	c.healthMu.Lock()
	defer c.healthMu.Unlock()
	return time.Now().Before(c.downUntil)
}

func (c *RerankClient) markDown() {
	c.healthMu.Lock()
	defer c.healthMu.Unlock()
	c.downUntil = time.Now().Add(rerankDownTTL)
}

func (c *RerankClient) markUp() {
	c.healthMu.Lock()
	defer c.healthMu.Unlock()
	c.downUntil = time.Time{}
}

// callWithRetry runs one rerank request with a single quick retry. ctx bounds
// the whole fan-out: if it is already gone we stop early so a tight caller
// budget fails open instead of burning another attempt. 2 attempts keep the
// healthy 30-doc fan-out (~36s) inside budget while an offline upstream fails
// fast instead of hanging the search path.
func (c *RerankClient) callWithRetry(ctx context.Context, baseURL string, req RerankRequest) ([]RerankResult, error) {
	const maxAttempts = 2
	var err error
	var results []RerankResult
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if ctx.Err() != nil {
			break
		}
		results, err = c.callEndpointWithClient(c.httpClient, baseURL, req)
		if err == nil {
			return results, nil
		}
		if !rerankRetryable(err) && attempt == 1 {
			// Non-retryable (auth/validation): one attempt is enough, but
			// still give the backup endpoint a chance via the caller.
			return nil, err
		}
		if attempt < maxAttempts {
			time.Sleep(time.Duration(attempt) * time.Second)
		}
	}
	return results, err
}

// rerankRetryable reports whether a rerank error deserves another attempt.
// Transport failures and 429/5xx are retryable; auth/validation fail fast.
// A client timeout is NOT retryable: the upstream is absent or overloaded,
// and a second full timeout would only double the stall (fail-open instead).
func rerankRetryable(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	if strings.Contains(msg, "context deadline exceeded") || strings.Contains(msg, "Client.Timeout exceeded") {
		return false
	}
	for _, code := range []string{" 429", " 500", " 502", " 503", " 504", "request failed"} {
		if strings.Contains(msg, code) {
			return true
		}
	}
	return false
}

// fatalRerankErr reports whether an error is FATAL: the request must never be
// retried and must not trip the breaker. These are client-side misconfig — a
// wrong API key (401) or an unparseable body — where hammering again would
// only repeat the same failure until a human fixes it.
func fatalRerankErr(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "401") || strings.Contains(msg, "unauthorized") ||
		strings.Contains(msg, "invalid api key") || strings.Contains(msg, "malformed") ||
		strings.Contains(msg, "failed to parse rerank") || strings.Contains(msg, "failed to marshal rerank")
}

// rerankExceeded reports whether err is a timeout (client deadline or context
// cancellation). A timed-out upstream must NOT latch the breaker: fail-open and
// fall back to vector order so a transient stall never turns into a persistent
// outage. This mirrors rerankRetryable's "timeout = absent/overloaded" stance,
// but as an explicit branch so it is impossible to be retried-then-latched by
// mistake (the old any-error -> markDown path).
func rerankExceeded(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "context deadline exceeded") ||
		strings.Contains(msg, "Client.Timeout exceeded") ||
		strings.Contains(strings.ToLower(err.Error()), "context canceled")
}

func (c *RerankClient) callEndpoint(baseURL string, req RerankRequest) ([]RerankResult, error) {
	return c.callEndpointWithClient(c.httpClient, baseURL, req)
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

// truncateTokens cuts s to at most max jina-v2 tokens, appending an ellipsis
// when clipped. Byte-safe because TokenLen/SplitByTokens work on runes.
func truncateTokens(s string, max int) string {
	if max <= 0 || chunking.TokenLen(s) <= max {
		return s
	}
	parts := chunking.SplitByTokens(s, max)
	if len(parts) == 0 {
		return ""
	}
	return parts[0] + "…"
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
