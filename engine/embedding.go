package engine

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// EmbeddingClient calls one-api's embedding endpoint
type EmbeddingClient struct {
	baseURL    string
	backupURL  string
	model      string
	apiKey     string
	httpClient *http.Client

	// timeout is the TOTAL budget for one logical embed across all retries.
	// It must stay comfortably above one-api's own channel-failover window:
	// cancelling mid-failover makes one-api see context.Canceled and treat the
	// request as a user abort — it then neither tries another channel nor
	// penalizes the bad one (see one-api controller/relay.go). rag-mcp never
	// cancels early; one-api decides.
	timeout time.Duration

	// limiter caps concurrent embedding POSTs so parallel searches / index
	// jobs cannot stampede a CPU-only fan-out. Its capacity follows the
	// compute mode through concurrency (GPU vs CPU fallback).
	limiter     *dynLimiter
	concurrency func() int
}

// defaultOneAPITimeout is the total per-call budget for embed/rerank when the
// config leaves it unset. It is deliberately larger than one-api's per-attempt
// RELAY_TIMEOUT (120s) so one-api gets its fallback chances before rag-mcp
// reports "unavailable" and lets the frontend fail open.
const defaultOneAPITimeout = 180 * time.Second

// defaultEmbedConcurrency bounds concurrent embedding POSTs when no compute
// mode is wired (standalone callers/tests). The engine overrides it.
const defaultEmbedConcurrency = 8

// dynLimiter is a counting semaphore whose capacity may change at runtime:
// GPU mode allows a few parallel embeds, the CPU fallback serializes.
type dynLimiter struct {
	mu    sync.Mutex
	cond  *sync.Cond
	limit int
	inUse int
}

func newDynLimiter(limit int) *dynLimiter {
	if limit < 1 {
		limit = 1
	}
	l := &dynLimiter{limit: limit}
	l.cond = sync.NewCond(&l.mu)
	return l
}

func (l *dynLimiter) setLimit(n int) {
	if n < 1 {
		n = 1
	}
	l.mu.Lock()
	if l.limit != n {
		l.limit = n
		l.cond.Broadcast()
	}
	l.mu.Unlock()
}

func (l *dynLimiter) acquire() {
	l.mu.Lock()
	for l.inUse >= l.limit {
		l.cond.Wait()
	}
	l.inUse++
	l.mu.Unlock()
}

func (l *dynLimiter) release() {
	l.mu.Lock()
	if l.inUse > 0 {
		l.inUse--
	}
	l.cond.Signal()
	l.mu.Unlock()
}

// SetConcurrencyProvider wires a callback returning the current concurrent
// embedding-request cap. It is consulted on every acquire so the cap tracks
// the compute mode (GPU vs CPU fallback) without a restart.
func (c *EmbeddingClient) SetConcurrencyProvider(f func() int) {
	c.concurrency = f
}

func (c *EmbeddingClient) acquire() {
	if c.limiter == nil {
		return
	}
	if c.concurrency != nil {
		c.limiter.setLimit(c.concurrency())
	}
	c.limiter.acquire()
}

func (c *EmbeddingClient) release() {
	if c.limiter != nil {
		c.limiter.release()
	}
}

// NewEmbeddingClient creates a new embedding client.
// Timeout is 120s to match one-api RELAY_TIMEOUT on m64: single-query
// embeds return in ~10s through GPU fan-out, but CPU-only fan-out
// (LOQ down, 5 weak nodes) needs tens of seconds per shard. Bulk index
// jobs are batched (32/batch) and async. Fallback across one-api channels
// is one-api's job, so this client only retries briefly (thin retry)
// instead of stacking a second fallback layer.
func NewEmbeddingClient(baseURL, backupURL, model, apiKey string) *EmbeddingClient {
	return &EmbeddingClient{
		baseURL:   baseURL,
		backupURL: backupURL,
		model:     model,
		apiKey:    apiKey,
		httpClient: &http.Client{
			Timeout: defaultOneAPITimeout,
		},
		timeout: defaultOneAPITimeout,
		limiter: newDynLimiter(defaultEmbedConcurrency),
	}
}

// SetTimeout sets the total per-call budget. 0 keeps the default. The value is
// capped so a misconfigured config cannot pin a request forever.
func (c *EmbeddingClient) SetTimeout(d time.Duration) {
	if d <= 0 {
		return
	}
	if d > 15*time.Minute {
		d = 15 * time.Minute
	}
	c.timeout = d
	c.httpClient.Timeout = d
}

func (c *EmbeddingClient) budget() time.Duration {
	if c.timeout <= 0 {
		return defaultOneAPITimeout
	}
	return c.timeout
}

// EmbeddingRequest represents the request to one-api's embedding endpoint
type EmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// EmbeddingResponse represents the response from one-api's embedding endpoint
type EmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

// defaultEmbedBatchSize is the GPU fast path (RTX4060). The CPU slow path
// (LOQ down or VRAM full) should use 8 or lower via BatchSize.
const defaultEmbedBatchSize = 32

// BatchSize caps texts per embedding POST. 0/negative = default (32).
// Set per index job from settings.embed_batch_size so operators can flip
// between GPU-fast (32) and CPU-slow (8) in the WebUI without a restart.
// Single-text query embeds never reach the batch loop, so search latency is
// unaffected by this value.
func (c *EmbeddingClient) batchSize() int {
	return defaultEmbedBatchSize
}

// CreateEmbeddings generates embeddings for the given texts with an explicit
// per-POST batch cap. Use CreateEmbeddings for the default (32); index jobs
// pass settings.embed_batch_size so CPU-only nights use 8.
func (c *EmbeddingClient) CreateEmbeddings(texts []string) ([][]float32, error) {
	return c.CreateEmbeddingsBatched(texts, 0)
}

// CreateEmbeddingsBatched is CreateEmbeddings with a caller-chosen batch cap
// (<=0 = default). Batches are sent sequentially to preserve chunk order.
func (c *EmbeddingClient) CreateEmbeddingsBatched(texts []string, batchSize int) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	bs := batchSize
	if bs <= 0 {
		bs = defaultEmbedBatchSize
	}

	req := EmbeddingRequest{
		Model: c.model,
		Input: texts,
	}

	// Small input: single request (fast path, preserves old behavior)
	if len(texts) <= bs {
		// Try primary endpoint first
		embeddings, err := c.callWithRetry(c.baseURL, req)
		if err != nil && c.backupURL != "" {
			// Fallback to backup endpoint
			embeddings, err = c.callWithRetry(c.backupURL, req)
		}
		return embeddings, err
	}

	// Large input: batch sequentially to avoid timeouts
	all := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += bs {
		end := start + bs
		if end > len(texts) {
			end = len(texts)
		}
		batchReq := EmbeddingRequest{Model: c.model, Input: texts[start:end]}
		embeddings, err := c.callWithRetry(c.baseURL, batchReq)
		if err != nil && c.backupURL != "" {
			embeddings, err = c.callWithRetry(c.backupURL, batchReq)
		}
		if err != nil {
			return nil, fmt.Errorf("batch [%d:%d]/%d (batch_size=%d): %w", start, end, len(texts), bs, err)
		}
		all = append(all, embeddings...)
	}

	return all, nil
}

// apiError carries the HTTP status so the retry wrapper can decide.
type apiError struct {
	status     int
	body       string
	retryAfter time.Duration
}

func (e *apiError) Error() string {
	return fmt.Sprintf("embedding API error %d: %s", e.status, e.body)
}

// retryable reports whether the error deserves another attempt:
// rate limits, bad gateways, and transport failures. Auth/validation
// errors (400/401/403/404/413/422) fail fast.
func retryable(err error) bool {
	if err == nil {
		return false
	}
	if ae, ok := err.(*apiError); ok {
		switch ae.status {
		case 429, 500, 502, 503, 504:
			return true
		}
		return false
	}
	// Transport errors (connection reset, timeouts) are retryable
	return true
}

// callWithRetry runs one logical embedding against one-api within a single
// total budget, retrying only on a genuine one-api error (5xx/429/transport)
// and only while time remains. It never cancels one-api mid-failover: one-api
// owns upstream channel selection, and a client-side cancel makes it treat the
// request as an user abort (context.Canceled) — no channel fallback, no health
// penalty. So rag-mcp waits the full budget, then reports "unavailable" and
// lets the frontend fail open.
func (c *EmbeddingClient) callWithRetry(baseURL string, req EmbeddingRequest) ([][]float32, error) {
	budget := c.budget()
	deadline := time.Now().Add(budget)
	backoff := 500 * time.Millisecond
	const maxAttempts = 3
	var err error
	var embeddings [][]float32
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			break
		}
		var apiErr *apiError
		embeddings, err = c.callEndpointWithBudget(baseURL, req, remaining)
		if err == nil {
			return embeddings, nil
		}
		if !retryable(err) {
			return nil, err
		}
		if attempt == maxAttempts {
			break
		}
		wait := backoff
		if errors.As(err, &apiErr) && apiErr.retryAfter > 0 {
			wait = apiErr.retryAfter
		}
		if time.Until(deadline) <= wait {
			break
		}
		log.Printf("Embedding attempt %d/%d failed (%v), retrying in %s", attempt, maxAttempts, err, wait)
		time.Sleep(wait)
		backoff *= 2
	}
	if err == nil {
		return nil, fmt.Errorf("embedding unavailable: budget of %.0fs exhausted", budget.Seconds())
	}
	return nil, fmt.Errorf("embedding unavailable after %.0fs (one-api did not complete; rag-mcp did not cancel early): %w", budget.Seconds(), err)
}

// callEndpointWithBudget runs one HTTP attempt under the remaining slice of the
// total budget and the concurrency limiter (so retries cannot stampede a
// saturated CPU-only fan-out). Ping bypasses this via callEndpointWithClient.
func (c *EmbeddingClient) callEndpointWithBudget(baseURL string, req EmbeddingRequest, remaining time.Duration) ([][]float32, error) {
	c.acquire()
	defer c.release()
	return c.callEndpointWithClient(&http.Client{Timeout: remaining}, baseURL, req)
}

func (c *EmbeddingClient) callEndpointWithClient(client *http.Client, baseURL string, req EmbeddingRequest) ([][]float32, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	url := baseURL + "/v1/embeddings"
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
		return nil, fmt.Errorf("embedding request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, &apiError{status: resp.StatusCode, body: string(body), retryAfter: parseRetryAfter(resp)}
	}

	var embeddingResp EmbeddingResponse
	if err := json.Unmarshal(body, &embeddingResp); err != nil {
		return nil, fmt.Errorf("failed to parse embedding response: %w", err)
	}

	// Sort by index and extract embeddings
	textCount := len(req.Input)
	embeddings := make([][]float32, textCount)
	for _, item := range embeddingResp.Data {
		if item.Index < textCount {
			embeddings[item.Index] = item.Embedding
		}
	}

	// Validate all embeddings were returned
	for i, emb := range embeddings {
		if emb == nil {
			return nil, fmt.Errorf("missing embedding for text %d", i)
		}
	}

	return embeddings, nil
}

// parseRetryAfter reads the Retry-After header (seconds or HTTP date).
func parseRetryAfter(resp *http.Response) time.Duration {
	v := strings.TrimSpace(resp.Header.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
		if secs > 120 {
			secs = 120
		}
		return time.Duration(secs) * time.Second
	}
	return 0
}

// Ping tests connectivity to the embedding endpoint. Uses a single attempt
// with a 15s budget (no retries): GPU answers in ~1s but a CPU-only fan-out
// needs 5-10s for one text. The engine's 8s probeWithTimeout still bounds
// dashboard latency (over-budget shows amber "slow", never "down"), so a
// generous client here turns slow-but-working into "slow", not a hard error.
func (c *EmbeddingClient) Ping() error {
	req := EmbeddingRequest{Model: c.model, Input: []string{"test"}}
	_, err := c.callEndpointWithClient(c.pingClient(), c.baseURL, req)
	if err != nil && c.backupURL != "" {
		_, err = c.callEndpointWithClient(c.pingClient(), c.backupURL, req)
	}
	return err
}

// pingClient is a short-budget client for health probes only.
func (c *EmbeddingClient) pingClient() *http.Client {
	return &http.Client{Timeout: 15 * time.Second}
}
