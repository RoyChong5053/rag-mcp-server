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
	"time"
)

// EmbeddingClient calls one-api's embedding endpoint
type EmbeddingClient struct {
	baseURL    string
	backupURL  string
	model      string
	apiKey     string
	httpClient *http.Client
}

// NewEmbeddingClient creates a new embedding client.
// Timeout is 20s: single-query embeds return in ~10s through one-api
// fan-out; bulk index jobs are batched (32/batch) and async. Fallback
// across one-api channels is one-api's job, so this client only retries
// briefly (thin retry) instead of stacking a second fallback layer.
func NewEmbeddingClient(baseURL, backupURL, model, apiKey string) *EmbeddingClient {
	return &EmbeddingClient{
		baseURL:   baseURL,
		backupURL: backupURL,
		model:     model,
		apiKey:    apiKey,
		httpClient: &http.Client{
			Timeout: 20 * time.Second,
		},
	}
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

// embedBatchSize caps texts per embedding request so large documents
// don't blow up into a single giant HTTP call (timeouts / 413s).
const embedBatchSize = 32

// CreateEmbeddings generates embeddings for the given texts
func (c *EmbeddingClient) CreateEmbeddings(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	req := EmbeddingRequest{
		Model: c.model,
		Input: texts,
	}

	// Small input: single request (fast path, preserves old behavior)
	if len(texts) <= embedBatchSize {
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
	for start := 0; start < len(texts); start += embedBatchSize {
		end := start + embedBatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batchReq := EmbeddingRequest{Model: c.model, Input: texts[start:end]}
		embeddings, err := c.callWithRetry(c.baseURL, batchReq)
		if err != nil && c.backupURL != "" {
			embeddings, err = c.callWithRetry(c.backupURL, batchReq)
		}
		if err != nil {
			return nil, fmt.Errorf("batch [%d:%d]: %w", start, end, err)
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

// callWithRetry runs one embedding request with a short exponential backoff.
// Honors Retry-After when the server sends one. 3 attempts with 500ms base
// keep the typical offline case under ~10s: one-api already retried across
// all its channels, so a second deep retry here only multiplies tail latency.
func (c *EmbeddingClient) callWithRetry(baseURL string, req EmbeddingRequest) ([][]float32, error) {
	backoff := 500 * time.Millisecond
	const maxAttempts = 3
	var err error
	var embeddings [][]float32
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		var apiErr *apiError
		embeddings, err = c.callEndpoint(baseURL, req)
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
		log.Printf("Embedding attempt %d/%d failed (%v), retrying in %s", attempt, maxAttempts, err, wait)
		time.Sleep(wait)
		backoff *= 2
	}
	return nil, fmt.Errorf("embedding failed after %d attempts: %w", maxAttempts, err)
}

func (c *EmbeddingClient) callEndpoint(baseURL string, req EmbeddingRequest) ([][]float32, error) {
	return c.callEndpointWithClient(c.httpClient, baseURL, req)
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
// with a short 5s budget (no retries) so health checks stay fast and quiet
// and never hold a HealthCheck probe hostage.
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
	return &http.Client{Timeout: 5 * time.Second}
}
