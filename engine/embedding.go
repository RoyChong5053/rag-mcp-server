package engine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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

// NewEmbeddingClient creates a new embedding client
func NewEmbeddingClient(baseURL, backupURL, model, apiKey string) *EmbeddingClient {
	return &EmbeddingClient{
		baseURL:   baseURL,
		backupURL: backupURL,
		model:     model,
		apiKey:    apiKey,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
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
		embeddings, err := c.callEndpoint(c.baseURL, req)
		if err != nil && c.backupURL != "" {
			// Fallback to backup endpoint
			embeddings, err = c.callEndpoint(c.backupURL, req)
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
		embeddings, err := c.callEndpoint(c.baseURL, batchReq)
		if err != nil && c.backupURL != "" {
			embeddings, err = c.callEndpoint(c.backupURL, batchReq)
		}
		if err != nil {
			return nil, fmt.Errorf("batch [%d:%d]: %w", start, end, err)
		}
		all = append(all, embeddings...)
	}

	return all, nil
}

func (c *EmbeddingClient) callEndpoint(baseURL string, req EmbeddingRequest) ([][]float32, error) {
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

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("embedding request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("embedding API error %d: %s", resp.StatusCode, string(body))
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

// Ping tests connectivity to the embedding endpoint
func (c *EmbeddingClient) Ping() error {
	_, err := c.CreateEmbeddings([]string{"test"})
	return err
}
