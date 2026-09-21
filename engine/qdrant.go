package engine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// QdrantClient wraps the Qdrant REST API
type QdrantClient struct {
	baseURL    string
	httpClient *http.Client
}

// QdrantCollectionInfo holds collection metadata from Qdrant
type QdrantCollectionInfo struct {
	Name       string `json:"name"`
	ChunkCount int    `json:"chunk_count"`
}

// Point represents a Qdrant point
type Point struct {
	ID      string         `json:"id"`
	Payload map[string]any `json:"payload"`
	Vector  []float32      `json:"vector,omitempty"`
}

// QdrantSearchResult represents a raw search result from Qdrant
type QdrantSearchResult struct {
	ID      string         `json:"id"`
	Score   float64        `json:"score"`
	Payload map[string]any `json:"payload"`
}

// NewQdrantClient creates a new Qdrant client
func NewQdrantClient(host string, port int) *QdrantClient {
	return &QdrantClient{
		baseURL: fmt.Sprintf("http://%s:%d", host, port),
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// CreateCollection creates a new Qdrant collection with cosine similarity
func (c *QdrantClient) CreateCollection(name string) error {
	body := map[string]any{
		"vectors": map[string]any{
			"size":     1024, // Qwen3-Embedding-0.6B dimension
			"distance": "Cosine",
		},
	}
	return c.put(fmt.Sprintf("/collections/%s", name), body)
}

// CollectionExists checks if a collection exists
func (c *QdrantClient) CollectionExists(name string) (bool, error) {
	_, err := c.get(fmt.Sprintf("/collections/%s", name))
	if err != nil {
		return false, nil // Collection doesn't exist
	}
	return true, nil
}

// GetCollectionInfo returns collection info including point count
func (c *QdrantClient) GetCollectionInfo(name string) (*QdrantCollectionInfo, error) {
	resp, err := c.get(fmt.Sprintf("/collections/%s", name))
	if err != nil {
		return nil, err
	}

	var result struct {
		Result struct {
			Status string `json:"status"`
			OptimizerStatus string `json:"optimizer_status"`
			VectorsCount int `json:"vectors_count"`
			PointsCount  int `json:"points_count"`
		} `json:"result"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}

	return &QdrantCollectionInfo{
		Name:       name,
		ChunkCount: result.Result.PointsCount,
	}, nil
}

// ListCollections returns all collections
func (c *QdrantClient) ListCollections() ([]QdrantCollectionInfo, error) {
	resp, err := c.get("/collections")
	if err != nil {
		return nil, err
	}

	var result struct {
		Result struct {
			Collections []struct {
				Name string `json:"name"`
			} `json:"collections"`
		} `json:"result"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}

	var collections []QdrantCollectionInfo
	for _, col := range result.Result.Collections {
		info, err := c.GetCollectionInfo(col.Name)
		if err != nil {
			continue
		}
		collections = append(collections, *info)
	}

	return collections, nil
}

// UpsertPoints inserts or updates points
func (c *QdrantClient) UpsertPoints(collection string, points []Point) error {
	body := map[string]any{
		"points": points,
	}
	return c.put(fmt.Sprintf("/collections/%s/points", collection), body)
}

// Search performs a vector search
func (c *QdrantClient) Search(collection string, vector []float32, limit int, threshold float64) ([]QdrantSearchResult, error) {
	body := map[string]any{
		"vector": vector,
		"limit":  limit,
		"score_threshold": threshold,
		"with_payload": true,
	}

	resp, err := c.post(fmt.Sprintf("/collections/%s/points/search", collection), body)
	if err != nil {
		return nil, err
	}

	var result struct {
		Result []QdrantSearchResult `json:"result"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}

	return result.Result, nil
}

// DeletePoints deletes points by filter
func (c *QdrantClient) DeletePoints(collection string, filter map[string]any) error {
	body := map[string]any{
		"filter": filter,
	}
	_, err := c.post(fmt.Sprintf("/collections/%s/points/delete", collection), body)
	return err
}

// Ping checks if Qdrant is reachable
func (c *QdrantClient) Ping() error {
	resp, err := c.get("/collections")
	if err != nil {
		return fmt.Errorf("qdrant unreachable: %w", err)
	}
	var result struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return fmt.Errorf("invalid qdrant response: %w", err)
	}
	if result.Status != "ok" {
		return fmt.Errorf("qdrant status: %s", result.Status)
	}
	return nil
}

// HTTP helpers

func (c *QdrantClient) get(path string) ([]byte, error) {
	resp, err := c.httpClient.Get(c.baseURL + path)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func (c *QdrantClient) put(path string, body any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPut, c.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant error %d: %s", resp.StatusCode, string(body))
	}
	return nil
}

func (c *QdrantClient) post(path string, body any) ([]byte, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient.Post(c.baseURL+path, "application/json", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("qdrant error %d: %s", resp.StatusCode, string(respBody))
	}
	return respBody, nil
}
