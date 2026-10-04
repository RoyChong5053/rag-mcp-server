package engine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"
)

// QdrantClient wraps the Qdrant REST API
type QdrantClient struct {
	baseURL    string
	httpClient *http.Client
	// probeClient is used only by Ping/health checks: it fails fast (~1.5s)
	// so a dead qdrant never stalls the dashboard for the full data timeout.
	probeClient *http.Client
	// Vector space for newly created collections. Defaults preserve the
	// historical Qwen3-Embedding-0.6B shape (1024/Cosine).
	vectorDim      int
	vectorDistance string
}

// NewQdrantClient creates a new Qdrant client
func NewQdrantClient(host string, port int) *QdrantClient {
	// A dead/blackholed host must fail fast so the engine can fail over to
	// vectra. Data requests get a 2s TCP connect budget (search responses can
	// still take up to the 10s overall cap); the probe client gets 1.5s total.
	dataTransport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   2 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          16,
		MaxIdleConnsPerHost:   8,
		IdleConnTimeout:       90 * time.Second,
		ResponseHeaderTimeout: 5 * time.Second,
		ExpectContinueTimeout: time.Second,
	}
	probeTransport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   1 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          4,
		MaxIdleConnsPerHost:   2,
		IdleConnTimeout:       30 * time.Second,
		ResponseHeaderTimeout: time.Second,
		ExpectContinueTimeout: time.Second,
	}
	return &QdrantClient{
		baseURL: fmt.Sprintf("http://%s:%d", host, port),
		httpClient: &http.Client{
			// Short enough that a dead/blackholed qdrant fails fast so the
			// engine can fail over to vectra; LAN round-trips are <1ms.
			Timeout:   10 * time.Second,
			Transport: dataTransport,
		},
		probeClient: &http.Client{
			Timeout:   1500 * time.Millisecond,
			Transport: probeTransport,
		},
	}
}

// setVectorSpace updates the shape used for newly created collections so a
// provider switch applies to collections created after the switch.
func (c *QdrantClient) setVectorSpace(dim int, distance string) {
	if dim > 0 {
		c.vectorDim = dim
	}
	if distance != "" {
		c.vectorDistance = distance
	}
}

// CreateCollection creates a new Qdrant collection with cosine similarity
func (c *QdrantClient) CreateCollection(name string) error {
	size := c.vectorDim
	if size == 0 {
		size = 1024
	}
	dist := c.vectorDistance
	if dist == "" {
		dist = "Cosine"
	}
	body := map[string]any{
		"vectors": map[string]any{
			"size":     size,
			"distance": dist,
		},
	}
	return c.put(fmt.Sprintf("/collections/%s", name), body)
}

// CollectionExists checks if a collection exists.
// A transport/5xx failure is surfaced as unavailable instead of being folded
// into "not found": the engine must be able to tell a down qdrant from a
// missing collection so it can mark the backend down and fail over. Only a
// genuine 404 (a plain, non-unavailable qdrant error) means "absent".
func (c *QdrantClient) CollectionExists(name string) (bool, error) {
	_, err := c.get(fmt.Sprintf("/collections/%s", name))
	if err != nil {
		if IsUnavailable(err) {
			return false, err
		}
		return false, nil // 404: collection doesn't exist
	}
	return true, nil
}

// GetCollectionInfo returns collection info including point count
func (c *QdrantClient) GetCollectionInfo(name string) (*StoreCollectionInfo, error) {
	resp, err := c.get(fmt.Sprintf("/collections/%s", name))
	if err != nil {
		return nil, err
	}

	var result struct {
		Result struct {
			Status          string `json:"status"`
			OptimizerStatus string `json:"optimizer_status"`
			VectorsCount    int    `json:"vectors_count"`
			PointsCount     int    `json:"points_count"`
		} `json:"result"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}

	return &StoreCollectionInfo{
		Name:       name,
		ChunkCount: result.Result.PointsCount,
	}, nil
}

// GetVectorSize returns the configured vector dimension of an existing
// collection. Qdrant fixes this at creation time and rejects any other size.
func (c *QdrantClient) GetVectorSize(name string) (int, error) {
	resp, err := c.get(fmt.Sprintf("/collections/%s", name))
	if err != nil {
		return 0, err
	}
	var result struct {
		Result struct {
			Config struct {
				Params struct {
					Vectors struct {
						Size int `json:"size"`
					} `json:"vectors"`
				} `json:"params"`
			} `json:"config"`
		} `json:"result"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, err
	}
	return result.Result.Config.Params.Vectors.Size, nil
}

// ListCollections returns all collections
func (c *QdrantClient) ListCollections() ([]StoreCollectionInfo, error) {
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

	var collections []StoreCollectionInfo
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

// Search performs a vector search. An optional Qdrant filter narrows the
// candidate set by payload (e.g. {"must":[{"key":"metadata.role","match":{"value":"user"}}]}).
func (c *QdrantClient) Search(collection string, vector []float32, limit int, threshold float64, filter map[string]any) ([]StoreSearchResult, error) {
	body := map[string]any{
		"vector":          vector,
		"limit":           limit,
		"score_threshold": threshold,
		"with_payload":    true,
	}
	if len(filter) > 0 {
		body["filter"] = filter
	}

	resp, err := c.post(fmt.Sprintf("/collections/%s/points/search", collection), body)
	if err != nil {
		return nil, err
	}

	var result struct {
		Result []StoreSearchResult `json:"result"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}

	return result.Result, nil
}

// KeywordSearch implements BM25 over a collection's payload text via the
// scroll API (Qdrant has no native BM25 without sparse vectors). In-memory
// per call keeps it simple; collections are small enough that this is fine.
func (c *QdrantClient) KeywordSearch(collection string, query string, limit int) ([]StoreSearchResult, error) {
	var docs []*bm25Doc
	payloads := map[string]map[string]any{}
	ids := map[string]any{}
	var offset any
	for {
		body := map[string]any{"limit": 1000, "with_payload": true, "with_vector": false}
		if offset != nil {
			body["offset"] = offset
		}
		resp, err := c.post(fmt.Sprintf("/collections/%s/points/scroll", collection), body)
		if err != nil {
			return nil, err
		}
		var result struct {
			Result struct {
				Points []struct {
					ID      any            `json:"id"`
					Payload map[string]any `json:"payload"`
				} `json:"points"`
				NextOffset any `json:"next_page_offset"`
			} `json:"result"`
		}
		if err := json.Unmarshal(resp, &result); err != nil {
			return nil, err
		}
		for _, p := range result.Result.Points {
			text, _ := p.Payload["text"].(string)
			id := fmt.Sprintf("%v", p.ID)
			docs = append(docs, &bm25Doc{id: id, text: text})
			payloads[id] = p.Payload
			ids[id] = p.ID
		}
		if result.Result.NextOffset == nil || len(result.Result.Points) == 0 {
			break
		}
		offset = result.Result.NextOffset
	}
	if len(docs) == 0 {
		return nil, nil
	}
	ranked := buildBM25(docs).Rank(query, limit)
	var out []StoreSearchResult
	for _, d := range ranked {
		out = append(out, StoreSearchResult{ID: ids[d.id], Score: 0, Payload: payloads[d.id]})
	}
	return out, nil
}

// DeletePoints deletes points by filter
func (c *QdrantClient) DeletePoints(collection string, filter map[string]any) error {
	body := map[string]any{
		"filter": filter,
	}
	_, err := c.post(fmt.Sprintf("/collections/%s/points/delete", collection), body)
	return err
}

// DeleteCollection drops an entire collection with all its points.
func (c *QdrantClient) DeleteCollection(name string) error {
	return c.delete(fmt.Sprintf("/collections/%s", name))
}

// SetPayload sets payload fields on points matching filter.
// An empty filter matches ALL points in the collection, which is exactly
// what backfill wants (tag every existing chunk without re-embedding).
func (c *QdrantClient) SetPayload(collection string, payload map[string]any, filter map[string]any) error {
	if filter == nil {
		filter = map[string]any{}
	}
	body := map[string]any{
		"payload": payload,
		"filter":  filter,
	}
	_, err := c.post(fmt.Sprintf("/collections/%s/points/payload", collection), body)
	return err
}

// Ping checks if Qdrant is reachable. Uses the short-timeout probe client so
// health checks and dashboard listings never hang on a blackholed host.
func (c *QdrantClient) Ping() error {
	resp, err := c.probeClient.Get(c.baseURL + "/collections")
	if err != nil {
		return unavailable(fmt.Errorf("qdrant unreachable: %w", err))
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return unavailable(err)
	}
	if resp.StatusCode >= 400 {
		return unavailable(fmt.Errorf("qdrant error %d: %s", resp.StatusCode, string(body)))
	}
	var result struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return fmt.Errorf("invalid qdrant response: %w", err)
	}
	if result.Status != "ok" {
		return fmt.Errorf("qdrant status: %s", result.Status)
	}
	return nil
}

// HTTP helpers

func (c *QdrantClient) delete(path string) error {
	req, err := http.NewRequest(http.MethodDelete, c.baseURL+path, nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return unavailable(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		err := fmt.Errorf("qdrant error %d: %s", resp.StatusCode, string(body))
		if resp.StatusCode >= 500 {
			return unavailable(err)
		}
		return err
	}
	return nil
}

func (c *QdrantClient) get(path string) ([]byte, error) {
	resp, err := c.httpClient.Get(c.baseURL + path)
	if err != nil {
		return nil, unavailable(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, unavailable(err)
	}
	if resp.StatusCode >= 400 {
		err := fmt.Errorf("qdrant error %d: %s", resp.StatusCode, string(body))
		if resp.StatusCode >= 500 {
			return nil, unavailable(err)
		}
		return nil, err
	}
	return body, nil
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
		return unavailable(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		err := fmt.Errorf("qdrant error %d: %s", resp.StatusCode, string(body))
		if resp.StatusCode >= 500 {
			return unavailable(err)
		}
		return err
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
		return nil, unavailable(err)
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, unavailable(err)
	}
	if resp.StatusCode >= 400 {
		err := fmt.Errorf("qdrant error %d: %s", resp.StatusCode, string(respBody))
		if resp.StatusCode >= 500 {
			return nil, unavailable(err)
		}
		return nil, err
	}
	return respBody, nil
}
