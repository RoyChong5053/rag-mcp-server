package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// maxErrorBodyBytes caps how much of a provider's error body is read into the
// error message. Provider error bodies are usually a short JSON object, but
// some gateways return HTML pages.
const maxErrorBodyBytes = 4 << 10

// httpClient is the shared client for provider calls. Per-request timeouts are
// applied with context deadlines, so the client itself only needs a sane
// ceiling for slow/hanging bodies.
func httpClient(timeout time.Duration) *http.Client {
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	return &http.Client{Timeout: timeout}
}

// postJSON sends body as JSON to url with the given auth header and decodes
// the JSON response into out. A non-2xx status becomes an error carrying a
// truncated slice of the response body.
func postJSON(ctx context.Context, client *http.Client, url string, headers map[string]string, body any, out any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return fmt.Errorf("http %d: %s", resp.StatusCode, truncateRunes(strings.TrimSpace(string(data)), 500))
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("decode response: %w (body: %s)", err, truncateRunes(strings.TrimSpace(string(data)), 300))
	}
	return nil
}

// getJSON issues a GET and decodes the JSON response into out.
func getJSON(ctx context.Context, client *http.Client, url string, headers map[string]string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return fmt.Errorf("http %d: %s", resp.StatusCode, truncateRunes(strings.TrimSpace(string(data)), 500))
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("decode response: %w (body: %s)", err, truncateRunes(strings.TrimSpace(string(data)), 300))
	}
	return nil
}
