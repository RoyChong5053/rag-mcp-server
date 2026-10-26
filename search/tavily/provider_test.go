package tavily

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/search"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// The provider reads its WebUI options on every call, and the request must
// carry exactly those values (plus the common query fields).
func TestSearchUsesLiveOptions(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-key" {
			t.Errorf("Authorization = %q", auth)
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"results":[]}`))
	}))
	defer srv.Close()

	st := settings.New("", settings.Settings{
		TavilySearchDepth: "advanced",
		TavilyCountry:     "SG",
	})
	p := New(Config{BaseURL: srv.URL, APIKey: "test-key", Timeout: 5 * time.Second}, st)
	if _, err := p.Search(context.Background(), search.Query{
		Text: "opencode", Count: 5, Topic: "general", FreshnessDays: 7, NeedContent: true,
	}); err != nil {
		t.Fatalf("Search: %v", err)
	}

	if got["search_depth"] != "advanced" {
		t.Fatalf("search_depth = %v", got["search_depth"])
	}
	if got["country"] != "SG" {
		t.Fatalf("country = %v", got["country"])
	}
	if got["time_range"] != "week" {
		t.Fatalf("time_range = %v", got["time_range"])
	}
	if got["include_raw_content"] != "markdown" {
		t.Fatalf("include_raw_content = %v", got["include_raw_content"])
	}
	if got["max_results"] != float64(5) {
		t.Fatalf("max_results = %v", got["max_results"])
	}
	// domains were removed from the tool; the request must not carry a
	// stale include_domains field.
	if _, ok := got["include_domains"]; ok {
		t.Fatal("include_domains must not be sent")
	}
}
