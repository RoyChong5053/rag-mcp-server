package firecrawl

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

func TestSearchUsesLiveGeoOptions(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"success":true,"data":{"web":[],"news":[]}}`))
	}))
	defer srv.Close()

	st := settings.New("", settings.Settings{
		FirecrawlCountry:  "sg",
		FirecrawlLocation: "Singapore",
	})
	p := New(Config{BaseURL: srv.URL, APIKey: "fc-key", Timeout: 5 * time.Second}, st)
	if _, err := p.Search(context.Background(), search.Query{
		Text: "opencode", Count: 7, NeedContent: true, FreshnessDays: 3,
	}); err != nil {
		t.Fatalf("Search: %v", err)
	}

	if got["country"] != "SG" {
		t.Fatalf("country = %v, want SG", got["country"])
	}
	if got["location"] != "Singapore" {
		t.Fatalf("location = %v", got["location"])
	}
	if got["limit"] != float64(7) {
		t.Fatalf("limit = %v", got["limit"])
	}
	if got["tbs"] != "qdr:w" {
		t.Fatalf("tbs = %v", got["tbs"])
	}
	if _, ok := got["scrapeOptions"]; !ok {
		t.Fatal("NeedContent should request scrapeOptions")
	}
	if _, ok := got["includeDomains"]; ok {
		t.Fatal("includeDomains must not be sent")
	}
}

// An empty settings file must not silently erase the configured country; the
// settings store is seeded from config, so defaults survive. Here we check
// the constructor fallback used when no store is wired.
func TestNewDefaultCountry(t *testing.T) {
	p := New(Config{APIKey: "k"}, nil)
	if got := p.options().Country; got != "MY" {
		t.Fatalf("default country = %q, want MY", got)
	}
}
