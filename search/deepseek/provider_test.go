package deepseek

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/search"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

func TestSearchCapsCountAndAppliesFreshness(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "Bearer ds-key" {
			t.Errorf("Authorization = %q", auth)
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"results":[]}`))
	}))
	defer srv.Close()

	st := settings.New("", settings.Settings{DeepSeekMaxCount: 3})
	p := New(Config{BaseURL: srv.URL, APIKey: "ds-key", Timeout: 5 * time.Second}, st)
	if _, err := p.Search(context.Background(), search.Query{
		Text: "opencode", Count: 10, FreshnessDays: 7, NeedContent: true,
	}); err != nil {
		t.Fatalf("Search: %v", err)
	}

	if got["count"] != float64(3) {
		t.Fatalf("count = %v, want 3 (settings cap)", got["count"])
	}
	if got["need_content"] != true {
		t.Fatalf("need_content = %v", got["need_content"])
	}
	q, _ := got["query"].(string)
	if !strings.HasPrefix(q, "opencode after:") {
		t.Fatalf("query = %q, want an after: date bound", q)
	}
	if _, ok := got["include_domains"]; ok {
		t.Fatal("deepseek request must not carry domains")
	}
}
