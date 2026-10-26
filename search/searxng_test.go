package search

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"
)

// A live SearXNG answer carries two shapes that are not plain strings:
// answers is a list of objects and unresponsive_engines is a list of
// [engine, error] pairs. Decoding must survive both: one captcha'd engine
// otherwise takes the whole search down, and the results that did come back
// are perfectly usable.
func TestSearXNGSearchDecodesRealResponse(t *testing.T) {
	const body = `{
	  "query": "opencode",
	  "number_of_results": 0,
	  "results": [
	    {"url":"https://example.com/a","title":"A","content":"snippet a","engine":"bing","score":1.5,"category":"general","publishedDate":"2026-10-20T10:00:00+00:00"},
	    {"url":"https://example.com/b","title":"B","content":"snippet b","engine":"bing","score":0,"category":"general","publishedDate":null},
	    {"url":"","title":"no link","content":"","engine":"bing","score":0,"category":"general","publishedDate":null}
	  ],
	  "answers": [{"answer":"42","url":"https://example.com"}],
	  "unresponsive_engines": [["google","SearxEngineAccessDenied"],["duckduckgo","SearxEngineTooManyRequests"]]
	}`
	var gotQuery url.Values
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.Query()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	p := NewSearXNG(srv.URL, 5*time.Second)
	res, err := p.Search(context.Background(), Query{Text: "opencode", Count: 1})
	if err != nil {
		t.Fatalf("Search() must not fail on a real response: %v", err)
	}
	if got := gotQuery.Get("format"); got != "json" {
		t.Fatalf("format = %q, want json", got)
	}
	// Two results have a URL; Count=1 must cut them down.
	if len(res) != 1 {
		t.Fatalf("len(res) = %d, want 1 (count truncation)", len(res))
	}
	if res[0].URL != "https://example.com/a" {
		t.Fatalf("URL = %q, want the first result", res[0].URL)
	}
	if res[0].Provider != "searxng" {
		t.Fatalf("Provider = %q, want searxng", res[0].Provider)
	}
	if res[0].PublishedAt.IsZero() {
		t.Fatal("publishedDate should parse")
	}
	if res[0].Snippet != "snippet a" {
		t.Fatalf("Snippet = %q", res[0].Snippet)
	}
}

// A result without a score still has to be orderable: SearXNG does not rank
// the way the other providers do.
func TestSearXNGSearchPositionalScore(t *testing.T) {
	const body = `{"results":[{"url":"https://example.com/a","title":"A","score":0}],"answers":[],"unresponsive_engines":[]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	res, err := NewSearXNG(srv.URL, 5*time.Second).Search(context.Background(), Query{Text: "x", Count: 10})
	if err != nil {
		t.Fatalf("Search(): %v", err)
	}
	if len(res) != 1 || res[0].Score != 1.0 {
		t.Fatalf("res = %+v, want one result scored 1.0", res)
	}
}
