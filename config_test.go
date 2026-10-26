package main

import (
	"os"
	"path/filepath"
	"testing"
)

// The provider configs are split per module, but the YAML stays flat:
// base_url/api_key/api_key_env are shared, and each provider owns its own
// behavioural keys.
func TestLoadConfigSplitsProviderOptions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	const body = `
search:
  enabled: true
  providers: [tavily, searxng]
  store: false
  deepseek:
    base_url: "https://ds.example"
    api_key_env: "DS_KEY"
    max_count: 20
  tavily:
    base_url: "https://tv.example"
    api_key: "tv-inline"
    search_depth: "advanced"
    country: "SG"
  firecrawl:
    country: "MY"
    location: "Kuala Lumpur"
  searxng:
    base_url: "http://127.0.0.1:8899"
    language: "zh-CN"
    categories: "general,news"
    safesearch: 2
    engines: [bing, brave]
`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Search.Store {
		t.Fatal("store should be false")
	}
	if cfg.Search.DeepSeek.MaxCount != 20 {
		t.Fatalf("deepseek.max_count = %d, want 20", cfg.Search.DeepSeek.MaxCount)
	}
	if cfg.Search.DeepSeek.BaseURL != "https://ds.example" {
		t.Fatalf("deepseek.base_url = %q", cfg.Search.DeepSeek.BaseURL)
	}
	if cfg.Search.Tavily.SearchDepth != "advanced" || cfg.Search.Tavily.Country != "SG" {
		t.Fatalf("tavily = %+v", cfg.Search.Tavily)
	}
	if cfg.Search.Tavily.Key() != "tv-inline" {
		t.Fatalf("tavily key = %q", cfg.Search.Tavily.Key())
	}
	if cfg.Search.Firecrawl.Location != "Kuala Lumpur" {
		t.Fatalf("firecrawl.location = %q", cfg.Search.Firecrawl.Location)
	}
	if cfg.Search.SearXNG.SafeSearch != 2 || len(cfg.Search.SearXNG.Engines) != 2 {
		t.Fatalf("searxng = %+v", cfg.Search.SearXNG)
	}
}

// Missing keys must keep the built-in defaults (Store defaults to true).
func TestLoadConfigKeepsSearchDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("search:\n  providers: [searxng]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if !cfg.Search.Store {
		t.Fatal("store should default to true")
	}
	if cfg.Search.DeepSeek.MaxCount != 50 {
		t.Fatalf("deepseek.max_count = %d, want 50", cfg.Search.DeepSeek.MaxCount)
	}
	if cfg.Search.Tavily.SearchDepth != "basic" {
		t.Fatalf("tavily.search_depth = %q, want basic", cfg.Search.Tavily.SearchDepth)
	}
	if cfg.Search.SearXNG.Categories != "general" {
		t.Fatalf("searxng.categories = %q, want general", cfg.Search.SearXNG.Categories)
	}
}
