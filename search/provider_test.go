package search

import (
	"context"
	"testing"
)

func TestCanonicalizeURL(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"https://a.com/x?utm_source=mail&utm_medium=e&id=7", "https://a.com/x?id=7"},
		{"https://a.com/x#section", "https://a.com/x"},
		{"https://a.com/x/", "https://a.com/x"},
		{"HTTPS://A.com/x", "https://A.com/x"},
		{"https://a.com/x?fbclid=abc", "https://a.com/x"},
		{"https://a.com/x?ref=abc&keep=1", "https://a.com/x?keep=1"},
		{"", ""},
	}
	for _, tc := range cases {
		if got := CanonicalizeURL(tc.in); got != tc.want {
			t.Errorf("CanonicalizeURL(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestParseDate(t *testing.T) {
	if ParseDate("2026-10-17T00:00:00+00:00").IsZero() {
		t.Fatal("RFC3339 date should parse")
	}
	if ParseDate("2026-10-17").IsZero() {
		t.Fatal("plain date should parse")
	}
	if !ParseDate("not a date").IsZero() {
		t.Fatal("garbage should yield the zero time")
	}
	if !ParseDate("").IsZero() {
		t.Fatal("empty should yield the zero time")
	}
}

type stubProvider struct {
	name      string
	available bool
}

func (p stubProvider) Name() string    { return p.name }
func (p stubProvider) Available() bool { return p.available }
func (p stubProvider) Search(_ context.Context, _ Query) ([]Result, error) {
	return nil, nil
}

func TestRegistryLookup(t *testing.T) {
	r := NewRegistry(
		stubProvider{name: "deepseek", available: false},
		stubProvider{name: "searxng", available: true},
	)
	if len(r.Providers()) != 2 {
		t.Fatalf("expected 2 providers, got %d", len(r.Providers()))
	}
	if got := r.Names(); len(got) != 2 || got[0] != "deepseek" || got[1] != "searxng" {
		t.Fatalf("Names() = %v", got)
	}
	if p := r.Get("SEARXNG"); p == nil || p.Name() != "searxng" {
		t.Fatalf("Get should match case-insensitively, got %v", p)
	}
	if r.Get("nope") != nil {
		t.Fatal("Get should return nil for an unknown name")
	}
}
