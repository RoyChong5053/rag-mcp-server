package search

import "testing"

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
	if parseDate("2026-10-17T00:00:00+00:00").IsZero() {
		t.Fatal("RFC3339 date should parse")
	}
	if parseDate("2026-10-17").IsZero() {
		t.Fatal("plain date should parse")
	}
	if !parseDate("not a date").IsZero() {
		t.Fatal("garbage should yield the zero time")
	}
	if !parseDate("").IsZero() {
		t.Fatal("empty should yield the zero time")
	}
}

func TestChainSkipsUnavailableProviders(t *testing.T) {
	c := NewChain(NewDeepSeek("https://example.invalid", "", 0))
	if len(c.Providers()) != 1 {
		t.Fatalf("expected 1 provider, got %d", len(c.Providers()))
	}
	if c.Providers()[0].Available() {
		t.Fatal("a provider without a key must not be available")
	}
	if c.Get("deepseek") == nil {
		t.Fatal("Get should find the provider by name")
	}
	if c.Get("nope") != nil {
		t.Fatal("Get should return nil for an unknown name")
	}
}
