// Package search wraps external web-search providers behind one interface.
//
// The rag-mcp-server uses it for one job: pull live web pages in and turn them
// into vectors. The provider details (query operators, result shapes, credit
// rules) differ wildly between services, so everything service-specific lives
// behind Provider and the caller only ever sees Result.
package search

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// ErrUnsupported is returned by a provider that cannot do the requested
// operation (e.g. SearXNG has no fetch endpoint). Callers treat it as "this
// provider cannot serve that need" and move on to the next one.
var ErrUnsupported = errors.New("operation not supported by provider")

// Query is the provider-agnostic search request. Each provider maps what it
// can honour and ignores the rest.
type Query struct {
	// Text is the search query. Providers that support operators (site:,
	// after:, lang:) receive it verbatim so callers can steer the search.
	Text string
	// Count is the desired number of results (1-50 depending on provider).
	Count int
	// NeedContent asks for the full page text alongside the snippet. Most
	// providers can return it in the same request; SearXNG cannot.
	NeedContent bool
	// FreshnessDays restricts results to the last N days (0 = no limit).
	FreshnessDays int
	// Domains restricts results to these hosts (empty = no restriction).
	Domains []string
	// Lang is an ISO 639-1 code used to bias/filter results by language.
	Lang string
	// Topic is a provider-specific bucket: "general" (default) or "news".
	Topic string
}

// Result is one search hit, normalized across providers.
type Result struct {
	Title   string
	URL     string
	Snippet string
	// Summary is a model-readable medium-length summary. DeepSeek returns
	// one; other providers leave it empty and the snippet carries the load.
	Summary string
	// Content is the full page text (markdown) when the provider returned
	// it. Empty when the provider only gives snippets.
	Content string
	// Chunks holds provider-extracted relevant passages (Tavily's
	// chunks_per_source). They are already sized for retrieval.
	Chunks []string
	// PublishedAt is the publication time when known, else zero.
	PublishedAt time.Time
	// Authority is a 0-4 source-quality tier when known (DeepSeek), else -1.
	Authority int
	// Score is the provider's own relevance score.
	Score float64
	// Provider is the name of the provider that produced this result.
	Provider string
}

// Page is a fetched web page.
type Page struct {
	Title   string
	URL     string
	Content string
}

// Provider is one external search service.
type Provider interface {
	// Name is the short identifier used in logs, metadata and config.
	Name() string
	// Available reports whether the provider is configured and usable
	// (e.g. an API key is present). Unavailable providers are skipped by
	// the chain instead of failing the call.
	Available() bool
	// Search runs the query. It returns an empty slice when the provider
	// worked but found nothing; that is not an error.
	Search(ctx context.Context, q Query) ([]Result, error)
	// Fetch retrieves one page as markdown. Providers without that ability
	// return ErrUnsupported.
	Fetch(ctx context.Context, url string) (Page, error)
}

// CanonicalizeURL strips tracking parameters and fragments so the same article
// reached through different links produces the same identity. It is exported
// so the ingest path can record a stable URL alongside the original one.
func CanonicalizeURL(raw string) string {
	s := strings.TrimSpace(raw)
	if s == "" {
		return ""
	}
	// Drop the fragment outright.
	if i := strings.IndexByte(s, '#'); i >= 0 {
		s = s[:i]
	}
	// Drop tracking query parameters. Kept as a suffix list so a query
	// string that carries meaning (e.g. "id=123") survives.
	drop := map[string]bool{
		"utm_source": true, "utm_medium": true, "utm_campaign": true,
		"utm_term": true, "utm_content": true, "utm_id": true,
		"utm_name": true, "utm_reader": true, "utm_social": true,
		"utm_brand": true, "utm_pubreferrer": true, "utm_swu": true,
		"gclid": true, "fbclid": true, "msclkid": true, "dclid": true,
		"yclid": true, "twclid": true, "igshid": true, "mc_cid": true,
		"mc_eid": true, "_hsenc": true, "_hsmi": true, "ref": true,
		"ref_src": true, "spm": true, "from_source": true,
	}
	if i := strings.IndexByte(s, '?'); i >= 0 {
		base, query := s[:i], s[i+1:]
		keep := make([]string, 0, 8)
		for _, part := range strings.Split(query, "&") {
			if part == "" {
				continue
			}
			key := part
			if j := strings.IndexByte(part, '='); j >= 0 {
				key = part[:j]
			}
			if drop[strings.ToLower(key)] {
				continue
			}
			keep = append(keep, part)
		}
		if len(keep) == 0 {
			s = base
		} else {
			s = base + "?" + strings.Join(keep, "&")
		}
	}
	// Normalize the scheme casing and a trailing slash on the path.
	s = strings.Replace(s, "HTTP://", "http://", 1)
	s = strings.Replace(s, "HTTPS://", "https://", 1)
	if i := strings.IndexByte(s, '?'); i >= 0 {
		if strings.HasSuffix(s[:i], "/") {
			s = s[:i-1] + s[i:]
		}
	} else if strings.HasSuffix(s, "/") && !strings.HasSuffix(s, "://") {
		s = strings.TrimSuffix(s, "/")
	}
	return s
}

// hostOf returns the lowercase hostname of a URL, or "" when there is none.
func hostOf(raw string) string {
	s := raw
	if i := strings.Index(s, "://"); i >= 0 {
		s = s[i+3:]
	}
	if i := strings.IndexAny(s, "/?#"); i >= 0 {
		s = s[:i]
	}
	if i := strings.IndexByte(s, '@'); i >= 0 {
		s = s[i+1:]
	}
	if i := strings.IndexByte(s, ':'); i >= 0 {
		s = s[:i]
	}
	return strings.ToLower(s)
}

// joinTitle builds a display title from a provider's title and site name
// when the title is missing.
func joinTitle(title, site string) string {
	t := strings.TrimSpace(title)
	if t != "" {
		return t
	}
	if site = strings.TrimSpace(site); site != "" {
		return site
	}
	return ""
}

// truncateRunes cuts s to at most n runes, appending an ellipsis marker when
// it had to cut. Used to keep provider payloads bounded.
func truncateRunes(s string, n int) string {
	if n <= 0 {
		return s
	}
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

// parseDate tries the date layouts providers use. It returns the zero time
// when nothing parses.
func parseDate(s string) time.Time {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}
	}
	layouts := []string{
		time.RFC3339,
		time.RFC3339Nano,
		"2006-01-02T15:04:05Z0700",
		"2006-01-02T15:04:05",
		"2006-01-02 15:04:05",
		"2006-01-02",
		"Mon, 02 Jan 2006 15:04:05 MST",
		"Mon, 02 Jan 2006 15:04:05 -0700",
		"Mon, 2 Jan 2006 15:04:05 -0700",
		"2006/01/02",
	}
	for _, l := range layouts {
		if t, err := time.Parse(l, s); err == nil {
			return t.UTC()
		}
	}
	return time.Time{}
}

// errProvider builds a provider-scoped error.
func errProvider(name string, err error) error {
	return fmt.Errorf("%s: %w", name, err)
}
