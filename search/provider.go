// Package search holds the provider-agnostic web-search contract: the query
// the tool layer sends, the result it gets back, and the small registry that
// maps a provider name to its module.
//
// Each provider lives in its own subpackage (deepseek, tavily, firecrawl,
// searxng) and owns its URL, authentication, request body, response shape,
// parameters and error mapping. The APIs change independently, so no provider
// logic is shared here: a change to Tavily's payload touches only
// search/tavily.
package search

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Query is the provider-agnostic request built by web_search. Each provider
// maps the fields it can honour and ignores the rest. Provider-specific
// knobs (Tavily's search_depth, SearXNG's categories, ...) are deliberately
// absent: each module reads its own settings.
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

// Provider is one external search service. Each implementation lives in its
// own subpackage and is wired in main.go from config.
type Provider interface {
	// Name is the short identifier used in logs, metadata and settings.
	Name() string
	// Available reports whether the provider is configured and usable
	// (e.g. an API key is present). An unavailable provider is never
	// silently replaced by another one.
	Available() bool
	// Search runs the query. It returns an empty slice when the provider
	// worked but found nothing; that is not an error.
	Search(ctx context.Context, q Query) ([]Result, error)
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

// JoinTitle builds a display title from a provider's title and site name when
// the title is missing. Providers use it to normalize their own response.
func JoinTitle(title, site string) string {
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
// it had to cut. Used to keep provider payloads bounded in error messages.
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

// ParseDate tries the date layouts providers use. It returns the zero time
// when nothing parses.
func ParseDate(s string) time.Time {
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

// ProviderError builds a provider-scoped error.
func ProviderError(name string, err error) error {
	return fmt.Errorf("%s: %w", name, err)
}

var (
	isoDateRe = regexp.MustCompile(`\d{4}-\d{2}-\d{2}`)
	dayMonRe  = regexp.MustCompile(`\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b`)
	monDayRe  = regexp.MustCompile(`\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b`)
)

var monthNames = map[string]time.Month{
	"jan": time.January, "january": time.January,
	"feb": time.February, "february": time.February,
	"mar": time.March, "march": time.March,
	"apr": time.April, "april": time.April,
	"may": time.May,
	"jun": time.June, "june": time.June,
	"jul": time.July, "july": time.July,
	"aug": time.August, "august": time.August,
	"sep": time.September, "sept": time.September, "september": time.September,
	"oct": time.October, "october": time.October,
	"nov": time.November, "november": time.November,
	"dec": time.December, "december": time.December,
}

// ParseDateFromText makes a best effort at finding a publication date in the
// head of a page body.
//
// Some search sources do not date their results at all — Firecrawl's web
// source never does — yet the page itself usually carries the date in its
// first lines. Recovering it there is worth doing: a reader who cannot tell
// yesterday's news from today's is reading noise.
//
// It is deliberately conservative. Only the head of the text is scanned, a
// match has to look like a date rather than any number, and a year outside
// 2000-2099 is rejected. Nothing is guessed when nothing matches: a wrong
// date is worse than no date.
func ParseDateFromText(text string) time.Time {
	text = strings.TrimSpace(text)
	if text == "" {
		return time.Time{}
	}
	head := text
	if len(head) > 1500 {
		head = head[:1500]
	}

	for _, loc := range isoDateRe.FindAllStringIndex(head, -1) {
		if !dateBoundaryOK(head, loc[0], loc[1]) {
			continue
		}
		t, err := time.Parse("2006-01-02", head[loc[0]:loc[1]])
		if err != nil || t.Year() < 2000 || t.Year() > 2099 {
			continue
		}
		return t.UTC()
	}

	if m := dayMonRe.FindStringSubmatch(head); m != nil {
		if t, ok := monthDayYear(m[2], m[1], m[3]); ok {
			return t
		}
	}
	if m := monDayRe.FindStringSubmatch(head); m != nil {
		if t, ok := monthDayYear(m[1], m[2], m[3]); ok {
			return t
		}
	}
	return time.Time{}
}

// dateBoundaryOK rejects a numeric match that is part of a longer token, so
// "v2026-10-16-rc1" or "/2026/10/16/" is not read as a date.
func dateBoundaryOK(s string, start, end int) bool {
	bad := func(c byte) bool {
		return c == '-' || c == '/' || c == '.' || c == '_' || c == ':'
	}
	if start > 0 && (bad(s[start-1]) || isDigit(s[start-1])) {
		return false
	}
	if end < len(s) && (bad(s[end]) || isDigit(s[end])) {
		return false
	}
	return true
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

// monthDayYear turns a month name, a day and a year into a date.
func monthDayYear(mon, day, year string) (time.Time, bool) {
	m, ok := monthNames[strings.ToLower(strings.TrimSuffix(mon, "."))]
	if !ok {
		return time.Time{}, false
	}
	d, err := strconv.Atoi(day)
	if err != nil || d < 1 || d > 31 {
		return time.Time{}, false
	}
	y, err := strconv.Atoi(year)
	if err != nil || y < 2000 || y > 2099 {
		return time.Time{}, false
	}
	t := time.Date(y, m, d, 0, 0, 0, 0, time.UTC)
	// Reject impossible dates like 31 February, which time.Date would roll over.
	if t.Day() != d || t.Month() != m {
		return time.Time{}, false
	}
	return t, true
}
