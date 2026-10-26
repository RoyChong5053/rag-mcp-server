package search

import (
	"testing"
	"time"
)

// TestParseDateFromText covers the best-effort date recovery used for
// providers that do not date their results (Firecrawl's web source).
func TestParseDateFromText(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string // "" = no date
	}{
		{"iso date", "# Headline\n\n2026-10-16\n\nBody text", "2026-10-16"},
		{"day month name", "16 Oct 2026, 09:10\n\nPETALING JAYA: Services", "2026-10-16"},
		{"full month name", "Published on October 16, 2026 9:47AM", "2026-10-16"},
		{"month abbrev with dot", "Oct. 16, 2026", "2026-10-16"},
		{"day first full month", "23 September 2026", "2026-09-23"},
		{"no date", "# Specs\n\n| CPU | 20-core |\n", ""},
		{"version string is not a date", "build v2026-10-16-rc1 released", ""},
		{"url path is not a date", "see https://site.com/2026/10/16/story", ""},
		{"impossible day", "31 February 2026", ""},
		{"out of range year", "16 Oct 1899", ""},
		{"empty", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseDateFromText(tc.in)
			if tc.want == "" {
				if !got.IsZero() {
					t.Fatalf("ParseDateFromText(%q) = %v, want no date", tc.in, got)
				}
				return
			}
			if got.IsZero() {
				t.Fatalf("ParseDateFromText(%q) = no date, want %s", tc.in, tc.want)
			}
			if got.Format("2006-01-02") != tc.want {
				t.Fatalf("ParseDateFromText(%q) = %s, want %s", tc.in, got.Format("2006-01-02"), tc.want)
			}
			if got.Location() != time.UTC {
				t.Fatalf("ParseDateFromText(%q) is not UTC: %v", tc.in, got.Location())
			}
		})
	}
}

// TestParseDateFromTextIgnoresLateDates makes sure a date far down the page
// (usually a footer, an unrelated article, a copyright line) is not mistaken
// for the publication date.
func TestParseDateFromTextIgnoresLateDates(t *testing.T) {
	body := "# Headline\n\n" + string(make([]byte, 0))
	// Pad past the scan window with filler.
	for i := 0; i < 200; i++ {
		body += "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
	}
	body += "\n\n2026-10-16\n"
	if got := ParseDateFromText(body); !got.IsZero() {
		t.Fatalf("a date below the scan window was picked up: %v", got)
	}
}
