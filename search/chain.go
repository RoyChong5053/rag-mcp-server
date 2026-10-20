package search

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// Chain is an ordered list of providers tried one after another. It is the
// failover mechanism: the first configured+available provider answers, and
// the next one is tried only if the previous fails.
//
// The order matters and is configurable (config.yaml: search.providers). The
// default puts the cheapest, most capable provider first.
type Chain struct {
	providers []Provider
}

// NewChain builds a chain from the given providers, skipping nil entries.
func NewChain(providers ...Provider) *Chain {
	c := &Chain{}
	for _, p := range providers {
		if p != nil {
			c.providers = append(c.providers, p)
		}
	}
	return c
}

// Providers returns the chain members in order.
func (c *Chain) Providers() []Provider {
	if c == nil {
		return nil
	}
	return c.providers
}

// Names lists the configured provider names in order.
func (c *Chain) Names() []string {
	out := make([]string, 0, len(c.providers))
	for _, p := range c.providers {
		out = append(out, p.Name())
	}
	return out
}

// Search runs the query against the chain. It returns the first non-empty
// result set. If every provider fails, the last error is returned along with
// the list of providers that were tried.
func (c *Chain) Search(ctx context.Context, q Query) ([]Result, error) {
	if c == nil || len(c.providers) == 0 {
		return nil, errors.New("no search providers configured")
	}
	tried := make([]string, 0, len(c.providers))
	var lastErr error
	for _, p := range c.providers {
		if !p.Available() {
			tried = append(tried, p.Name()+"(unconfigured)")
			continue
		}
		results, err := p.Search(ctx, q)
		if err != nil {
			lastErr = err
			tried = append(tried, p.Name()+"(error)")
			continue
		}
		if len(results) == 0 {
			// The provider worked but found nothing. That is a real answer,
			// but a broader provider might still have hits, so keep going
			// and only fall back to this empty set if nobody else answers.
			tried = append(tried, p.Name()+"(empty)")
			if lastErr == nil {
				lastErr = nil
			}
			continue
		}
		for i := range results {
			if results[i].Provider == "" {
				results[i].Provider = p.Name()
			}
		}
		return results, nil
	}
	if lastErr != nil {
		return nil, fmt.Errorf("all search providers failed (tried: %s): %w", strings.Join(tried, ", "), lastErr)
	}
	return nil, nil
}

// Fetch retrieves one page, trying each provider that supports fetching.
func (c *Chain) Fetch(ctx context.Context, url string) (Page, error) {
	if c == nil || len(c.providers) == 0 {
		return Page{}, errors.New("no search providers configured")
	}
	tried := make([]string, 0, len(c.providers))
	var lastErr error
	for _, p := range c.providers {
		if !p.Available() {
			continue
		}
		page, err := p.Fetch(ctx, url)
		if err != nil {
			if errors.Is(err, ErrUnsupported) {
				tried = append(tried, p.Name()+"(no fetch)")
				continue
			}
			lastErr = err
			tried = append(tried, p.Name()+"(error)")
			continue
		}
		if strings.TrimSpace(page.Content) == "" {
			tried = append(tried, p.Name()+"(empty)")
			continue
		}
		return page, nil
	}
	if lastErr != nil {
		return Page{}, fmt.Errorf("all fetch providers failed (tried: %s): %w", strings.Join(tried, ", "), lastErr)
	}
	return Page{}, fmt.Errorf("no provider could fetch the page (tried: %s)", strings.Join(tried, ", "))
}

// Get returns the provider with the given name, or nil.
func (c *Chain) Get(name string) Provider {
	if c == nil {
		return nil
	}
	for _, p := range c.providers {
		if strings.EqualFold(p.Name(), name) {
			return p
		}
	}
	return nil
}
