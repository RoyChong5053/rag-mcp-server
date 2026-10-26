package search

import "strings"

// Registry holds the providers this server was built with. It is not a
// failover chain: web_search resolves exactly the selected provider and
// calls it, or fails. The registry exists only to look a provider up by name
// and to report the names the dashboard may offer.
type Registry struct {
	providers []Provider
}

// NewRegistry builds a registry, skipping nil entries.
func NewRegistry(providers ...Provider) *Registry {
	r := &Registry{}
	for _, p := range providers {
		if p != nil {
			r.providers = append(r.providers, p)
		}
	}
	return r
}

// Providers returns the registered providers in build order.
func (r *Registry) Providers() []Provider {
	if r == nil {
		return nil
	}
	return r.providers
}

// Names lists the registered provider names.
func (r *Registry) Names() []string {
	if r == nil {
		return nil
	}
	out := make([]string, 0, len(r.providers))
	for _, p := range r.providers {
		out = append(out, p.Name())
	}
	return out
}

// Get returns the provider with the given name, or nil. The match is
// case-insensitive because the name comes from a settings file.
func (r *Registry) Get(name string) Provider {
	if r == nil {
		return nil
	}
	for _, p := range r.providers {
		if strings.EqualFold(p.Name(), name) {
			return p
		}
	}
	return nil
}
