package settings

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
)

// Settings holds runtime-adjustable search defaults. Persisted to
// settings.json and edited through the WebUI; every frontend (ST plugin,
// opencode, raw HTTP callers) reads these instead of hardcoding values.
type Settings struct {
	// ActiveBackend is kept for compatibility and always normalizes to
	// "vectra" (single-backend server).
	ActiveBackend string `json:"active_backend"`
	// DefaultCollectionQdrant is a legacy alias: honored when
	// DefaultCollectionVectra is empty.
	DefaultCollectionQdrant string `json:"default_collection_qdrant"`
	// DefaultCollectionVectra is the default scope for omitted collection_id.
	// Empty means "search all enabled collections".
	DefaultCollectionVectra string `json:"default_collection_vectra"`
	// FailoverEnabled is deprecated (no second backend) and ignored.
	FailoverEnabled bool `json:"failover_enabled"`
	// DefaultTopK is how many results search_memory returns when top_k is omitted.
	DefaultTopK int `json:"default_top_k"`
	// DefaultThreshold is the minimum similarity when threshold is omitted.
	DefaultThreshold float64 `json:"default_threshold"`
	// RerankEnabled gates the rerank stage for search_memory.
	RerankEnabled bool `json:"rerank_enabled"`
	// RerankRecall is the vector over-fetch count fed to the reranker.
	RerankRecall int `json:"rerank_recall"`
	// QueryMaxChars caps the query text (runes) before it is embedded and
	// reranked; the head is kept (the actual question usually precedes pasted
	// source material). 0 = unlimited.
	QueryMaxChars int `json:"query_max_chars"`
	// DocMaxChars caps each document handed to the reranker (runes).
	// 0 = unlimited.
	DocMaxChars int `json:"doc_max_chars"`
	// EmbedBatchSize caps texts per embedding POST from index jobs when the
	// GPU path is active. 0 = auto (GPU 32, CPU fallback capped to 4). A
	// positive value overrides the GPU size; the CPU fallback still caps it.
	EmbedBatchSize int `json:"embed_batch_size"`
	// EmbedProviders lists every embed endpoint the server can talk to
	// (one-api fan-out, openrouter, ...). Switching providers only affects
	// new writes: existing collections keep the provenance recorded at
	// vectorize time and the engine refuses to mix vector spaces.
	EmbedProviders []EmbedProvider `json:"embed_providers,omitempty"`
	// ActiveEmbedProvider selects which EmbedProviders entry embedQuery and
	// index jobs use. Empty = the built-in one-api config (legacy default).
	ActiveEmbedProvider string `json:"active_embed_provider,omitempty"`
	// BM25Enabled adds a keyword-retrieval leg to searches; results merge
	// with vector hits via RRF before rerank. Off = pure vector+rerank.
	BM25Enabled bool `json:"bm25_enabled"`
	// FusionMethod picks how the vector and BM25 lists merge: "rrf" (default)
	// or "weighted".
	FusionMethod string `json:"fusion_method"`
	// SearchProviders lists the web-search provider modules this server was
	// built with. It is seeded from config.yaml and restored on every load;
	// it is not editable, so the dashboard reports what the server can
	// actually call instead of inventing providers that would fail.
	SearchProviders []string `json:"search_providers,omitempty"`
	// DefaultSearchProvider is the provider web_search uses. There is no
	// failover chain: an empty value is a configuration error, and a
	// selected provider that is unavailable fails the call instead of
	// switching to another one.
	DefaultSearchProvider string `json:"default_search_provider"`
	// DefaultSearchCount is how many results web_search asks for when the
	// caller does not specify. 0 = use the server default.
	DefaultSearchCount int `json:"default_search_count"`
	// DefaultSearchFreshnessDays is the default age limit on results.
	// 0 = no limit.
	DefaultSearchFreshnessDays int `json:"default_search_freshness_days"`
	// DefaultSearchTopic is "general" or "news".
	DefaultSearchTopic string `json:"default_search_topic"`
	// DefaultSearchLang is the ISO 639-1 language bias. Empty = provider
	// default.
	DefaultSearchLang string `json:"default_search_lang"`
	// DefaultSearchStore files what a search returned into the web
	// collection. False searches without filing anything (and the
	// providers are not asked for full text).
	DefaultSearchStore bool `json:"default_search_store"`
	// DefaultSearchIngest is how many of the returned results get filed.
	// 0 = all of them.
	DefaultSearchIngest int `json:"default_search_ingest"`

	// DeepSeekMaxCount caps the result count requested from DeepSeek
	// (the API allows 1-50).
	DeepSeekMaxCount int `json:"deepseek_max_count"`
	// TavilySearchDepth is "basic", "advanced" or "fast".
	TavilySearchDepth string `json:"tavily_search_depth"`
	// TavilyCountry is the ISO 3166-1 alpha-2 geo bias (general topic only).
	TavilyCountry string `json:"tavily_country"`
	// FirecrawlCountry / FirecrawlLocation bias Firecrawl results.
	FirecrawlCountry  string `json:"firecrawl_country"`
	FirecrawlLocation string `json:"firecrawl_location"`
	// SearXNGLanguage / SearXNGCategories restrict the SearXNG search.
	// Empty language falls back to DefaultSearchLang.
	SearXNGLanguage   string `json:"searxng_language"`
	SearXNGCategories string `json:"searxng_categories"`
	// SearXNGSafeSearch is 0 (off), 1 (moderate) or 2 (strict).
	SearXNGSafeSearch int `json:"searxng_safesearch"`
	// SearXNGEngines restricts the search to these engines. Empty = all
	// engines enabled in the instance's settings.yml.
	SearXNGEngines []string `json:"searxng_engines,omitempty"`
}

// EmbedProvider describes one embedding endpoint. APIKeyEnv names an
// environment variable holding the key; empty falls back to the one-api key.
type EmbedProvider struct {
	ID        string `json:"id"`
	BaseURL   string `json:"base_url"`
	BackupURL string `json:"backup_url,omitempty"`
	Model     string `json:"model"`
	Dim       int    `json:"dim,omitempty"`
	Distance  string `json:"distance,omitempty"`
	APIKeyEnv string `json:"api_key_env,omitempty"`
}

// Patch carries optional settings updates; nil means "leave unchanged".
// Pointers distinguish an explicit zero from an omitted field.
type Patch struct {
	ActiveBackend           *string          `json:"active_backend"`
	DefaultCollectionQdrant *string          `json:"default_collection_qdrant"`
	DefaultCollectionVectra *string          `json:"default_collection_vectra"`
	FailoverEnabled         *bool            `json:"failover_enabled"`
	DefaultTopK             *int             `json:"default_top_k"`
	DefaultThreshold        *float64         `json:"default_threshold"`
	RerankEnabled           *bool            `json:"rerank_enabled"`
	RerankRecall            *int             `json:"rerank_recall"`
	QueryMaxChars           *int             `json:"query_max_chars"`
	DocMaxChars             *int             `json:"doc_max_chars"`
	EmbedBatchSize          *int             `json:"embed_batch_size"`
	EmbedProviders          *[]EmbedProvider `json:"embed_providers"`
	ActiveEmbedProvider     *string          `json:"active_embed_provider"`
	BM25Enabled             *bool            `json:"bm25_enabled"`
	FusionMethod            *string          `json:"fusion_method"`
	DefaultSearchProvider   *string          `json:"default_search_provider"`
	DefaultSearchCount      *int             `json:"default_search_count"`

	DefaultSearchFreshnessDays *int      `json:"default_search_freshness_days"`
	DefaultSearchTopic         *string   `json:"default_search_topic"`
	DefaultSearchLang          *string   `json:"default_search_lang"`
	DefaultSearchStore         *bool     `json:"default_search_store"`
	DefaultSearchIngest        *int      `json:"default_search_ingest"`
	DeepSeekMaxCount           *int      `json:"deepseek_max_count"`
	TavilySearchDepth          *string   `json:"tavily_search_depth"`
	TavilyCountry              *string   `json:"tavily_country"`
	FirecrawlCountry           *string   `json:"firecrawl_country"`
	FirecrawlLocation          *string   `json:"firecrawl_location"`
	SearXNGLanguage            *string   `json:"searxng_language"`
	SearXNGCategories          *string   `json:"searxng_categories"`
	SearXNGSafeSearch          *int      `json:"searxng_safesearch"`
	SearXNGEngines             *[]string `json:"searxng_engines"`
}

// Store is a concurrency-safe runtime settings store backed by a JSON file.
type Store struct {
	mu   sync.RWMutex
	path string
	data Settings
}

// New builds a store seeded with base, then overlays path if it exists.
// Missing fields in the file keep their base value; a missing file is fine.
func New(path string, base Settings) *Store {
	s := &Store{path: path, data: base}
	s.load()
	return s
}

func (s *Store) load() {
	if s.path == "" {
		return
	}
	raw, err := os.ReadFile(s.path)
	if err != nil {
		return
	}
	// SearchProviders reports what this build actually registered. It is
	// seeded from config.yaml, never from settings.json, so a stale file
	// cannot make the dashboard offer a provider that no longer exists.
	registered := append([]string(nil), s.data.SearchProviders...)
	// Pre-dual-backend files used a single "default_collection" key, which was
	// the qdrant scope. Migrate it to default_collection_qdrant so existing
	// settings.json files keep working without manual edits.
	var legacy struct {
		DefaultCollection *string `json:"default_collection"`
	}
	_ = json.Unmarshal(raw, &legacy)
	_ = json.Unmarshal(raw, &s.data)
	s.data.SearchProviders = registered
	if s.data.DefaultCollectionQdrant == "" && legacy.DefaultCollection != nil {
		s.data.DefaultCollectionQdrant = *legacy.DefaultCollection
	}
}

// Get returns a snapshot of the current settings (safe for concurrent use).
func (s *Store) Get() Settings {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.data
}

// Update applies a partial patch and persists the result. A write failure is
// returned but the in-memory value is already updated, so the running service
// stays consistent with what the operator just saw.
func (s *Store) Update(p Patch) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if p.ActiveBackend != nil {
		s.data.ActiveBackend = "vectra"
	}
	if p.DefaultCollectionQdrant != nil {
		s.data.DefaultCollectionQdrant = *p.DefaultCollectionQdrant
	}
	if p.DefaultCollectionVectra != nil {
		s.data.DefaultCollectionVectra = *p.DefaultCollectionVectra
	}
	if p.FailoverEnabled != nil {
		s.data.FailoverEnabled = *p.FailoverEnabled
	}
	if p.DefaultTopK != nil {
		s.data.DefaultTopK = *p.DefaultTopK
	}
	if p.DefaultThreshold != nil {
		s.data.DefaultThreshold = *p.DefaultThreshold
	}
	if p.RerankEnabled != nil {
		s.data.RerankEnabled = *p.RerankEnabled
	}
	if p.RerankRecall != nil {
		s.data.RerankRecall = *p.RerankRecall
	}
	if p.QueryMaxChars != nil {
		s.data.QueryMaxChars = *p.QueryMaxChars
	}
	if p.DocMaxChars != nil {
		s.data.DocMaxChars = *p.DocMaxChars
	}
	if p.EmbedBatchSize != nil {
		s.data.EmbedBatchSize = *p.EmbedBatchSize
	}
	if p.EmbedProviders != nil {
		s.data.EmbedProviders = *p.EmbedProviders
	}
	if p.ActiveEmbedProvider != nil {
		s.data.ActiveEmbedProvider = *p.ActiveEmbedProvider
	}
	if p.BM25Enabled != nil {
		s.data.BM25Enabled = *p.BM25Enabled
	}
	if p.FusionMethod != nil {
		s.data.FusionMethod = *p.FusionMethod
	}
	if p.DefaultSearchProvider != nil {
		s.data.DefaultSearchProvider = strings.TrimSpace(*p.DefaultSearchProvider)
	}
	if p.DefaultSearchCount != nil {
		if *p.DefaultSearchCount < 0 {
			s.data.DefaultSearchCount = 0
		} else {
			s.data.DefaultSearchCount = *p.DefaultSearchCount
		}
	}
	if p.DefaultSearchFreshnessDays != nil {
		s.data.DefaultSearchFreshnessDays = nonNegative(*p.DefaultSearchFreshnessDays)
	}
	if p.DefaultSearchTopic != nil {
		s.data.DefaultSearchTopic = strings.TrimSpace(*p.DefaultSearchTopic)
	}
	if p.DefaultSearchLang != nil {
		s.data.DefaultSearchLang = strings.TrimSpace(*p.DefaultSearchLang)
	}
	if p.DefaultSearchStore != nil {
		s.data.DefaultSearchStore = *p.DefaultSearchStore
	}
	if p.DefaultSearchIngest != nil {
		s.data.DefaultSearchIngest = nonNegative(*p.DefaultSearchIngest)
	}
	if p.DeepSeekMaxCount != nil {
		v := *p.DeepSeekMaxCount
		if v < 0 {
			v = 0
		}
		if v > 50 {
			v = 50
		}
		s.data.DeepSeekMaxCount = v
	}
	if p.TavilySearchDepth != nil {
		s.data.TavilySearchDepth = normalizeSearchDepth(*p.TavilySearchDepth)
	}
	if p.TavilyCountry != nil {
		s.data.TavilyCountry = upperCountry(*p.TavilyCountry)
	}
	if p.FirecrawlCountry != nil {
		s.data.FirecrawlCountry = upperCountry(*p.FirecrawlCountry)
	}
	if p.FirecrawlLocation != nil {
		s.data.FirecrawlLocation = strings.TrimSpace(*p.FirecrawlLocation)
	}
	if p.SearXNGLanguage != nil {
		s.data.SearXNGLanguage = strings.TrimSpace(*p.SearXNGLanguage)
	}
	if p.SearXNGCategories != nil {
		s.data.SearXNGCategories = strings.TrimSpace(*p.SearXNGCategories)
	}
	if p.SearXNGSafeSearch != nil {
		v := *p.SearXNGSafeSearch
		if v < 0 {
			v = 0
		}
		if v > 2 {
			v = 2
		}
		s.data.SearXNGSafeSearch = v
	}
	if p.SearXNGEngines != nil {
		s.data.SearXNGEngines = cleanList(*p.SearXNGEngines)
	}

	return s.save()
}

func nonNegative(v int) int {
	if v < 0 {
		return 0
	}
	return v
}

func normalizeSearchDepth(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "advanced":
		return "advanced"
	case "fast":
		return "fast"
	default:
		return "basic"
	}
}

func upperCountry(v string) string {
	return strings.ToUpper(strings.TrimSpace(v))
}

func cleanList(in []string) []string {
	out := make([]string, 0, len(in))
	seen := make(map[string]bool, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}

func (s *Store) save() error {
	if s.path == "" {
		return nil
	}
	data, err := json.MarshalIndent(s.data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, data, 0644)
}
