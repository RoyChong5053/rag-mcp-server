package main

import (
	"os"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/RoyChong5053/rag-mcp-server/engine"
)

type Config struct {
	Server       ServerConfig    `yaml:"server"`
	Admin        AdminConfig     `yaml:"admin"`
	Storage      StorageConfig   `yaml:"storage"`
	OneAPI       OneAPIConfig    `yaml:"oneapi"`
	Chunking     ChunkingConfig  `yaml:"chunking"`
	Rerank       RerankConfig    `yaml:"rerank"`
	Search       SearchConfig    `yaml:"search"`
	Documents    DocumentsConfig `yaml:"documents"`
	RegistryPath string          `yaml:"registry_path"`
	// AuditPath is the append-only log the dashboard tails (MCP method calls +
	// tool invocations). Relative paths resolve against the config file's
	// directory so it survives a working-directory change (systemd/setsid).
	// Empty defaults to logs/rag-mcp.log.
	AuditPath string `yaml:"audit_path"`
	// AuditMaxMB caps one audit file before startup rotation; AuditKeep is how
	// many rotated backups (.1, .2, ...) to retain. 0 = defaults (20MB, 3).
	AuditMaxMB int `yaml:"audit_max_mb"`
	AuditKeep  int `yaml:"audit_keep"`
	// SettingsPath is the runtime settings file (settings.json). It holds the
	// search defaults edited via the WebUI; missing file = built-in defaults.
	SettingsPath string `yaml:"settings_path"`
	// DocsDir is the data-bank root browsed by the dashboard.
	// Relative paths resolve against the process working directory;
	// prefer absolute on servers (ssh CWD is $HOME, not the app dir).
	DocsDir string `yaml:"docs_dir"`
}

// StorageConfig selects the vector backend. The server is vectra-only:
// backend is accepted for compatibility ("qdrant" maps to "vectra") but the
// file store under VectraDir is always used.
type StorageConfig struct {
	Backend   string `yaml:"backend"`
	VectraDir string `yaml:"vectra_dir"`
	// MemoryDir is where store_memory raw text is persisted. Empty defaults to
	// <docs_dir>/memory (agent journal: <memory>/agent/YYYY-MM-DD.md).
	MemoryDir string `yaml:"memory_dir"`
}

// DocumentsConfig configures the documents vault: the raw-text corpus that
// backs the "search the original" path.
//
// The vault is never curated by hand. Documents are filed by machine under
// <dir>/<yyyy-mm-dd>/<channel>/<category>/<time>-<hash>-<slug>.md, and
// ripgrep (or the built-in scanner) is how anything is found again.
type DocumentsConfig struct {
	// Dir is the vault root. Empty defaults to <docs_dir>/documents.
	Dir string `yaml:"dir"`
	// SearchBackend picks the grep engine:
	//   auto   - use ripgrep when the host has it, else the built-in scanner
	//   rg     - insist on ripgrep (fail loudly when it is missing)
	//   builtin - always use the built-in scanner
	SearchBackend string `yaml:"search_backend"`
	// RGPath is an explicit ripgrep binary. Empty = look it up on PATH.
	RGPath string `yaml:"rg_path"`
	// TimeoutSeconds bounds one document search. 0 = 60.
	TimeoutSeconds int `yaml:"timeout_seconds"`
	// MaxMatches caps the matches one search returns. 0 = 200.
	MaxMatches int `yaml:"max_matches"`
	// PerFile caps the matches reported per document. 0 = 5.
	PerFile int `yaml:"per_file"`
	// LedgerWindowDays is how far back the in-memory ledger index reaches.
	// Older documents are still found (the ledger files hold them), but a
	// body that reappears after the window is filed as a new document.
	// 0 = 30.
	LedgerWindowDays int `yaml:"ledger_window_days"`
	// Categories are the rules that file a document into a category folder.
	// The first matching rule wins; a document that matches none lands in
	// "_".
	Categories []engine.CategoryRule `yaml:"categories"`
}

type ServerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
}

// AdminConfig is the management UI + API. Defaults to 0.0.0.0 so all LAN
// devices can reach it during development; there is no TLS, so restrict the
// host or firewall the port before exposing it beyond a trusted network.
//
// Auth (one-api style): when username + password_sha256 are both set, every
// /api/* except /api/health, /api/info and /api/login requires
// `Authorization: Bearer <token>`. The browser keeps the token in
// localStorage (remember-me) or sessionStorage, so the password is only typed
// once. password_sha256 = hex(sha256(password)), e.g.
// `echo -n 's3cret' | sha256sum`. Empty username = auth disabled (legacy).
type AdminConfig struct {
	Host           string `yaml:"host"`
	Port           int    `yaml:"port"`
	Username       string `yaml:"username"`
	PasswordSHA256 string `yaml:"password_sha256"`
	SessionDays    int    `yaml:"session_days"`
}

type OneAPIConfig struct {
	BaseURL     string `yaml:"base_url"`
	BackupURL   string `yaml:"backup_url"`
	EmbedModel  string `yaml:"embed_model"`
	RerankModel string `yaml:"rerank_model"`
	APIKey      string `yaml:"api_key"`
	// ProviderName labels the embedding provider recorded in collection
	// provenance (e.g. "one-api", "openrouter"). Default "one-api".
	ProviderName string `yaml:"provider_name"`
	// VectorDim/VectorDistance define the vector space new collections are
	// created with. Recorded in registry provenance so a later provider/model
	// switch is caught by the consistency gate instead of silently mixing.
	VectorDim      int    `yaml:"vector_dim"`
	VectorDistance string `yaml:"vector_distance"`
	// EmbedTimeoutSeconds / RerankTimeoutSeconds bound one call to one-api's
	// embed/rerank endpoint. They must stay comfortably above one-api's own
	// channel-failover window (RELAY_TIMEOUT x retries): if rag-mcp cancels
	// mid-flight, one-api treats it as a client abort, stops failing over and
	// never penalizes the bad channel. 0 = default 180.
	//
	// EmbedTimeoutSeconds is the INTERACTIVE budget: a recall has a user
	// waiting, so it stays short. BulkEmbedTimeoutSeconds is the background
	// budget for index jobs and web ingests, where nobody waits and a slow
	// channel should be survived rather than abort a whole document.
	EmbedTimeoutSeconds     int `yaml:"embed_timeout_seconds"`
	BulkEmbedTimeoutSeconds int `yaml:"bulk_embed_timeout_seconds"`
	RerankTimeoutSeconds    int `yaml:"rerank_timeout_seconds"`
	// BulkConcurrency is how many BACKGROUND embedding requests may be in
	// flight at once. The interactive query path has its own slot, so
	// one-api sees at most this + 1 concurrent requests. 0 = default 2.
	BulkConcurrency int `yaml:"bulk_concurrency"`
	// SearchBudgetSeconds bounds one whole search (embed + stores + rerank)
	// so the server answers before the caller's outer fail-safe (TavernLab
	// mcp_timeout 180s). Checked only at stage boundaries; an in-flight HTTP
	// call is never cancelled. 0 = default 170.
	SearchBudgetSeconds int `yaml:"search_budget_seconds"`
}

type ChunkingConfig struct {
	ChunkSize      int `yaml:"chunk_size"`
	OverlapPercent int `yaml:"overlap_percent"`
	// Strategy selects the chunking algorithm: window (default, deterministic
	// sliding window), adaptive, paragraph, section, sentence, dialogue.
	Strategy string `yaml:"strategy"`
}

type RerankConfig struct {
	Enabled bool `yaml:"enabled"`
	Recall  int  `yaml:"recall"`
	// QueryMaxChars bounds the query (runes) before embedding and reranking;
	// the head is kept. 0 = unlimited. DocMaxChars bounds rerank documents.
	QueryMaxChars int `yaml:"query_max_chars"`
	DocMaxChars   int `yaml:"doc_max_chars"`
}

// ProviderConfig configures one external search provider.
//
// The API key is read from api_key_env first and only falls back to the
// inline api_key. The environment variable is the preferred form because
// config.yaml is tracked in git while the key must not be.
type ProviderConfig struct {
	BaseURL string `yaml:"base_url"`
	// APIKey is an inline key. Leave it empty and set api_key_env instead.
	APIKey string `yaml:"api_key"`
	// APIKeyEnv names an environment variable holding the key.
	APIKeyEnv string `yaml:"api_key_env"`
	// Country / Location bias geo-targeted results (ISO 3166-1 alpha-2).
	Country  string `yaml:"country"`
	Location string `yaml:"location"`
}

// Key resolves the provider's API key.
func (p ProviderConfig) Key() string {
	if env := p.APIKeyEnv; env != "" {
		if v := os.Getenv(env); v != "" {
			return strings.TrimSpace(v)
		}
	}
	return strings.TrimSpace(p.APIKey)
}

// SearXNGConfig configures a self-hosted SearXNG instance. It needs no API
// key, only a reachable base URL with the json format enabled.
type SearXNGConfig struct {
	BaseURL string `yaml:"base_url"`
	// Language is the SearXNG language code ("zh-CN", "en", ...).
	Language string `yaml:"language"`
	// Categories restricts which engine groups run ("general").
	Categories string `yaml:"categories"`
}

// SearchConfig configures the web-search side of the server: which providers
// are tried in which order, where the fetched pages are stored, and how
// aggressively duplicates are filtered out.
type SearchConfig struct {
	// Enabled turns the web tools on. When false, web_search reports that
	// the feature is off instead of calling out.
	Enabled bool `yaml:"enabled"`
	// Collection is where ingested pages are vectorized.
	Collection string `yaml:"collection"`
	// Providers is the ordered failover chain ("deepseek", "tavily",
	// "firecrawl", "searxng"). The first one that answers wins.
	Providers []string `yaml:"providers"`
	// Count is the default number of results to request.
	Count int `yaml:"count"`
	// IngestTopN is how many of the ranked results get fetched and stored.
	// 0 = every result.
	IngestTopN int `yaml:"ingest_top_n"`
	// FreshnessDays is the default age limit on search results: 0 = no
	// limit. It is only a default; web_search's freshness_days argument
	// overrides it per call.
	FreshnessDays int `yaml:"freshness_days"`
	// StaleAfterDays is the age after which a recalled web page is
	// flagged as stale. 0 = never flag. Flagging never hides a page:
	// knowing that something is old is more useful than not knowing it
	// exists.
	StaleAfterDays int `yaml:"stale_after_days"`
	// MaxContentChars caps how much of a page body is stored.
	MaxContentChars int `yaml:"max_content_chars"`
	// TimeoutSeconds bounds one provider request.
	TimeoutSeconds int `yaml:"timeout_seconds"`
	// Topic is the default search topic ("general" or "news").
	Topic string `yaml:"topic"`
	// Lang is the default language bias passed to the providers.
	Lang string `yaml:"lang"`
	// Tags are attached to every ingested chunk.
	Tags []string `yaml:"tags"`

	DeepSeek  ProviderConfig `yaml:"deepseek"`
	Tavily    ProviderConfig `yaml:"tavily"`
	Firecrawl ProviderConfig `yaml:"firecrawl"`
	SearXNG   SearXNGConfig  `yaml:"searxng"`
}

func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host: "0.0.0.0",
			Port: 8199,
		},
		Admin: AdminConfig{
			Host: "0.0.0.0",
			Port: 8198,
		},
		RegistryPath: "collections.json",
		SettingsPath: "settings.json",
		AuditPath:    "logs/rag-mcp.log",
		AuditMaxMB:   20,
		AuditKeep:    3,
		DocsDir:      "docs",
		Storage: StorageConfig{
			Backend:   "vectra",
			VectraDir: "Vectra",
		},
		OneAPI: OneAPIConfig{
			BaseURL:                 "http://192.168.100.20:3000",
			BackupURL:               "http://192.168.10.2:3000",
			EmbedModel:              "embedding",
			RerankModel:             "reranker",
			ProviderName:            "one-api",
			VectorDim:               1024,
			VectorDistance:          "Cosine",
			EmbedTimeoutSeconds:     180,
			BulkEmbedTimeoutSeconds: 180,
			RerankTimeoutSeconds:    180,
			SearchBudgetSeconds:     170,
			BulkConcurrency:         2,
		},
		Chunking: ChunkingConfig{
			ChunkSize:      500,
			OverlapPercent: 30,
			Strategy:       "window",
		},
		Rerank: RerankConfig{
			Enabled:       true,
			Recall:        30,
			QueryMaxChars: 500,
			DocMaxChars:   1000,
		},
		Search: SearchConfig{
			Enabled:         true,
			Collection:      "web_cache",
			Providers:       []string{"deepseek", "tavily", "firecrawl", "searxng"},
			Count:           10,
			IngestTopN:      0,
			FreshnessDays:   7,
			MaxContentChars: 20000,
			TimeoutSeconds:  45,
			Topic:           "general",
			Lang:            "",
			Tags:            []string{"web"},
			DeepSeek:        ProviderConfig{BaseURL: "https://api.deepseek.com"},
			Tavily:          ProviderConfig{BaseURL: "https://api.tavily.com"},
			Firecrawl:       ProviderConfig{BaseURL: "https://api.firecrawl.dev/v2", Country: "MY"},
			SearXNG:         SearXNGConfig{Categories: "general"},
		},
		Documents: DocumentsConfig{
			SearchBackend:    "auto",
			TimeoutSeconds:   60,
			MaxMatches:       200,
			PerFile:          5,
			LedgerWindowDays: 30,
		},
	}
}

func LoadConfig(path string) (*Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, err
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}
