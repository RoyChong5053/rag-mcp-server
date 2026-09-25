package main

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server       ServerConfig   `yaml:"server"`
	Admin        AdminConfig    `yaml:"admin"`
	Storage      StorageConfig  `yaml:"storage"`
	Qdrant       QdrantConfig   `yaml:"qdrant"`
	OneAPI       OneAPIConfig   `yaml:"oneapi"`
	Chunking     ChunkingConfig `yaml:"chunking"`
	Rerank       RerankConfig   `yaml:"rerank"`
	RegistryPath string         `yaml:"registry_path"`
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

// StorageConfig selects the vector backend. backend is the global default
// ("qdrant" or "vectra"); a collection can override it via its registry entry.
type StorageConfig struct {
	Backend   string `yaml:"backend"`
	VectraDir string `yaml:"vectra_dir"`
	// MemoryDir is where store_memory raw text is persisted. Empty defaults to
	// <docs_dir>/memory.
	MemoryDir string `yaml:"memory_dir"`
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

type QdrantConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
}

type OneAPIConfig struct {
	BaseURL     string `yaml:"base_url"`
	BackupURL   string `yaml:"backup_url"`
	EmbedModel  string `yaml:"embed_model"`
	RerankModel string `yaml:"rerank_model"`
	APIKey      string `yaml:"api_key"`
}

type ChunkingConfig struct {
	ChunkSize      int `yaml:"chunk_size"`
	OverlapPercent int `yaml:"overlap_percent"`
}

type RerankConfig struct {
	Enabled       bool `yaml:"enabled"`
	Recall        int  `yaml:"recall"`
	QueryMaxChars int  `yaml:"query_max_chars"`
	DocMaxChars   int  `yaml:"doc_max_chars"`
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
			Backend:   "qdrant",
			VectraDir: "Vectra",
		},
		Qdrant: QdrantConfig{
			Host: "localhost",
			Port: 6333,
		},
		OneAPI: OneAPIConfig{
			BaseURL:     "http://192.168.100.20:3000",
			BackupURL:   "http://192.168.10.2:3000",
			EmbedModel:  "embedding",
			RerankModel: "reranker",
		},
		Chunking: ChunkingConfig{
			ChunkSize:      500,
			OverlapPercent: 30,
		},
		Rerank: RerankConfig{
			Enabled:       true,
			Recall:        30,
			QueryMaxChars: 2000,
			DocMaxChars:   1000,
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
