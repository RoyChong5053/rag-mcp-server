package main

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server       ServerConfig   `yaml:"server"`
	Admin        AdminConfig    `yaml:"admin"`
	Qdrant       QdrantConfig   `yaml:"qdrant"`
	OneAPI       OneAPIConfig   `yaml:"oneapi"`
	Chunking     ChunkingConfig `yaml:"chunking"`
	Rerank       RerankConfig   `yaml:"rerank"`
	RegistryPath string         `yaml:"registry_path"`
}

type ServerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
}

// AdminConfig is the localhost-only management UI + API.
// Never bind this to 0.0.0.0: m64 has no TLS; reach it via ssh tunnel.
type AdminConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
}

type QdrantConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
}

type OneAPIConfig struct {
	BaseURL    string `yaml:"base_url"`
	BackupURL  string `yaml:"backup_url"`
	EmbedModel string `yaml:"embed_model"`
	RerankModel string `yaml:"rerank_model"`
	APIKey     string `yaml:"api_key"`
}

type ChunkingConfig struct {
	ChunkSize      int `yaml:"chunk_size"`
	OverlapPercent int `yaml:"overlap_percent"`
}

type RerankConfig struct {
	Enabled         bool `yaml:"enabled"`
	Recall          int  `yaml:"recall"`
	QueryMaxChars   int  `yaml:"query_max_chars"`
	DocMaxChars     int  `yaml:"doc_max_chars"`
}

func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host: "0.0.0.0",
			Port: 8199,
		},
		Admin: AdminConfig{
			Host: "127.0.0.1",
			Port: 8198,
		},
		RegistryPath: "collections.json",
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
