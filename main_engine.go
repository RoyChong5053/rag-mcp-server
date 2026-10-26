package main

import (
	"fmt"
	"path/filepath"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// buildEngine creates the engine from the config.
//
// The server and the maintenance commands share it, so both see the same
// store, the same registry and the same documents vault. Relative paths are
// resolved against the config file's directory, not the process working
// directory: a service started from $HOME must still find its data.
func buildEngine(cfg *Config, cfgDir string) (*engine.Engine, error) {
	// Runtime settings (search defaults + rerank behavior). config.yaml seeds
	// the initial values; settings.json persists WebUI edits on top.
	settingsPath := cfg.SettingsPath
	if settingsPath == "" {
		settingsPath = "settings.json"
	}
	if !filepath.IsAbs(settingsPath) {
		settingsPath = filepath.Join(cfgDir, settingsPath)
	}
	settingsStore := settings.New(settingsPath, settings.Settings{
		ActiveBackend:              "",
		DefaultTopK:                10,
		DefaultThreshold:           0.25,
		RerankEnabled:              cfg.Rerank.Enabled,
		RerankRecall:               cfg.Rerank.Recall,
		QueryMaxChars:              cfg.Rerank.QueryMaxChars,
		DocMaxChars:                cfg.Rerank.DocMaxChars,
		EmbedBatchSize:             0,
		FailoverEnabled:            true,
		BM25Enabled:                true,
		FusionMethod:               "rrf",
		SearchProviders:            cfg.Search.Providers,
		DefaultSearchCount:         cfg.Search.Count,
		DefaultSearchFreshnessDays: cfg.Search.FreshnessDays,
		DefaultSearchTopic:         cfg.Search.Topic,
		DefaultSearchLang:          cfg.Search.Lang,
		DefaultSearchStore:         cfg.Search.Store,
		DefaultSearchIngest:        cfg.Search.IngestTopN,
		DefaultSearchMaxChars:      cfg.Search.MaxContentChars,
		DeepSeekMaxCount:           cfg.Search.DeepSeek.MaxCount,
		TavilySearchDepth:          cfg.Search.Tavily.SearchDepth,
		TavilyCountry:              cfg.Search.Tavily.Country,
		FirecrawlCountry:           cfg.Search.Firecrawl.Country,
		FirecrawlLocation:          cfg.Search.Firecrawl.Location,
		SearXNGLanguage:            cfg.Search.SearXNG.Language,
		SearXNGCategories:          cfg.Search.SearXNG.Categories,
		SearXNGSafeSearch:          cfg.Search.SearXNG.SafeSearch,
		SearXNGEngines:             cfg.Search.SearXNG.Engines,
		EmbedProviders: []settings.EmbedProvider{{
			ID: cfg.OneAPI.ProviderName, BaseURL: cfg.OneAPI.BaseURL, BackupURL: cfg.OneAPI.BackupURL,
			Model: cfg.OneAPI.EmbedModel, Dim: cfg.OneAPI.VectorDim, Distance: cfg.OneAPI.VectorDistance,
			APIKeyEnv: "ONE_API_KEY",
		}},
	})

	vectraDir := cfg.Storage.VectraDir
	if vectraDir != "" && !filepath.IsAbs(vectraDir) {
		vectraDir = filepath.Join(cfgDir, vectraDir)
	}
	memoryDir := cfg.Storage.MemoryDir
	if memoryDir != "" && !filepath.IsAbs(memoryDir) {
		memoryDir = filepath.Join(cfgDir, memoryDir)
	}
	// The data bank and the documents vault live next to each other, so the
	// vault travels with the raw files it indexes.
	docsDir := cfg.DocsDir
	if docsDir == "" {
		docsDir = "docs"
	}
	if !filepath.IsAbs(docsDir) {
		docsDir = filepath.Join(cfgDir, docsDir)
	}
	documentsDir := cfg.Documents.Dir
	if documentsDir == "" {
		documentsDir = filepath.Join(docsDir, "documents")
	}
	if !filepath.IsAbs(documentsDir) {
		documentsDir = filepath.Join(cfgDir, documentsDir)
	}
	registryPath := cfg.RegistryPath
	if registryPath == "" {
		registryPath = "collections.json"
	}
	if !filepath.IsAbs(registryPath) {
		registryPath = filepath.Join(cfgDir, registryPath)
	}

	engineConfig := &engine.EngineConfig{
		VectraDir:        vectraDir,
		Backend:          cfg.Storage.Backend,
		DocsDir:          docsDir,
		MemoryDir:        memoryDir,
		OneAPIBaseURL:    cfg.OneAPI.BaseURL,
		OneAPIBackupURL:  cfg.OneAPI.BackupURL,
		EmbedModel:       cfg.OneAPI.EmbedModel,
		EmbedProvider:    cfg.OneAPI.ProviderName,
		VectorDim:        cfg.OneAPI.VectorDim,
		VectorDistance:   cfg.OneAPI.VectorDistance,
		RerankModel:      cfg.OneAPI.RerankModel,
		APIKey:           cfg.OneAPI.APIKey,
		EmbedTimeout:     time.Duration(cfg.OneAPI.EmbedTimeoutSeconds) * time.Second,
		BulkEmbedTimeout: time.Duration(cfg.OneAPI.BulkEmbedTimeoutSeconds) * time.Second,
		RerankTimeout:    time.Duration(cfg.OneAPI.RerankTimeoutSeconds) * time.Second,
		SearchBudget:     time.Duration(cfg.OneAPI.SearchBudgetSeconds) * time.Second,
		BulkConcurrency:  cfg.OneAPI.BulkConcurrency,
		ChunkSize:        cfg.Chunking.ChunkSize,
		OverlapPercent:   cfg.Chunking.OverlapPercent,
		ChunkStrategy:    cfg.Chunking.Strategy,
		RerankEnabled:    cfg.Rerank.Enabled,
		RerankRecall:     cfg.Rerank.Recall,
		QueryMaxChars:    cfg.Rerank.QueryMaxChars,
		DocMaxChars:      cfg.Rerank.DocMaxChars,
		RegistryPath:     registryPath,
		SettingsPath:     settingsPath,
		HistoryPath:      filepath.Join(cfgDir, "logs", "index_history.jsonl"),
		DocumentsDir:     documentsDir,
		DocCategories:    cfg.Documents.Categories,
		DocSearchBackend: cfg.Documents.SearchBackend,
		DocRGPath:        cfg.Documents.RGPath,
		DocSearchTimeout: time.Duration(cfg.Documents.TimeoutSeconds) * time.Second,
		DocMaxMatches:    cfg.Documents.MaxMatches,
		DocPerFile:       cfg.Documents.PerFile,
		LedgerWindowDays: cfg.Documents.LedgerWindowDays,
	}

	eng, err := engine.NewEngine(engineConfig, settingsStore)
	if err != nil {
		return nil, fmt.Errorf("create engine: %w", err)
	}
	return eng, nil
}
