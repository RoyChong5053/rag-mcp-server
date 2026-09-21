package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"path/filepath"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/mcp"
	"github.com/RoyChong5053/rag-mcp-server/settings"
	"github.com/RoyChong5053/rag-mcp-server/webui"
)

func main() {
	configPath := flag.String("config", "config.yaml", "Path to config file")
	flag.Parse()

	cfg, err := LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Runtime settings (search defaults + rerank behavior). config.yaml seeds
	// the initial values; settings.json persists WebUI edits on top.
	settingsPath := cfg.SettingsPath
	if settingsPath == "" {
		settingsPath = "settings.json"
	}
	settingsStore := settings.New(settingsPath, settings.Settings{
		DefaultCollection: "",
		DefaultTopK:       10,
		DefaultThreshold:  0.25,
		RerankEnabled:     cfg.Rerank.Enabled,
		RerankRecall:      cfg.Rerank.Recall,
		QueryMaxChars:     cfg.Rerank.QueryMaxChars,
		DocMaxChars:       cfg.Rerank.DocMaxChars,
	})

	// Create engine
	engineConfig := &engine.EngineConfig{
		QdrantHost:      cfg.Qdrant.Host,
		QdrantPort:      cfg.Qdrant.Port,
		OneAPIBaseURL:   cfg.OneAPI.BaseURL,
		OneAPIBackupURL: cfg.OneAPI.BackupURL,
		EmbedModel:      cfg.OneAPI.EmbedModel,
		RerankModel:     cfg.OneAPI.RerankModel,
		APIKey:          cfg.OneAPI.APIKey,
		ChunkSize:       cfg.Chunking.ChunkSize,
		OverlapPercent:  cfg.Chunking.OverlapPercent,
		RerankEnabled:   cfg.Rerank.Enabled,
		RerankRecall:    cfg.Rerank.Recall,
		QueryMaxChars:   cfg.Rerank.QueryMaxChars,
		DocMaxChars:     cfg.Rerank.DocMaxChars,
		RegistryPath:    cfg.RegistryPath,
	}

	eng, err := engine.NewEngine(engineConfig, settingsStore)
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}

	// Create MCP server
	mcpServer := mcp.NewServer()

	// Register all RAG tools
	mcp.RegisterTools(mcpServer, eng)

	// HTTP handler
	http.HandleFunc("/mcp", mcpServer.HandleMCP)

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	log.Printf("RAG MCP Server starting on %s", addr)
	log.Printf("MCP endpoint: http://%s/mcp", addr)
	log.Printf("Qdrant: %s:%d", cfg.Qdrant.Host, cfg.Qdrant.Port)
	log.Printf("one-api: %s", cfg.OneAPI.BaseURL)
	log.Printf("Registry: %s", cfg.RegistryPath)
	log.Printf("Settings: %s", settingsPath)

	// Management dashboard + API. Binds per cfg.Admin.Host (default 0.0.0.0 so
	// any LAN device can reach it during development; there is no TLS).
	adminAddr := fmt.Sprintf("%s:%d", cfg.Admin.Host, cfg.Admin.Port)
	if cfg.Admin.Host != "" {
		docsAbs, err := filepath.Abs(cfg.DocsDir)
		if err != nil {
			log.Fatalf("Resolve docs dir: %v", err)
		}
		admin := webui.New(eng, "/tmp/rag-mcp.log", docsAbs)
		go func() {
			log.Printf("Admin dashboard: http://%s (no TLS; docs=%s)", adminAddr, docsAbs)
			if err := http.ListenAndServe(adminAddr, admin.Routes()); err != nil {
				log.Fatalf("Admin server failed: %v", err)
			}
		}()
	} else {
		log.Printf("Admin dashboard disabled (empty admin.host)")
	}

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
