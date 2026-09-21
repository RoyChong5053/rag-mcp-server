package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/mcp"
)

func main() {
	configPath := flag.String("config", "config.yaml", "Path to config file")
	flag.Parse()

	cfg, err := LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

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
	}

	eng := engine.NewEngine(engineConfig)

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

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
