package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/RoyChong5053/rag-mcp-server/mcp"
)

func main() {
	configPath := flag.String("config", "config.yaml", "Path to config file")
	flag.Parse()

	cfg, err := LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Create MCP server
	mcpServer := mcp.NewServer()

	// TODO: Register tools in Phase 4
	// For now, register a placeholder tool
	mcpServer.RegisterTool(mcp.Tool{
		Name:        "health_check",
		Description: "Check if the RAG MCP server is running",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	}, func(args map[string]any) (any, error) {
		return "RAG MCP Server is running", nil
	})

	// HTTP handler
	http.HandleFunc("/mcp", mcpServer.HandleMCP)

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	log.Printf("RAG MCP Server starting on %s", addr)
	log.Printf("MCP endpoint: http://%s/mcp", addr)

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
