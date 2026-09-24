package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime/debug"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
	"github.com/RoyChong5053/rag-mcp-server/mcp"
	"github.com/RoyChong5053/rag-mcp-server/settings"
	"github.com/RoyChong5053/rag-mcp-server/webui"
)

// version is stamped at build time with -ldflags "-X main.version=...";
// commit falls back to the VCS revision Go embeds for git builds.
var (
	version = "dev"
	commit  = "none"
)

// buildInfo resolves the version/commit shown in the dashboard.
func buildInfo() (string, string) {
	c := commit
	if c == "none" {
		if bi, ok := debug.ReadBuildInfo(); ok {
			for _, s := range bi.Settings {
				if s.Key == "vcs.revision" && s.Value != "" {
					c = s.Value
					if len(c) > 12 {
						c = c[:12]
					}
				}
			}
		}
	}
	return version, c
}

func main() {
	configPath := flag.String("config", "config.yaml", "Path to config file")
	flag.Parse()

	cfg, err := LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Resolve the file-backed store dir (and optional memory/audit dirs) against
	// the config file's directory, so a relative path lands in the project
	// folder even when the process CWD differs (e.g. setsid from $HOME).
	cfgDir, err := filepath.Abs(filepath.Dir(*configPath))
	if err != nil {
		log.Fatalf("Resolve config dir: %v", err)
	}

	// Persist the audit log ourselves instead of relying on stdout redirection
	// (systemd journal / /tmp), which is volatile and/or root-owned.
	auditPath := cfg.AuditPath
	if auditPath == "" {
		auditPath = filepath.Join("logs", "rag-mcp.log")
	}
	if !filepath.IsAbs(auditPath) {
		auditPath = filepath.Join(cfgDir, auditPath)
	}
	if err := os.MkdirAll(filepath.Dir(auditPath), 0o755); err != nil {
		log.Fatalf("Create audit dir: %v", err)
	}
	auditFile, err := os.OpenFile(auditPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatalf("Open audit log %s: %v", auditPath, err)
	}
	defer auditFile.Close()
	log.SetOutput(io.MultiWriter(os.Stdout, auditFile))

	ver, com := buildInfo()

	// Runtime settings (search defaults + rerank behavior). config.yaml seeds
	// the initial values; settings.json persists WebUI edits on top.
	settingsPath := cfg.SettingsPath
	if settingsPath == "" {
		settingsPath = "settings.json"
	}
	settingsStore := settings.New(settingsPath, settings.Settings{
		ActiveBackend:    "",
		DefaultTopK:      10,
		DefaultThreshold: 0.25,
		RerankEnabled:    cfg.Rerank.Enabled,
		RerankRecall:     cfg.Rerank.Recall,
		QueryMaxChars:    cfg.Rerank.QueryMaxChars,
		DocMaxChars:      cfg.Rerank.DocMaxChars,
		FailoverEnabled:  true,
	})

	// Resolve the file-backed store dir (and optional memory dir) against the
	// config file's directory, so a relative "Vectra" lands in the project
	// folder even when the process CWD differs (e.g. setsid from $HOME).
	vectraDir := cfg.Storage.VectraDir
	if vectraDir != "" && !filepath.IsAbs(vectraDir) {
		vectraDir = filepath.Join(cfgDir, vectraDir)
	}
	memoryDir := cfg.Storage.MemoryDir
	if memoryDir != "" && !filepath.IsAbs(memoryDir) {
		memoryDir = filepath.Join(cfgDir, memoryDir)
	}

	// Create engine
	engineConfig := &engine.EngineConfig{
		QdrantHost:      cfg.Qdrant.Host,
		QdrantPort:      cfg.Qdrant.Port,
		VectraDir:       vectraDir,
		Backend:         cfg.Storage.Backend,
		DocsDir:         cfg.DocsDir,
		MemoryDir:       memoryDir,
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
	// Warm FileStore indexes off the request path: cold loads of large
	// collections otherwise stall the first search_memory for many seconds.
	go eng.PrewarmVectra()

	// Create MCP server
	mcpServer := mcp.NewServer()

	// Register all RAG tools
	mcp.RegisterTools(mcpServer, eng)

	// HTTP handler
	http.HandleFunc("/mcp", mcpServer.HandleMCP)

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	log.Printf("RAG MCP Server %s (%s) starting on %s", ver, com, addr)
	log.Printf("MCP endpoint: http://%s/mcp", addr)
	log.Printf("Qdrant: %s:%d", cfg.Qdrant.Host, cfg.Qdrant.Port)
	log.Printf("one-api: %s", cfg.OneAPI.BaseURL)
	log.Printf("Storage backend: %s (vectra_dir=%s)", cfg.Storage.Backend, vectraDir)
	log.Printf("Registry: %s", cfg.RegistryPath)
	log.Printf("Settings: %s", settingsPath)
	log.Printf("Audit: %s", auditPath)

	// Management dashboard + API. Binds per cfg.Admin.Host (default 0.0.0.0 so
	// any LAN device can reach it during development; there is no TLS).
	adminAddr := fmt.Sprintf("%s:%d", cfg.Admin.Host, cfg.Admin.Port)
	if cfg.Admin.Host != "" {
		docsAbs, err := filepath.Abs(cfg.DocsDir)
		if err != nil {
			log.Fatalf("Resolve docs dir: %v", err)
		}
		admin := webui.New(eng, auditPath, docsAbs, webui.BuildInfo{
			Version:   ver,
			Commit:    com,
			StartedAt: time.Now(),
		})
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
