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
	"strings"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/jobs"
	"github.com/RoyChong5053/rag-mcp-server/mcp"
	"github.com/RoyChong5053/rag-mcp-server/search"
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

// rotateAudit renames an overgrown audit log to .1 (.1->.2, etc.) at startup.
// maxMB<=0 defaults to 20, keep<=0 defaults to 3. Missing file = no-op.
func rotateAudit(path string, maxMB, keep int) {
	if maxMB <= 0 {
		maxMB = 20
	}
	if keep <= 0 {
		keep = 3
	}
	st, err := os.Stat(path)
	if err != nil || st.IsDir() {
		return
	}
	if st.Size() <= int64(maxMB)<<20 {
		return
	}
	// Drop the oldest beyond keep, then shift down.
	oldest := path + "." + itoa(keep)
	_ = os.Remove(oldest)
	for i := keep - 1; i >= 1; i-- {
		_ = os.Rename(path+"."+itoa(i), path+"."+itoa(i+1))
	}
	_ = os.Rename(path, path+".1")
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	b := make([]byte, 0, 4)
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

func main() {
	// Maintenance subcommands. They are operational work, not agent work, so
	// they live outside the MCP tool surface.
	if len(os.Args) > 1 && !strings.HasPrefix(os.Args[1], "-") {
		switch os.Args[1] {
		case "serve":
			os.Args = append(os.Args[:1], os.Args[2:]...)
		case "index-verify", "index-prune", "index-import", "verify-deep",
			"docs-migrate", "docs-verify", "docs-search":
			runMaintenance(os.Args[1], os.Args[2:])
			return
		default:
			fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
			usage()
			os.Exit(2)
		}
	}

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
	// Startup rotation so the audit log can't grow without bound across
	// restarts/deploys. Keeps auditPath + .1 ... .N (configurable).
	rotateAudit(auditPath, cfg.AuditMaxMB, cfg.AuditKeep)
	auditFile, err := os.OpenFile(auditPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatalf("Open audit log %s: %v", auditPath, err)
	}
	defer auditFile.Close()
	log.SetOutput(io.MultiWriter(os.Stdout, auditFile))

	ver, com := buildInfo()

	eng, err := buildEngine(cfg, cfgDir)
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}
	// Warm FileStore indexes off the request path: cold loads of large
	// collections otherwise stall the first search_memory for many seconds.
	go eng.PrewarmVectra()

	// Create MCP server
	mcpServer := mcp.NewServer()

	// Background-task registry for the dashboard's async file indexing. The MCP
	// tools are synchronous and do not use it.
	jobsMgr := jobs.NewManager(eng, 2)

	// Register all RAG tools
	mcp.RegisterTools(mcpServer, eng)
	mcp.RegisterDocumentTools(mcpServer, eng)

	// The window after which a recalled web page is flagged as stale.
	mcp.SetWebStaleAfterDays(cfg.Search.StaleAfterDays)

	// Web search: build the provider chain from config and register the
	// web tools. The chain is ordered, so the first provider that answers
	// wins and the rest are only used as failover.
	if cfg.Search.Enabled {
		timeout := time.Duration(cfg.Search.TimeoutSeconds) * time.Second
		if timeout <= 0 {
			timeout = 45 * time.Second
		}
		providers := make([]search.Provider, 0, len(cfg.Search.Providers))
		for _, name := range cfg.Search.Providers {
			switch strings.ToLower(strings.TrimSpace(name)) {
			case "deepseek":
				providers = append(providers, search.NewDeepSeek(cfg.Search.DeepSeek.BaseURL, cfg.Search.DeepSeek.Key(), timeout))
			case "tavily":
				providers = append(providers, search.NewTavily(cfg.Search.Tavily.BaseURL, cfg.Search.Tavily.Key(), timeout))
			case "firecrawl":
				p := search.NewFirecrawl(cfg.Search.Firecrawl.BaseURL, cfg.Search.Firecrawl.Key(), timeout)
				if c := cfg.Search.Firecrawl.Country; c != "" {
					p.Country = c
				}
				p.Location = cfg.Search.Firecrawl.Location
				providers = append(providers, p)
			case "searxng":
				providers = append(providers, search.NewSearXNG(cfg.Search.SearXNG.BaseURL, timeout))
			default:
				log.Printf("Unknown search provider in config: %q (ignored)", name)
			}
		}
		chain := search.NewChain(providers...)
		webSvc := mcp.NewWebService(eng, chain, mcp.WebOptions{
			Collection:      cfg.Search.Collection,
			Count:           cfg.Search.Count,
			IngestTopN:      cfg.Search.IngestTopN,
			FreshnessDays:   cfg.Search.FreshnessDays,
			MaxContentChars: cfg.Search.MaxContentChars,
			Timeout:         timeout,
			Topic:           cfg.Search.Topic,
			Lang:            cfg.Search.Lang,
			Tags:            cfg.Search.Tags,
		})
		// The pages a search found are filed in the background. The job
		// registry is what makes that visible: the dashboard's Jobs panel
		// shows the filing as it runs, instead of the operator guessing from
		// the CPU load.
		webSvc.WithJobs(jobsMgr)
		webSvc.Register(mcpServer)
		log.Printf("Web search: collection=%s providers=[%s]", cfg.Search.Collection, strings.Join(webSvc.Providers(), ", "))
	} else {
		log.Printf("Web search: disabled (search.enabled=false)")
	}

	// HTTP handler
	http.HandleFunc("/mcp", mcpServer.HandleMCP)

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	log.Printf("RAG MCP Server %s (%s) starting on %s", ver, com, addr)
	log.Printf("MCP endpoint: http://%s/mcp", addr)
	log.Printf("one-api: %s", cfg.OneAPI.BaseURL)
	log.Printf("Storage backend: %s (vectra_dir=%s)", cfg.Storage.Backend, eng.Config().VectraDir)
	log.Printf("Registry: %s", cfg.RegistryPath)
	log.Printf("Settings: %s", eng.Config().SettingsPath)
	log.Printf("Audit: %s", auditPath)
	if cfg.Admin.Username != "" {
		log.Printf("Admin auth: enabled (user=%s)", cfg.Admin.Username)
	} else {
		log.Printf("Admin auth: disabled (set admin.username + password_sha256 to enable)")
	}

	// Management dashboard + API. Binds per cfg.Admin.Host (default 0.0.0.0 so
	// any LAN device can reach it during development; there is no TLS).
	adminAddr := fmt.Sprintf("%s:%d", cfg.Admin.Host, cfg.Admin.Port)
	if cfg.Admin.Host != "" {
		// The dashboard and the engine must agree on the data-bank root,
		// otherwise a document the dashboard lists is not the document the
		// engine indexed.
		docsAbs := eng.Config().DocsDir
		admin := webui.NewWithJobs(eng, auditPath, docsAbs, webui.BuildInfo{
			Version:   ver,
			Commit:    com,
			StartedAt: time.Now(),
		}, webui.AdminAuth{
			Username:       cfg.Admin.Username,
			PasswordSHA256: cfg.Admin.PasswordSHA256,
			SessionDays:    cfg.Admin.SessionDays,
			SessionFile:    filepath.Join(cfgDir, "sessions.json"),
		}, jobsMgr)
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
