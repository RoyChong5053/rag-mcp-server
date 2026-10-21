package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/RoyChong5053/rag-mcp-server/engine"
)

// The maintenance commands are operational work, not agent work.
//
// The MCP tool surface stays small on purpose: a tool an agent can call is a
// tool an agent will call, and "check whether the vector index is still
// consistent" is not something an agent needs at 3am. These commands are run
// by a human, from a shell, when something looks wrong.

// usage prints the command list.
func usage() {
	fmt.Fprintf(os.Stderr, `rag-mcp-server - MCP server over a local vector store

  rag-mcp-server [serve] [-config config.yaml]
      Run the server (the default).

  rag-mcp-server index-verify [-config f] [collection]
      Check the vector index for one collection (or all).

  rag-mcp-server index-prune [-config f] <collection>
      Drop the empty sources of a collection.

  rag-mcp-server index-import [-config f] <path>
      Make a copied vector store live again (after an rclone copy).

  rag-mcp-server verify-deep [-config f] <collection> [-source path] [-samples n]
      Three-layer health check: structure, text coverage, vector drift.

  rag-mcp-server docs-migrate [-config f] [-apply]
      Move the legacy raw files into the documents vault.

  rag-mcp-server docs-verify [-config f]
      Check the documents vault against its ledger.

  rag-mcp-server docs-repair [-config f]
      Re-point the collections at the documents that now hold their text.

  rag-mcp-server docs-search [-config f] -query <text> [-mode literal|regex] [-channel c]
      Search the raw documents (the same search the dashboard uses).
`)
}

// runMaintenance runs one maintenance command.
func runMaintenance(cmd string, args []string) {
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	configPath := fs.String("config", "config.yaml", "Path to config file")
	apply := fs.Bool("apply", false, "Actually perform the change (default: report only)")
	source := fs.String("source", "", "Raw source file for the text-coverage layer")
	samples := fs.Int("samples", 20, "Vector spot-check samples")
	query := fs.String("query", "", "Search text")
	mode := fs.String("mode", "literal", "Search mode: literal or regex")
	channel := fs.String("channel", "", "Restrict to one channel")
	fs.Parse(args)

	cfg, err := LoadConfig(*configPath)
	if err != nil {
		fatal("load config: %v", err)
	}
	cfgDir, err := filepath.Abs(filepath.Dir(*configPath))
	if err != nil {
		fatal("resolve config dir: %v", err)
	}
	eng, err := buildEngine(cfg, cfgDir)
	if err != nil {
		fatal("create engine: %v", err)
	}

	positional := fs.Args()
	switch cmd {
	case "index-verify":
		if len(positional) > 0 && strings.TrimSpace(positional[0]) != "" {
			emit(eng.VerifyCollection(strings.TrimSpace(positional[0])))
			return
		}
		res := eng.VerifyAll()
		if res == nil {
			res = []engine.VaultVerifyResult{}
		}
		emit(res)

	case "index-prune":
		if len(positional) == 0 {
			fatal("index-prune needs a collection name")
		}
		name := strings.TrimSpace(positional[0])
		n, err := eng.PruneCollection(name)
		if err != nil {
			fatal("prune: %v", err)
		}
		emit(map[string]any{"pruned": n, "collection": name})

	case "index-import":
		if len(positional) == 0 {
			fatal("index-import needs a path")
		}
		res, err := eng.ImportVault(strings.TrimSpace(positional[0]))
		if err != nil {
			fatal("import: %v", err)
		}
		emit(res)

	case "verify-deep":
		if len(positional) == 0 {
			fatal("verify-deep needs a collection name")
		}
		res, err := eng.VerifyDeep(strings.TrimSpace(positional[0]), strings.TrimSpace(*source), *samples)
		if err != nil {
			fatal("verify-deep: %v", err)
		}
		emit(res)

	case "docs-migrate":
		res, err := eng.MigrateDocuments(!*apply)
		if err != nil {
			fatal("docs-migrate: %v", err)
		}
		emit(res)
		if res.DryRun {
			fmt.Fprintln(os.Stderr, "\n(dry run: nothing was moved. re-run with -apply to do it.)")
		}

	case "docs-verify":
		emit(eng.VerifyLedger())

	case "docs-repair":
		res, err := eng.RepairSources()
		if err != nil {
			fatal("docs-repair: %v", err)
		}
		emit(res)

	case "docs-search":
		if strings.TrimSpace(*query) == "" {
			fatal("docs-search needs -query")
		}
		res, err := eng.SearchDocuments(engine.DocSearchOptions{
			Query:   *query,
			Mode:    *mode,
			Channel: *channel,
		})
		if err != nil {
			fatal("docs-search: %v", err)
		}
		emit(res)

	default:
		usage()
		os.Exit(2)
	}
}

// emit prints a result as JSON.
func emit(v any) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", v)
		return
	}
	fmt.Println(string(data))
}

// fatal prints an error and stops.
func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
