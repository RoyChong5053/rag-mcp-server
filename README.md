# RAG MCP Server

An independent Retrieval-Augmented Generation engine exposed via the Model Context Protocol (MCP). Any MCP-compatible frontend — SillyTavern, OpenClaw, Hermes, Pi Agent, Claude Code, etc. — can query a shared persistent memory layer through this service.

## Why

RAG shouldn't be locked inside a single chat frontend. Your "memory" should be a shared service that any agent can access. This project decouples RAG from SillyTavern and exposes it as a standard MCP toolset, backed by:

- **Qdrant** for vector storage
- **one-api** for high-availability embedding and reranking (6-device fan-out)
- **MCP Streamable HTTP** for universal client compatibility

## Architecture

```
┌─────────────────────────────────────────────┐
│  MCP Clients                                │
│  ST / OpenClaw / Hermes / Claude Code / ... │
└──────────────────┬──────────────────────────┘
                   │ MCP (Streamable HTTP)
                   ▼
┌─────────────────────────────────────────────┐
│  rag-mcp-server  (Go, :8199)               │
│  Tools: search, index, delete, list, sync   │
└──────┬──────────────────────┬───────────────┘
       │                      │
  ┌────▼─────┐    ┌──────────▼──────────────┐
  │ Qdrant   │    │ one-api (:3000)          │
  │ (:6333)  │    │ /v1/embeddings (fan-out) │
  │          │    │ /v1/rerank   (fan-out)   │
  └──────────┘    └─────────────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/RoyChong5053/rag-mcp-server.git
cd rag-mcp-server

# Build
go build -o rag-mcp-server .

# Run
./rag-mcp-server -config config.yaml
```

## Configuration

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8199

qdrant:
  host: "localhost"
  port: 6333

oneapi:
  base_url: "http://192.168.100.20:3000"
  embed_model: "embedding"
  rerank_model: "reranker"

chunking:
  chunk_size: 500       # chars
  overlap_percent: 30   # %

rerank:
  enabled: true
  recall: 30
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_memory` | Semantic search; `collections[]` scopes it, empty = all enabled; disabled always skipped |
| `index_document` | Index a file into a Qdrant collection |
| `index_text` | Index raw text directly |
| `delete_memory` | Delete by filter or collection |
| `list_collections` | List all collections with chunk counts |
| `collection_info` | Live stats + registry metadata, one or all |
| `set_collection_meta` | Display name, tags, consumers, enabled flag |
| `delete_collection` | Drop whole collection, requires `confirm:true` |
| `health_check` | Verify all components are healthy |

## Management dashboard

Localhost-only admin UI + JSON API on `:8198` (reach via `ssh -L 8198:localhost:8198 m64`):
list/search collections, edit metadata, backfill payloads, recall test with
threshold-kill report, audit tail.
Collection metadata lives in `collections.json` (server-local, gitignored).

## Data Bank (dashboard)

- `docs/` browser with fresh/stale/unindexed badges (sha256 vs registry provenance)
- Upload to `docs/staging/`, zero-token chunk preview, index into new/existing
  collection with per-job `chunk_size`/`overlap_percent` (append or rebuild)
- Background jobs (max 2 concurrent, search never blocked) with sidebar progress
- Provenance snapshot per collection: chunk size/overlap, embed model, source sha

## Register with one-api

After deploying, register this server in one-api's MCP Aggregator:

```bash
curl -X POST http://192.168.100.20:3000/api/mcp_servers \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag-memory",
    "base_url": "http://127.0.0.1:8199/mcp",
    "protocol": "streamable_http",
    "auth_type": "none",
    "auto_sync_enabled": true
  }'
```

## Chunking Parameters

Matches Vector-Storage-5053 behavior exactly:

- **Chunk size**: 500 chars (configurable)
- **Overlap**: 30%, trimmed to sentence boundaries
- **Delimiters**: `\n\n` → `\n` → ` ` → `""` (recursive fallback)
- **Rerank recall**: 30 candidates fetched,精排后取 top N

## Deployment

```bash
# Build for target
CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o rag-mcp-server .

# Deploy to m64
scp rag-mcp-server m64:~/app/rag-mcp-server/
ssh m64 "sudo systemctl restart rag-mcp-server"
```

## License

MIT
