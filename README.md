# RAG MCP Server

An independent Retrieval-Augmented Generation engine exposed via the Model Context Protocol (MCP). Any MCP-compatible frontend — SillyTavern, OpenClaw, Hermes, Pi Agent, Claude Code, etc. — can query a shared persistent memory layer through this service.

## Why

RAG shouldn't be locked inside a single chat frontend. Your "memory" should be a shared service that any agent can access. This project decouples RAG from SillyTavern and exposes it as a standard MCP toolset, backed by:

- **Pluggable vector storage**: Qdrant for full deployments, or a local
  file-based Vectra-compatible store for lightweight/portable setups
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
│  Tools: search, store, delete, list         │
└──────┬──────────────────────┬───────────────┘
       │                      │
  ┌────▼──────────────┐  ┌────▼─────────────────┐
  │ VectorStore       │  │ one-api (:3000)      │
  │ Qdrant :6333  or  │  │ /v1/embeddings       │
  │ Vectra/ (files)   │  │ /v1/rerank (fan-out) │
  └───────────────────┘  └──────────────────────┘
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

admin:                    # management UI + JSON API
  host: "0.0.0.0"         # no TLS; restrict the host/firewall if exposed
  port: 8198

registry_path: "collections.json"
settings_path: "settings.json"
docs_dir: "docs"

# Vector backend: "qdrant" (default) or "vectra" (local files).
# A collection can override the default via its registry `backend` field.
storage:
  backend: "qdrant"
  vectra_dir: "Vectra"
  # memory_dir: "docs/memory"   # store_memory raw text (default <docs_dir>/memory)

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

### Runtime settings (`settings.json`)

`config.yaml` covers infrastructure (bind addresses, Qdrant/one-api endpoints,
chunking). A separate `settings.json` holds **runtime search defaults** that all
frontends share and that take effect immediately, without a restart:

- `active_backend` — which backend's default collection the omitted-`collection_id`
  scope uses (`qdrant`|`vectra`; empty = follow `storage.backend`). Search/default
  scope only — it never changes where new named collections are created
- `default_collection_qdrant` — qdrant scope for calls that omit `collection_id`
  (empty = search all enabled collections)
- `default_collection_vectra` — vectra scope; also the failover target when qdrant
  is unreachable
- `failover_enabled` — fall back to the vectra default on a qdrant outage
- `default_top_k`, `default_threshold`
- `rerank_enabled`, `rerank_recall` (vector over-fetch before reranking)
- `query_max_chars`, `doc_max_chars` (rerank truncation, runes)

Edit these from the dashboard's **Search Defaults** panel; `config.yaml` only
seeds the initial values. A default collection that does not exist in its own
backend is rejected loudly, so a typo can never silently widen searches into a
global scan. A pre-existing single `default_collection` key is migrated to
`default_collection_qdrant` on load.

## Storage Backends

`storage.backend` selects the global default; a collection can override it with
its registry `backend` field (dashboard "Backend" selector or
`set_collection_meta`). Both backends share one `VectorStore` interface, so
search/index/delete behave identically.

| Backend | Storage | Best for |
|---------|---------|----------|
| `qdrant` | Qdrant REST (`:6333`) | large corpora, shared deployments |
| `vectra` | Local JSON files under `vectra_dir/` | lightweight/portable, no server |

The file backend mirrors the Vectra / SillyTavern on-disk shape:

```
Vectra/<collection>/catalog.json
Vectra/<collection>/<source_key>/index.json
```

`index.json` is `{version, metadata_config, items:[{id, metadata, vector, norm}]}`.
Source keys mirror the `docs/` tree, so `docs/chat_history/Leer.md` becomes
`Vectra/<collection>/chat_history/Leer.md/index.json`. Writes are serialized and
land atomically (temp + rename). It is only ever written inside our own
`Vectra/` root; SillyTavern's `data/.../vectors` is never touched.

`store_memory` now persists the raw text under `<memory_dir>/<YYYY-MM-DD>/`
before indexing, in **both** backends, so memory writes have an on-disk source
document (browsable in the data bank) instead of living only in the vector
store.

### Failover (qdrant → vectra)

For setups where Qdrant runs on a machine that is not always on, the default
scope can fall back to a local vectra collection:

- Calls that omit `collection_id` use `active_backend`'s default collection.
  If that is qdrant and it is unreachable (connection refused/timeout/5xx) and
  `failover_enabled` is set, the `default_collection_vectra` is searched/written
  instead. Results are tagged with their `[backend]`.
- An explicit `collection_id` whose backend is down falls back to a **same-name**
  collection on the other backend when one exists; otherwise it errors loudly.
  A disabled backend is never retried into a global scan.
- Unreachable backends are remembered for ~15s so a dead qdrant does not cost a
  timeout on every call. Logical errors (missing collection, bad request) never
  trigger failover.
- There is no automatic reconciliation: memories written to the vectra fallback
  during an outage stay there. The raw source files are under `docs/memory/`, so
  they can be re-indexed into qdrant later.

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_memory` | Semantic search. Omitted `collection_id` uses the active backend's default, with automatic vectra fallback when qdrant is down; optional `top_k`/`threshold` override the server defaults. Disabled collections are always skipped |
| `store_memory` | Vectorize and store text for later recall (the write counterpart of `search_memory`). Raw text is persisted on disk by date, then indexed. Omitted `collection_id` uses the active backend's default with qdrant→vectra failover; errors loudly if there is none |
| `delete_memory` | Delete by filter or collection |
| `list_collections` | List all collections with chunk counts |
| `collection_info` | Live stats + registry metadata, one or all |
| `set_collection_meta` | Display name, tags, consumers, enabled flag, backend |
| `delete_collection` | Drop whole collection, requires `confirm:true` |
| `health_check` | Verify all components are healthy and report the active backend |

## Management dashboard

Admin UI + JSON API on `:8198` (defaults to `0.0.0.0` so every LAN device can
reach it; there is no TLS — restrict the bind or firewall it on untrusted
networks). Includes a **Search Defaults** panel (active backend, per-backend
default collections, failover toggle, top_k, threshold, rerank/recall, truncation
limits) shared live by every frontend, plus:
list/search collections, edit metadata, backfill payloads, recall test with
threshold-kill report, and an audit tail.
Collection metadata lives in `collections.json` (server-local, gitignored).

## Data Bank (dashboard)

- `docs/` browser with fresh/stale/unindexed badges (sha256 vs registry provenance)
- Upload to `docs/staging/`, zero-token chunk preview, index into new/existing
  collection with per-job `chunk_size`/`overlap_percent` (append or rebuild)
- Background jobs (max 2 concurrent, search never blocked) with sidebar progress
  and a **clear** button for finished jobs
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
