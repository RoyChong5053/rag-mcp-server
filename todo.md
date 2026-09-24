# RAG MCP Server - Implementation TODO

## Overview

An independent RAG (Retrieval-Augmented Generation) engine exposed via MCP (Model Context Protocol). Any MCP-compatible frontend (SillyTavern, OpenClaw, Hermes, Pi Agent, etc.) can query persistent "memory" through this service.

**Core value**: Decouple RAG from any single frontend. Share one memory layer across all agents.

---

## Dashboard responsiveness + ops (2026-09-25, second pass)

- [x] WebUI renders the skeleton first; health/collections probed in background
      with client-side timeouts; compact per-component health badges
- [x] concurrent `HealthCheck` + cached-down backends; qdrant 1.5s probe client;
      `listAllStores` short-probes before listing (no more 10s stall)
- [x] embed/rerank probe timeouts report `slow` (amber) not `down`
- [x] phone layout: card tables, 16px controls, larger tap targets, stacked
      rows, jobs panel first, safe-area; in-page modal replaces alert/confirm
- [x] jobs polling pauses when the tab is hidden
- [x] persistent audit log (`audit_path`, default `logs/rag-mcp.log`)
- [x] `GET /api/info` (version/commit/uptime) shown in the header
- [x] `search_memory` `metadata`/`filter` args, both backends
- [ ] one-api: disable/derank the offline fan-out node (gt20) to kill the ~20s
      embedding tail (external; biggest remaining latency)
- [ ] one-api panel: Sync now to pick up the 9-tool schema incl. filter/metadata
- [ ] Path B verification

---

## Dual backend (2026-09-23)

- [x] `VectorStore` interface; Qdrant and file-based Vectra-compatible backends
- [x] `storage.backend` global default + per-collection registry `backend` override
- [x] `Vectra/<collection>/<source_key>/index.json` (ST-compatible shape), atomic writes
- [x] `store_memory` persists raw text under `docs/memory/<date>/` in both backends
- [x] Dashboard: backend column/selector, jobs "clear" button
- [ ] qdrant → vectra export helper (optional)

## Vectorize-time backend choice (2026-09-25)

- [x] Data Bank panel: explicit `database` selector (qdrant/vectra) at Vectorize
- [x] `IndexDocumentOn` / `ReindexDocumentOn` route to an explicit backend;
      `recordIndex` stamps the backend into the registry
- [x] Manage panel backend is read-only; `set_collection_meta` refuses to switch
      a collection that already holds vectors (no more silent orphaning)
- [x] Reject cross-backend name collisions (same name in qdrant and vectra)
- [x] UI: 500/30 chunk defaults, "index" renamed to "Vectorize", busy-spinner +
      toast feedback on all buttons

## Failover + dual defaults (2026-09-24)

- [x] settings: `active_backend`, `default_collection_qdrant`,
      `default_collection_vectra`, `failover_enabled`; legacy
      `default_collection` migrates to the qdrant default
- [x] engine: `IsUnavailable` classifier (conn refused/timeout/5xx) + 15s health
      cache so a dead qdrant doesn't cost a timeout per call
- [x] default scope: active backend's default, auto-fallback to vectra default
- [x] explicit collection_id: same-name fallback when its backend is down
- [x] results tagged with `[backend]`; health_check reports active backend
- [x] responsive dashboard: table scroll-wrap, long-name wrapping, flex sidebar
- [ ] optional: "re-index docs/memory into qdrant" catch-up job after an outage

---

## Architecture

```
MCP Clients → rag-mcp-server (Go, :8199) → Qdrant (:6333)
                                         → one-api (:3000) → Embed/Rerank fan-out (6 devices)
```

---

## Phase 1: Project Skeleton + MCP Protocol + Qdrant Connection

**Goal**: Runnable Go binary that accepts MCP Streamable HTTP requests and connects to Qdrant.

### Tasks

- [ ] `go mod init github.com/RoyChong5053/rag-mcp-server`
- [ ] `main.go` — HTTP server entry point (net/http or gin)
- [ ] `config.go` — YAML config loading (env override support)
- [ ] `config.yaml` — Default configuration file
- [ ] `mcp/protocol.go` — JSON-RPC 2.0 message types (Request, Response, Error)
- [ ] `mcp/server.go` — Streamable HTTP transport handler
  - [ ] `POST /mcp` — Handle JSON-RPC messages
  - [ ] `initialize` — Return server info + capabilities
  - [ ] `tools/list` — Return registered tools
  - [ ] `tools/call` — Dispatch to tool handlers
- [ ] `engine/qdrant.go` — Qdrant REST client wrapper
  - [ ] `CreateCollection(name)` — Create collection with cosine similarity
  - [ ] `CollectionExists(name)` — Check collection existence
  - [ ] `GetCollectionInfo(name)` — Get chunk count, config
  - [ ] `ListCollections()` — List all collections
- [ ] Unit tests for protocol parsing and Qdrant client

### Dependencies

```
go get gopkg.in/yaml.v3
```

Qdrant client: use REST API directly (simpler, no CGO), not gRPC client.

### Qdrant REST API Endpoints Used

```
GET  /collections                        — list all
PUT  /collections/{name}                 — create collection
GET  /collections/{name}                 — collection info
PUT  /collections/{name}/points          — upsert points
POST /collections/{name}/points/search   — search
DELETE /collections/{name}/points        — delete by filter
```

---

## Phase 2: Chunking Engine

**Goal**: Port Vector-Storage-5053's chunking logic to Go. Exact behavioral parity.

### Tasks

- [ ] `chunking/splitter.go`
  - [ ] `splitRecursive(text, maxChunkSize, delimiters)` — Recursive splitting on delimiters
  - [ ] Delimiter priority: `["\n\n", "\n", " ", ""]`
  - [ ] When forced delimiter set: prepend it to priority list
  - [ ] Handle edge cases: empty text, single-char chunks, no-delimiter text
- [ ] `chunking/overlap.go`
  - [ ] `overlapChunks(chunks, overlapSize)` — Add sentence-boundary overlap
  - [ ] `trimToStartSentence(text)` — Find sentence start (。！？\n or beginning)
  - [ ] `trimToEndSentence(text)` — Find sentence end (。！？\n or end)
  - [ ] Half overlap from previous chunk end, half from next chunk start
- [ ] `chunking/validator.go`
  - [ ] Detect broken emoji (lone surrogate pairs)
  - [ ] Detect empty chunks
  - [ ] Detect tiny chunks (<8 chars)
  - [ ] Detect duplicate hashes
  - [ ] Report problems without mutating data
- [ ] `chunking/splitter_test.go` — Table-driven tests matching ST behavior

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| chunk_size | 500 chars | Max chars per chunk (overlap excluded) |
| overlap_percent | 30% | Overlap size = chunk_size * percent / 100 |
| delimiters | `\n\n`, `\n`, ` `, `""` | Priority order for splitting |

### Behavioral Notes

- Overlap is EXCLUDED from chunk_size calculation: `effective_chunk = chunk_size - overlap_size`
- Overlap text is trimmed to sentence boundaries (not mid-sentence)
- `splitRecursive` tries largest delimiter first; if resulting chunk still too big, tries next delimiter
- Empty-string delimiter means "split at every character" (last resort)

---

## Phase 3: Embedding + Rerank via one-api

**Goal**: Call one-api's embedding and rerank endpoints with fan-out support.

### Tasks

- [ ] `engine/embedding.go`
  - [ ] `CreateEmbeddings(ctx, texts) ([][]float32, error)`
  - [ ] POST `{oneapi_base_url}/v1/embeddings`
  - [ ] Body: `{ "model": "embedding", "input": ["text1", ...] }`
  - [ ] Parse response: `data[].embedding` → `[][]float32`
  - [ ] Batch support (one-api handles fan-out internally)
  - [ ] Error handling with context
- [ ] `engine/rerank.go`
  - [ ] `Rerank(ctx, query, documents, topN) ([]RerankResult, error)`
  - [ ] POST `{oneapi_base_url}/v1/rerank`
  - [ ] Body: `{ "model": "reranker", "query": "...", "documents": [...], "top_n": N }`
  - [ ] Parse response: `results[].index`, `results[].relevance_score`
  - [ ] Query truncation: 2000 chars max
  - [ ] Document truncation: 1000 chars max
  - [ ] Score mode detection: probability [0,1] vs logit (needs sigmoid)
- [ ] `engine/hasher.go`
  - [ ] `GetStringHash(text) uint32` — Match ST's `calculateHash` behavior
  - [ ] Need to reverse-engineer ST's hash function or use compatible algorithm
- [ ] Integration tests with mock one-api responses

### one-api Endpoints

| Endpoint | Model | Fan-out |
|----------|-------|---------|
| `POST /v1/embeddings` | `embedding` | All embed devices |
| `POST /v1/rerank` | `reranker` | All rerank devices |

Primary: `http://192.168.100.20:3000`
Backup: `http://192.168.10.2:3000`

---

## Phase 4: Core MCP Tools

**Goal**: Implement all 6 MCP tools. End-to-end working RAG pipeline.

### Tools

- [ ] `search_memory`
  - Params: `query` (required), `collection_id?`, `top_k?=10`, `rerank?=true`, `threshold?=0.25`
  - Flow: query → embed → Qdrant search (top_k * 3 if rerank) → optional rerank → return top_k
  - Returns: `[{ text, score, source, metadata }]`

- [ ] `index_document`
  - Params: `path` (required), `collection_id` (required), `metadata?={}`
  - Flow: read file → chunking → embed batch → upsert to Qdrant
  - Returns: `{ chunks_indexed, collection }`

- [ ] `index_text`
  - Params: `text` (required), `collection_id` (required), `metadata?={}`
  - Flow: chunking → embed batch → upsert to Qdrant
  - Returns: `{ chunks_indexed, collection }`

- [ ] `delete_memory`
  - Params: `filter?`, `collection_id?`
  - Flow: build Qdrant filter → delete points
  - Returns: `{ deleted_count }`

- [ ] `list_collections`
  - Params: none
  - Flow: Qdrant GET /collections → return names + chunk counts
  - Returns: `[{ name, chunk_count, created_at }]`

- [ ] `health_check`
  - Params: none
  - Flow: check Qdrant ping + one-api embedding test + one-api rerank test
  - Returns: `{ qdrant, embedding, rerank }` with status per component

### Implementation Order

1. `list_collections` (simplest, validates Qdrant connection)
2. `index_text` (validates chunking + embedding pipeline)
3. `search_memory` (validates full search + optional rerank)
4. `index_document` (adds file I/O on top of index_text)
5. `delete_memory` (filter-based deletion)
6. `health_check` (composite check)

---

## Phase 5: Obsidian Vault Sync

**Goal**: Incremental sync of Obsidian vault to Qdrant collections.

### Tasks

- [ ] `watcher/vault_watcher.go`
  - [ ] `SyncVault(vaultPath, collectionId, force) — SyncResult`
  - [ ] Scan vault for `*.md` files
  - [ ] Compute file hash (SHA256 of content)
  - [ ] Compare with stored hashes in Qdrant metadata
  - [ ] New/modified files → chunk + embed + upsert
  - [ ] Deleted files → remove corresponding points
  - [ ] Dry-run mode for preview
- [ ] MCP tool `sync_vault` wrapping the above
- [ ] Cronjob-ready: exit code 0 = no changes, 1 = error, 2 = changes applied

### Hash Tracking

Store file hashes in a dedicated Qdrant collection `rag_file_hashes`:
```
Points:
  - id: file_path hash
  - payload: { path, content_hash, last_synced_at, chunk_count }
```

---

## Phase 6: Deploy + Register to one-api

### Tasks

- [ ] Build for linux/amd64 (m64) and linux/arm64 (phones)
- [ ] `rag-mcp-server.service` — systemd unit file
- [ ] Deploy to m64: `scp` → `systemctl enable --now`
- [ ] Register in one-api admin:
  ```
  POST /api/mcp_servers
  {
    "name": "rag-memory",
    "base_url": "http://127.0.0.1:8199/mcp",
    "protocol": "streamable_http",
    "auth_type": "none"
  }
  ```
- [ ] Verify: one-api syncs tools, `/mcp` proxy exposes them
- [ ] Test from Claude Desktop / Claude Code / ST

### systemd Service

```ini
[Unit]
Description=RAG MCP Server
After=network.target qdrant.service

[Service]
Type=simple
ExecStart=/usr/local/bin/rag-mcp-server -config /etc/rag-mcp-server/config.yaml
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## Open Questions

- [ ] Hash function compatibility: need to verify ST's `calculateHash` algorithm for dedup
- [ ] Qdrant collection naming convention: `rag_` prefix? or bare names?
- [ ] Multi-tenancy: separate collections per user vs shared with metadata filter?
- [ ] ST plugin handoff: keep Vector-Storage-5053 for chat RAG, or migrate chat to RAG MCP?
