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
│  Tools: recall, store, delete, list,        │
│         search_document, read_document      │
└──────┬──────────────────────┬───────────────┘
       │                      │
  ┌────▼──────────────┐  ┌────▼─────────────────┐
  │ VectorStore       │  │ one-api (:3000)      │
  │ Qdrant :6333  or  │  │ /v1/embeddings       │
  │ Vectra/ (files)   │  │ /v1/rerank (fan-out) │
  └───────────────────┘  └──────────────────────┘
       ▲
       │  the same raw text, searchable by literal string
┌──────┴───────────────────────────────┐
│  documents/ (the vault)              │
│  <date>/<channel>/<category>/<doc>   │
│  indexed by ripgrep, not by a tree    │
└──────────────────────────────────────┘
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

# The raw-text corpus behind search_document. Every document is filed by
# machine under <dir>/<yyyy-mm-dd>/<channel>/<category>/, so the date comes
# first: a date range is then a list of directories, not a walk.
documents:
  dir: ""                    # empty = <docs_dir>/documents
  search_backend: "auto"     # auto (ripgrep if present) | rg | builtin
  rg_path: ""                # explicit ripgrep binary; empty = PATH
  timeout_seconds: 60
  max_matches: 200
  per_file: 5
  ledger_window_days: 30
  categories:                # rules; the first match wins
    - match: "^github\\.com"
      category: code

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
- `query_max_chars` (query truncation before embedding **and** reranking;
  runes, head kept, `0` = unlimited), `doc_max_chars` (rerank document
  truncation, runes)

Edit these from the dashboard's **Search Defaults** panel; `config.yaml` only
seeds the initial values. A default collection that does not exist in its own
backend is rejected loudly, so a typo can never silently widen searches into a
global scan. A pre-existing single `default_collection` key is migrated to
`default_collection_qdrant` on load.

## Storage Backends

`storage.backend` selects the global default; a collection can override it with
its registry `backend` field. Both backends share one `VectorStore` interface,
so search/index/delete behave identically.

**Choosing a backend happens at vectorize time.** The Data Bank's
`Vectorize → job` panel has a `database` selector (server default / `qdrant` /
`vectra`); the choice is written into the collection's registry entry as a
**fixed** property. The dashboard's manage panel shows it read-only, and
`set_collection_meta` refuses to switch a collection that already holds vectors
(that would strand its chunks in the old store). To move a collection, re-index
it into the other backend. Creating a same-name collection in the other backend
is rejected too, so a name never silently lives in two stores at once.

The global `active_backend` setting is unrelated: it only picks which default
collection an omitted-`collection_id` search/write uses, and never decides where
a new named collection is created.

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
| `search_memory` | Semantic search. Omitted `collection_id` uses the active backend's default, with automatic vectra fallback when qdrant is down; optional `top_k`/`threshold` override the server defaults. Disabled collections are always skipped. For exact strings (URLs, IDs, error codes, version numbers, names) use `search_document` instead — embedding similarity fuzzes over those |
| `store_memory` | Vectorize and store text for later recall (the write counterpart of `search_memory`). The raw text is filed in the documents vault as well, then indexed. Omitted `collection_id` uses the active backend's default with qdrant→vectra failover; errors loudly if there is none |
| `delete_memory` | Delete by filter or collection |
| `list_collections` | List all collections with chunk counts |
| `collection_info` | Live stats + registry metadata, one or all |
| `set_collection_meta` | Display name, tags, consumers, enabled flag, backend (backend refused once the collection holds vectors) |
| `delete_collection` | Drop whole collection, requires `confirm:true` |
| `health_check` | Verify all components are healthy and report the active backend |
| `search_document` | Search the raw documents of the vault for an exact string. Returns the document's title, URL and dates next to every hit. Literal by default, `mode:"regex"` for a regular expression; `channel`/`category`/`since`/`until` narrow the search |
| `read_document` | Read one document out of the vault (optionally a line range) |
| `list_documents` | List the documents of the vault, newest first. Reads the index, not the corpus, so it stays fast at any size |
| `documents_import` | File a folder of raw material into the vault: every text file is dated, categorized and named after its content hash; files that are already there are skipped |
| `web_search` | Search the live web with the provider selected in the dashboard, and return its hits. It always queries the live web — it never answers from stored material — and every hit carries its publication date. The result text is also vectorized into the web collection in the background, so later turns can recall it with `search_memory`. The tool takes only `query`; count, freshness, topic, language, store, ingest and the provider are set in the dashboard |

### The documents vault

Semantic recall and literal search answer different questions. The vector side
answers *"what did I learn about X"*; it is bad at exact strings, because
embedding similarity happily puts `v1.2.3` next to `v1.2.4`. The vault
answers *"where is that exact string"*.

The vault is the raw-text corpus: every page a web search filed, every memory
an agent stored, every file that was uploaded or imported. It is never curated
by hand. Documents are filed by machine under

```
<documents_dir>/<yyyy-mm-dd>/<channel>/<category>/<HHMMSS>-<sha8>-<slug>.<ext>
```

- **the date comes first** so a date range is a list of directories instead of
  a walk over the whole corpus — "search the last week" touches seven folders
  no matter how large the corpus is;
- **the channel** says where the document came from (`web`, `agent`, `upload`,
  `chat`, `import`);
- **the category** comes from a rule table in the config (host, title or tag
  match); a document that matches no rule lands in `_`;
- **the file name** carries the time and the first 8 hex of the body hash, so
  the same text is always the same document.

The body hash is the identity: a body that is already in the vault refreshes
that document instead of adding a second copy, and a document whose text
changed becomes a new one (the corpus keeps the versions).

Each document carries a YAML front-matter with its title, URL, dates, channel
and collection. The front-matter is searchable too, so looking for a title or
a URL finds the document that carries them.

### Searching the vault

`search_document` uses ripgrep when the host has it, and a built-in scanner
when it does not. Both return the same shape, and a test asserts they agree.
ripgrep is never bundled into the binary: it is a normal program (the server
this was written for has rg 14.1.0), and the built-in scanner is what keeps
the feature independent of the host.

```
search_document("error code 404")        → the exact string, in the raw document
search_document("code \\d+", mode=regex) → a pattern
search_document("ripgrep", channel=web)  → only what the web search filed
```

### Web search

The server can pull live web pages in and turn them into vectors, so a shared
knowledge base stays current without every client owning a search key.

```
web_search(query)   →  live search → the provider's results, with dates → filed in the background
search_memory(query)→  recall what was searched before, with its dates
```

- **web_search is honest.** It always queries the live web and reports what
  the provider returned. It never answers from the library: a tool named
  `web_search` that quietly serves stored material lies to its caller, and
  yesterday's weather is not today's weather. Recalling what was searched
  before is `search_memory`'s job, and the caller decides when it needs
  fresh truth — a package that was renamed again since the last search, for
  instance.
- **Filing happens in the background.** The call returns as soon as the
  provider answered; the pages it found are then chunked and embedded into the
  web collection by a background job — one job at a time, pages in rank
  order, so the most relevant material lands first. The reply names the job
  and the dashboard's Jobs panel shows it running. This used to happen
  inline: a search that vectorized ten pages took longer than any caller's
  timeout, so the fresh material was lost even though the server had it.
  A page whose content is unchanged is not embedded again, and a page whose
  embedding fails keeps its previous version until the new one is ready.
- **Dates travel with everything.** A search hit shows its publication date
  (or `unknown`), and a recalled web page shows its publication date, its
  fetch date, its age, and a stale flag once it is older than
  `stale_after_days`. A stale page is still returned: knowing that something
  is old is more useful than not knowing it exists. Some providers do not
  date their results at all (Firecrawl's web source never does); the server
  makes a best effort at recovering the date from the page itself and leaves
  it `unknown` when it cannot.
- **Everything the selected provider returned is stored** (`ingest` in the
  WebUI, 0 = all of it). Deciding what is worth keeping is the operator's
  job; a server that silently drops material makes the knowledge base
  impossible to reason about. One page lives per canonical URL, so
  re-searching the same article replaces it instead of piling up. Turning
  `store` off searches without filing anything. Lowering `ingest` or
  `max chars` does NOT save search-API credits — the provider has already
  fetched (and billed for) the full pages by the time the results come back;
  it only shortens the embedding work and throws material away.
- **The selected provider is the only provider called.** The four providers
  are independent modules: `deepseek` (search + full text in one request,
  with publication date and a source-authority tier), `tavily` (page-level
  chunks and a `search_depth` cost trade-off), `firecrawl` (anti-bot
  scraping, news results, country/location), `searxng` (self-hosted, free, no
  key). `search.providers` in config.yaml only says which modules to build;
  order is irrelevant.
- **No failover.** The *Search Defaults* panel has a "web search provider"
  dropdown, and `web_search` calls exactly that provider. An empty selection
  is an error telling the operator to choose; a selected provider that is not
  configured fails with its name. The call never switches to another backend
  behind the caller's back, because then the answer would look like it came
  from a provider that never ran. Changing the setting takes effect
  immediately, no restart.
- **Provider options live in the WebUI too**: DeepSeek max count, Tavily
  search depth and country, Firecrawl country and location, SearXNG language,
  categories, safesearch and engines. Only the selected provider's block is
  read; the others are kept for switching. API keys and base URLs stay in
  config.yaml/env and never enter `settings.json`.
- **API keys** come from environment variables (`api_key_env`), never from
  the tracked config file.

```yaml
search:
  enabled: true
  collection: "web_cache"      # keep web content apart from personal memory
  providers: [deepseek, tavily, firecrawl, searxng]  # modules to build; order irrelevant
  store: true                # seed for the WebUI toggle
  ingest_top_n: 0           # 0 = store every result
  freshness_days: 0         # default age limit on search results (0 = none)
  stale_after_days: 7       # older than this is flagged stale when recalled
  max_content_chars: 20000
  topic: "general"
  lang: ""
  deepseek:
    base_url: "https://api.deepseek.com"
    api_key_env: "DEEPSEEK_API_KEY"
    max_count: 50
  tavily:
    base_url: "https://api.tavily.com"
    api_key_env: "TAVILY_API_KEY"
    search_depth: "basic"
    country: ""
  firecrawl:
    base_url: "https://api.firecrawl.dev/v2"
    api_key_env: "FIRECRAWL_API_KEY"
    country: "MY"
    location: ""
  searxng:
    base_url: "http://127.0.0.1:8899"
    language: ""
    categories: "general"
    safesearch: 0
    engines: []
```

These values only seed `settings.json` on first run. After that the
dashboard is authoritative: changing the provider, count, freshness, topic,
language, store/ingest, max chars or any per-provider option takes effect on
the next `web_search`, without a restart.

### Self-hosting SearXNG

SearXNG is the free, key-less provider module. Two settings decide whether
it works at all: its `json` output format (off by default) and its bind
address. Deploy it as a container next to the server, on loopback only —
`web_search` runs on the same host, so nothing else needs to reach it:

```yaml
# docker-compose.yml (pin the image to the commit you read the docs for)
services:
  core:
    image: docker.io/searxng/searxng:2026.10.25-54ca1d9cf
    restart: unless-stopped
    ports: ["127.0.0.1:8899:8899"]
    env_file: ./.env          # SEARXNG_PORT=8899, SEARXNG_SECRET=<random>
    volumes: ["./core-config/:/etc/searxng/:rw"]
```

```yaml
# core-config/settings.yml
use_default_settings: true
search:
  formats: [html, json]       # without json the API returns an HTML page
server:
  limiter: false              # private instance; no valkey needed
  method: "GET"               # the Go client uses GET
engines:
  - name: bing
    disabled: false           # disabled by default in recent releases
```

Then point `search.searxng.base_url` at it (`http://127.0.0.1:8899`) and
restart the server. Engine availability depends on the host's egress IP: the
`unresponsive_engines` list in the JSON response is logged, and `/stats` in
SearXNG's own UI shows per-engine reliability. Preferences set in that UI are
cookie-scoped and do **not** affect API calls — an engine has to be enabled in
`settings.yml` to be used by `web_search`. The dashboard's SearXNG options
(`language`, `categories`, `safesearch`, `engines`) narrow the search further;
`engines` is the quickest way to exclude a flaky upstream.

## Management dashboard

Admin UI + JSON API on `:8198` (defaults to `0.0.0.0` so every LAN device can
reach it; there is no TLS — restrict the bind or firewall it on untrusted
networks). Includes a **Search Defaults** panel (active backend, per-backend
default collections, failover toggle, top_k, threshold, rerank/recall, truncation
limits) shared live by every frontend, plus the full web-search panel —
provider, results, freshness, topic, language, store/ingest, max chars and the
per-provider options (DeepSeek max count, Tavily depth/country, Firecrawl
country/location, SearXNG language/categories/safesearch/engines):
list/search collections, edit metadata, backfill payloads, recall test with
threshold-kill report, and an audit tail.
Collection metadata lives in `collections.json` (server-local, gitignored).

### Admin login (optional, one-api style)

Set both `admin.username` and `admin.password_sha256` in `config.yaml`
(`password_sha256 = hex(sha256(password))`, e.g. `echo -n 's3cret' | sha256sum`).
All `/api/*` except `health`, `info` and `login` then require
`Authorization: Bearer <token>`; `/mcp` stays open for one-api/clients.
The browser keeps the token (`remember-me` = localStorage ~30d, otherwise the
tab's sessionStorage), so the password is typed once. Sessions persist in
`sessions.json` (gitignored, 0600) across restarts.

### Audit log rotation

The audit log rotates at startup once it exceeds `audit_max_mb` (default 20,
keep 3 backups: `.1/.2/.3`), and `/api/audit` tails via seek (last 256KB)
instead of reading the whole file, so a large log never stalls the dashboard.

## Data Bank (dashboard)

- **Documents** panel: search the raw corpus with a keyword (literal or
  regex, case and whole-word toggles, channel and date filters), see the
  matching lines, and open the document to read it with the hits highlighted.
  The listing comes from the vault's index, never from a walk of the corpus,
  so the dashboard stays usable when the corpus has millions of files
- Upload a file: it is filed into the vault by machine (dated, categorized,
  named after its content hash)
- **Vectorize** any document into a collection with an explicit `database`
  (qdrant/vectra), per-job `chunk_size`/`overlap_percent` (defaults 500/30),
  append or rebuild, plus a zero-token chunk preview
- Background jobs (max 2 concurrent, search never blocked) with sidebar progress
  and a **clear** button for finished jobs
- Provenance snapshot per collection: chunk size/overlap, embed model, source sha

## Maintenance commands

Maintenance is operational work, not agent work, so it lives in the server's
CLI instead of the MCP tool surface. A tool an agent can call is a tool an
agent will call — and "check whether the vector index is still consistent" is
not something an agent needs at 3am.

```bash
rag-mcp-server index-verify [collection]      # catalog<->index consistency
rag-mcp-server index-prune <collection>        # drop empty sources
rag-mcp-server index-import <path>             # make a copied store live again
rag-mcp-server verify-deep <collection>        # structure + text coverage + vector drift
rag-mcp-server docs-verify                     # vault against its ledger
rag-mcp-server docs-migrate [-apply]           # move legacy raw files into the vault
rag-mcp-server docs-search -query <text>       # the same search the dashboard uses
```

`docs-migrate` is a dry run unless `-apply` is given. It moves the legacy
`docs/memory/web/`, `docs/staging/` and `docs/chat_history/` files into the
vault, puts the old files aside in `docs/.migrated/`, and re-points the
vector index at the new files so a recall still knows where the text lives.


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
