# rag-mcp-server · Plan

> 最后更新：2026-11-17
> 状态：**设计定案，待实施**（本轮只做规划，未改动代码）
> 配套计划：`TavernLab/plan.md`（agent 模式、记账 app、记账数据模型）

---

## 0. 现状与问题

### 0.1 当前工具面（16 个）

```
search_memory, store_memory, search_document, read_document, list_documents,
documents_import, get_chunk, update_memory, delete_memory, tag_tree,
collection_info, list_collections, set_collection_meta, delete_collection,
web_search, health_check
```

### 0.2 实测使用率

- TavernLab 实际只调用 **4 个**：`search_memory`、`store_memory`、`web_search`、`collection_info`。
- 即 **25% 使用率**，其余 12 个基本闲置。

### 0.3 工具膨胀的代价

每个暴露给模型的工具都不免费：

- **占 system prompt token**：16 个 schema 在 16k 窗口里能吃掉 1–2k，直接挤占聊天/记忆预算。
- **污染选择**：工具越多，越容易挑错、乱调、编造参数。
- **扩大风险面**：每个写工具都是一个可能被误触的洞。

> 原则：**给模型看的菜单要小，能力不能丢。**

---

## 1. 目标：精简、高频、可组合

### 1.1 三条原则

1. **Agent 面 ≠ Admin 面。** 模型每轮要用的走 MCP；生命周期操作（建/删 collection、改 meta、批量导入、健康检查、schema 管理）归人/CLI/cron，不进模型菜单。
2. **一个动词，多种操作，靠参数区分。** 不要 `record_append` / `record_update` / `record_delete` 三件套；一个 `record_write`，靠每条记录的 `op`。不要 `record_aggregate` / `record_get` / `record_history` 三件套；一个 `record_query`，靠 `aggregate` / `id` / `history` 参数。
3. **可组合优先于可枚举。** 能用一个带参数的通用工具表达的，就不再开专用工具。

### 1.2 目标 surface（16 → 8，含新增 2 个）

**Agent 面（模型可见，≤ 8）：**

| 工具 | 状态 | 吸收 / 说明 |
|---|---|---|
| `search_memory` | 保留 | 核心语义召回 |
| `store_memory` | 保留 | 写记忆；纠正可用参数或 admin `update_memory` |
| `search_document` | 保留 | **吸收 `list_documents`**（空 query + `sort:"recent"`） |
| `read_document` | 保留 | 定位后精读（可接受 chunk_id，吸收 `get_chunk`） |
| `web_search` | 保留 | 独立能力 |
| `collection_info` | 保留 | **吸收 `list_collections` + `tag_tree`**（`include:"tags"`） |
| `record_write` | 🆕 | 结构化写：append / update / delete 靠 `op` |
| `record_query` | 🆕 | 结构化读：过滤 / 聚合 / 单条 / 历史，靠参数 |

**Admin / CLI 面（模型不可见）：** `delete_collection`、`set_collection_meta`、`documents_import`、`update_memory`、`delete_memory`、`health_check`、schema 管理。

---

## 2. 合并 / 降级映射

| 现在 | 处置 |
|---|---|
| `list_collections` | 并进 `collection_info(collection_id:"")`（TavernLab 已这么用） |
| `tag_tree` | 并进 `collection_info(include:"tags")` 或 `record_query` 的标签维度 |
| `delete_memory` | 并进 `update_memory(new_text:"")` 或 admin |
| `update_memory` | 降为 admin/UI（纠错是用户动作，不是每轮 agent 动作） |
| `health_check` | 挪到 HTTP `/health`，**MCP 里本就不该有它** |
| `documents_import` | admin / CLI（批量导入） |
| `set_collection_meta` / `delete_collection` | admin / CLI |
| `list_documents` | 并进 `search_document`（空 query + `sort:"recent"`） |
| `get_chunk` | 保留或并进 `read_document`（按 chunk_id 读）；二选一 |

---

## 3. 新增：通用结构化记录层（`record_*`）

### 3.1 定位

现有两档能力已经很全：

- `search_memory` = **语义**（向量，模糊召回）
- `search_document` = **字面量**（ripgrep，精确字符串）

缺的是**第三档：结构化**（精确过滤 + 确定性聚合）。补齐后 =「语义 / 字面量 / 结构化」三档清晰分工。钱、蒸馏日记、习惯、健康、阅读日志……都是它的 collection。

### 3.2 复用而非新建

不要造新引擎，泛化已有件：

- **存储/窗口/rg 回退** ← `engine/docledger.go` 的 `docLedger`（按月追加 JSONL `.ledger/YYYY-MM.jsonl` + 内存窗口 + 超窗口用 rg）。泛化为 `engine/records.go`。
- **provenance/审计** ← `engine/history.go` 的 `HistoryEntry`（op/source/ok/error）。
- **更新/删除/标签** ← 现有 `update_memory`（supersedes、dry_run）、`tag_tree`（层级标签）的模型。
- **语义桥** ← `record_query` 可触发对指定字段的向量投影（调 `store_memory`），**只索引摘要，不向量化每条明细**。

**新增文件仅两个兄弟文件：** `engine/records.go` + `mcp/record_tools.go`（含测试 `engine/records_test.go`）。核心 fold/query/aggregate 写成纯函数、独立单测，**不碰 bm25 / embedding / fusion**。

### 3.3 通用信封（envelope）

```json
{"v":1,"id":"01J...","ts":"2026-11-03T...","op":"append|update|delete",
 "kind":"txn|diary|habit","data":{...typed...},
 "tags":["餐饮"],"source":"app|agent|ocr|distill","supersedes":null}
```

### 3.4 schema（声明式配置，不是工具）

```yaml
record:
  money:
    version: 1
    fields:
      occurred_at:   {type: ts}
      direction:     {type: enum, values: [in, out]}
      amount_minor:  {type: int}          # 整数最小单位，禁止 float
      currency:      {type: str}
      category:      {type: ref, ref: categories}
      merchant:      {type: str}
      note:          {type: text}
    indexes:   [occurred_at, category]
    searchable: [note, merchant]          # 可选：投影进向量的字段
  memory_diary:
    version: 1
    fields:
      occurred_at: {type: ts}
      session:     {type: str}
      kind:        {type: enum, values: [state, log]}
      text:        {type: text}
      tags:        {type: tag}
      source:      {type: str}
```

写入时按 schema 校验类型（`amount_minor` 非 int 直接拒绝）。schema 放配置，不占工具位。

### 3.5 `record_write`

```
record_write(collection, records[], source?)
```
- 每条：`{op:"append|update|delete", id?, data?, tags?}`。
- 批量、按 id 幂等；`update` 追加带 `supersedes` 的事件（不改历史）；`delete` 写墓碑。
- 校验类型；返回写成功的 id 列表。

### 3.6 `record_query`

一个读工具覆盖四种意图：

```
record_query(collection,
  where?, from?, to?, tags?, text_match?, order?, limit?,   # 默认：返回行
  aggregate?: {group_by?:[], metrics:[{field,op}]},          # 聚合：sum/count/min/max/avg
  id?,                                                       # 单条折叠状态
  history?: true)                                            # 该条全部事件
```
- 默认返回过滤后的行（确定性，不走向量）。
- 带 `aggregate` → Go 里精确计算，返回紧凑表格。
- 带 `id` → 单条当前状态；带 `history` → 事件流（审计/replay）。

**schema 不单独开工具**——写成配置，少一个工具、多一层强约束。

---

## 4. toolset 分层

不要「全开 / 全关」，用配置分档：

```yaml
mcp:
  toolsets:
    agent:
      - search_memory
      - store_memory
      - search_document
      - read_document
      - web_search
      - collection_info
      - record_write
      - record_query
    admin:
      - delete_collection
      - set_collection_meta
      - documents_import
      - update_memory
```

- TavernLab 的 agent 连 `agent` 档 → 模型只看到 8 个。
- WebUI / CLI / cron 连 `admin` 档 → 拿全量。
- 能力一个没丢，**模型的菜单变小**。
- MCP 没有 per-tool ACL，落地用「按配置注册子集」或 `/mcp` vs `/mcp/admin` 两条挂载。

---

## 5. 治理规则（防再度膨胀）

> **新工具只有在「本周真实会话会调用」时才进 agent 档；否则先做 CLI 命令。**

- 给每个工具记调用计数（复用 `history.go` 的日志底子）。
- **连续 N 天零调用的从 agent 档降级**，工具面自动收敛。

---

## 6. 分阶段

### P0 — 配置与收口
- 引入 `toolsets` 配置；按档注册工具子集。
- `health_check` 移到 HTTP；`list_collections` / `tag_tree` 并入 `collection_info`。

### P1 — record 层 MVP（只读不值钱，先打通写）
- `engine/records.go`：泛化 `docLedger` 的「按月 JSONL + 窗口 + rg 回退」为通用记录存储。
- `record_write` + `record_query`（先支持 append + 行查询 + 简单 aggregate）。
- schema 配置 + 类型校验（重点：money=整数）。

### P2 — 聚合与审计完善
- `record_query` 支持 `aggregate`(group_by/metrics)、`id`、`history`。
- `record_index`（可选向量投影，只索引摘要）——或作为 `record_query` 的参数。
- snapshot / compact 维护。

### P3 — 迁移与工具面收敛
- 蒸馏日记从 TavernLab 本地文件迁到 `memory_diary` collection（写穿缓存，见 TavernLab plan.md）。
- 记账 app 接入 `money` collection。
- 依使用计数把闲置工具降级到 admin/CLI。

---

## 7. 风险与护栏

| 风险 | 护栏 |
|---|---|
| MCP 变成 god-service（向量+文档+web+卡片+记录+财务） | record 层是干净兄弟包；业务规则留 app；纯函数可单测 |
| 与 bm25/embedding 耦合 | `engine/records` 不依赖检索内部，只做 fold/query/aggregate |
| 聚合类型错（钱算成 float） | schema 强制 `int`；写入校验；确定性计算 |
| 记录无限增长 | 月度文件 + snapshot + compact |
| 工具面又被加回来 | toolsets + 调用计数降级 + 「本周会被调用」准入规则 |
| 远端路径不可读 | `documents_import` 只读**服务端**路径；本机文件需经 `store_memory` 传文本（TavernLab 归档即如此） |
| 破坏现有向量/文档行为 | record 层新增、独立文件；不改 `search_memory`/`search_document` 代码路径 |

---

## 8. 一句话总结

**把 16+ 个工具收敛为「8 个 agent 工具 + 若干 admin 命令」**：核心为 `search_memory / store_memory / search_document / read_document / web_search / collection_info / record_write / record_query`。靠**参数复用**（op/aggregate/id/history）而非**堆工具**，靠 **toolset 分层** 把 admin 挡在模型菜单外。新增的 `record_*` 只在通用结构化层，服务钱、日记及未来任意结构化流——能力更全，菜单更瘦。
