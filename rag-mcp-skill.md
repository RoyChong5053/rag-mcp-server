# rag-mcp-skill

给 agent 的使用指南：怎么用 rag-mcp-server 的这 17 个 MCP 工具，以及每个坑在哪。
本文是**实测出来的**，不是从代码推的。凡是标了「实测」的数字，都是跑出来的。

---

## 一、先选对工具：语义 vs 字面量

这是整套东西最重要的一条。其余细节都是它的推论。

| 你要找的东西 | 用 | 为什么 |
|---|---|---|
| 「我之前学过什么关于 X 的」 | `search_memory` | 向量检索，按语义相似度排序 |
| 「这个精确字符串在哪」 | `search_document` | ripgrep 字面量匹配 |
| URL、ID、错误码、版本号、命令、函数名 | `search_document` | 向量检索会**模糊掉**它们 |
| 人名、概念、话题、观点 | `search_memory` | 字面量匹配找不到 |

**实测过的反面例子**：查 connectionID `a1556d6a-925b-4ddf-94f9-bb5acfc6e0b9`：
- `search_memory` 返回 3 条结果，但逐条核对，**没有一条包含这个 UUID** —— 纯按主题相似度排的噪音，看着像命中，其实不是。
- `search_document` 返回 **0 documents** —— 这是**正确**的，因为语料里只有截断形式 `a1556d6a-...`。

向量的相似度对字面量没有约束力。别拿语义检索的结果当「找到了」。

---

## 二、三个索引，同一个语料

```
向量索引  search_memory   →  语义召回（chunk，带 Source）
字面量索引 search_document →  精确命中（文件 + 行号 + 标题 + URL）
语料本身  vault           →  所有原始文档，都是文件
```

vault 里的文件按机器归档：

```
<documents_dir>/<yyyy-mm-dd>/<channel>/<category>/<HHMMSS>-<sha8>-<slug>.<ext>
```

- **日期在最前**：查日期范围 = 列目录，不是遍历全语料
- **channel**：`web` / `agent` / `upload` / `chat` / `import`
- **category**：按 config 里的规则表分；不匹配任何规则的落 `_`
- **文件名带正文哈希前 8 位**：同一段文字永远是同一篇文档

---

## 三、标准工作流：召回 → 定位 → 读取

### 1. `search_memory` 拿线索

```python
search_memory(collection_id="global_memory", query="MCP 工具列表不刷新", top_k=5)
```

返回每条带 `Source` 字段。**这个字段有坑，见第五节。**

### 2. `search_document` 定位到具体文件 + 行号

```python
search_document(query="Tool not found: index_text", limit=5, context=2)
```

返回形如：

```
Found 1 documents (2 matches, 1 files scanned in 180ms via rg)

--- 1. opencode - all - 20260323-20261004.md ---
Path: 2026-10-22/upload/_/012547-b6a35296-opencode---all---20260323-20261004-md.md
Channel: upload/_  Filed: 2026-10-22T01:25:47+08:00
L132579: MCP 返回 `Tool not found: index_text`...
```

### 3. `read_document` 读那一段

```python
read_document(path="2026-10-22/upload/_/012547-b6a35296-...md",
              from_line=132592, to_line=132597)
```

**`search_document` 返回的 `Path` 可以直接喂给 `read_document`**，这是闭环。

---

## 四、坑（都会真实误导 agent）

### 坑 1：`search_document` 搜的是**内容**，不是文件名

文件名里有的字符串，内容里不一定有。

- 搜 `opencode - all - 20260323-20261004.md` → ✅ 命中（因为 front-matter 的 `Title:` 行就是它）
- 搜 `031915-6bfa50fe-opencode-看不到-rag-mcp-server-新工具.md` → ❌ **0 命中**（内容里的 Title 不带 hash 前缀）

**正确姿势**：搜 **Title**（不带 hash 的那部分），或搜正文里任何一段独特的文字。

### 坑 2：`Source` 里的 `#行号` 是**陈旧坐标**，不能直接用

实测：`Source: opencode - all - 20260323-20261004.md#11658`
→ 那段文字实际在 **L127399**。

11658 既不是行号也不是字节偏移（L11658 在字节 458,522；L127399 在字节 5,444,971）。
这是 vault 迁移重写了文件行结构之前的旧坐标 —— 路径被 repair 修对了，行号没跟着修。

**结论：`Source#lineno` 只能当线索，不能当定位。** 要真实行号，用 `search_document` 反查正文。

### 坑 3：`Source` 有三种形态，provenance 没统一

| 来源 | `Source` 长什么样 | 怎么落到实体文档 |
|---|---|---|
| `store_memory` → agent journal | `031915-6bfa50fe-<标题>.md`（无行号） | 文件名就是 vault 文件名 ✅，但要用 Title 去搜（坑 1） |
| `web_search` → web 集合 | `https://example.com/article` | 搜这个 URL ✅ |
| 导入的聊天日志 | `opencode - all - 20260323-20261004.md#11658` | 文件名被改写过 + 行号陈旧 ⚠️ |

**要跳一跳**（search_memory → search_document → read_document），不是一步到位。

### 坑 4：`read_document` 不给行号 = 一次拉 6MB

实测：`read_document` 不带 `from_line`/`to_line`，读那篇 13 万行的 chat log：

```
chars: 6,061,373   lines: 167,009
```

**永远带上行号范围。**

### 坑 5：`list_documents` 的 `limit` 没有服务端上限

- 默认 `limit=50` → **12,209 字符 / 180 行**，与 vault 多大无关（服务端硬回落 50）
- `limit=99999` → 返回全部（实测 104 篇 = 26,826 字符 / 389 行）

单篇约 244 字节 / 3.6 行，所以：
- 超过 **~210 篇** 才可能撞上 opencode 的 51200 字节阈值
- 超过 **~555 篇** 才可能撞上 2000 行阈值

### 坑 6：opencode 的输出截断**包含 MCP 工具**，但 code mode 例外

opencode v2.1.2 有 `tool_output` 配置：`max_bytes` 默认 **51200**、`max_lines` 默认 **2000**。
超限就落盘到 `~/.local/share/opencode/tool-output/tool_*`，送给模型的是截断版 +
`[showing lines 1-N of M; full output saved to <path>]`。

- 实测：`chrome-devtools_take_snapshot`（MCP）被截断过 1 次 → **MCP 工具确实走这套**
- 实测：code mode 里 `read_document` 返回 6MB **完全没被截断** → code mode 把 MCP 原始值直接塞进沙箱，
  截断只作用于沙箱最终 print 的内容

**在 code mode 里不能指望 opencode 兜底，得自己控参数。**

### 坑 7：MCP 工具列表是**客户端缓存**的

服务端换了二进制、`systemctl restart` 过，客户端**不会**重新拉工具列表 —— 除非服务端发
`notifications/tools/list_changed`（本服务端没实现，无下行通知通道）。

表现：连接还是 `connected`，但工具是旧的。

**解法**（按代价排序）：
1. TUI `/mcps` → 选服务器 → reconnect
2. API：`POST /api/experimental/mcp/<server>/connect?location[directory]=<dir>`
   （认证是 Basic `opencode:<service.json 里的 password>`；`location` 必须用方括号写法）
3. 重启 opencode server

**别只比工具数量** —— 新旧两版可能都是 17 个，只是 4 换 4 对调。

---

## 五、写入路径：存进去就自动落文档

- `store_memory` → 原始文本先进 agent journal，再向量化。vault 里会出现
  `<date>/agent/_/<HHMMSS>-<sha8>-<标题>.md`
- `web_search` → **后台**归档，vault 里出现 `<date>/web/_/...`，带 `url:` 字段。
  调用在 provider 应答后就返回，归档在后台继续，稍后 recall 可能还不完整
- `documents_import` → 把一个目录的文件按机器规则归档进 vault，已存在的跳过

所以「写进去」和「能搜到」是自动连着的，不需要额外动作。

---

## 六、`web_search` 的诚实性

`web_search` **永远查实时网页**，绝不拿库里存的材料冒充 —— 它叫 web_search，
要是偷偷返回存储内容就是在骗调用方。昨天 vs 今天的天气，调用方要自己判断。

要召回「之前搜过什么」，用 `search_memory`，不是 `web_search`。
每个 web 命中都带发布日期（或 `unknown`）。

---

## 七、怎么找到这份文档

这份文档在两处：

- **仓库**：`~/app/rag-mcp-server/rag-mcp-skill.md`（也存进了 `global_memory` 集合）
- **vault**：`store_memory` 之后，会有一篇 `<date>/agent/_/<hash>-rag-mcp-skill.md`

召回它：

```python
search_memory(collection_id="global_memory", query="rag-mcp-skill 如何使用 MCP 工具", top_k=5)
search_document(query="rag-mcp-skill", limit=5)      # 搜内容，不是文件名
```

---

## 速查

| 想干的事 | 调用 |
|---|---|
| 语义召回 | `search_memory(collection_id, query, top_k)` |
| 精确找字符串 | `search_document(query, mode="literal"\|"regex", channel, since, until, word, limit, context)` |
| 读一段 | `read_document(path, from_line, to_line)` ← **务必带行号** |
| 看最近来了什么 | `list_documents(limit=50, channel)` |
| 归档一个目录 | `documents_import(path, collection, reindex, max_files)` |
| 查实时网页 | `web_search(query, count, domains, freshness_days, store, provider, lang, topic)` |
| 抓单页 | `fetch_url(url, ...)` |
| 看集合 | `list_collections()` / `collection_info(collection_id)` |
| 改记忆 | `update_memory(collection_id, ids\|filter, new_text, metadata, dry_run)` |
| 健康检查 | `health_check()` |
