package mcp

import (
	"fmt"
	"strings"

	"github.com/RoyChong5053/rag-mcp-server/engine"
)

// The documents vault is the raw-text half of the system. The vector side
// answers "what did I learn about X"; the vault answers "where is that exact
// string". These tools are the second half.

// RegisterDocumentTools registers the document tools with the server.
func RegisterDocumentTools(server *Server, eng *engine.Engine) {
	server.RegisterTool(Tool{
		Name: "search_document",
		Description: "Search the raw documents of the vault for an exact string. " +
			"Use this when the question is about a literal thing - a URL, an ID, an error code, a version number, a name, a command - " +
			"because semantic search (search_memory) fuzzes over those and a literal search does not. " +
			"Every hit comes back with the document's title, URL and dates, so the document is identified, not just its file name. " +
			"The default mode is a literal, case-insensitive search; pass mode=\"regex\" for a regular expression. " +
			"The corpus is filed by machine under <date>/<channel>/<category>/, so channel and category narrow the search, " +
			"and since/until (yyyy-mm-dd) restrict it to an ingest-date range.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "The string to look for.",
				},
				"mode": map[string]any{
					"type":        "string",
					"description": "Optional: \"literal\" (default) or \"regex\".",
				},
				"channel": map[string]any{
					"type":        "string",
					"description": "Optional: only search one channel (web, agent, upload, chat, import).",
				},
				"category": map[string]any{
					"type":        "string",
					"description": "Optional: only search one category folder.",
				},
				"path": map[string]any{
					"type":        "string",
					"description": "Optional: narrow inside the matched folders (sub-path of the vault).",
				},
				"since": map[string]any{
					"type":        "string",
					"description": "Optional: only documents filed on or after this date (yyyy-mm-dd).",
				},
				"until": map[string]any{
					"type":        "string",
					"description": "Optional: only documents filed on or before this date (yyyy-mm-dd).",
				},
				"word": map[string]any{
					"type":        "boolean",
					"description": "Optional: the match must sit on a word boundary.",
				},
				"case_sensitive": map[string]any{
					"type":        "boolean",
					"description": "Optional: match the case exactly. By default a lower-case query is case-insensitive and an upper-case one is exact.",
				},
				"context": map[string]any{
					"type":        "integer",
					"description": "Optional: how many lines of context each hit carries.",
				},
				"limit": map[string]any{
					"type":        "integer",
					"description": "Optional: cap on the number of returned matches. Defaults to the server setting.",
				},
				"per_file": map[string]any{
					"type":        "integer",
					"description": "Optional: cap on the matches reported per document. Defaults to the server setting.",
				},
			},
			"required": []string{"query"},
		},
	}, func(args map[string]any) (any, error) {
		opts := engine.DocSearchOptions{
			Query:         getStringArg(args, "query", ""),
			Mode:          getStringArg(args, "mode", "literal"),
			Path:          getStringArg(args, "path", ""),
			Channel:       getStringArg(args, "channel", ""),
			Category:      getStringArg(args, "category", ""),
			Since:         getStringArg(args, "since", ""),
			Until:         getStringArg(args, "until", ""),
			Word:          getBoolArg(args, "word", false),
			CaseSensitive: getBoolArg(args, "case_sensitive", false),
			Context:       getIntArg(args, "context", 0),
			Limit:         getIntArg(args, "limit", 0),
			PerFile:       getIntArg(args, "per_file", 0),
		}
		res, err := eng.SearchDocuments(opts)
		if err != nil {
			return nil, err
		}
		return formatDocumentSearch(res), nil
	})

	server.RegisterTool(Tool{
		Name: "read_document",
		Description: "Read one document out of the vault. " +
			"Use it after search_document when the matching line is not enough and the whole document is needed. " +
			"The path is the one search_document returned, relative to the vault root.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Document path relative to the vault root.",
				},
				"from_line": map[string]any{
					"type":        "integer",
					"description": "Optional: first line to return (1-based).",
				},
				"to_line": map[string]any{
					"type":        "integer",
					"description": "Optional: last line to return (1-based).",
				},
			},
			"required": []string{"path"},
		},
	}, func(args map[string]any) (any, error) {
		path := strings.TrimSpace(getStringArg(args, "path", ""))
		if path == "" {
			return nil, fmt.Errorf("path is required")
		}
		doc, err := eng.ReadDocument(path)
		if err != nil {
			return nil, err
		}
		body := doc.Body
		from := getIntArg(args, "from_line", 0)
		to := getIntArg(args, "to_line", 0)
		if from > 0 || to > 0 {
			lines := strings.Split(body, "\n")
			if from < 1 {
				from = 1
			}
			if to < from {
				to = from
			}
			if to > len(lines) {
				to = len(lines)
			}
			body = strings.Join(lines[from-1:to], "\n")
		}
		var b strings.Builder
		b.WriteString(fmt.Sprintf("Path: %s\n", doc.Rel))
		if doc.Meta.Title != "" {
			b.WriteString(fmt.Sprintf("Title: %s\n", doc.Meta.Title))
		}
		if doc.Meta.URL != "" {
			b.WriteString(fmt.Sprintf("URL: %s\n", doc.Meta.URL))
		}
		b.WriteString(fmt.Sprintf("Channel: %s/%s\n", doc.Meta.Channel, doc.Meta.Category))
		if doc.Meta.PublishedAt != "" {
			b.WriteString(fmt.Sprintf("Published: %s\n", doc.Meta.PublishedAt))
		}
		if doc.Meta.FetchedAt != "" {
			b.WriteString(fmt.Sprintf("Filed: %s\n", doc.Meta.FetchedAt))
		}
		b.WriteString("\n")
		b.WriteString(body)
		return b.String(), nil
	})

	server.RegisterTool(Tool{
		Name: "list_documents",
		Description: "List the documents in the vault, newest first. " +
			"It reads the vault's index, not the corpus, so it stays fast when the corpus is huge. " +
			"Use it to see what arrived recently; use search_document to find a specific string.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"channel": map[string]any{
					"type":        "string",
					"description": "Optional: only list one channel (web, agent, upload, chat, import).",
				},
				"limit": map[string]any{
					"type":        "integer",
					"description": "Optional: how many documents to return. Defaults to 50.",
				},
			},
		},
	}, func(args map[string]any) (any, error) {
		channel := strings.TrimSpace(getStringArg(args, "channel", ""))
		limit := getIntArg(args, "limit", 50)
		if limit <= 0 {
			limit = 50
		}
		entries := eng.ListDocuments(channel, limit)
		if len(entries) == 0 {
			return "No documents found.", nil
		}
		var b strings.Builder
		b.WriteString(fmt.Sprintf("Found %d documents:\n\n", len(entries)))
		for _, e := range entries {
			b.WriteString(fmt.Sprintf("- %s\n", e.Path))
			if e.Title != "" {
				b.WriteString(fmt.Sprintf("  title: %s\n", e.Title))
			}
			if e.URL != "" {
				b.WriteString(fmt.Sprintf("  url: %s\n", e.URL))
			}
			b.WriteString(fmt.Sprintf("  channel: %s/%s  filed: %s\n", e.Channel, e.Category, e.FetchedAt))
		}
		return b.String(), nil
	})

	server.RegisterTool(Tool{
		Name: "documents_import",
		Description: "File a folder of raw material into the documents vault. " +
			"Every text file in the folder is filed by machine: dated, categorized and named after its content hash. " +
			"Files that are already in the vault are skipped instead of copied twice. " +
			"Pass a collection to vectorize them at the same time.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Folder on the server holding the raw documents.",
				},
				"collection": map[string]any{
					"type":        "string",
					"description": "Optional: collection to vectorize the imported documents into.",
				},
				"reindex": map[string]any{
					"type":        "boolean",
					"description": "Optional: vectorize the documents into the collection. Default true when a collection is given.",
				},
				"max_files": map[string]any{
					"type":        "integer",
					"description": "Optional: cap on how many files one call files. 0 = 200.",
				},
			},
			"required": []string{"path"},
		},
	}, func(args map[string]any) (any, error) {
		path := strings.TrimSpace(getStringArg(args, "path", ""))
		if path == "" {
			return nil, fmt.Errorf("path is required")
		}
		collection := strings.TrimSpace(getStringArg(args, "collection", ""))
		reindex := getBoolArg(args, "reindex", collection != "")
		maxFiles := getIntArg(args, "max_files", 200)
		if maxFiles <= 0 {
			maxFiles = 200
		}
		res, err := eng.ImportDocuments(path, "import", collection, reindex, maxFiles, nil)
		if err != nil {
			return nil, err
		}
		out := fmt.Sprintf("Filed %d documents (%d skipped as duplicates, %d failed, %d vectorized) from %s",
			res.Imported, res.Skipped, res.Failed, res.Indexed, res.Root)
		if len(res.Errors) > 0 {
			out += "\n\nProblems:\n" + strings.Join(res.Errors, "\n")
		}
		return out, nil
	})
}

// formatDocumentSearch renders a document search for the caller.
func formatDocumentSearch(res *engine.DocSearchResult) string {
	if res == nil || len(res.Documents) == 0 {
		return fmt.Sprintf("No documents matched (%d files scanned in %dms via %s).",
			res.ScannedFiles, res.ElapsedMS, res.Engine)
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Found %d documents (%d matches, %d files scanned in %dms via %s)\n\n",
		len(res.Documents), res.TotalMatches, res.ScannedFiles, res.ElapsedMS, res.Engine))
	for i, doc := range res.Documents {
		b.WriteString(fmt.Sprintf("--- %d. %s ---\n", i+1, docTitle(doc)))
		b.WriteString(fmt.Sprintf("Path: %s\n", doc.Path))
		if doc.URL != "" {
			b.WriteString(fmt.Sprintf("URL: %s\n", doc.URL))
		}
		b.WriteString(fmt.Sprintf("Channel: %s/%s", doc.Channel, doc.Category))
		if doc.PublishedAt != "" {
			b.WriteString(fmt.Sprintf("  Published: %s", doc.PublishedAt))
		}
		if doc.FetchedAt != "" {
			b.WriteString(fmt.Sprintf("  Filed: %s", doc.FetchedAt))
		}
		b.WriteString("\n")
		for _, m := range doc.Matches {
			b.WriteString(fmt.Sprintf("L%d: %s\n", m.Line, m.Text))
			for _, c := range m.Context {
				b.WriteString(fmt.Sprintf("    | %s\n", c))
			}
		}
		b.WriteString("\n")
	}
	if res.Truncated {
		b.WriteString("(more matches were found; narrow the search with channel/category/since/until)\n")
	}
	return b.String()
}

// docTitle names a document for the reader.
func docTitle(doc engine.DocSearchHit) string {
	if doc.Title != "" {
		return doc.Title
	}
	return doc.Path
}
