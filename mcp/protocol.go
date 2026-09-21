package mcp

import "encoding/json"

// JSON-RPC 2.0 message types

type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type Response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  any             `json:"result,omitempty"`
	Error   *ResponseError  `json:"error,omitempty"`
}

type ResponseError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// MCP protocol constants

const (
	JSONRPCVersion = "2.0"
	ProtocolVersion = "2025-06-18"
	ServerName      = "rag-mcp-server"
	ServerVersion   = "0.1.0"
)

// MCP method names

const (
	MethodInitialize          = "initialize"
	MethodNotificationsInit   = "notifications/initialized"
	MethodToolsList           = "tools/list"
	MethodToolsCall           = "tools/call"
)

// Standard JSON-RPC error codes

const (
	ErrParseError     = -32700
	ErrInvalidRequest = -32600
	ErrMethodNotFound = -32601
	ErrInvalidParams  = -32602
	ErrInternal       = -32603
)

// Tool represents an MCP tool definition

type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

// ToolCallRequest represents a tools/call request

type ToolCallRequest struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments,omitempty"`
}

// ToolCallResult represents a tools/call response

type ToolCallResult struct {
	Content []ContentBlock `json:"content"`
	IsError bool           `json:"isError,omitempty"`
}

// ContentBlock represents a content block in MCP response

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// InitializeParams represents the params for initialize request

type InitializeParams struct {
	ProtocolVersion string    `json:"protocolVersion"`
	Capabilities    ClientCapabilities `json:"capabilities"`
	ClientInfo      ClientInfo `json:"clientInfo"`
}

type ClientCapabilities struct {
	Tools any `json:"tools,omitempty"`
}

type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// InitializeResult represents the response to initialize

type InitializeResult struct {
	ProtocolVersion string           `json:"protocolVersion"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	ServerInfo      ServerInfo       `json:"serverInfo"`
}

type ServerCapabilities struct {
	Tools any `json:"tools,omitempty"`
}

type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// ToolsListResult represents the response to tools/list

type ToolsListResult struct {
	Tools []Tool `json:"tools"`
}
