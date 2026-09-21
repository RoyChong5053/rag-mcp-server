package mcp

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
)

// ToolHandler is a function that handles a tool call
type ToolHandler func(args map[string]any) (any, error)

// Server is the MCP Streamable HTTP server
type Server struct {
	mu       sync.RWMutex
	tools    map[string]Tool
	handlers map[string]ToolHandler
}

// NewServer creates a new MCP server
func NewServer() *Server {
	return &Server{
		tools:    make(map[string]Tool),
		handlers: make(map[string]ToolHandler),
	}
}

// RegisterTool registers a tool with its handler
func (s *Server) RegisterTool(tool Tool, handler ToolHandler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools[tool.Name] = tool
	s.handlers[tool.Name] = handler
}

// HandleMCP handles POST /mcp requests
func (s *Server) HandleMCP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		s.sendError(w, nil, ErrInternal, "Failed to read request body")
		return
	}
	defer r.Body.Close()

	var req Request
	if err := json.Unmarshal(body, &req); err != nil {
		s.sendError(w, nil, ErrParseError, "Invalid JSON")
		return
	}

	if req.JSONRPC != JSONRPCVersion {
		s.sendError(w, req.ID, ErrInvalidRequest, "Invalid JSON-RPC version")
		return
	}

	switch req.Method {
	case MethodInitialize:
		s.handleInitialize(w, req)
	case MethodNotificationsInit:
		// No response needed for notifications
		w.WriteHeader(http.StatusNoContent)
	case MethodToolsList:
		s.handleToolsList(w, req)
	case MethodToolsCall:
		s.handleToolsCall(w, req)
	default:
		s.sendError(w, req.ID, ErrMethodNotFound, fmt.Sprintf("Method not found: %s", req.Method))
	}
}

func (s *Server) handleInitialize(w http.ResponseWriter, req Request) {
	result := InitializeResult{
		ProtocolVersion: ProtocolVersion,
		Capabilities: ServerCapabilities{
			Tools: struct{}{},
		},
		ServerInfo: ServerInfo{
			Name:    ServerName,
			Version: ServerVersion,
		},
	}
	s.sendResponse(w, req.ID, result)
}

func (s *Server) handleToolsList(w http.ResponseWriter, req Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	tools := make([]Tool, 0, len(s.tools))
	for _, tool := range s.tools {
		tools = append(tools, tool)
	}

	result := ToolsListResult{Tools: tools}
	s.sendResponse(w, req.ID, result)
}

func (s *Server) handleToolsCall(w http.ResponseWriter, req Request) {
	var params ToolCallRequest
	if err := json.Unmarshal(req.Params, &params); err != nil {
		s.sendError(w, req.ID, ErrInvalidParams, "Invalid tool call params")
		return
	}

	s.mu.RLock()
	handler, ok := s.handlers[params.Name]
	s.mu.RUnlock()

	if !ok {
		s.sendError(w, req.ID, ErrMethodNotFound, fmt.Sprintf("Tool not found: %s", params.Name))
		return
	}

	result, err := handler(params.Arguments)
	if err != nil {
		log.Printf("Tool %s error: %v", params.Name, err)
		s.sendResponse(w, req.ID, ToolCallResult{
			Content: []ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Error: %v", err)},
			},
			IsError: true,
		})
		return
	}

	// Convert result to content blocks
	var content []ContentBlock
	switch v := result.(type) {
	case string:
		content = []ContentBlock{{Type: "text", Text: v}}
	case []ContentBlock:
		content = v
	default:
		data, _ := json.MarshalIndent(v, "", "  ")
		content = []ContentBlock{{Type: "text", Text: string(data)}}
	}

	s.sendResponse(w, req.ID, ToolCallResult{Content: content})
}

func (s *Server) sendResponse(w http.ResponseWriter, id json.RawMessage, result any) {
	resp := Response{
		JSONRPC: JSONRPCVersion,
		ID:      id,
		Result:  result,
	}
	s.sendJSON(w, resp)
}

func (s *Server) sendError(w http.ResponseWriter, id json.RawMessage, code int, message string) {
	resp := Response{
		JSONRPC: JSONRPCVersion,
		ID:      id,
		Error: &ResponseError{
			Code:    code,
			Message: message,
		},
	}
	s.sendJSON(w, resp)
}

func (s *Server) sendJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Mcp-Protocol-Version", ProtocolVersion)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}
