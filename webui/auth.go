package webui

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"time"
)

// SessionManager issues opaque Bearer tokens after a username/password login
// (one-api style). Tokens persist to a JSON file so a restart doesn't log
// every browser out. No new dependencies: passwords are hex(sha256(password))
// compared in constant time; LAN-only dashboard, same threat model as before.
type SessionManager struct {
	mu       sync.Mutex
	sessions map[string]time.Time // token -> expiry (UTC)
	path     string
}

// NewSessionManager loads persisted sessions (missing/corrupt file = empty).
func NewSessionManager(path string) *SessionManager {
	m := &SessionManager{sessions: map[string]time.Time{}, path: path}
	if path == "" {
		return m
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return m
	}
	var raw map[string]string
	if err := json.Unmarshal(data, &raw); err != nil {
		return m
	}
	now := time.Now()
	for tok, expStr := range raw {
		exp, err := time.Parse(time.RFC3339, expStr)
		if err != nil || !exp.After(now) {
			continue
		}
		m.sessions[tok] = exp
	}
	return m
}

// Issue creates a token valid for ttl and persists the table.
func (m *SessionManager) Issue(ttl time.Duration) (token string, exp time.Time) {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		// Fallback: timestamp-derived (still unpredictable enough for LAN use).
		sum := sha256.Sum256([]byte(time.Now().UTC().String()))
		copy(b[:], sum[:])
	}
	token = hex.EncodeToString(b[:])
	exp = time.Now().Add(ttl).UTC()
	m.mu.Lock()
	m.sessions[token] = exp
	m.persistLocked()
	m.mu.Unlock()
	return token, exp
}

// Valid reports whether token exists and hasn't expired.
func (m *SessionManager) Valid(token string) bool {
	if token == "" {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	exp, ok := m.sessions[token]
	if !ok {
		return false
	}
	if time.Now().After(exp) {
		delete(m.sessions, token)
		m.persistLocked()
		return false
	}
	return true
}

// Revoke drops one token (logout). Returns true if it existed.
func (m *SessionManager) Revoke(token string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.sessions[token]; !ok {
		return false
	}
	delete(m.sessions, token)
	m.persistLocked()
	return true
}

func (m *SessionManager) persistLocked() {
	if m.path == "" {
		return
	}
	raw := make(map[string]string, len(m.sessions))
	now := time.Now()
	for tok, exp := range m.sessions {
		if exp.After(now) {
			raw[tok] = exp.Format(time.RFC3339)
		}
	}
	data, err := json.MarshalIndent(raw, "", "  ")
	if err != nil {
		return
	}
	tmp := m.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return
	}
	_ = os.Rename(tmp, m.path)
}

// VerifyPassword compares hex(sha256(password)) against the stored hash in
// constant time. Accepts upper/lowercase hex.
func VerifyPassword(storedHex, password string) bool {
	storedHex = strings.TrimSpace(strings.ToLower(storedHex))
	if len(storedHex) != 64 {
		return false
	}
	sum := sha256.Sum256([]byte(password))
	got := hex.EncodeToString(sum[:])
	return subtle.ConstantTimeCompare([]byte(got), []byte(storedHex)) == 1
}

// BearerToken extracts the token from "Bearer <token>".
func BearerToken(h string) string {
	if h == "" {
		return ""
	}
	parts := strings.SplitN(h, " ", 2)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "bearer") {
		return ""
	}
	return strings.TrimSpace(parts[1])
}
