package engine

import (
	"net"
	"testing"
)

// closedPort returns a localhost port that is guaranteed to refuse connections,
// so a Qdrant client pointed at it behaves like a down backend.
func closedPort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	if err := l.Close(); err != nil {
		t.Fatalf("close listener: %v", err)
	}
	return port
}

// A down qdrant must surface as unavailable, not as "collection absent":
// otherwise the engine cannot mark the backend down and fail over.
func TestQdrantCollectionExistsUnavailable(t *testing.T) {
	c := NewQdrantClient("127.0.0.1", closedPort(t))
	exists, err := c.CollectionExists("global_memory")
	if exists {
		t.Fatal("collection should not exist on a down backend")
	}
	if !IsUnavailable(err) {
		t.Fatalf("expected unavailable error, got %v", err)
	}
}

// Ping must likewise report the dead backend as unavailable.
func TestQdrantPingUnavailable(t *testing.T) {
	c := NewQdrantClient("127.0.0.1", closedPort(t))
	if err := c.Ping(); !IsUnavailable(err) {
		t.Fatalf("expected unavailable ping error, got %v", err)
	}
}
