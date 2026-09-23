package engine

import "errors"

// backendUnavailableError marks a backend as unreachable (connection refused,
// timeout, 5xx). The engine uses IsUnavailable to decide whether a request may
// fail over to another backend. Logical errors (404 collection missing, 400
// bad request) are deliberately NOT wrapped this way.
type backendUnavailableError struct{ err error }

func (e *backendUnavailableError) Error() string { return e.err.Error() }
func (e *backendUnavailableError) Unwrap() error { return e.err }

func unavailable(err error) error {
	if err == nil {
		return nil
	}
	var already *backendUnavailableError
	if errors.As(err, &already) {
		return err
	}
	return &backendUnavailableError{err: err}
}

// IsUnavailable reports whether err means the backend could not be reached, as
// opposed to a logical error such as a missing collection.
func IsUnavailable(err error) bool {
	var u *backendUnavailableError
	return errors.As(err, &u)
}

// otherBackend returns the counterpart backend used for same-name failover.
func otherBackend(b string) string {
	switch b {
	case BackendQdrant:
		return BackendVectra
	case BackendVectra:
		return BackendQdrant
	}
	return ""
}

// VectorStore is the storage surface shared by every backend (Qdrant REST,
// local file-based Vectra-compatible store). The RAG pipeline is written
// against this interface so a deployment can pick a backend without touching
// search/index logic. Methods keep the historical Qdrant client signatures.
type VectorStore interface {
	CreateCollection(name string) error
	CollectionExists(name string) (bool, error)
	GetCollectionInfo(name string) (*StoreCollectionInfo, error)
	ListCollections() ([]StoreCollectionInfo, error)
	UpsertPoints(collection string, points []Point) error
	Search(collection string, vector []float32, limit int, threshold float64) ([]StoreSearchResult, error)
	DeletePoints(collection string, filter map[string]any) error
	DeleteCollection(name string) error
	SetPayload(collection string, payload map[string]any, filter map[string]any) error
	Ping() error
}

// Backend names accepted by storage.backend and registry entries.
const (
	BackendQdrant = "qdrant"
	BackendVectra = "vectra"
)

var _ VectorStore = (*QdrantClient)(nil)

// Point is one stored vector plus its payload.
// ID must be an unsigned integer or UUID (Qdrant rejects arbitrary strings);
// the file backend stringifies it.
type Point struct {
	ID      uint64         `json:"id"`
	Payload map[string]any `json:"payload"`
	Vector  []float32      `json:"vector,omitempty"`
}

// StoreCollectionInfo is a collection name plus its live vector count.
type StoreCollectionInfo struct {
	Name       string `json:"name"`
	ChunkCount int    `json:"chunk_count"`
}

// StoreSearchResult is a raw nearest-neighbour hit from a backend.
type StoreSearchResult struct {
	ID      any            `json:"id"`
	Score   float64        `json:"score"`
	Payload map[string]any `json:"payload"`
}
