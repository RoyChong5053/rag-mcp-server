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

// VectorStore is the storage surface for the local file-based
// Vectra-compatible store. The RAG pipeline is written against this interface.
type VectorStore interface {
	CreateCollection(name string) error
	CollectionExists(name string) (bool, error)
	GetCollectionInfo(name string) (*StoreCollectionInfo, error)
	ListCollections() ([]StoreCollectionInfo, error)
	UpsertPoints(collection string, points []Point) error
	Search(collection string, vector []float32, limit int, threshold float64, filter map[string]any) ([]StoreSearchResult, error)
	DeletePoints(collection string, filter map[string]any) error
	DeleteCollection(name string) error
	SetPayload(collection string, payload map[string]any, filter map[string]any) error
	Ping() error
}

// Backend names accepted by storage.backend and registry entries.
// The server is vectra-only; BackendQdrant is kept as a deprecated alias so
// old settings.json / collections.json entries still parse (they are treated
// as vectra and normalized on write).
const (
	BackendVectra = "vectra"
	BackendQdrant = "qdrant" // deprecated: accepted on read, never used for storage
)

// NormalizeBackend maps legacy backend names to the single supported backend.
func NormalizeBackend(b string) string {
	return BackendVectra
}

// Point is one stored vector plus its payload.
// ID is a deterministic uint64 (collection hash + chunk hash); the file
// backend stringifies it into index.json.
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
