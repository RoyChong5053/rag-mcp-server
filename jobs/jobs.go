// Package jobs is the server's shared background-task registry. Both the
// management dashboard and the MCP tools submit here, so a long operation has
// one identity regardless of which surface started it.
//
// The design mirrors what makes rag-mcp-server a "real" server rather than a
// synchronous MCP shim: a tool call waits a bounded time for a result and, if
// the work is not done, returns a job handle the caller can poll.
package jobs

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
)

// Job states: queued (waiting for a worker slot) → running → done/error.
const (
	StateQueued  = "queued"
	StateRunning = "running"
	StateDone    = "done"
	StateError   = "error"
)

// Job kinds.
const (
	KindIndex  = "index"
	KindSearch = "search"
	KindStore  = "store"
)

// Job is one background task. Index jobs fill the dashboard fields
// (collection/backend/file/chunk options); generic MCP tasks (search/store)
// carry their result in Result. Every exported field is JSON so the existing
// dashboard API is unchanged.
type Job struct {
	ID          string `json:"id"`
	Kind        string `json:"kind"`
	State       string `json:"state"`
	Stage       string `json:"stage"`
	DoneChunks  int    `json:"done_chunks"`
	TotalChunks int    `json:"total_chunks"`
	Chunks      int    `json:"chunks"`
	Error       string `json:"error,omitempty"`
	Result      any    `json:"result,omitempty"`

	// Index-job dashboard fields.
	Collection string `json:"collection,omitempty"`
	Backend    string `json:"backend,omitempty"`
	File       string `json:"file,omitempty"`
	ChunkSize  int    `json:"chunk_size,omitempty"`
	Overlap    int    `json:"overlap,omitempty"`
	Rebuild    bool   `json:"rebuild,omitempty"`

	CreatedAt  string `json:"created_at"`
	FinishedAt string `json:"finished_at,omitempty"`

	done chan struct{} // closed when the job reaches done/error
}

// Manager runs background jobs. Index jobs are bounded by a worker pool so
// several large vectorizations cannot stampede the embedding backend; generic
// tasks rely on the engine's own embedding limiter for safety.
type Manager struct {
	eng   *engine.Engine
	sem   chan struct{} // index-job worker slots
	mu    sync.Mutex
	seq   atomic.Uint64
	jobs  map[string]*Job
	order []string
}

const maxKeptJobs = 50

// NewManager builds a manager with maxConcurrent index workers (<=0 => 2).
// eng may be nil for tests that only exercise generic tasks.
func NewManager(eng *engine.Engine, maxConcurrent int) *Manager {
	if maxConcurrent <= 0 {
		maxConcurrent = 2
	}
	return &Manager{
		eng:  eng,
		sem:  make(chan struct{}, maxConcurrent),
		jobs: make(map[string]*Job),
	}
}

// newJob registers a queued job, trims the oldest finished jobs beyond the
// cap (never an in-flight one), and returns it.
func (m *Manager) newJob(kind string) *Job {
	job := &Job{
		ID:        fmt.Sprintf("job-%d", m.seq.Add(1)),
		Kind:      kind,
		State:     StateQueued,
		Stage:     "queued",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		done:      make(chan struct{}),
	}
	m.mu.Lock()
	m.jobs[job.ID] = job
	m.order = append(m.order, job.ID)
	for len(m.order) > maxKeptJobs {
		oldest := m.order[0]
		j := m.jobs[oldest]
		if j != nil && (j.State == StateQueued || j.State == StateRunning) {
			break
		}
		delete(m.jobs, oldest)
		m.order = m.order[1:]
	}
	m.mu.Unlock()
	return job
}

func (m *Manager) update(id string, fn func(*Job)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if j, ok := m.jobs[id]; ok {
		fn(j)
	}
}

// finish marks a job terminal and wakes any Await waiter. Exactly once.
func (m *Manager) finish(id string, fn func(*Job)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok := m.jobs[id]
	if !ok {
		return
	}
	fn(j)
	j.FinishedAt = time.Now().UTC().Format(time.RFC3339)
	close(j.done)
}

// Submit runs a generic background task and returns the queued job. It does
// not consume an index worker slot: the expensive embedding step is bounded by
// the engine's limiter instead.
func (m *Manager) Submit(kind string, run func() (any, error)) *Job {
	job := m.newJob(kind)
	go func() {
		m.update(job.ID, func(j *Job) {
			j.State = StateRunning
			if j.Stage == "queued" {
				j.Stage = "running"
			}
		})
		res, err := run()
		m.finish(job.ID, func(j *Job) {
			if err != nil {
				j.State = StateError
				j.Stage = "failed"
				j.Error = err.Error()
				return
			}
			j.State = StateDone
			j.Stage = "done"
			j.Result = res
		})
	}()
	return m.Get(job.ID)
}

// SubmitIndex queues a file index job. absPath must already be jailed.
// backend may be empty (follow registry/default routing) or "qdrant"/"vectra"
// to force where the vectors land.
func (m *Manager) SubmitIndex(absPath, collection, backend string, opts *engine.ChunkOptions, rebuild bool) *Job {
	size, overlap := 0, 0
	if opts != nil {
		size, overlap = opts.Size, opts.OverlapPercent
	}
	job := m.newJob(KindIndex)
	m.update(job.ID, func(j *Job) {
		j.Collection = collection
		j.Backend = backend
		j.File = absPath
		j.ChunkSize = size
		j.Overlap = overlap
		j.Rebuild = rebuild
	})
	go m.runIndex(job, absPath, collection, backend, opts, rebuild)
	return m.Get(job.ID)
}

func (m *Manager) runIndex(job *Job, absPath, collection, backend string, opts *engine.ChunkOptions, rebuild bool) {
	m.sem <- struct{}{} // wait for a worker slot
	defer func() { <-m.sem }()

	m.update(job.ID, func(j *Job) { j.State = StateRunning; j.Stage = "reading" })
	prog := func(done, total int) {
		m.update(job.ID, func(j *Job) {
			j.Stage = "upserting"
			j.DoneChunks = done
			j.TotalChunks = total
		})
	}
	m.update(job.ID, func(j *Job) { j.Stage = "chunking+embedding" })
	var (
		res *engine.IndexResult
		err error
	)
	if rebuild {
		res, err = m.eng.ReindexDocumentOn(backend, absPath, collection, nil, opts, prog)
	} else {
		res, err = m.eng.IndexDocumentOn(backend, absPath, collection, nil, opts, prog)
	}
	m.finish(job.ID, func(j *Job) {
		if err != nil {
			j.State = StateError
			j.Stage = "failed"
			j.Error = err.Error()
		} else {
			j.State = StateDone
			j.Stage = "done"
			j.Chunks = res.ChunksIndexed
			j.DoneChunks = res.ChunksIndexed
			j.TotalChunks = res.ChunksIndexed
		}
		if m.eng != nil {
			op := "index"
			if rebuild {
				op = "reindex"
			}
			prov, model, dim, _ := m.eng.ActiveEmbedProvenance()
			var sum string
			if data, rerr := readFileSHA(absPath); rerr == nil {
				sum = data
			}
			if herr := m.eng.AppendHistory(engine.HistoryEntry{
				Op: op, File: absPath, SHA256: sum, Collection: collection, Backend: backend,
				Mode: map[bool]string{true: "rebuild", false: "append"}[rebuild],
				Chunks: resChunks(res), EmbedProvider: prov, EmbedModel: model, VectorDim: dim,
				OK: err == nil, Error: errString(err),
			}); herr != nil {
				log.Printf("history append failed: %v", herr)
			}
		}
	})
}

// Await blocks until the job reaches done/error or d elapses. d<0 waits
// indefinitely, d==0 returns immediately. The bool reports whether the job is
// terminal. A missing job counts as terminal.
func (m *Manager) Await(id string, d time.Duration) (*Job, bool) {
	m.mu.Lock()
	j, ok := m.jobs[id]
	var ch chan struct{}
	if ok {
		ch = j.done
	}
	m.mu.Unlock()
	if !ok {
		return nil, true
	}
	// Fast path: already finished (avoids the d==0 select race).
	select {
	case <-ch:
		return m.Get(id), true
	default:
	}
	if d == 0 {
		return m.Get(id), false
	}
	if d < 0 {
		<-ch
		return m.Get(id), true
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ch:
		return m.Get(id), true
	case <-t.C:
		return m.Get(id), false
	}
}

// Get returns a copy of the job, or nil.
func (m *Manager) Get(id string) *Job {
	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok := m.jobs[id]
	if !ok {
		return nil
	}
	cp := *j
	return &cp
}

// List returns copies newest-first.
func (m *Manager) List() []*Job {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.listLocked(func(*Job) bool { return true })
}

// ListKind returns copies newest-first, filtered to one kind.
func (m *Manager) ListKind(kind string) []*Job {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.listLocked(func(j *Job) bool { return j.Kind == kind })
}

func (m *Manager) listLocked(keep func(*Job) bool) []*Job {
	out := make([]*Job, 0, len(m.order))
	for i := len(m.order) - 1; i >= 0; i-- {
		j := m.jobs[m.order[i]]
		if j == nil || !keep(j) {
			continue
		}
		cp := *j
		out = append(out, &cp)
	}
	return out
}

// Clear removes finished jobs and returns how many were dropped. Queued/running
// jobs are kept so an in-flight task is never hidden.
func (m *Manager) Clear() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	kept := make([]string, 0, len(m.order))
	removed := 0
	for _, id := range m.order {
		j := m.jobs[id]
		if j != nil && (j.State == StateQueued || j.State == StateRunning) {
			kept = append(kept, id)
			continue
		}
		delete(m.jobs, id)
		removed++
	}
	m.order = kept
	return removed
}

func resChunks(res *engine.IndexResult) int {
	if res == nil {
		return 0
	}
	return res.ChunksIndexed
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func readFileSHA(absPath string) (string, error) {
	f, err := os.Open(absPath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
