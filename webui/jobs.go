package webui

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/engine"
)

// Job states: queued (waiting for a worker slot) → running → done/error.
type Job struct {
	ID          string `json:"id"`
	Kind        string `json:"kind"`
	Collection  string `json:"collection"`
	Backend     string `json:"backend,omitempty"`
	File        string `json:"file"`
	ChunkSize   int    `json:"chunk_size"`
	Overlap     int    `json:"overlap"`
	Rebuild     bool   `json:"rebuild"`
	State       string `json:"state"`
	Stage       string `json:"stage"`
	DoneChunks  int    `json:"done_chunks"`
	TotalChunks int    `json:"total_chunks"`
	Chunks      int    `json:"chunks"`
	Error       string `json:"error,omitempty"`
	CreatedAt   string `json:"created_at"`
	FinishedAt  string `json:"finished_at,omitempty"`
}

// Manager runs index jobs in the background with bounded concurrency.
// Search traffic is never blocked: jobs only consume worker slots, and
// Qdrant upserts land progressively (visible to recall as batches commit).
type Manager struct {
	eng   *engine.Engine
	sem   chan struct{}
	mu    sync.Mutex
	seq   atomic.Uint64
	jobs  map[string]*Job
	order []string
}

const maxKeptJobs = 50

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

// SubmitIndex queues a file index job. absPath must already be jailed.
// backend may be empty (follow registry/default routing) or "qdrant"/"vectra"
// to force where the vectors land.
func (m *Manager) SubmitIndex(absPath, collection, backend string, opts *engine.ChunkOptions, rebuild bool) *Job {
	id := fmt.Sprintf("job-%d", m.seq.Add(1))
	size, overlap := 0, 0
	if opts != nil {
		size, overlap = opts.Size, opts.OverlapPercent
	}
	job := &Job{
		ID:         id,
		Kind:       "index",
		Collection: collection,
		Backend:    backend,
		File:       absPath,
		ChunkSize:  size,
		Overlap:    overlap,
		Rebuild:    rebuild,
		State:      "queued",
		Stage:      "queued",
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
	}
	m.mu.Lock()
	m.jobs[id] = job
	m.order = append(m.order, id)
	// Drop oldest finished jobs beyond cap
	for len(m.order) > maxKeptJobs {
		oldest := m.order[0]
		if m.jobs[oldest].State == "queued" || m.jobs[oldest].State == "running" {
			break
		}
		delete(m.jobs, oldest)
		m.order = m.order[1:]
	}
	m.mu.Unlock()

	go m.run(job, absPath, collection, backend, opts, rebuild)
	return m.Get(id)
}

func (m *Manager) run(job *Job, absPath, collection, backend string, opts *engine.ChunkOptions, rebuild bool) {
	m.sem <- struct{}{} // wait for a worker slot
	defer func() { <-m.sem }()

	m.update(job.ID, func(j *Job) {
		j.State = "running"
		j.Stage = "reading"
	})
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
	m.update(job.ID, func(j *Job) {
		j.FinishedAt = time.Now().UTC().Format(time.RFC3339)
		if err != nil {
			j.State = "error"
			j.Stage = "failed"
			j.Error = err.Error()
			return
		}
		j.State = "done"
		j.Stage = "done"
		j.Chunks = res.ChunksIndexed
		j.DoneChunks = res.ChunksIndexed
		j.TotalChunks = res.ChunksIndexed
	})
}

func (m *Manager) update(id string, fn func(*Job)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if j, ok := m.jobs[id]; ok {
		fn(j)
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
	out := make([]*Job, 0, len(m.order))
	for i := len(m.order) - 1; i >= 0; i-- {
		cp := *m.jobs[m.order[i]]
		out = append(out, &cp)
	}
	return out
}

// Clear removes finished jobs (done/error) and returns how many were dropped.
// Queued/running jobs are kept so an in-flight index is never hidden.
func (m *Manager) Clear() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	kept := make([]string, 0, len(m.order))
	removed := 0
	for _, id := range m.order {
		j := m.jobs[id]
		if j != nil && (j.State == "queued" || j.State == "running") {
			kept = append(kept, id)
			continue
		}
		delete(m.jobs, id)
		removed++
	}
	m.order = kept
	return removed
}
