package engine

import (
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// Active vectra, or an unreachable qdrant, means the CPU fallback is in charge.
func TestUsesCPUCompute(t *testing.T) {
	vectraActive := newTestEngine(t, settings.Settings{ActiveBackend: BackendVectra}, newFakeStore(true), newFakeStore(true))
	if !vectraActive.usesCPUCompute() {
		t.Fatal("active vectra should be CPU mode")
	}

	qdrantUp := newTestEngine(t, settings.Settings{ActiveBackend: BackendQdrant}, newFakeStore(true), newFakeStore(true))
	if qdrantUp.usesCPUCompute() {
		t.Fatal("reachable qdrant should be GPU mode")
	}

	qdrantDown := newTestEngine(t, settings.Settings{ActiveBackend: BackendQdrant}, newFakeStore(false), newFakeStore(true))
	qdrantDown.healthMu.Lock()
	qdrantDown.downUntil[BackendQdrant] = time.Now().Add(time.Minute)
	qdrantDown.healthMu.Unlock()
	if !qdrantDown.usesCPUCompute() {
		t.Fatal("cached-down qdrant should be CPU mode")
	}
}

// The batch size follows the compute mode: GPU honours the setting (0 => 32),
// the CPU fallback is capped to embedBatchSizeCPU.
func TestEffectiveEmbedBatchSizeByMode(t *testing.T) {
	cpu := newTestEngine(t, settings.Settings{ActiveBackend: BackendVectra, EmbedBatchSize: 32}, newFakeStore(true), newFakeStore(true))
	if got := cpu.effectiveEmbedBatchSize(cpu.cpuComputeMode()); got != embedBatchSizeCPU {
		t.Fatalf("cpu mode batch size = %d, want %d", got, embedBatchSizeCPU)
	}

	gpu := newTestEngine(t, settings.Settings{ActiveBackend: BackendQdrant, EmbedBatchSize: 0}, newFakeStore(true), newFakeStore(true))
	if gpu.cpuComputeMode() {
		t.Fatal("reachable qdrant should be GPU mode")
	}
	if got := gpu.effectiveEmbedBatchSize(false); got != embedBatchSizeGPU {
		t.Fatalf("gpu default batch size = %d, want %d", got, embedBatchSizeGPU)
	}

	// A CPU cap does not raise a smaller explicit setting.
	small := newTestEngine(t, settings.Settings{ActiveBackend: BackendVectra, EmbedBatchSize: 2}, newFakeStore(true), newFakeStore(true))
	if got := small.effectiveEmbedBatchSize(true); got != 2 {
		t.Fatalf("explicit small batch size = %d, want 2", got)
	}
}

// With a limit of 1 the limiter serializes: a second acquire must block until
// the first releases.
func TestEmbeddingLimiterSerializes(t *testing.T) {
	c := NewEmbeddingClient("http://127.0.0.1:0", "", "m", "")
	c.SetConcurrencyProvider(func() int { return 1 })

	entered := make(chan struct{})
	release := make(chan struct{})
	go func() { c.acquire(); close(entered); <-release; c.release() }()
	<-entered

	second := make(chan struct{})
	go func() { c.acquire(); close(second); c.release() }()

	select {
	case <-second:
		t.Fatal("second acquire must block while the limit is 1")
	case <-time.After(80 * time.Millisecond):
	}

	close(release)
	select {
	case <-second:
	case <-time.After(time.Second):
		t.Fatal("second acquire did not proceed after release")
	}
}
