package engine

import (
	"testing"
	"time"

	"github.com/RoyChong5053/rag-mcp-server/settings"
)

// Compute mode: vectra-only servers always use the CPU batch cap.
func TestUsesCPUCompute(t *testing.T) {
	vectraActive := newTestEngine(t, settings.Settings{ActiveBackend: BackendVectra}, newFakeStore(true), newFakeStore(true))
	if !vectraActive.usesCPUCompute() {
		t.Fatal("active vectra should be CPU mode")
	}
}

// The batch size follows the compute mode: GPU honours the setting (0 => 32),
// the CPU fallback is capped to embedBatchSizeCPU.
func TestEffectiveEmbedBatchSizeByMode(t *testing.T) {
	cpu := newTestEngine(t, settings.Settings{ActiveBackend: BackendVectra, EmbedBatchSize: 32}, newFakeStore(true), newFakeStore(true))
	if got := cpu.effectiveEmbedBatchSize(cpu.usesCPUCompute()); got != embedBatchSizeCPU {
		t.Fatalf("cpu mode batch size = %d, want %d", got, embedBatchSizeCPU)
	}

	if got := newTestEngine(t, settings.Settings{EmbedBatchSize: 0}, newFakeStore(true)).effectiveEmbedBatchSize(false); got != embedBatchSizeGPU {
		t.Fatalf("default batch size = %d, want %d", got, embedBatchSizeGPU)
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
