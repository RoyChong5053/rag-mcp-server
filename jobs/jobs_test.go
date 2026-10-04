package jobs

import (
	"fmt"
	"testing"
	"time"
)

func TestSubmitAwaitResult(t *testing.T) {
	m := NewManager(nil, 2)
	j := m.Submit(KindSearch, func() (any, error) { return "ok", nil })
	got, done := m.Await(j.ID, time.Second)
	if !done {
		t.Fatal("expected the job to finish")
	}
	if got.State != StateDone || got.Result != "ok" {
		t.Fatalf("got %+v", got)
	}
}

func TestSubmitError(t *testing.T) {
	m := NewManager(nil, 2)
	j := m.Submit(KindSearch, func() (any, error) { return nil, fmt.Errorf("boom") })
	got, _ := m.Await(j.ID, time.Second)
	if got.State != StateError || got.Error != "boom" {
		t.Fatalf("got %+v", got)
	}
}

func TestAwaitTimesOutThenCompletes(t *testing.T) {
	m := NewManager(nil, 2)
	release := make(chan struct{})
	j := m.Submit(KindStore, func() (any, error) { <-release; return "late", nil })

	got, done := m.Await(j.ID, 30*time.Millisecond)
	if done {
		t.Fatal("expected a timeout while the task is blocked")
	}
	if got == nil || got.State != StateRunning {
		t.Fatalf("mid-flight job = %+v, want running", got)
	}

	close(release)
	got, done = m.Await(j.ID, time.Second)
	if !done || got.Result != "late" {
		t.Fatalf("after release: done=%v job=%+v", done, got)
	}
}

func TestListKindFilters(t *testing.T) {
	m := NewManager(nil, 2)
	a := m.Submit(KindSearch, func() (any, error) { return nil, nil })
	b := m.Submit(KindStore, func() (any, error) { return nil, nil })
	_, _ = m.Await(a.ID, time.Second)
	_, _ = m.Await(b.ID, time.Second)

	search := m.ListKind(KindSearch)
	if len(search) != 1 || search[0].Kind != KindSearch {
		t.Fatalf("ListKind(search) = %+v", search)
	}
	if n := len(m.List()); n != 2 {
		t.Fatalf("List() = %d, want 2", n)
	}
	if removed := m.Clear(); removed != 2 {
		t.Fatalf("Clear() = %d, want 2", removed)
	}
}
