package server

import (
	"context"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func newTestSSE(t *testing.T) (*sseWriter, *httptest.ResponseRecorder) {
	t.Helper()
	rec := httptest.NewRecorder()
	sse, ok := newSSEWriter(rec)
	if !ok {
		t.Fatal("recorder should support flushing")
	}
	return sse, rec
}

// A heartbeat must not fire during the warmup, so a fast upstream rejection
// can still be reported as a clean HTTP error instead of an in-stream one.
func TestKeepAliveWaitsForWarmup(t *testing.T) {
	sse, rec := newTestSSE(t)

	stop := sse.keepAlive(context.Background(), time.Hour, 10*time.Millisecond, func() {
		_ = sse.send("ping", map[string]any{"type": "ping"})
	})
	time.Sleep(60 * time.Millisecond)
	stop()

	if body := rec.Body.String(); body != "" {
		t.Fatalf("heartbeat fired during warmup: %q", body)
	}
}

// Once the warmup passes, a client waiting on a thinking model must still see
// traffic even though nothing has been written yet.
func TestKeepAliveEmitsBeforeFirstContent(t *testing.T) {
	sse, rec := newTestSSE(t)

	stop := sse.keepAlive(context.Background(), 10*time.Millisecond, 10*time.Millisecond, func() {
		sse.begin()
		_ = sse.send("ping", map[string]any{"type": "ping"})
	})
	time.Sleep(80 * time.Millisecond)
	stop()

	if n := strings.Count(rec.Body.String(), "event: ping"); n == 0 {
		t.Fatalf("expected a heartbeat while waiting for the first event, body = %q", rec.Body.String())
	}
}

func TestKeepAliveEmitsWhenIdle(t *testing.T) {
	sse, rec := newTestSSE(t)
	sse.begin()
	_ = sse.send("message_start", map[string]any{"type": "message_start"})

	stop := sse.keepAlive(context.Background(), 10*time.Millisecond, 10*time.Millisecond, func() {
		_ = sse.send("ping", map[string]any{"type": "ping"})
	})
	time.Sleep(80 * time.Millisecond)
	stop()

	if n := strings.Count(rec.Body.String(), "event: ping"); n == 0 {
		t.Fatalf("expected heartbeats while idle, body = %q", rec.Body.String())
	}

	// After stopping, no further pings may appear.
	before := strings.Count(rec.Body.String(), "event: ping")
	time.Sleep(50 * time.Millisecond)
	if after := strings.Count(rec.Body.String(), "event: ping"); after != before {
		t.Errorf("heartbeat continued after stop: %d -> %d", before, after)
	}
}

// A write landing just after a poll must not push the next beat out to almost
// twice the interval. Claude Code abandoned a stream 130ms before a beat that
// was due at 2x the interval, so the real gap is what matters here.
func TestKeepAliveGapStaysNearInterval(t *testing.T) {
	sse, rec := newTestSSE(t)
	sse.begin()

	const every = 40 * time.Millisecond
	stop := sse.keepAlive(context.Background(), every, every, func() {
		_ = sse.send("ping", map[string]any{"type": "ping"})
	})

	// Write just after the stream opens, the worst case for a coarse ticker.
	time.Sleep(5 * time.Millisecond)
	_ = sse.send("message_start", map[string]any{"type": "message_start"})
	last := time.Now()

	for rec.Body.Len() >= 0 {
		if strings.Contains(rec.Body.String(), "event: ping") {
			break
		}
		if time.Since(last) > 4*every {
			stop()
			t.Fatalf("no heartbeat within %v of the last write", time.Since(last))
		}
		time.Sleep(2 * time.Millisecond)
	}
	gap := time.Since(last)
	stop()

	// Allow scheduling slack, but nowhere near the 2x the old ticker permitted.
	if gap > 2*every {
		t.Errorf("silence before the heartbeat was %v, want close to %v", gap, every)
	}
}

func TestHeartbeatResolutionIsFinerThanInterval(t *testing.T) {
	if got := heartbeatResolution(10 * time.Second); got != time.Second {
		t.Errorf("resolution for 10s = %v, want 1s", got)
	}
	if got := heartbeatResolution(2 * time.Second); got != 200*time.Millisecond {
		t.Errorf("resolution for 2s = %v, want 200ms", got)
	}
	if got := heartbeatResolution(time.Millisecond); got != time.Millisecond {
		t.Errorf("resolution for 1ms = %v, want 1ms", got)
	}
}

func TestKeepAliveStopsWithContext(t *testing.T) {
	sse, rec := newTestSSE(t)
	sse.begin()
	_ = sse.send("message_start", map[string]any{"type": "message_start"})

	ctx, cancel := context.WithCancel(context.Background())
	stop := sse.keepAlive(ctx, 10*time.Millisecond, 10*time.Millisecond, func() {
		_ = sse.send("ping", map[string]any{"type": "ping"})
	})
	defer stop()

	cancel()
	time.Sleep(50 * time.Millisecond)
	count := strings.Count(rec.Body.String(), "event: ping")
	time.Sleep(50 * time.Millisecond)
	if now := strings.Count(rec.Body.String(), "event: ping"); now != count {
		t.Errorf("heartbeat ignored context cancellation: %d -> %d", count, now)
	}
}

// The heartbeat goroutine writes to the same stream as the request goroutine,
// so frames must never interleave.
func TestSSEWritesDoNotInterleave(t *testing.T) {
	sse, rec := newTestSSE(t)
	sse.begin()
	_ = sse.send("message_start", map[string]any{"type": "message_start"})

	stop := sse.keepAlive(context.Background(), time.Millisecond, time.Millisecond, func() {
		_ = sse.send("ping", map[string]any{"type": "ping"})
	})

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				_ = sse.send("content_block_delta", map[string]any{"text": "chunk"})
			}
		}()
	}
	wg.Wait()
	stop()

	// Every line must be a complete frame component, never a spliced one.
	for _, line := range strings.Split(strings.TrimSpace(rec.Body.String()), "\n") {
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "event: ") && !strings.HasPrefix(line, "data: ") {
			t.Fatalf("corrupted frame line: %q", line)
		}
	}
	if n := strings.Count(rec.Body.String(), "event: content_block_delta"); n != 200 {
		t.Errorf("expected 200 delta events, got %d", n)
	}
}
