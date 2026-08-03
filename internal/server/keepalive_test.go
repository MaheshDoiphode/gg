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

// A heartbeat before the first real event would commit a 200 response and stop
// upstream failures from surfacing as a clean HTTP error.
func TestKeepAliveWaitsForFirstWrite(t *testing.T) {
	sse, rec := newTestSSE(t)

	stop := sse.keepAlive(context.Background(), 10*time.Millisecond, func() {
		_ = sse.send("ping", map[string]any{"type": "ping"})
	})
	time.Sleep(60 * time.Millisecond)
	stop()

	if body := rec.Body.String(); body != "" {
		t.Fatalf("heartbeat fired before any content: %q", body)
	}
}

func TestKeepAliveEmitsWhenIdle(t *testing.T) {
	sse, rec := newTestSSE(t)
	sse.begin()
	_ = sse.send("message_start", map[string]any{"type": "message_start"})

	stop := sse.keepAlive(context.Background(), 10*time.Millisecond, func() {
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

func TestKeepAliveStopsWithContext(t *testing.T) {
	sse, rec := newTestSSE(t)
	sse.begin()
	_ = sse.send("message_start", map[string]any{"type": "message_start"})

	ctx, cancel := context.WithCancel(context.Background())
	stop := sse.keepAlive(ctx, 10*time.Millisecond, func() {
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

	stop := sse.keepAlive(context.Background(), time.Millisecond, func() {
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
