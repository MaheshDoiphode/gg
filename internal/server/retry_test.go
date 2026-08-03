package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"syscall"
	"testing"
	"time"
)

// The exact error text from a Windows client when AWS drops the connection.
const windowsReset = "read tcp 10.78.116.59:2155->44.229.152.64:443: wsarecv: " +
	"An existing connection was forcibly closed by the remote host."

func TestIsTransientNetworkErr(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"windows reset", errors.New(windowsReset), true},
		{"wrapped windows reset", fmt.Errorf("stream failed: %w", errors.New(windowsReset)), true},
		{"url error", &url.Error{Op: "Post", URL: "https://x", Err: errors.New("connection reset by peer")}, true},
		{"unexpected eof", io.ErrUnexpectedEOF, true},
		{"eof", io.EOF, true},
		{"econnreset", syscall.ECONNRESET, true},
		{"broken pipe", errors.New("write: broken pipe"), true},
		{"http2 lost", errors.New("http2: client connection lost"), true},
		{"net timeout", &net.OpError{Op: "read", Err: &timeoutErr{}}, true},

		// The client leaving is not a transport failure and must not be retried.
		{"context canceled", context.Canceled, false},
		{"wrapped cancel", fmt.Errorf("stream: %w", context.Canceled), false},
		{"deadline", context.DeadlineExceeded, false},
		{"nil", nil, false},
		{"validation", errors.New("messages: at least one message is required"), false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isTransientNetworkErr(c.err); got != c.want {
				t.Errorf("isTransientNetworkErr(%v) = %v, want %v", c.err, got, c.want)
			}
		})
	}
}

type timeoutErr struct{}

func (e *timeoutErr) Error() string   { return "i/o timeout" }
func (e *timeoutErr) Timeout() bool   { return true }
func (e *timeoutErr) Temporary() bool { return true }

func TestRetryBackoffGrowsAndIsBounded(t *testing.T) {
	var prev time.Duration
	for attempt := 0; attempt < 6; attempt++ {
		d := retryBackoff(attempt)
		if d <= 0 {
			t.Fatalf("attempt %d produced %v", attempt, d)
		}
		if d > 3*time.Second {
			t.Errorf("attempt %d backoff unbounded: %v", attempt, d)
		}
		if attempt > 0 && attempt < 4 && d < prev/2 {
			t.Errorf("attempt %d backoff shrank: %v -> %v", attempt, prev, d)
		}
		prev = d
	}
}

func TestSleepForStopsOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	start := time.Now()
	if err := sleepFor(ctx, 5*time.Second); err == nil {
		t.Fatal("expected an error when the context is already cancelled")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("sleepFor ignored cancellation, waited %v", elapsed)
	}
}
