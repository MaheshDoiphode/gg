package server

import (
	"context"
	"errors"
	"io"
	"math/rand"
	"net"
	"strings"
	"syscall"
	"time"
)

// A dropped TCP connection is not the credential's fault and not the request's
// fault, so it is retried on the same credential rather than failing the turn.
// AWS documents a 350s idle timeout on NAT gateways and VPC endpoints, and a
// reasoning model can easily exceed that without sending a byte.

var resetPhrases = []string{
	"connection reset",
	"forcibly closed", // Windows WSAECONNRESET
	"broken pipe",
	"unexpected eof",
	"connection was aborted",
	"server closed idle connection",
	"http2: client connection lost",
	"use of closed network connection",
}

// isTransientNetworkErr reports whether err is a dropped connection that is
// worth retrying. A cancelled context means the client left and is never
// retried.
func isTransientNetworkErr(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
		return true
	}
	if errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.ECONNABORTED) ||
		errors.Is(err, syscall.EPIPE) {
		return true
	}

	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}

	// Windows and HTTP/2 surface resets as plain text, so the message is the
	// only signal left.
	msg := strings.ToLower(err.Error())
	for _, phrase := range resetPhrases {
		if strings.Contains(msg, phrase) {
			return true
		}
	}
	return false
}

// retryBackoff grows the wait between attempts and adds jitter, which is the
// approach AWS recommends for transient failures.
func retryBackoff(attempt int) time.Duration {
	base := time.Duration(1<<attempt) * 250 * time.Millisecond
	if base > 2*time.Second {
		base = 2 * time.Second
	}
	return base + time.Duration(rand.Int63n(int64(base/2+1)))
}

// sleepFor waits unless the caller goes away first.
func sleepFor(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
