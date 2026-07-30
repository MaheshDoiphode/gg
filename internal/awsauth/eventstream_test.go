package awsauth

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"testing"
)

// buildFrame assembles a valid event-stream message for testing.
func buildFrame(headers map[string]string, payload []byte) []byte {
	var hb bytes.Buffer
	for name, value := range headers {
		hb.WriteByte(byte(len(name)))
		hb.WriteString(name)
		hb.WriteByte(7) // string
		_ = binary.Write(&hb, binary.BigEndian, uint16(len(value)))
		hb.WriteString(value)
	}

	headersLen := uint32(hb.Len())
	totalLen := 16 + headersLen + uint32(len(payload))

	var prelude bytes.Buffer
	_ = binary.Write(&prelude, binary.BigEndian, totalLen)
	_ = binary.Write(&prelude, binary.BigEndian, headersLen)
	preludeCRC := crc32.ChecksumIEEE(prelude.Bytes())
	_ = binary.Write(&prelude, binary.BigEndian, preludeCRC)

	var msg bytes.Buffer
	msg.Write(prelude.Bytes())
	msg.Write(hb.Bytes())
	msg.Write(payload)
	_ = binary.Write(&msg, binary.BigEndian, crc32.ChecksumIEEE(msg.Bytes()))
	return msg.Bytes()
}

func TestEventStreamDecodesFrames(t *testing.T) {
	var raw bytes.Buffer
	raw.Write(buildFrame(map[string]string{":event-type": "contentBlockDelta", ":message-type": "event"},
		[]byte(`{"contentBlockIndex":0,"delta":{"text":"hello"}}`)))
	raw.Write(buildFrame(map[string]string{":event-type": "messageStop", ":message-type": "event"},
		[]byte(`{"stopReason":"end_turn"}`)))

	s := NewEventStream(&raw)

	first, err := s.Next()
	if err != nil {
		t.Fatalf("first frame: %v", err)
	}
	if first.EventType() != "contentBlockDelta" {
		t.Errorf("event type = %q", first.EventType())
	}
	if string(first.Payload) != `{"contentBlockIndex":0,"delta":{"text":"hello"}}` {
		t.Errorf("payload = %s", first.Payload)
	}

	second, err := s.Next()
	if err != nil {
		t.Fatalf("second frame: %v", err)
	}
	if second.EventType() != "messageStop" {
		t.Errorf("event type = %q", second.EventType())
	}
	// The decoder reuses its scratch buffer, so earlier payloads must be copies.
	if string(first.Payload) != `{"contentBlockIndex":0,"delta":{"text":"hello"}}` {
		t.Error("first payload was clobbered by the second read")
	}

	if _, err := s.Next(); !errors.Is(err, io.EOF) {
		t.Errorf("expected EOF, got %v", err)
	}
}

func TestEventStreamRejectsCorruptedPayload(t *testing.T) {
	frame := buildFrame(map[string]string{":event-type": "chunk"}, []byte(`{"a":1}`))
	frame[len(frame)-6] ^= 0xFF // corrupt the payload, leaving the CRC stale

	if _, err := NewEventStream(bytes.NewReader(frame)).Next(); !errors.Is(err, ErrChecksum) {
		t.Fatalf("expected checksum error, got %v", err)
	}
}

func TestEventStreamRejectsBadPrelude(t *testing.T) {
	frame := buildFrame(map[string]string{":event-type": "chunk"}, []byte(`{}`))
	frame[0] = 0xFF // corrupt total length

	if _, err := NewEventStream(bytes.NewReader(frame)).Next(); err == nil {
		t.Fatal("expected an error for a corrupt prelude")
	}
}

func TestEventStreamExceptionHeaders(t *testing.T) {
	frame := buildFrame(map[string]string{
		":message-type":   "exception",
		":exception-type": "throttlingException",
	}, []byte(`{"message":"slow down"}`))

	ev, err := NewEventStream(bytes.NewReader(frame)).Next()
	if err != nil {
		t.Fatal(err)
	}
	if ev.MessageType() != "exception" || ev.ExceptionType() != "throttlingException" {
		t.Fatalf("headers = %#v", ev.Headers)
	}
}
