package awsauth

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

// EventStream decodes the `application/vnd.amazon.eventstream` binary framing
// that Bedrock uses for InvokeModelWithResponseStream. Implemented directly on
// io.Reader so nothing is buffered ahead of what has actually arrived.
type EventStream struct {
	r   io.Reader
	buf []byte
}

// Event is one decoded frame.
type Event struct {
	Headers map[string]string
	Payload []byte
}

// MessageType / EventType / ExceptionType read the reserved headers.
func (e Event) MessageType() string   { return e.Headers[":message-type"] }
func (e Event) EventType() string     { return e.Headers[":event-type"] }
func (e Event) ExceptionType() string { return e.Headers[":exception-type"] }

const (
	preludeLen  = 12
	minFrameLen = 16
	maxFrameLen = 24 * 1024 * 1024
)

var (
	ErrInvalidFrame = errors.New("eventstream: invalid frame")
	ErrChecksum     = errors.New("eventstream: checksum mismatch")
)

func NewEventStream(r io.Reader) *EventStream {
	return &EventStream{r: r, buf: make([]byte, 0, 64*1024)}
}

// Next reads one frame. It returns io.EOF when the stream ends cleanly.
func (s *EventStream) Next() (Event, error) {
	var prelude [preludeLen]byte
	if _, err := io.ReadFull(s.r, prelude[:]); err != nil {
		return Event{}, err
	}

	totalLen := binary.BigEndian.Uint32(prelude[0:4])
	headersLen := binary.BigEndian.Uint32(prelude[4:8])
	preludeCRC := binary.BigEndian.Uint32(prelude[8:12])

	if crc32.ChecksumIEEE(prelude[0:8]) != preludeCRC {
		return Event{}, ErrChecksum
	}
	if totalLen < minFrameLen || totalLen > maxFrameLen || uint64(headersLen)+minFrameLen > uint64(totalLen) {
		return Event{}, fmt.Errorf("%w: total=%d headers=%d", ErrInvalidFrame, totalLen, headersLen)
	}

	rest := int(totalLen) - preludeLen
	if cap(s.buf) < rest {
		s.buf = make([]byte, rest)
	}
	body := s.buf[:rest]
	if _, err := io.ReadFull(s.r, body); err != nil {
		return Event{}, err
	}

	msgCRC := binary.BigEndian.Uint32(body[rest-4:])
	running := crc32.ChecksumIEEE(prelude[:])
	running = crc32.Update(running, crc32.IEEETable, body[:rest-4])
	if running != msgCRC {
		return Event{}, ErrChecksum
	}

	headers, err := parseHeaders(body[:headersLen])
	if err != nil {
		return Event{}, err
	}
	payload := body[headersLen : rest-4]

	// Copy: s.buf is reused by the next Next() call.
	out := make([]byte, len(payload))
	copy(out, payload)
	return Event{Headers: headers, Payload: out}, nil
}

func parseHeaders(b []byte) (map[string]string, error) {
	headers := make(map[string]string, 4)
	for i := 0; i < len(b); {
		if i+1 > len(b) {
			return nil, ErrInvalidFrame
		}
		nameLen := int(b[i])
		i++
		if i+nameLen+1 > len(b) {
			return nil, ErrInvalidFrame
		}
		name := string(b[i : i+nameLen])
		i += nameLen
		valueType := b[i]
		i++

		switch valueType {
		case 0: // bool true
			headers[name] = "true"
		case 1: // bool false
			headers[name] = "false"
		case 2: // byte
			if i+1 > len(b) {
				return nil, ErrInvalidFrame
			}
			headers[name] = fmt.Sprint(int8(b[i]))
			i++
		case 3: // int16
			if i+2 > len(b) {
				return nil, ErrInvalidFrame
			}
			headers[name] = fmt.Sprint(int16(binary.BigEndian.Uint16(b[i:])))
			i += 2
		case 4: // int32
			if i+4 > len(b) {
				return nil, ErrInvalidFrame
			}
			headers[name] = fmt.Sprint(int32(binary.BigEndian.Uint32(b[i:])))
			i += 4
		case 5, 8: // int64 / timestamp
			if i+8 > len(b) {
				return nil, ErrInvalidFrame
			}
			headers[name] = fmt.Sprint(int64(binary.BigEndian.Uint64(b[i:])))
			i += 8
		case 6, 7: // byte array / string
			if i+2 > len(b) {
				return nil, ErrInvalidFrame
			}
			n := int(binary.BigEndian.Uint16(b[i:]))
			i += 2
			if i+n > len(b) {
				return nil, ErrInvalidFrame
			}
			headers[name] = string(b[i : i+n])
			i += n
		case 9: // uuid
			if i+16 > len(b) {
				return nil, ErrInvalidFrame
			}
			headers[name] = fmt.Sprintf("%x", b[i:i+16])
			i += 16
		default:
			return nil, fmt.Errorf("%w: header type %d", ErrInvalidFrame, valueType)
		}
	}
	return headers, nil
}
