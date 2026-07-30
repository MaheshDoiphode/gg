// Package anthropic holds the Anthropic Messages API shapes. Bedrock's
// InvokeModel accepts this exact body for Claude models, so these structs serve
// double duty as the inbound /v1/messages schema and the upstream request body.
package anthropic

import (
	"encoding/json"
	"fmt"
)

// Request is the Anthropic Messages API request / Bedrock native body.
type Request struct {
	AnthropicVersion string          `json:"anthropic_version,omitempty"`
	Model            string          `json:"-"`
	Messages         []Message       `json:"messages"`
	System           []SystemBlock   `json:"system,omitempty"`
	MaxTokens        int             `json:"max_tokens"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	TopK             *int            `json:"top_k,omitempty"`
	StopSequences    []string        `json:"stop_sequences,omitempty"`
	Stream           bool            `json:"-"`
	Tools            []Tool          `json:"tools,omitempty"`
	ToolChoice       json.RawMessage `json:"tool_choice,omitempty"`
	Thinking         *Thinking       `json:"thinking,omitempty"`
}

// Thinking enables extended reasoning.
type Thinking struct {
	Type         string `json:"type"` // "enabled" | "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// SystemBlock is one entry of the system prompt array.
type SystemBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Tool is an Anthropic tool definition.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// Message is one conversation turn.
type Message struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

// ContentBlock covers every block type this proxy produces or consumes.
type ContentBlock struct {
	Type string `json:"type"`

	// text
	Text string `json:"text,omitempty"`

	// thinking
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"` // redacted_thinking

	// image
	Source *ImageSource `json:"source,omitempty"`

	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

// ImageSource is an inline base64 image or a remote URL.
type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// UnmarshalJSON accepts the shorthand form where content is a bare string.
func (m *Message) UnmarshalJSON(b []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	m.Role = raw.Role
	m.Content = nil
	if len(raw.Content) == 0 || string(raw.Content) == "null" {
		return nil
	}
	if raw.Content[0] == '"' {
		var s string
		if err := json.Unmarshal(raw.Content, &s); err != nil {
			return err
		}
		m.Content = []ContentBlock{{Type: "text", Text: s}}
		return nil
	}
	return json.Unmarshal(raw.Content, &m.Content)
}

// UnmarshalJSON accepts a bare string for the system field.
func (r *Request) UnmarshalJSON(b []byte) error {
	type alias Request
	var raw struct {
		alias
		Model  string          `json:"model"`
		Stream bool            `json:"stream"`
		System json.RawMessage `json:"system"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	*r = Request(raw.alias)
	r.Model = raw.Model
	r.Stream = raw.Stream
	r.System = nil

	if len(raw.System) == 0 || string(raw.System) == "null" {
		return nil
	}
	if raw.System[0] == '"' {
		var s string
		if err := json.Unmarshal(raw.System, &s); err != nil {
			return err
		}
		if s != "" {
			r.System = []SystemBlock{{Type: "text", Text: s}}
		}
		return nil
	}
	return json.Unmarshal(raw.System, &r.System)
}

// Response is a non-streaming Anthropic message response.
type Response struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Model        string         `json:"model"`
	Content      []ContentBlock `json:"content"`
	StopReason   string         `json:"stop_reason"`
	StopSequence *string        `json:"stop_sequence"`
	Usage        Usage          `json:"usage"`
}

// Usage is Anthropic token accounting.
type Usage struct {
	InputTokens             int `json:"input_tokens"`
	OutputTokens            int `json:"output_tokens"`
	CacheCreationInputTokes int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens    int `json:"cache_read_input_tokens,omitempty"`
}

// StreamEvent is a decoded Anthropic SSE event.
type StreamEvent struct {
	Type string `json:"type"`

	Message *Response `json:"message,omitempty"`

	Index        int           `json:"index,omitempty"`
	ContentBlock *ContentBlock `json:"content_block,omitempty"`

	Delta *StreamDelta `json:"delta,omitempty"`
	Usage *Usage       `json:"usage,omitempty"`

	Error *APIError `json:"error,omitempty"`
}

// StreamDelta carries both content_block_delta and message_delta payloads.
type StreamDelta struct {
	Type        string `json:"type,omitempty"`
	Text        string `json:"text,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	Signature   string `json:"signature,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`

	StopReason   string  `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

// APIError is the Anthropic error envelope body.
type APIError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func (e *APIError) Error() string { return fmt.Sprintf("%s: %s", e.Type, e.Message) }

// ErrorEnvelope wraps APIError for the wire.
type ErrorEnvelope struct {
	Type  string   `json:"type"`
	Error APIError `json:"error"`
}

// NewErrorEnvelope builds the standard {"type":"error","error":{...}} body.
func NewErrorEnvelope(kind, msg string) ErrorEnvelope {
	return ErrorEnvelope{Type: "error", Error: APIError{Type: kind, Message: msg}}
}
