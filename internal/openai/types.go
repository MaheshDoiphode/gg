// Package openai holds the OpenAI Chat Completions API shapes.
package openai

import (
	"encoding/json"
	"fmt"
)

// ChatRequest is POST /v1/chat/completions.
type ChatRequest struct {
	Model            string          `json:"model"`
	Messages         []ChatMessage   `json:"messages"`
	MaxTokens        *int            `json:"max_tokens,omitempty"`
	MaxCompletionTok *int            `json:"max_completion_tokens,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	StreamOptions    *StreamOptions  `json:"stream_options,omitempty"`
	Stop             json.RawMessage `json:"stop,omitempty"`
	Tools            []Tool          `json:"tools,omitempty"`
	ToolChoice       json.RawMessage `json:"tool_choice,omitempty"`
	ReasoningEffort  string          `json:"reasoning_effort,omitempty"`
	User             string          `json:"user,omitempty"`
}

// StreamOptions currently only carries include_usage.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ChatMessage is one message. Content is raw because it may be a string or an
// array of multimodal parts.
type ChatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

// ContentPart is one element of a multimodal content array.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL is either an https URL or a data: URL.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// Tool is an OpenAI function tool.
type Tool struct {
	Type     string       `json:"type"`
	Function FunctionSpec `json:"function"`
}

// FunctionSpec describes a callable function.
type FunctionSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// UnmarshalJSON tolerates clients that send a bare function object without the
// {"type":"function","function":{...}} wrapper.
func (t *Tool) UnmarshalJSON(b []byte) error {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(b, &probe); err != nil {
		return err
	}
	if fn, ok := probe["function"]; ok {
		t.Type = "function"
		if raw, ok := probe["type"]; ok {
			_ = json.Unmarshal(raw, &t.Type)
		}
		return json.Unmarshal(fn, &t.Function)
	}
	t.Type = "function"
	return json.Unmarshal(b, &t.Function)
}

// ToolCall is an assistant-issued function call.
type ToolCall struct {
	Index    *int         `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

// FunctionCall carries the name and JSON-encoded arguments.
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

// ChatResponse is a non-streaming completion.
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Choice is one completion candidate.
type Choice struct {
	Index        int             `json:"index"`
	Message      *ResponseMsg    `json:"message,omitempty"`
	Delta        *ResponseMsg    `json:"delta,omitempty"`
	FinishReason *string         `json:"finish_reason"`
	Logprobs     json.RawMessage `json:"logprobs,omitempty"`
}

// ResponseMsg is the assistant message (or a streaming delta of one).
type ResponseMsg struct {
	Role             string  `json:"role,omitempty"`
	Content          *string `json:"content,omitempty"`
	ReasoningContent string  `json:"reasoning_content,omitempty"`
	// Reasoning is Bedrock Mantle's spelling of reasoning_content.
	Reasoning string     `json:"reasoning,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Refusal   *string    `json:"refusal,omitempty"`
}

// ReasoningText returns whichever reasoning field the upstream populated.
func (m *ResponseMsg) ReasoningText() string {
	if m.ReasoningContent != "" {
		return m.ReasoningContent
	}
	return m.Reasoning
}

// Usage is OpenAI token accounting.
type Usage struct {
	PromptTokens        int           `json:"prompt_tokens"`
	CompletionTokens    int           `json:"completion_tokens"`
	TotalTokens         int           `json:"total_tokens"`
	PromptTokensDetails *PromptDetail `json:"prompt_tokens_details,omitempty"`
}

// PromptDetail reports the cached portion of the prompt.
type PromptDetail struct {
	CachedTokens int `json:"cached_tokens"`
}

// Model is one entry of GET /v1/models.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelList is the GET /v1/models envelope.
type ModelList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// ErrorEnvelope is the OpenAI error body.
type ErrorEnvelope struct {
	Error ErrorBody `json:"error"`
}

// ErrorBody describes the failure.
type ErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
	Param   string `json:"param,omitempty"`
}

func (e ErrorBody) Error() string { return fmt.Sprintf("%s: %s", e.Type, e.Message) }

// NewError builds an OpenAI-shaped error envelope.
func NewError(kind, code, msg string) ErrorEnvelope {
	return ErrorEnvelope{Error: ErrorBody{Message: msg, Type: kind, Code: code}}
}
