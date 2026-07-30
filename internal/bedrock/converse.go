package bedrock

import "encoding/json"

// Converse API shapes. This is Bedrock's model-agnostic messages protocol, so
// one code path covers Claude, Nova, Llama, Mistral, DeepSeek and the rest.

// ConverseRequest is the body of POST /model/{id}/converse[-stream].
type ConverseRequest struct {
	Messages                     []Message        `json:"messages,omitempty"`
	System                       []SystemBlock    `json:"system,omitempty"`
	InferenceConfig              *InferenceConfig `json:"inferenceConfig,omitempty"`
	ToolConfig                   *ToolConfig      `json:"toolConfig,omitempty"`
	AdditionalModelRequestFields map[string]any   `json:"additionalModelRequestFields,omitempty"`

	// ReasoningEffort is a side channel, never serialised: Converse has no
	// portable field for it and every upstream spells it differently.
	ReasoningEffort string `json:"-"`
}

// Reasoning effort levels, as accepted by the Mantle endpoints.
const (
	EffortNone   = "none"
	EffortLow    = "low"
	EffortMedium = "medium"
	EffortHigh   = "high"
)

// Message is one conversation turn.
type Message struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

// ContentBlock is a tagged union; exactly one field is set.
type ContentBlock struct {
	Text             *string           `json:"text,omitempty"`
	Image            *ImageBlock       `json:"image,omitempty"`
	Document         *DocumentBlock    `json:"document,omitempty"`
	ToolUse          *ToolUseBlock     `json:"toolUse,omitempty"`
	ToolResult       *ToolResultBlock  `json:"toolResult,omitempty"`
	ReasoningContent *ReasoningContent `json:"reasoningContent,omitempty"`
	CachePoint       *CachePoint       `json:"cachePoint,omitempty"`
}

// SystemBlock is one entry of the system prompt array.
type SystemBlock struct {
	Text       *string     `json:"text,omitempty"`
	CachePoint *CachePoint `json:"cachePoint,omitempty"`
}

// CachePoint marks a prompt-caching breakpoint.
type CachePoint struct {
	Type string `json:"type"` // "default"
}

// ImageBlock is an inline image. Bedrock base64-encodes blob members in JSON.
type ImageBlock struct {
	Format string      `json:"format"` // png | jpeg | gif | webp
	Source ImageSource `json:"source"`
}

// ImageSource carries the raw image bytes.
type ImageSource struct {
	Bytes string `json:"bytes,omitempty"`
}

// DocumentBlock is an attached file.
type DocumentBlock struct {
	Format string         `json:"format"`
	Name   string         `json:"name"`
	Source DocumentSource `json:"source"`
}

// DocumentSource carries the raw document bytes.
type DocumentSource struct {
	Bytes string `json:"bytes,omitempty"`
}

// ToolUseBlock is a model-issued tool call.
type ToolUseBlock struct {
	ToolUseID string          `json:"toolUseId"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

// ToolResultBlock returns a tool's output to the model.
type ToolResultBlock struct {
	ToolUseID string              `json:"toolUseId"`
	Content   []ToolResultContent `json:"content"`
	Status    string              `json:"status,omitempty"` // success | error
}

// ToolResultContent is one part of a tool result.
type ToolResultContent struct {
	Text  *string         `json:"text,omitempty"`
	JSON  json.RawMessage `json:"json,omitempty"`
	Image *ImageBlock     `json:"image,omitempty"`
}

// ReasoningContent holds extended-thinking output.
type ReasoningContent struct {
	ReasoningText   *ReasoningText `json:"reasoningText,omitempty"`
	RedactedContent string         `json:"redactedContent,omitempty"`
}

// ReasoningText is the visible reasoning plus its integrity signature.
type ReasoningText struct {
	Text      string `json:"text"`
	Signature string `json:"signature,omitempty"`
}

// InferenceConfig is the portable subset of sampling parameters.
type InferenceConfig struct {
	MaxTokens     *int     `json:"maxTokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"topP,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

// ToolConfig declares the callable tools.
type ToolConfig struct {
	Tools      []Tool      `json:"tools"`
	ToolChoice *ToolChoice `json:"toolChoice,omitempty"`
}

// Tool wraps a tool specification.
type Tool struct {
	ToolSpec *ToolSpec `json:"toolSpec,omitempty"`
}

// ToolSpec describes one callable tool.
type ToolSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema ToolInputSchema `json:"inputSchema"`
}

// ToolInputSchema wraps a JSON Schema object.
type ToolInputSchema struct {
	JSON json.RawMessage `json:"json,omitempty"`
}

// ToolChoice is a union of auto / any / a named tool.
type ToolChoice struct {
	Auto *struct{}       `json:"auto,omitempty"`
	Any  *struct{}       `json:"any,omitempty"`
	Tool *ToolChoiceTool `json:"tool,omitempty"`
}

// ToolChoiceTool forces a specific tool.
type ToolChoiceTool struct {
	Name string `json:"name"`
}

// ConverseResponse is the non-streaming reply.
type ConverseResponse struct {
	Output struct {
		Message *Message `json:"message"`
	} `json:"output"`
	StopReason string     `json:"stopReason"`
	Usage      TokenUsage `json:"usage"`
	Metrics    struct {
		LatencyMs int64 `json:"latencyMs"`
	} `json:"metrics"`
}

// TokenUsage is Bedrock token accounting.
type TokenUsage struct {
	InputTokens           int `json:"inputTokens"`
	OutputTokens          int `json:"outputTokens"`
	TotalTokens           int `json:"totalTokens"`
	CacheReadInputTokens  int `json:"cacheReadInputTokens,omitempty"`
	CacheWriteInputTokens int `json:"cacheWriteInputTokens,omitempty"`
}

// StreamEvent is one decoded ConverseStream frame. Type comes from the
// :event-type header; the rest is the union of every event body.
type StreamEvent struct {
	Type string `json:"-"`

	Role              string             `json:"role,omitempty"`
	ContentBlockIndex int                `json:"contentBlockIndex,omitempty"`
	Start             *ContentBlockStart `json:"start,omitempty"`
	Delta             *ContentBlockDelta `json:"delta,omitempty"`
	StopReason        string             `json:"stopReason,omitempty"`
	Usage             *TokenUsage        `json:"usage,omitempty"`
}

// ContentBlockStart opens a block; only tool use carries start data.
type ContentBlockStart struct {
	ToolUse *ToolUseStart `json:"toolUse,omitempty"`
}

// ToolUseStart names the tool call being opened.
type ToolUseStart struct {
	ToolUseID string `json:"toolUseId"`
	Name      string `json:"name"`
}

// ContentBlockDelta is an incremental update to an open block.
type ContentBlockDelta struct {
	Text             string          `json:"text,omitempty"`
	ToolUse          *ToolUseDelta   `json:"toolUse,omitempty"`
	ReasoningContent *ReasoningDelta `json:"reasoningContent,omitempty"`
}

// ToolUseDelta carries a fragment of the tool input JSON.
type ToolUseDelta struct {
	Input string `json:"input"`
}

// ReasoningDelta carries a fragment of reasoning output.
type ReasoningDelta struct {
	Text            string `json:"text,omitempty"`
	Signature       string `json:"signature,omitempty"`
	RedactedContent string `json:"redactedContent,omitempty"`
}

// Stop reasons returned by Converse.
const (
	StopEndTurn      = "end_turn"
	StopToolUse      = "tool_use"
	StopMaxTokens    = "max_tokens"
	StopStopSequence = "stop_sequence"
)

// Ptr is a helper for the many optional Converse fields.
func Ptr[T any](v T) *T { return &v }
