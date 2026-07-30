package convert

import (
	"encoding/json"
	"strings"

	"bedrock-simple/internal/bedrock"
	"bedrock-simple/internal/openai"
)

// Converse is this proxy's internal hub format. Mantle speaks OpenAI, so these
// convert in the upstream direction: hub -> OpenAI request, OpenAI response ->
// hub. That keeps both inbound APIs working over either upstream.

// ConverseToOpenAIRequest renders the hub request as an OpenAI chat completion.
func ConverseToOpenAIRequest(model string, in *bedrock.ConverseRequest, stream bool) *openai.ChatRequest {
	out := &openai.ChatRequest{Model: model, Stream: stream}

	var system strings.Builder
	for _, s := range in.System {
		if s.Text != nil && *s.Text != "" {
			if system.Len() > 0 {
				system.WriteString("\n\n")
			}
			system.WriteString(*s.Text)
		}
	}
	if system.Len() > 0 {
		out.Messages = append(out.Messages, openai.ChatMessage{
			Role:    "system",
			Content: mustJSON(system.String()),
		})
	}

	for _, m := range in.Messages {
		var text strings.Builder
		var parts []openai.ContentPart
		var toolCalls []openai.ToolCall
		var hasImage bool

		for _, b := range m.Content {
			switch {
			case b.Text != nil:
				text.WriteString(*b.Text)
				parts = append(parts, openai.ContentPart{Type: "text", Text: *b.Text})

			case b.Image != nil:
				hasImage = true
				parts = append(parts, openai.ContentPart{
					Type: "image_url",
					ImageURL: &openai.ImageURL{
						URL: "data:image/" + b.Image.Format + ";base64," + b.Image.Source.Bytes,
					},
				})

			case b.ToolUse != nil:
				args := string(b.ToolUse.Input)
				if args == "" {
					args = "{}"
				}
				toolCalls = append(toolCalls, openai.ToolCall{
					ID:       b.ToolUse.ToolUseID,
					Type:     "function",
					Function: openai.FunctionCall{Name: b.ToolUse.Name, Arguments: args},
				})

			case b.ToolResult != nil:
				// A tool result has to become its own OpenAI message.
				out.Messages = append(out.Messages, openai.ChatMessage{
					Role:       "tool",
					ToolCallID: b.ToolResult.ToolUseID,
					Content:    mustJSON(toolResultText(b.ToolResult)),
				})
			}
		}

		if text.Len() == 0 && len(toolCalls) == 0 && !hasImage {
			continue
		}
		msg := openai.ChatMessage{Role: m.Role, ToolCalls: toolCalls}
		if hasImage {
			msg.Content = mustJSON(parts)
		} else if text.Len() > 0 {
			msg.Content = mustJSON(text.String())
		}
		out.Messages = append(out.Messages, msg)
	}

	if ic := in.InferenceConfig; ic != nil {
		out.MaxTokens = ic.MaxTokens
		out.Temperature = ic.Temperature
		out.TopP = ic.TopP
		if len(ic.StopSequences) > 0 {
			out.Stop = mustJSON(ic.StopSequences)
		}
	}
	out.ReasoningEffort = in.ReasoningEffort

	if in.ToolConfig != nil {
		for _, t := range in.ToolConfig.Tools {
			if t.ToolSpec == nil {
				continue
			}
			out.Tools = append(out.Tools, openai.Tool{
				Type: "function",
				Function: openai.FunctionSpec{
					Name:        t.ToolSpec.Name,
					Description: t.ToolSpec.Description,
					Parameters:  t.ToolSpec.InputSchema.JSON,
				},
			})
		}
		switch tc := in.ToolConfig.ToolChoice; {
		case tc == nil:
		case tc.Any != nil:
			out.ToolChoice = mustJSON("required")
		case tc.Auto != nil:
			out.ToolChoice = mustJSON("auto")
		case tc.Tool != nil:
			out.ToolChoice = mustJSON(map[string]any{
				"type":     "function",
				"function": map[string]string{"name": tc.Tool.Name},
			})
		}
	}
	return out
}

func toolResultText(res *bedrock.ToolResultBlock) string {
	var b strings.Builder
	for _, c := range res.Content {
		switch {
		case c.Text != nil:
			b.WriteString(*c.Text)
		case len(c.JSON) > 0:
			b.Write(c.JSON)
		}
	}
	return b.String()
}

// OpenAIResponseToConverse maps a completed OpenAI response back to the hub.
func OpenAIResponseToConverse(in *openai.ChatResponse) *bedrock.ConverseResponse {
	out := &bedrock.ConverseResponse{StopReason: bedrock.StopEndTurn}
	msg := &bedrock.Message{Role: "assistant"}

	if len(in.Choices) > 0 {
		ch := in.Choices[0]
		if m := ch.Message; m != nil {
			if r := m.ReasoningText(); r != "" {
				msg.Content = append(msg.Content, bedrock.ContentBlock{
					ReasoningContent: &bedrock.ReasoningContent{
						ReasoningText: &bedrock.ReasoningText{Text: r},
					},
				})
			}
			if m.Content != nil && *m.Content != "" {
				msg.Content = append(msg.Content, bedrock.ContentBlock{Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				input := json.RawMessage(strings.TrimSpace(tc.Function.Arguments))
				if len(input) == 0 || !json.Valid(input) {
					input = json.RawMessage(`{}`)
				}
				msg.Content = append(msg.Content, bedrock.ContentBlock{ToolUse: &bedrock.ToolUseBlock{
					ToolUseID: tc.ID, Name: tc.Function.Name, Input: input,
				}})
			}
		}
		if ch.FinishReason != nil {
			out.StopReason = finishReasonToConverse(*ch.FinishReason)
		}
	}
	out.Output.Message = msg

	if u := in.Usage; u != nil {
		out.Usage = bedrock.TokenUsage{
			InputTokens:  u.PromptTokens,
			OutputTokens: u.CompletionTokens,
			TotalTokens:  u.TotalTokens,
		}
		if u.PromptTokensDetails != nil {
			out.Usage.CacheReadInputTokens = u.PromptTokensDetails.CachedTokens
		}
	}
	return out
}

func finishReasonToConverse(reason string) string {
	switch reason {
	case "tool_calls", "function_call":
		return bedrock.StopToolUse
	case "length":
		return bedrock.StopMaxTokens
	case "content_filter":
		return "content_filtered"
	default:
		return bedrock.StopEndTurn
	}
}

// MantleStream turns OpenAI chunks back into hub stream events, so the existing
// OpenAI and Anthropic renderers work over Mantle without modification.
type MantleStream struct {
	started   bool
	blockOpen bool
	toolBlock map[int]int // OpenAI tool index -> hub content block index
	nextBlock int
}

// NewMantleStream creates the upstream stream adapter.
func NewMantleStream() *MantleStream {
	return &MantleStream{toolBlock: map[int]int{}}
}

// Handle converts one OpenAI chunk into zero or more hub events.
func (s *MantleStream) Handle(chunk *openai.ChatResponse) []bedrock.StreamEvent {
	var out []bedrock.StreamEvent

	if !s.started {
		s.started = true
		out = append(out, bedrock.StreamEvent{Type: "messageStart", Role: "assistant"})
	}

	for _, ch := range chunk.Choices {
		d := ch.Delta
		if d != nil {
			if r := d.ReasoningText(); r != "" {
				out = append(out, bedrock.StreamEvent{
					Type:              "contentBlockDelta",
					ContentBlockIndex: 0,
					Delta: &bedrock.ContentBlockDelta{
						ReasoningContent: &bedrock.ReasoningDelta{Text: r},
					},
				})
			}
			if d.Content != nil && *d.Content != "" {
				s.blockOpen = true
				out = append(out, bedrock.StreamEvent{
					Type:              "contentBlockDelta",
					ContentBlockIndex: 0,
					Delta:             &bedrock.ContentBlockDelta{Text: *d.Content},
				})
			}
			for _, tc := range d.ToolCalls {
				idx := 0
				if tc.Index != nil {
					idx = *tc.Index
				}
				block, known := s.toolBlock[idx]
				if !known {
					s.nextBlock++
					block = s.nextBlock
					s.toolBlock[idx] = block
					out = append(out, bedrock.StreamEvent{
						Type:              "contentBlockStart",
						ContentBlockIndex: block,
						Start: &bedrock.ContentBlockStart{ToolUse: &bedrock.ToolUseStart{
							ToolUseID: tc.ID, Name: tc.Function.Name,
						}},
					})
				}
				if tc.Function.Arguments != "" {
					out = append(out, bedrock.StreamEvent{
						Type:              "contentBlockDelta",
						ContentBlockIndex: block,
						Delta: &bedrock.ContentBlockDelta{
							ToolUse: &bedrock.ToolUseDelta{Input: tc.Function.Arguments},
						},
					})
				}
			}
		}

		if ch.FinishReason != nil {
			if s.blockOpen {
				out = append(out, bedrock.StreamEvent{Type: "contentBlockStop", ContentBlockIndex: 0})
				s.blockOpen = false
			}
			for _, block := range s.toolBlock {
				out = append(out, bedrock.StreamEvent{Type: "contentBlockStop", ContentBlockIndex: block})
			}
			s.toolBlock = map[int]int{}
			out = append(out, bedrock.StreamEvent{
				Type: "messageStop", StopReason: finishReasonToConverse(*ch.FinishReason),
			})
		}
	}

	// Mantle sends usage in a final choice-less chunk.
	if u := chunk.Usage; u != nil {
		out = append(out, bedrock.StreamEvent{Type: "metadata", Usage: &bedrock.TokenUsage{
			InputTokens:  u.PromptTokens,
			OutputTokens: u.CompletionTokens,
			TotalTokens:  u.TotalTokens,
		}})
	}
	return out
}

func mustJSON(v any) json.RawMessage {
	raw, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage(`""`)
	}
	return raw
}
