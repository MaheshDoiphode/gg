package convert

import (
	"encoding/json"
	"fmt"
	"strings"

	"bedrock-simple/internal/anthropic"
	"bedrock-simple/internal/bedrock"
)

// AnthropicToConverse maps an Anthropic Messages request onto Converse.
// resolvedModel gates the provider-specific fields, which are not portable.
func AnthropicToConverse(req *anthropic.Request, resolvedModel string, defaultMaxTokens int) (*bedrock.ConverseRequest, error) {
	out := &bedrock.ConverseRequest{}

	for _, s := range req.System {
		if s.Text != "" {
			out.System = append(out.System, bedrock.SystemBlock{Text: bedrock.Ptr(s.Text)})
		}
	}

	for _, m := range req.Messages {
		blocks, err := anthropicBlocks(m.Content)
		if err != nil {
			return nil, err
		}
		if len(blocks) == 0 {
			continue
		}
		if n := len(out.Messages); n > 0 && out.Messages[n-1].Role == m.Role {
			out.Messages[n-1].Content = append(out.Messages[n-1].Content, blocks...)
			continue
		}
		out.Messages = append(out.Messages, bedrock.Message{Role: m.Role, Content: blocks})
	}
	if len(out.Messages) == 0 {
		return nil, fmt.Errorf("no usable messages in request")
	}

	out.InferenceConfig = &bedrock.InferenceConfig{
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		StopSequences: req.StopSequences,
	}
	if req.MaxTokens > 0 {
		out.InferenceConfig.MaxTokens = &req.MaxTokens
	}

	anthropicModel := isAnthropicModel(resolvedModel)
	if req.Thinking != nil {
		if req.Thinking.Type == "enabled" {
			out.ReasoningEffort = bedrock.EffortForBudget(req.Thinking.BudgetTokens)
		} else {
			out.ReasoningEffort = bedrock.EffortNone
		}
	}
	if req.TopK != nil && anthropicModel {
		out.AdditionalModelRequestFields = map[string]any{"top_k": *req.TopK}
	}
	if req.Thinking != nil && req.Thinking.Type == "enabled" && anthropicModel {
		if out.AdditionalModelRequestFields == nil {
			out.AdditionalModelRequestFields = map[string]any{}
		}
		budget := req.Thinking.BudgetTokens
		if budget <= 0 {
			budget = 1024
		}
		out.AdditionalModelRequestFields["thinking"] = map[string]any{
			"type":          "enabled",
			"budget_tokens": budget,
		}
		out.InferenceConfig.Temperature = nil
		out.InferenceConfig.TopP = nil
		if mt := out.InferenceConfig.MaxTokens; mt == nil || *mt <= budget {
			out.InferenceConfig.MaxTokens = bedrock.Ptr(budget + 4096)
		}
	}

	if len(req.Tools) > 0 {
		tools := make([]bedrock.Tool, 0, len(req.Tools))
		for _, t := range req.Tools {
			if t.Name == "" {
				continue
			}
			schema := t.InputSchema
			if len(schema) == 0 || !json.Valid(schema) {
				schema = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			tools = append(tools, bedrock.Tool{ToolSpec: &bedrock.ToolSpec{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: bedrock.ToolInputSchema{JSON: schema},
			}})
		}
		if len(tools) > 0 {
			out.ToolConfig = &bedrock.ToolConfig{Tools: tools, ToolChoice: anthropicToolChoice(req.ToolChoice)}
		}
	}
	return out, nil
}

func anthropicToolChoice(raw json.RawMessage) *bedrock.ToolChoice {
	if len(raw) == 0 {
		return nil
	}
	var tc struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &tc) != nil {
		return nil
	}
	switch tc.Type {
	case "any":
		return &bedrock.ToolChoice{Any: &struct{}{}}
	case "auto":
		return &bedrock.ToolChoice{Auto: &struct{}{}}
	case "tool":
		if tc.Name != "" {
			return &bedrock.ToolChoice{Tool: &bedrock.ToolChoiceTool{Name: tc.Name}}
		}
	}
	return nil
}

func anthropicBlocks(in []anthropic.ContentBlock) ([]bedrock.ContentBlock, error) {
	var out []bedrock.ContentBlock
	for _, b := range in {
		switch b.Type {
		case "text":
			if b.Text != "" {
				out = append(out, bedrock.ContentBlock{Text: bedrock.Ptr(b.Text)})
			}

		case "thinking":
			if b.Thinking != "" {
				out = append(out, bedrock.ContentBlock{ReasoningContent: &bedrock.ReasoningContent{
					ReasoningText: &bedrock.ReasoningText{Text: b.Thinking, Signature: b.Signature},
				}})
			}

		case "redacted_thinking":
			if b.Data != "" {
				out = append(out, bedrock.ContentBlock{ReasoningContent: &bedrock.ReasoningContent{
					RedactedContent: b.Data,
				}})
			}

		case "image":
			if b.Source == nil || b.Source.Data == "" {
				continue
			}
			format := strings.TrimPrefix(b.Source.MediaType, "image/")
			if format == "jpg" {
				format = "jpeg"
			}
			switch format {
			case "png", "jpeg", "gif", "webp":
			default:
				continue
			}
			out = append(out, bedrock.ContentBlock{Image: &bedrock.ImageBlock{
				Format: format,
				Source: bedrock.ImageSource{Bytes: b.Source.Data},
			}})

		case "tool_use":
			input := b.Input
			if len(input) == 0 || !json.Valid(input) {
				input = json.RawMessage(`{}`)
			}
			out = append(out, bedrock.ContentBlock{ToolUse: &bedrock.ToolUseBlock{
				ToolUseID: b.ID, Name: b.Name, Input: input,
			}})

		case "tool_result":
			res := &bedrock.ToolResultBlock{ToolUseID: b.ToolUseID}
			if b.IsError {
				res.Status = "error"
			}
			res.Content = toolResultContent(b.Content)
			out = append(out, bedrock.ContentBlock{ToolResult: res})
		}
	}
	return out, nil
}

// toolResultContent accepts the string form and the block-array form.
func toolResultContent(raw json.RawMessage) []bedrock.ToolResultContent {
	if len(raw) == 0 || string(raw) == "null" {
		return []bedrock.ToolResultContent{{Text: bedrock.Ptr("")}}
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return []bedrock.ToolResultContent{{Text: &s}}
	}
	var blocks []anthropic.ContentBlock
	if json.Unmarshal(raw, &blocks) == nil {
		var out []bedrock.ToolResultContent
		for _, b := range blocks {
			if b.Type == "text" {
				out = append(out, bedrock.ToolResultContent{Text: bedrock.Ptr(b.Text)})
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	return []bedrock.ToolResultContent{{Text: bedrock.Ptr(string(raw))}}
}

// ConverseToAnthropic builds a non-streaming Anthropic response.
func ConverseToAnthropic(resp *bedrock.ConverseResponse, id, model string) *anthropic.Response {
	out := &anthropic.Response{
		ID:         id,
		Type:       "message",
		Role:       "assistant",
		Model:      model,
		StopReason: stopReasonToAnthropic(resp.StopReason),
		Usage: anthropic.Usage{
			InputTokens:             resp.Usage.InputTokens,
			OutputTokens:            resp.Usage.OutputTokens,
			CacheReadInputTokens:    resp.Usage.CacheReadInputTokens,
			CacheCreationInputTokes: resp.Usage.CacheWriteInputTokens,
		},
	}

	if resp.Output.Message != nil {
		for _, b := range resp.Output.Message.Content {
			switch {
			case b.Text != nil:
				out.Content = append(out.Content, anthropic.ContentBlock{Type: "text", Text: *b.Text})
			case b.ReasoningContent != nil && b.ReasoningContent.ReasoningText != nil:
				out.Content = append(out.Content, anthropic.ContentBlock{
					Type:      "thinking",
					Thinking:  b.ReasoningContent.ReasoningText.Text,
					Signature: b.ReasoningContent.ReasoningText.Signature,
				})
			case b.ReasoningContent != nil && b.ReasoningContent.RedactedContent != "":
				out.Content = append(out.Content, anthropic.ContentBlock{
					Type: "redacted_thinking", Data: b.ReasoningContent.RedactedContent,
				})
			case b.ToolUse != nil:
				input := b.ToolUse.Input
				if len(input) == 0 {
					input = json.RawMessage(`{}`)
				}
				out.Content = append(out.Content, anthropic.ContentBlock{
					Type: "tool_use", ID: b.ToolUse.ToolUseID, Name: b.ToolUse.Name, Input: input,
				})
			}
		}
	}
	if out.Content == nil {
		out.Content = []anthropic.ContentBlock{{Type: "text", Text: ""}}
	}
	return out
}

// stopReasonToAnthropic normalises Converse stop reasons, which mostly already
// use Anthropic's vocabulary.
func stopReasonToAnthropic(reason string) string {
	switch reason {
	case bedrock.StopEndTurn, bedrock.StopToolUse, bedrock.StopMaxTokens, bedrock.StopStopSequence:
		return reason
	case "model_context_window_exceeded":
		return "max_tokens"
	case "":
		return "end_turn"
	default:
		return "end_turn"
	}
}

// AnthropicStream converts Converse events into the Anthropic SSE event
// sequence. Converse omits contentBlockStart for plain text, so blocks are
// opened lazily on their first delta.
type AnthropicStream struct {
	ID    string
	Model string

	started    bool
	openBlocks map[int]string
	Usage      bedrock.TokenUsage
	stopReason string
}

// SSE is one server-sent event to write out.
type SSE struct {
	Event string
	Data  any
}

// NewAnthropicStream creates a stream converter.
func NewAnthropicStream(id, model string) *AnthropicStream {
	return &AnthropicStream{ID: id, Model: model, openBlocks: map[int]string{}}
}

// Start emits the message_start event.
func (s *AnthropicStream) Start(inputTokens int) []SSE {
	s.started = true
	return []SSE{{Event: "message_start", Data: map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": s.ID, "type": "message", "role": "assistant", "model": s.Model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": inputTokens, "output_tokens": 0},
		},
	}}}
}

func (s *AnthropicStream) openBlock(idx int, kind string) []SSE {
	if _, ok := s.openBlocks[idx]; ok {
		return nil
	}
	s.openBlocks[idx] = kind
	var block map[string]any
	switch kind {
	case "thinking":
		block = map[string]any{"type": "thinking", "thinking": ""}
	default:
		block = map[string]any{"type": "text", "text": ""}
	}
	return []SSE{{Event: "content_block_start", Data: map[string]any{
		"type": "content_block_start", "index": idx, "content_block": block,
	}}}
}

// Handle turns one Converse event into zero or more Anthropic SSE events.
func (s *AnthropicStream) Handle(ev bedrock.StreamEvent) []SSE {
	var out []SSE
	idx := ev.ContentBlockIndex

	switch ev.Type {
	case "contentBlockStart":
		if ev.Start != nil && ev.Start.ToolUse != nil {
			s.openBlocks[idx] = "tool_use"
			out = append(out, SSE{Event: "content_block_start", Data: map[string]any{
				"type": "content_block_start", "index": idx,
				"content_block": map[string]any{
					"type":  "tool_use",
					"id":    ev.Start.ToolUse.ToolUseID,
					"name":  ev.Start.ToolUse.Name,
					"input": map[string]any{},
				},
			}})
		}

	case "contentBlockDelta":
		if ev.Delta == nil {
			break
		}
		switch {
		case ev.Delta.Text != "":
			out = append(out, s.openBlock(idx, "text")...)
			out = append(out, SSE{Event: "content_block_delta", Data: map[string]any{
				"type": "content_block_delta", "index": idx,
				"delta": map[string]any{"type": "text_delta", "text": ev.Delta.Text},
			}})

		case ev.Delta.ReasoningContent != nil && ev.Delta.ReasoningContent.Text != "":
			out = append(out, s.openBlock(idx, "thinking")...)
			out = append(out, SSE{Event: "content_block_delta", Data: map[string]any{
				"type": "content_block_delta", "index": idx,
				"delta": map[string]any{"type": "thinking_delta", "thinking": ev.Delta.ReasoningContent.Text},
			}})

		case ev.Delta.ReasoningContent != nil && ev.Delta.ReasoningContent.Signature != "":
			out = append(out, s.openBlock(idx, "thinking")...)
			out = append(out, SSE{Event: "content_block_delta", Data: map[string]any{
				"type": "content_block_delta", "index": idx,
				"delta": map[string]any{"type": "signature_delta", "signature": ev.Delta.ReasoningContent.Signature},
			}})

		case ev.Delta.ToolUse != nil:
			out = append(out, SSE{Event: "content_block_delta", Data: map[string]any{
				"type": "content_block_delta", "index": idx,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": ev.Delta.ToolUse.Input},
			}})
		}

	case "contentBlockStop":
		if _, ok := s.openBlocks[idx]; ok {
			delete(s.openBlocks, idx)
			out = append(out, SSE{Event: "content_block_stop", Data: map[string]any{
				"type": "content_block_stop", "index": idx,
			}})
		}

	case "messageStop":
		s.stopReason = stopReasonToAnthropic(ev.StopReason)

	case "metadata":
		if ev.Usage != nil {
			s.Usage = *ev.Usage
		}
	}
	return out
}

// Finish closes any dangling blocks and emits message_delta + message_stop.
func (s *AnthropicStream) Finish() []SSE {
	var out []SSE
	for idx := range s.openBlocks {
		out = append(out, SSE{Event: "content_block_stop", Data: map[string]any{
			"type": "content_block_stop", "index": idx,
		}})
	}
	s.openBlocks = map[int]string{}

	reason := s.stopReason
	if reason == "" {
		reason = "end_turn"
	}
	out = append(out, SSE{Event: "message_delta", Data: map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": reason, "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": s.Usage.OutputTokens},
	}})
	out = append(out, SSE{Event: "message_stop", Data: map[string]any{"type": "message_stop"}})
	return out
}
