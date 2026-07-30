// Package convert translates the client-facing APIs to and from Bedrock's
// Converse protocol, which is the single upstream shape this proxy speaks.
package convert

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"bedrock-simple/internal/bedrock"
	"bedrock-simple/internal/openai"
)

// OpenAIToConverse maps an OpenAI chat completion request onto Converse.
func OpenAIToConverse(req *openai.ChatRequest, resolvedModel string, defaultMaxTokens int) (*bedrock.ConverseRequest, error) {
	out := &bedrock.ConverseRequest{}

	var system []bedrock.SystemBlock
	var msgs []bedrock.Message

	// appendTurn merges into the previous turn when roles match, because
	// Converse requires the conversation to alternate user/assistant.
	appendTurn := func(role string, blocks []bedrock.ContentBlock) {
		if len(blocks) == 0 {
			return
		}
		if n := len(msgs); n > 0 && msgs[n-1].Role == role {
			msgs[n-1].Content = append(msgs[n-1].Content, blocks...)
			return
		}
		msgs = append(msgs, bedrock.Message{Role: role, Content: blocks})
	}

	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
			if txt := plainText(m.Content); txt != "" {
				system = append(system, bedrock.SystemBlock{Text: bedrock.Ptr(txt)})
			}

		case "user":
			blocks, err := userContentBlocks(m.Content)
			if err != nil {
				return nil, err
			}
			appendTurn("user", blocks)

		case "assistant":
			var blocks []bedrock.ContentBlock
			if txt := plainText(m.Content); txt != "" {
				blocks = append(blocks, bedrock.ContentBlock{Text: bedrock.Ptr(txt)})
			}
			for _, tc := range m.ToolCalls {
				input := json.RawMessage(strings.TrimSpace(tc.Function.Arguments))
				if len(input) == 0 || !json.Valid(input) {
					input = json.RawMessage(`{}`)
				}
				blocks = append(blocks, bedrock.ContentBlock{ToolUse: &bedrock.ToolUseBlock{
					ToolUseID: tc.ID,
					Name:      tc.Function.Name,
					Input:     input,
				}})
			}
			appendTurn("assistant", blocks)

		case "tool", "function":
			// Converse carries tool results on a user turn.
			text := plainText(m.Content)
			appendTurn("user", []bedrock.ContentBlock{{ToolResult: &bedrock.ToolResultBlock{
				ToolUseID: m.ToolCallID,
				Content:   []bedrock.ToolResultContent{{Text: bedrock.Ptr(text)}},
			}}})
		}
	}

	if len(msgs) == 0 {
		return nil, fmt.Errorf("no usable messages in request")
	}
	out.System = system
	out.Messages = msgs

	// Left nil when the client did not ask for a cap, so the model decides.
	out.InferenceConfig = &bedrock.InferenceConfig{
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		StopSequences: parseStop(req.Stop),
	}
	if req.MaxCompletionTok != nil && *req.MaxCompletionTok > 0 {
		out.InferenceConfig.MaxTokens = req.MaxCompletionTok
	} else if req.MaxTokens != nil && *req.MaxTokens > 0 {
		out.InferenceConfig.MaxTokens = req.MaxTokens
	}

	if len(req.Tools) > 0 {
		tools := make([]bedrock.Tool, 0, len(req.Tools))
		for _, t := range req.Tools {
			if t.Function.Name == "" {
				continue
			}
			schema := t.Function.Parameters
			if len(schema) == 0 || !json.Valid(schema) {
				schema = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			tools = append(tools, bedrock.Tool{ToolSpec: &bedrock.ToolSpec{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				InputSchema: bedrock.ToolInputSchema{JSON: schema},
			}})
		}
		if len(tools) > 0 {
			out.ToolConfig = &bedrock.ToolConfig{Tools: tools, ToolChoice: openAIToolChoice(req.ToolChoice)}
		}
	}

	out.ReasoningEffort = bedrock.NormalizeEffort(req.ReasoningEffort)
	applyReasoning(out, resolvedModel, req.ReasoningEffort)
	return out, nil
}

// IsAnthropicModel reports whether provider-specific Anthropic fields such as
// thinking and top_k are safe to send.
func IsAnthropicModel(model string) bool {
	l := strings.ToLower(model)
	return strings.Contains(l, "anthropic") || strings.Contains(l, "claude")
}

func isAnthropicModel(model string) bool { return IsAnthropicModel(model) }

// applyReasoning turns OpenAI's reasoning_effort into the provider-specific
// extended-thinking field. Only Anthropic models are touched, since the field
// name is not portable.
func applyReasoning(out *bedrock.ConverseRequest, model, effort string) {
	budget := 0
	switch strings.ToLower(effort) {
	case "low":
		budget = 1024
	case "medium":
		budget = 4096
	case "high":
		budget = 16384
	default:
		return
	}
	if !isAnthropicModel(model) {
		return
	}
	if out.AdditionalModelRequestFields == nil {
		out.AdditionalModelRequestFields = map[string]any{}
	}
	out.AdditionalModelRequestFields["thinking"] = map[string]any{
		"type":          "enabled",
		"budget_tokens": budget,
	}
	if out.InferenceConfig != nil {
		// Anthropic rejects temperature/top_p while thinking, and requires
		// max_tokens to leave room beyond the thinking budget.
		out.InferenceConfig.Temperature = nil
		out.InferenceConfig.TopP = nil
		if out.InferenceConfig.MaxTokens == nil || *out.InferenceConfig.MaxTokens <= budget {
			out.InferenceConfig.MaxTokens = bedrock.Ptr(budget + 4096)
		}
	}
}

func openAIToolChoice(raw json.RawMessage) *bedrock.ToolChoice {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch s {
		case "required":
			return &bedrock.ToolChoice{Any: &struct{}{}}
		case "auto":
			return &bedrock.ToolChoice{Auto: &struct{}{}}
		default: // "none" has no Converse equivalent; let the model decide.
			return nil
		}
	}
	var named struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &named) == nil && named.Function.Name != "" {
		return &bedrock.ToolChoice{Tool: &bedrock.ToolChoiceTool{Name: named.Function.Name}}
	}
	return nil
}

func parseStop(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		if one == "" {
			return nil
		}
		return []string{one}
	}
	var many []string
	if json.Unmarshal(raw, &many) == nil {
		return many
	}
	return nil
}

// plainText flattens a string-or-parts content field into text.
func plainText(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []openai.ContentPart
	if json.Unmarshal(raw, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Text != "" {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return ""
}

// userContentBlocks handles multimodal user content.
func userContentBlocks(raw json.RawMessage) ([]bedrock.ContentBlock, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "" {
			return nil, nil
		}
		return []bedrock.ContentBlock{{Text: &s}}, nil
	}

	var parts []openai.ContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("unsupported message content: %w", err)
	}

	var blocks []bedrock.ContentBlock
	for _, p := range parts {
		switch p.Type {
		case "text", "input_text":
			if p.Text != "" {
				blocks = append(blocks, bedrock.ContentBlock{Text: bedrock.Ptr(p.Text)})
			}
		case "image_url", "input_image":
			if p.ImageURL == nil {
				continue
			}
			format, data, ok := parseDataURL(p.ImageURL.URL)
			if !ok {
				// Bedrock cannot fetch remote URLs; tell the model instead of
				// silently dropping the image.
				blocks = append(blocks, bedrock.ContentBlock{
					Text: bedrock.Ptr("[image omitted: only inline data: URLs are supported]"),
				})
				continue
			}
			blocks = append(blocks, bedrock.ContentBlock{Image: &bedrock.ImageBlock{
				Format: format,
				Source: bedrock.ImageSource{Bytes: data},
			}})
		}
	}
	return blocks, nil
}

// parseDataURL splits "data:image/png;base64,AAAA" into ("png", "AAAA").
func parseDataURL(u string) (format, data string, ok bool) {
	if !strings.HasPrefix(u, "data:") {
		return "", "", false
	}
	comma := strings.Index(u, ",")
	if comma < 0 {
		return "", "", false
	}
	meta, payload := u[5:comma], u[comma+1:]
	if !strings.Contains(meta, "base64") {
		return "", "", false
	}
	mime := strings.TrimSuffix(meta, ";base64")
	format = strings.TrimPrefix(mime, "image/")
	switch format {
	case "png", "jpeg", "gif", "webp":
	case "jpg":
		format = "jpeg"
	default:
		return "", "", false
	}
	if _, err := base64.StdEncoding.DecodeString(payload); err != nil {
		return "", "", false
	}
	return format, payload, true
}

// stopReasonToOpenAI maps a Converse stop reason to an OpenAI finish_reason.
func stopReasonToOpenAI(reason string, hadToolCall bool) string {
	switch reason {
	case bedrock.StopToolUse:
		return "tool_calls"
	case bedrock.StopMaxTokens, "model_context_window_exceeded":
		return "length"
	case "guardrail_intervened", "content_filtered":
		return "content_filter"
	case bedrock.StopEndTurn, bedrock.StopStopSequence:
		return "stop"
	}
	if hadToolCall {
		return "tool_calls"
	}
	return "stop"
}

// ConverseToOpenAI builds a non-streaming chat completion from a Converse reply.
func ConverseToOpenAI(resp *bedrock.ConverseResponse, id, model string, created int64) *openai.ChatResponse {
	var text, reasoning strings.Builder
	var toolCalls []openai.ToolCall

	if resp.Output.Message != nil {
		for _, b := range resp.Output.Message.Content {
			switch {
			case b.Text != nil:
				text.WriteString(*b.Text)
			case b.ReasoningContent != nil && b.ReasoningContent.ReasoningText != nil:
				reasoning.WriteString(b.ReasoningContent.ReasoningText.Text)
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
			}
		}
	}

	content := text.String()
	msg := &openai.ResponseMsg{Role: "assistant", Content: &content, ToolCalls: toolCalls}
	if r := reasoning.String(); r != "" {
		msg.ReasoningContent = r
	}
	finish := stopReasonToOpenAI(resp.StopReason, len(toolCalls) > 0)

	return &openai.ChatResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: []openai.Choice{{Index: 0, Message: msg, FinishReason: &finish}},
		Usage:   usageToOpenAI(&resp.Usage),
	}
}

func usageToOpenAI(u *bedrock.TokenUsage) *openai.Usage {
	if u == nil {
		return nil
	}
	out := &openai.Usage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.TotalTokens,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = u.InputTokens + u.OutputTokens
	}
	if u.CacheReadInputTokens > 0 {
		out.PromptTokensDetails = &openai.PromptDetail{CachedTokens: u.CacheReadInputTokens}
	}
	return out
}

// OpenAIStream converts Converse stream events into chat.completion.chunk
// frames. Usage is captured so the caller can record it after the stream ends.
type OpenAIStream struct {
	ID      string
	Model   string
	Created int64

	sentRole  bool
	nextTool  int
	blockTool map[int]int
	hadTool   bool
	Usage     *bedrock.TokenUsage
}

// NewOpenAIStream creates a stream converter.
func NewOpenAIStream(id, model string, created int64) *OpenAIStream {
	return &OpenAIStream{ID: id, Model: model, Created: created, blockTool: map[int]int{}}
}

func (s *OpenAIStream) chunk(delta *openai.ResponseMsg, finish *string) openai.ChatResponse {
	return openai.ChatResponse{
		ID:      s.ID,
		Object:  "chat.completion.chunk",
		Created: s.Created,
		Model:   s.Model,
		Choices: []openai.Choice{{Index: 0, Delta: delta, FinishReason: finish}},
	}
}

// Handle turns one Converse event into zero or more OpenAI chunks.
func (s *OpenAIStream) Handle(ev bedrock.StreamEvent) []openai.ChatResponse {
	var out []openai.ChatResponse

	if !s.sentRole && ev.Type != "metadata" {
		s.sentRole = true
		empty := ""
		out = append(out, s.chunk(&openai.ResponseMsg{Role: "assistant", Content: &empty}, nil))
	}

	switch ev.Type {
	case "contentBlockStart":
		if ev.Start != nil && ev.Start.ToolUse != nil {
			idx := s.nextTool
			s.nextTool++
			s.blockTool[ev.ContentBlockIndex] = idx
			s.hadTool = true
			out = append(out, s.chunk(&openai.ResponseMsg{ToolCalls: []openai.ToolCall{{
				Index:    &idx,
				ID:       ev.Start.ToolUse.ToolUseID,
				Type:     "function",
				Function: openai.FunctionCall{Name: ev.Start.ToolUse.Name, Arguments: ""},
			}}}, nil))
		}

	case "contentBlockDelta":
		if ev.Delta == nil {
			break
		}
		switch {
		case ev.Delta.Text != "":
			t := ev.Delta.Text
			out = append(out, s.chunk(&openai.ResponseMsg{Content: &t}, nil))
		case ev.Delta.ReasoningContent != nil && ev.Delta.ReasoningContent.Text != "":
			out = append(out, s.chunk(&openai.ResponseMsg{ReasoningContent: ev.Delta.ReasoningContent.Text}, nil))
		case ev.Delta.ToolUse != nil:
			idx, ok := s.blockTool[ev.ContentBlockIndex]
			if !ok {
				idx = 0
			}
			out = append(out, s.chunk(&openai.ResponseMsg{ToolCalls: []openai.ToolCall{{
				Index:    &idx,
				Function: openai.FunctionCall{Arguments: ev.Delta.ToolUse.Input},
			}}}, nil))
		}

	case "messageStop":
		finish := stopReasonToOpenAI(ev.StopReason, s.hadTool)
		out = append(out, s.chunk(&openai.ResponseMsg{}, &finish))

	case "metadata":
		if ev.Usage != nil {
			s.Usage = ev.Usage
		}
	}
	return out
}

// UsageChunk is the final usage-only frame emitted for stream_options.include_usage.
func (s *OpenAIStream) UsageChunk() *openai.ChatResponse {
	if s.Usage == nil {
		return nil
	}
	return &openai.ChatResponse{
		ID:      s.ID,
		Object:  "chat.completion.chunk",
		Created: s.Created,
		Model:   s.Model,
		Choices: []openai.Choice{},
		Usage:   usageToOpenAI(s.Usage),
	}
}
