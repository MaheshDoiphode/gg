package convert

import (
	"encoding/json"
	"strings"

	"bedrock-simple/internal/bedrock"
)

// ConverseToResponsesRequest renders the hub request for the Responses API.
func ConverseToResponsesRequest(model string, in *bedrock.ConverseRequest) *bedrock.ResponsesRequest {
	out := &bedrock.ResponsesRequest{Model: model, Input: []bedrock.ResponsesInput{}}

	var instructions strings.Builder
	for _, s := range in.System {
		if s.Text != nil && *s.Text != "" {
			if instructions.Len() > 0 {
				instructions.WriteString("\n")
			}
			instructions.WriteString(*s.Text)
		}
	}
	out.Instructions = instructions.String()

	for _, m := range in.Messages {
		var parts []bedrock.ResponsesContent
		// Text part types are role-specific; the API rejects the wrong one.
		textType := "input_text"
		if m.Role == "assistant" {
			textType = "output_text"
		}

		// Text explaining a call has to stay ahead of it, so pending parts are
		// flushed before each call rather than after the whole message.
		flush := func() {
			if len(parts) == 0 {
				return
			}
			out.Input = append(out.Input, bedrock.ResponsesInput{Role: m.Role, Content: parts})
			parts = nil
		}

		for _, b := range m.Content {
			switch {
			case b.Text != nil && *b.Text != "":
				parts = append(parts, bedrock.ResponsesContent{Type: textType, Text: *b.Text})

			case b.Image != nil:
				parts = append(parts, bedrock.ResponsesContent{
					Type:     "input_image",
					ImageURL: "data:image/" + b.Image.Format + ";base64," + b.Image.Source.Bytes,
				})

			case b.ToolUse != nil:
				flush()
				args := string(b.ToolUse.Input)
				if args == "" {
					args = "{}"
				}
				out.Input = append(out.Input, bedrock.ResponsesInput{
					Type:      "function_call",
					CallID:    b.ToolUse.ToolUseID,
					Name:      b.ToolUse.Name,
					Arguments: args,
				})

			case b.ToolResult != nil:
				flush()
				out.Input = append(out.Input, bedrock.ResponsesInput{
					Type:   "function_call_output",
					CallID: b.ToolResult.ToolUseID,
					Output: toolResultText(b.ToolResult),
				})
			}
		}
		flush()
	}

	if ic := in.InferenceConfig; ic != nil {
		out.MaxOutputTokens = ic.MaxTokens
		out.Temperature = ic.Temperature
		out.TopP = ic.TopP
	}
	if in.ReasoningEffort != "" {
		out.Reasoning = &bedrock.ResponsesReasoning{Effort: in.ReasoningEffort}
	}

	if in.ToolConfig != nil {
		for _, t := range in.ToolConfig.Tools {
			if t.ToolSpec == nil {
				continue
			}
			out.Tools = append(out.Tools, bedrock.ResponsesTool{
				Type:        "function",
				Name:        t.ToolSpec.Name,
				Description: t.ToolSpec.Description,
				Parameters:  t.ToolSpec.InputSchema.JSON,
			})
		}
		switch tc := in.ToolConfig.ToolChoice; {
		case tc == nil:
		case tc.Any != nil:
			out.ToolChoice = mustJSON("required")
		case tc.Auto != nil:
			out.ToolChoice = mustJSON("auto")
		case tc.Tool != nil:
			out.ToolChoice = mustJSON(map[string]string{"type": "function", "name": tc.Tool.Name})
		}
	}
	return out
}

// ResponsesStream converts Responses API events into hub events, so the OpenAI
// and Anthropic renderers work over this upstream unchanged.
type ResponsesStream struct {
	started     bool
	textBlock   int // -1 until the model emits text
	reasonBlock int // -1 until the model emits reasoning
	sawContent  bool
	toolBlock   map[string]int // function_call item id -> hub content block index
	nextBlock   int
	stopped     bool
}

// NewResponsesStream creates the upstream stream adapter.
func NewResponsesStream() *ResponsesStream {
	return &ResponsesStream{toolBlock: map[string]int{}, textBlock: -1, reasonBlock: -1}
}

// block assigns an index on first use. Anthropic requires content block indices
// to start at 0, so reserving 0 for text would strand a tool-only reply at 1.
func (s *ResponsesStream) block(slot *int) int {
	if *slot < 0 {
		*slot = s.nextBlock
		s.nextBlock++
	}
	return *slot
}

// Handle converts one Responses event into zero or more hub events.
func (s *ResponsesStream) Handle(ev bedrock.ResponsesEvent) []bedrock.StreamEvent {
	var out []bedrock.StreamEvent

	if !s.started {
		s.started = true
		out = append(out, bedrock.StreamEvent{Type: "messageStart", Role: "assistant"})
	}

	switch ev.Type {
	case "response.output_text.delta":
		if ev.Delta == "" {
			break
		}
		s.sawContent = true
		out = append(out, bedrock.StreamEvent{
			Type: "contentBlockDelta", ContentBlockIndex: s.block(&s.textBlock),
			Delta: &bedrock.ContentBlockDelta{Text: ev.Delta},
		})

	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		if ev.Delta == "" {
			break
		}
		out = append(out, bedrock.StreamEvent{
			Type: "contentBlockDelta", ContentBlockIndex: s.block(&s.reasonBlock),
			Delta: &bedrock.ContentBlockDelta{
				ReasoningContent: &bedrock.ReasoningDelta{Text: ev.Delta},
			},
		})

	case "response.output_item.added":
		if ev.Item == nil || ev.Item.Type != "function_call" {
			break
		}
		block := s.nextBlock
		s.nextBlock++
		s.toolBlock[ev.Item.ID] = block
		s.sawContent = true
		out = append(out, bedrock.StreamEvent{
			Type: "contentBlockStart", ContentBlockIndex: block,
			Start: &bedrock.ContentBlockStart{ToolUse: &bedrock.ToolUseStart{
				ToolUseID: firstNonEmptyStr(ev.Item.CallID, ev.Item.ID),
				Name:      ev.Item.Name,
			}},
		})

	case "response.function_call_arguments.delta":
		block, ok := s.toolBlock[ev.ItemID]
		if !ok || ev.Delta == "" {
			break
		}
		out = append(out, bedrock.StreamEvent{
			Type: "contentBlockDelta", ContentBlockIndex: block,
			Delta: &bedrock.ContentBlockDelta{ToolUse: &bedrock.ToolUseDelta{Input: ev.Delta}},
		})

	case "response.completed", "response.incomplete":
		out = append(out, s.finish(ev)...)
	}
	return out
}

// Produced reports whether the model emitted any content. An empty answer
// breaks agent loops silently, so callers treat it as a failure. Reasoning
// alone does not count, since it carries no answer for the caller.
func (s *ResponsesStream) Produced() bool {
	return s.sawContent || len(s.toolBlock) > 0
}

func (s *ResponsesStream) finish(ev bedrock.ResponsesEvent) []bedrock.StreamEvent {
	if s.stopped {
		return nil
	}
	s.stopped = true

	var out []bedrock.StreamEvent
	if s.reasonBlock >= 0 {
		out = append(out, bedrock.StreamEvent{Type: "contentBlockStop", ContentBlockIndex: s.reasonBlock})
		s.reasonBlock = -1
	}
	if s.textBlock >= 0 {
		out = append(out, bedrock.StreamEvent{Type: "contentBlockStop", ContentBlockIndex: s.textBlock})
		s.textBlock = -1
	}
	hadTool := len(s.toolBlock) > 0
	for _, block := range s.toolBlock {
		out = append(out, bedrock.StreamEvent{Type: "contentBlockStop", ContentBlockIndex: block})
	}
	s.toolBlock = map[string]int{}

	stop := bedrock.StopEndTurn
	switch {
	case hadTool:
		stop = bedrock.StopToolUse
	case ev.Type == "response.incomplete":
		stop = bedrock.StopMaxTokens
	}
	out = append(out, bedrock.StreamEvent{Type: "messageStop", StopReason: stop})

	if ev.Response != nil && ev.Response.Usage != nil {
		u := ev.Response.Usage
		out = append(out, bedrock.StreamEvent{Type: "metadata", Usage: &bedrock.TokenUsage{
			InputTokens:  u.InputTokens,
			OutputTokens: u.OutputTokens,
			TotalTokens:  u.TotalTokens,
		}})
	}
	return out
}

// Finish closes the stream if the upstream ended without a terminal event.
func (s *ResponsesStream) Finish() []bedrock.StreamEvent {
	return s.finish(bedrock.ResponsesEvent{})
}

// AggregateToConverse folds hub stream events into a complete response, for
// clients that asked for a non-streaming reply on a stream-only upstream.
func AggregateToConverse(events []bedrock.StreamEvent) *bedrock.ConverseResponse {
	out := &bedrock.ConverseResponse{StopReason: bedrock.StopEndTurn}
	var text, reasoning strings.Builder
	toolArgs := map[int]*strings.Builder{}
	toolMeta := map[int]*bedrock.ToolUseStart{}
	var order []int

	for _, ev := range events {
		switch ev.Type {
		case "contentBlockStart":
			if ev.Start != nil && ev.Start.ToolUse != nil {
				toolMeta[ev.ContentBlockIndex] = ev.Start.ToolUse
				toolArgs[ev.ContentBlockIndex] = &strings.Builder{}
				order = append(order, ev.ContentBlockIndex)
			}
		case "contentBlockDelta":
			if ev.Delta == nil {
				continue
			}
			switch {
			case ev.Delta.Text != "":
				text.WriteString(ev.Delta.Text)
			case ev.Delta.ReasoningContent != nil:
				reasoning.WriteString(ev.Delta.ReasoningContent.Text)
			case ev.Delta.ToolUse != nil:
				if b, ok := toolArgs[ev.ContentBlockIndex]; ok {
					b.WriteString(ev.Delta.ToolUse.Input)
				}
			}
		case "messageStop":
			out.StopReason = ev.StopReason
		case "metadata":
			if ev.Usage != nil {
				out.Usage = *ev.Usage
			}
		}
	}

	msg := &bedrock.Message{Role: "assistant"}
	if r := reasoning.String(); r != "" {
		msg.Content = append(msg.Content, bedrock.ContentBlock{
			ReasoningContent: &bedrock.ReasoningContent{
				ReasoningText: &bedrock.ReasoningText{Text: r},
			},
		})
	}
	if t := text.String(); t != "" {
		msg.Content = append(msg.Content, bedrock.ContentBlock{Text: &t})
	}
	for _, idx := range order {
		args := json.RawMessage(strings.TrimSpace(toolArgs[idx].String()))
		if len(args) == 0 || !json.Valid(args) {
			args = json.RawMessage(`{}`)
		}
		msg.Content = append(msg.Content, bedrock.ContentBlock{ToolUse: &bedrock.ToolUseBlock{
			ToolUseID: toolMeta[idx].ToolUseID, Name: toolMeta[idx].Name, Input: args,
		}})
	}
	out.Output.Message = msg
	return out
}

func firstNonEmptyStr(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
