package convert

import (
	"encoding/json"
	"strings"
	"testing"

	"bedrock-simple/internal/bedrock"
)

func TestConverseToResponsesRequestShapes(t *testing.T) {
	in := &bedrock.ConverseRequest{
		System: []bedrock.SystemBlock{{Text: bedrock.Ptr("be terse")}},
		Messages: []bedrock.Message{
			{Role: "user", Content: []bedrock.ContentBlock{{Text: bedrock.Ptr("weather?")}}},
			{Role: "assistant", Content: []bedrock.ContentBlock{
				{Text: bedrock.Ptr("checking")},
				{ToolUse: &bedrock.ToolUseBlock{ToolUseID: "c1", Name: "f", Input: json.RawMessage(`{"a":1}`)}},
			}},
			{Role: "user", Content: []bedrock.ContentBlock{
				{ToolResult: &bedrock.ToolResultBlock{
					ToolUseID: "c1",
					Content:   []bedrock.ToolResultContent{{Text: bedrock.Ptr("31C")}},
				}},
			}},
		},
		InferenceConfig: &bedrock.InferenceConfig{MaxTokens: bedrock.Ptr(500)},
		ToolConfig: &bedrock.ToolConfig{Tools: []bedrock.Tool{{ToolSpec: &bedrock.ToolSpec{
			Name: "f", Description: "d", InputSchema: bedrock.ToolInputSchema{JSON: json.RawMessage(`{"type":"object"}`)},
		}}}},
	}

	out := ConverseToResponsesRequest("xai.grok-4.3", in)

	if out.Instructions != "be terse" {
		t.Errorf("instructions = %q", out.Instructions)
	}
	if *out.MaxOutputTokens != 500 {
		t.Errorf("max_output_tokens = %d", *out.MaxOutputTokens)
	}
	// Tools are flattened, not nested under "function".
	if len(out.Tools) != 1 || out.Tools[0].Name != "f" || out.Tools[0].Type != "function" {
		t.Fatalf("tools = %#v", out.Tools)
	}

	// The API rejects the wrong text part type per role.
	var sawInputText, sawOutputText, sawCall, sawCallOutput bool
	for _, item := range out.Input {
		switch item.Type {
		case "function_call":
			sawCall = item.CallID == "c1" && item.Name == "f"
		case "function_call_output":
			sawCallOutput = item.CallID == "c1" && item.Output == "31C"
		default:
			for _, p := range item.Content {
				if item.Role == "user" && p.Type == "input_text" {
					sawInputText = true
				}
				if item.Role == "assistant" && p.Type == "output_text" {
					sawOutputText = true
				}
			}
		}
	}
	if !sawInputText || !sawOutputText {
		t.Errorf("text part types wrong: input=%v output=%v", sawInputText, sawOutputText)
	}
	if !sawCall || !sawCallOutput {
		t.Errorf("function call round trip: call=%v output=%v", sawCall, sawCallOutput)
	}
}

func TestResponsesStreamTextAndUsage(t *testing.T) {
	s := NewResponsesStream()
	var events []bedrock.StreamEvent

	for _, ev := range []bedrock.ResponsesEvent{
		{Type: "response.created"},
		{Type: "response.output_text.delta", Delta: "GROK"},
		{Type: "response.output_text.delta", Delta: " OK"},
		{Type: "response.completed", Response: &bedrock.ResponsesResult{
			Usage: &bedrock.ResponsesUsage{InputTokens: 9, OutputTokens: 3, TotalTokens: 12},
		}},
	} {
		events = append(events, s.Handle(ev)...)
	}

	var kinds []string
	var text strings.Builder
	var usage *bedrock.TokenUsage
	for _, e := range events {
		kinds = append(kinds, e.Type)
		if e.Delta != nil {
			text.WriteString(e.Delta.Text)
		}
		if e.Usage != nil {
			usage = e.Usage
		}
	}
	if text.String() != "GROK OK" {
		t.Errorf("text = %q", text.String())
	}
	if usage == nil || usage.OutputTokens != 3 {
		t.Errorf("usage = %#v", usage)
	}
	want := "messageStart,contentBlockDelta,contentBlockDelta,contentBlockStop,messageStop,metadata"
	if got := strings.Join(kinds, ","); got != want {
		t.Errorf("events\n got: %s\nwant: %s", got, want)
	}
}

func TestResponsesStreamToolCall(t *testing.T) {
	s := NewResponsesStream()
	var events []bedrock.StreamEvent

	for _, ev := range []bedrock.ResponsesEvent{
		{Type: "response.output_item.added", Item: &bedrock.ResponsesItem{
			Type: "function_call", ID: "fc_1", CallID: "call_1", Name: "get_weather",
		}},
		{Type: "response.function_call_arguments.delta", ItemID: "fc_1", Delta: `{"city":`},
		{Type: "response.function_call_arguments.delta", ItemID: "fc_1", Delta: `"Pune"}`},
		{Type: "response.completed", Response: &bedrock.ResponsesResult{}},
	} {
		events = append(events, s.Handle(ev)...)
	}

	// Arguments are keyed by item_id, not call_id; mixing them loses the args.
	var args strings.Builder
	var start *bedrock.ToolUseStart
	var stop string
	for _, e := range events {
		if e.Start != nil && e.Start.ToolUse != nil {
			start = e.Start.ToolUse
		}
		if e.Delta != nil && e.Delta.ToolUse != nil {
			args.WriteString(e.Delta.ToolUse.Input)
		}
		if e.Type == "messageStop" {
			stop = e.StopReason
		}
	}
	if start == nil || start.ToolUseID != "call_1" || start.Name != "get_weather" {
		t.Fatalf("tool start = %#v", start)
	}
	if args.String() != `{"city":"Pune"}` {
		t.Errorf("args = %q", args.String())
	}
	if stop != bedrock.StopToolUse {
		t.Errorf("stop reason = %q", stop)
	}
}

// A reasoning model can burn the whole budget before emitting text.
func TestResponsesStreamIncompleteMapsToMaxTokens(t *testing.T) {
	s := NewResponsesStream()
	var stop string
	for _, e := range s.Handle(bedrock.ResponsesEvent{Type: "response.incomplete",
		Response: &bedrock.ResponsesResult{}}) {
		if e.Type == "messageStop" {
			stop = e.StopReason
		}
	}
	if stop != bedrock.StopMaxTokens {
		t.Errorf("stop reason = %q", stop)
	}
}

// If the upstream connection ends without a terminal event, the stream must
// still be closed or clients hang.
func TestResponsesStreamFinishIsIdempotent(t *testing.T) {
	s := NewResponsesStream()
	s.Handle(bedrock.ResponsesEvent{Type: "response.output_text.delta", Delta: "hi"})

	first := s.Finish()
	if len(first) == 0 {
		t.Fatal("Finish must close a dangling stream")
	}
	if second := s.Finish(); len(second) != 0 {
		t.Fatalf("Finish must not emit twice: %#v", second)
	}
}

func TestAggregateToConverse(t *testing.T) {
	s := NewResponsesStream()
	var events []bedrock.StreamEvent
	for _, ev := range []bedrock.ResponsesEvent{
		{Type: "response.output_text.delta", Delta: "hello"},
		{Type: "response.output_item.added", Item: &bedrock.ResponsesItem{
			Type: "function_call", ID: "fc_1", CallID: "call_1", Name: "f",
		}},
		{Type: "response.function_call_arguments.delta", ItemID: "fc_1", Delta: `{"a":1}`},
		{Type: "response.completed", Response: &bedrock.ResponsesResult{
			Usage: &bedrock.ResponsesUsage{InputTokens: 4, OutputTokens: 5},
		}},
	} {
		events = append(events, s.Handle(ev)...)
	}

	out := AggregateToConverse(events)
	if out.Usage.OutputTokens != 5 {
		t.Errorf("usage = %#v", out.Usage)
	}
	if out.StopReason != bedrock.StopToolUse {
		t.Errorf("stop = %q", out.StopReason)
	}

	var text string
	var tool *bedrock.ToolUseBlock
	for _, b := range out.Output.Message.Content {
		if b.Text != nil {
			text = *b.Text
		}
		if b.ToolUse != nil {
			tool = b.ToolUse
		}
	}
	if text != "hello" {
		t.Errorf("text = %q", text)
	}
	if tool == nil || tool.ToolUseID != "call_1" || string(tool.Input) != `{"a":1}` {
		t.Errorf("tool = %#v", tool)
	}
}
