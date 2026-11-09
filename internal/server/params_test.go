package server

import (
	"net/http"
	"testing"

	"bedrock-simple/internal/bedrock"
)

func sampleBody() *bedrock.ConverseRequest {
	return &bedrock.ConverseRequest{
		InferenceConfig: &bedrock.InferenceConfig{
			MaxTokens:     bedrock.Ptr(100),
			Temperature:   bedrock.Ptr(0.7),
			TopP:          bedrock.Ptr(0.9),
			StopSequences: []string{"END"},
		},
		AdditionalModelRequestFields: map[string]any{"top_k": 40},
	}
}

func validationErr(msg string) error {
	return &bedrock.APIError{Status: http.StatusBadRequest, ErrorType: "ValidationException", Message: msg}
}

func TestDropRejectedParam(t *testing.T) {
	cases := []struct {
		name, message, want string
	}{
		{"grok temperature", "Unsupported parameter: 'temperature'", "temperature"},
		{"top_p", "The model does not support top_p", "top_p"},
		{"top_k in additional fields", "Extraneous key [top_k] is not permitted", "top_k"},
		{"stop sequences", "unsupported field stopSequences", "stop_sequences"},
		{"unrelated validation error", "messages: at least one message is required", ""},
		{"throttling is not a param problem", "", ""},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			body := sampleBody()
			var err error = validationErr(c.message)
			if c.message == "" {
				err = &bedrock.APIError{Status: http.StatusTooManyRequests, Message: "slow down"}
			}
			if got := dropRejectedParam(body, err); got != c.want {
				t.Fatalf("dropRejectedParam = %q, want %q", got, c.want)
			}
		})
	}
}

func TestDropRejectedParamActuallyRemovesTheField(t *testing.T) {
	body := sampleBody()

	if got := dropRejectedParam(body, validationErr("Unsupported parameter: 'temperature'")); got != "temperature" {
		t.Fatalf("first drop = %q", got)
	}
	if body.InferenceConfig.Temperature != nil {
		t.Fatal("temperature was not removed")
	}
	// Reporting the same field twice must not loop forever.
	if got := dropRejectedParam(body, validationErr("Unsupported parameter: 'temperature'")); got != "" {
		t.Fatalf("second drop = %q, want \"\"", got)
	}

	if got := dropRejectedParam(body, validationErr("unsupported: top_k")); got != "top_k" {
		t.Fatalf("top_k drop = %q", got)
	}
	if body.AdditionalModelRequestFields != nil {
		t.Fatal("emptied additionalModelRequestFields should be nil, not an empty map")
	}
}

// Once bytes are on the wire the failure is wrapped as fatal, and retrying
// would concatenate two answers.
func TestDropRejectedParamRefusesAfterOutputStarted(t *testing.T) {
	body := sampleBody()
	err := &fatalError{validationErr("Unsupported parameter: 'temperature'")}

	if got := dropRejectedParam(body, err); got != "" {
		t.Fatalf("dropRejectedParam = %q, want \"\"", got)
	}
	if body.InferenceConfig.Temperature == nil {
		t.Fatal("body must not be modified once streaming has begun")
	}
}

func TestDropRejectedParamClampsUnsupportedEffortLevels(t *testing.T) {
	// GLM-5 / Kimi K2.5 reject xhigh and cap at high.
	b1 := sampleBody()
	b1.ReasoningEffort = bedrock.EffortXHigh
	b1.AdditionalModelRequestFields["reasoning_effort"] = bedrock.EffortXHigh
	got := dropRejectedParam(b1, validationErr("1 validation error for ChatCompletionRequest\nreasoning_effort\n  Input should be 'none', 'low', 'medium' or 'high'"))
	if got != "reasoning_effort(xhigh->high)" || b1.ReasoningEffort != bedrock.EffortHigh || b1.AdditionalModelRequestFields["reasoning_effort"] != bedrock.EffortHigh {
		t.Fatalf("xhigh clamp: got=%q effort=%q fields=%#v", got, b1.ReasoningEffort, b1.AdditionalModelRequestFields)
	}

	// Grok 4.6 rejects none and starts at low.
	b2 := sampleBody()
	b2.ReasoningEffort = bedrock.EffortNone
	got = dropRejectedParam(b2, validationErr("Unsupported value: 'none' is not supported with the 'xai.grok-4.6' model. Supported values are: 'low', 'medium', 'high', and 'xhigh'."))
	if got != "reasoning_effort(none->low)" || b2.ReasoningEffort != bedrock.EffortLow {
		t.Fatalf("none clamp: got=%q effort=%q", got, b2.ReasoningEffort)
	}
}
