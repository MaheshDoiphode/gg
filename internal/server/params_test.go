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
