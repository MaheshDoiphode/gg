package server

import (
	"errors"
	"net/http"
	"strings"
	"sync"

	"bedrock-simple/internal/bedrock"
)

// maxParamStripRetries bounds how many rejected fields we will drop before
// giving up and returning the error to the client.
const maxParamStripRetries = 3

// rejectionHints are the phrases Bedrock uses when a model does not accept a
// field that Converse itself allows.
var rejectionHints = []string{
	"unsupported", "not supported", "does not support", "doesn't support",
	"unknown", "unrecognized", "extraneous", "unexpected",
	"invalid value", "input should be", "validation error",
	"maximum context length", "exceeds the model limit", "exceeds model maximum",
	"rejected by the target",
}

// rejectedParams remembers fields each model has refused, so later requests
// skip them up front instead of paying a 400 round trip every time.
type rejectedParams struct {
	mu      sync.RWMutex
	byModel map[string]map[string]bool
}

// learnable fields are request-independent; clamps and effort levels are not.
func learnable(name string) bool {
	return !strings.Contains(name, "(") && name != "reasoning_effort"
}

func (p *rejectedParams) learn(model, name string) {
	if !learnable(name) {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.byModel == nil {
		p.byModel = map[string]map[string]bool{}
	}
	if p.byModel[model] == nil {
		p.byModel[model] = map[string]bool{}
	}
	p.byModel[model][name] = true
}

func (p *rejectedParams) apply(model string, body *bedrock.ConverseRequest) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	for name := range p.byModel[model] {
		if ic := body.InferenceConfig; ic != nil {
			switch name {
			case "temperature":
				ic.Temperature = nil
			case "top_p":
				ic.TopP = nil
			case "stop_sequences":
				ic.StopSequences = nil
			}
		}
		delete(body.AdditionalModelRequestFields, name)
	}
	if len(body.AdditionalModelRequestFields) == 0 {
		body.AdditionalModelRequestFields = nil
	}
}

// dropRejectedParam removes one field that the model complained about and
// returns its name, or "" if the error is not a recoverable parameter
// rejection. Some models reject sampling parameters that Converse accepts in
// general: xai.grok-4.3 400s on temperature and top_p, and several reasoning
// models reject them once thinking is enabled.
func dropRejectedParam(body *bedrock.ConverseRequest, err error) string {
	// A mid-stream validation failure arrives wrapped as fatal because bytes
	// are already on the wire; retrying would duplicate the answer.
	var fatal *fatalError
	if errors.As(err, &fatal) {
		return ""
	}
	var apiErr *bedrock.APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadRequest {
		return ""
	}
	msg := strings.ToLower(apiErr.Message)

	rejected := false
	for _, hint := range rejectionHints {
		if strings.Contains(msg, hint) {
			rejected = true
			break
		}
	}
	if !rejected {
		return ""
	}

	mentions := func(names ...string) bool {
		for _, n := range names {
			if strings.Contains(msg, n) {
				return true
			}
		}
		return false
	}

	if ic := body.InferenceConfig; ic != nil {
		switch {
		case ic.Temperature != nil && mentions("temperature"):
			ic.Temperature = nil
			return "temperature"
		case ic.TopP != nil && mentions("top_p", "topp"):
			ic.TopP = nil
			return "top_p"
		case len(ic.StopSequences) > 0 && mentions("stop_sequences", "stopsequences", "stop sequence"):
			ic.StopSequences = nil
			return "stop_sequences"
		case ic.MaxTokens != nil && *ic.MaxTokens > 4096 && mentions("maximum context length", "exceeds the model limit", "exceeds model maximum", "rejected by the target"):
			next := *ic.MaxTokens / 2
			if next > 65536 {
				next = 65536
			}
			if next < 4096 {
				next = 4096
			}
			ic.MaxTokens = bedrock.Ptr(next)
			return "max_tokens(clamped)"
		}
	}

	// Clamp or strip reasoning_effort when a model supports a narrower set of
	// levels (e.g. GLM-5/Kimi K2.5 cap at 'high', Grok 4.6 starts at 'low').
	if body.ReasoningEffort != "" && mentions("reasoning_effort", "reasoning.effort", "supported values are", "input should be") {
		switch {
		case body.ReasoningEffort == bedrock.EffortXHigh && strings.Contains(msg, "'high'") && !strings.Contains(msg, "'xhigh'"):
			body.ReasoningEffort = bedrock.EffortHigh
			if body.AdditionalModelRequestFields != nil {
				if _, ok := body.AdditionalModelRequestFields["reasoning_effort"]; ok {
					body.AdditionalModelRequestFields["reasoning_effort"] = bedrock.EffortHigh
				}
			}
			return "reasoning_effort(xhigh->high)"
		case body.ReasoningEffort == bedrock.EffortNone && strings.Contains(msg, "'low'") && (strings.Contains(msg, "'none' is not supported") || !strings.Contains(msg, "'none'")):
			body.ReasoningEffort = bedrock.EffortLow
			if body.AdditionalModelRequestFields != nil {
				if _, ok := body.AdditionalModelRequestFields["reasoning_effort"]; ok {
					body.AdditionalModelRequestFields["reasoning_effort"] = bedrock.EffortLow
				}
			}
			return "reasoning_effort(none->low)"
		default:
			body.ReasoningEffort = ""
			if body.AdditionalModelRequestFields != nil {
				delete(body.AdditionalModelRequestFields, "reasoning_effort")
				if len(body.AdditionalModelRequestFields) == 0 {
					body.AdditionalModelRequestFields = nil
				}
			}
			return "reasoning_effort"
		}
	}

	// additionalModelRequestFields carries provider-specific keys such as
	// top_k and thinking; match the offending one by name.
	for key := range body.AdditionalModelRequestFields {
		if mentions(strings.ToLower(key)) {
			delete(body.AdditionalModelRequestFields, key)
			if len(body.AdditionalModelRequestFields) == 0 {
				body.AdditionalModelRequestFields = nil
			}
			return key
		}
	}
	return ""
}
