package server

import (
	"errors"
	"net/http"
	"strings"

	"bedrock-simple/internal/bedrock"
)

// maxParamStripRetries bounds how many rejected fields we will drop before
// giving up and returning the error to the client.
const maxParamStripRetries = 3

// rejectionHints are the phrases Bedrock uses when a model does not accept a
// field that Converse itself allows.
var rejectionHints = []string{
	"unsupported", "not supported", "does not support",
	"unknown", "unrecognized", "extraneous", "unexpected",
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
