package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Claude Code calls count_tokens before each turn; a 404 breaks its context
// budgeting, so the route must exist and answer without touching upstream.
func TestCountTokens(t *testing.T) {
	h, fake := setup(t)

	rec := post(t, h, "/v1/messages/count_tokens", `{
		"model":"xai.grok-4.3",
		"system":"You are a helpful assistant.",
		"messages":[{"role":"user","content":"Explain the theory of relativity in detail."}],
		"tools":[{"name":"read_file","description":"Read a file from disk",
		          "input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}]
	}`, true)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var out struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.InputTokens <= 0 {
		t.Fatalf("input_tokens = %d", out.InputTokens)
	}
	if fake.calls != 0 {
		t.Errorf("counting must not call upstream, got %d calls", fake.calls)
	}
}

func TestCountTokensScalesWithInput(t *testing.T) {
	h, _ := setup(t)

	count := func(text string) int {
		body := `{"model":"m","messages":[{"role":"user","content":"` + text + `"}]}`
		rec := post(t, h, "/v1/messages/count_tokens", body, true)
		var out struct {
			InputTokens int `json:"input_tokens"`
		}
		_ = json.Unmarshal(rec.Body.Bytes(), &out)
		return out.InputTokens
	}

	small := count("hello")
	large := count(strings.Repeat("the quick brown fox jumps over the lazy dog ", 200))
	if large <= small*10 {
		t.Errorf("count did not scale: small=%d large=%d", small, large)
	}
}

func TestCountTokensRejectsBadBody(t *testing.T) {
	h, _ := setup(t)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader("{not json"))
	req.Header.Set("Authorization", "Bearer "+testKey)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d", rec.Code)
	}
}
