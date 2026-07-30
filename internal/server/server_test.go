package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"bedrock-simple/internal/bedrock"
	"bedrock-simple/internal/logx"
	"bedrock-simple/internal/store"
)

// eventFrame assembles one AWS event-stream message, as ConverseStream sends.
func eventFrame(eventType string, payload string) []byte {
	var hb bytes.Buffer
	writeHeader := func(name, value string) {
		hb.WriteByte(byte(len(name)))
		hb.WriteString(name)
		hb.WriteByte(7)
		_ = binary.Write(&hb, binary.BigEndian, uint16(len(value)))
		hb.WriteString(value)
	}
	writeHeader(":event-type", eventType)
	writeHeader(":message-type", "event")

	headersLen := uint32(hb.Len())
	totalLen := 16 + headersLen + uint32(len(payload))

	var prelude bytes.Buffer
	_ = binary.Write(&prelude, binary.BigEndian, totalLen)
	_ = binary.Write(&prelude, binary.BigEndian, headersLen)
	_ = binary.Write(&prelude, binary.BigEndian, crc32.ChecksumIEEE(prelude.Bytes()))

	var msg bytes.Buffer
	msg.Write(prelude.Bytes())
	msg.Write(hb.Bytes())
	msg.WriteString(payload)
	_ = binary.Write(&msg, binary.BigEndian, crc32.ChecksumIEEE(msg.Bytes()))
	return msg.Bytes()
}

// fakeBedrock stands in for bedrock-runtime and records the last request body.
type fakeBedrock struct {
	server   *httptest.Server
	lastPath string
	lastBody map[string]any
	status   int
	errBody  string
	// rejectParam makes the fake 400 while that inferenceConfig field is
	// present, the way xai.grok-4.3 rejects temperature.
	rejectParam string
	calls       int
}

func newFakeBedrock(t *testing.T) *fakeBedrock {
	t.Helper()
	f := &fakeBedrock{}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.lastPath = r.URL.Path
		f.calls++
		raw, _ := readAll(r)
		f.lastBody = nil
		_ = json.Unmarshal(raw, &f.lastBody)

		if f.status != 0 {
			w.Header().Set("x-amzn-ErrorType", "ThrottlingException:http://internal")
			w.WriteHeader(f.status)
			_, _ = w.Write([]byte(f.errBody))
			return
		}

		if f.rejectParam != "" && hasInferenceParam(f.lastBody, f.rejectParam) {
			w.Header().Set("x-amzn-ErrorType", "ValidationException:http://internal")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"message":"Unsupported parameter: '` + f.rejectParam + `'"}`))
			return
		}

		if strings.HasSuffix(r.URL.Path, "/converse-stream") {
			w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
			w.WriteHeader(http.StatusOK)
			for _, frame := range [][]byte{
				eventFrame("messageStart", `{"role":"assistant"}`),
				eventFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"Hello"}}`),
				eventFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":" world"}}`),
				eventFrame("contentBlockStop", `{"contentBlockIndex":0}`),
				eventFrame("messageStop", `{"stopReason":"end_turn"}`),
				eventFrame("metadata", `{"usage":{"inputTokens":9,"outputTokens":2,"totalTokens":11}}`),
			} {
				_, _ = w.Write(frame)
				w.(http.Flusher).Flush()
			}
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"output":{"message":{"role":"assistant","content":[{"text":"Hello world"}]}},
			"stopReason":"end_turn",
			"usage":{"inputTokens":9,"outputTokens":2,"totalTokens":11}}`))
	}))
	t.Cleanup(f.server.Close)
	return f
}

func readAll(r *http.Request) ([]byte, error) {
	var buf bytes.Buffer
	_, err := buf.ReadFrom(r.Body)
	return buf.Bytes(), err
}

func hasInferenceParam(body map[string]any, name string) bool {
	ic, ok := body["inferenceConfig"].(map[string]any)
	if !ok {
		return false
	}
	_, present := ic[name]
	return present
}

const testKey = "sk-test-key"

func setup(t *testing.T) (*Handler, *fakeBedrock) {
	t.Helper()
	logx.SetLevel("error")

	if err := store.Init(filepath.Join(t.TempDir(), "config.json")); err != nil {
		t.Fatal(err)
	}
	fake := newFakeBedrock(t)
	if _, err := store.AddCredential(store.Credential{
		Name: "test", Enabled: true, AuthMode: store.AuthBearer,
		Region: "us-east-1", BearerKey: "bk", EndpointURL: fake.server.URL,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.AddAPIKeyWithValue("test", testKey); err != nil {
		t.Fatal(err)
	}
	return New(bedrock.New(), bedrock.NewRegistry()), fake
}

func post(t *testing.T, h *Handler, path, body string, auth bool) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if auth {
		req.Header.Set("Authorization", "Bearer "+testKey)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestChatCompletions(t *testing.T) {
	h, fake := setup(t)

	rec := post(t, h, "/v1/chat/completions",
		`{"model":"claude-x","messages":[{"role":"user","content":"hi"}]}`, true)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out["object"] != "chat.completion" {
		t.Errorf("object = %v", out["object"])
	}
	choice := out["choices"].([]any)[0].(map[string]any)
	if got := choice["message"].(map[string]any)["content"]; got != "Hello world" {
		t.Errorf("content = %v", got)
	}
	if got := choice["finish_reason"]; got != "stop" {
		t.Errorf("finish_reason = %v", got)
	}
	usage := out["usage"].(map[string]any)
	if usage["prompt_tokens"].(float64) != 9 || usage["completion_tokens"].(float64) != 2 {
		t.Errorf("usage = %v", usage)
	}

	if !strings.HasSuffix(fake.lastPath, "/converse") {
		t.Errorf("upstream path = %s", fake.lastPath)
	}
	// Unknown model names must reach Bedrock untouched.
	if !strings.Contains(fake.lastPath, "claude-x") {
		t.Errorf("model id not passed through: %s", fake.lastPath)
	}
}

func TestChatCompletionsStreaming(t *testing.T) {
	h, _ := setup(t)

	rec := post(t, h, "/v1/chat/completions",
		`{"model":"m","stream":true,"stream_options":{"include_usage":true},
		  "messages":[{"role":"user","content":"hi"}]}`, true)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}

	body := rec.Body.String()
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Error("stream must terminate with [DONE]")
	}

	var text strings.Builder
	var sawUsage bool
	for _, line := range strings.Split(body, "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chunk %q: %v", payload, err)
		}
		if chunk["object"] != "chat.completion.chunk" {
			t.Errorf("object = %v", chunk["object"])
		}
		if u, ok := chunk["usage"].(map[string]any); ok {
			sawUsage = true
			if u["completion_tokens"].(float64) != 2 {
				t.Errorf("usage = %v", u)
			}
		}
		for _, c := range chunk["choices"].([]any) {
			choice := c.(map[string]any)
			delta, _ := choice["delta"].(map[string]any)
			if s, ok := delta["content"].(string); ok {
				text.WriteString(s)
			}
		}
	}
	if text.String() != "Hello world" {
		t.Errorf("streamed text = %q", text.String())
	}
	if !sawUsage {
		t.Error("include_usage requested but no usage chunk emitted")
	}
}

func TestAnthropicMessages(t *testing.T) {
	h, _ := setup(t)

	rec := post(t, h, "/v1/messages",
		`{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`, true)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)

	if out["type"] != "message" || out["role"] != "assistant" {
		t.Errorf("envelope = %v", out)
	}
	block := out["content"].([]any)[0].(map[string]any)
	if block["text"] != "Hello world" {
		t.Errorf("content = %v", block)
	}
	if out["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v", out["stop_reason"])
	}
}

func TestAnthropicMessagesStreaming(t *testing.T) {
	h, _ := setup(t)

	rec := post(t, h, "/v1/messages",
		`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`, true)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()

	var order []string
	for _, line := range strings.Split(body, "\n") {
		if name, ok := strings.CutPrefix(line, "event: "); ok {
			order = append(order, name)
		}
	}
	want := "message_start,content_block_start,content_block_delta,content_block_delta,content_block_stop,message_delta,message_stop"
	if got := strings.Join(order, ","); got != want {
		t.Fatalf("event order\n got: %s\nwant: %s", got, want)
	}
}

func TestAuthRequired(t *testing.T) {
	h, _ := setup(t)

	rec := post(t, h, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`, false)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d", rec.Code)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("x-api-key", "sk-wrong")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong key accepted: %d", rec.Code)
	}
}

// Upstream failures must surface with their own status, not a blanket 500.
func TestUpstreamErrorPropagates(t *testing.T) {
	h, fake := setup(t)
	fake.status = http.StatusTooManyRequests
	fake.errBody = `{"message":"Too many tokens"}`

	rec := post(t, h, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`, true)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	errObj := out["error"].(map[string]any)
	if errObj["type"] != "rate_limit_error" || errObj["message"] != "Too many tokens" {
		t.Errorf("error = %v", errObj)
	}
}

// xai.grok-4.3 rejects temperature outright. The proxy should drop it and
// retry rather than failing the request.
func TestRejectedParameterIsDroppedAndRetried(t *testing.T) {
	h, fake := setup(t)
	fake.rejectParam = "temperature"

	rec := post(t, h, "/v1/chat/completions",
		`{"model":"xai.grok-4.3","temperature":0.7,
		  "messages":[{"role":"user","content":"hi"}]}`, true)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if fake.calls != 2 {
		t.Errorf("expected one rejection then one success, got %d calls", fake.calls)
	}
	if hasInferenceParam(fake.lastBody, "temperature") {
		t.Error("the retry still carried temperature")
	}
	if !hasInferenceParam(fake.lastBody, "maxTokens") {
		t.Error("the retry dropped more than it should have")
	}
}

func TestRejectedParameterOnAnthropicEndpoint(t *testing.T) {
	h, fake := setup(t)
	fake.rejectParam = "topP"

	rec := post(t, h, "/v1/messages",
		`{"model":"xai.grok-4.3","max_tokens":100,"top_p":0.9,
		  "messages":[{"role":"user","content":"hi"}]}`, true)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if hasInferenceParam(fake.lastBody, "topP") {
		t.Error("the retry still carried topP")
	}
}

// A validation error that is not about a parameter must surface immediately.
func TestNonParameterValidationErrorIsNotRetried(t *testing.T) {
	h, fake := setup(t)
	fake.status = http.StatusBadRequest
	fake.errBody = `{"message":"messages: at least one message is required"}`

	rec := post(t, h, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`, true)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d", rec.Code)
	}
	if fake.calls != 1 {
		t.Errorf("expected exactly 1 call, got %d", fake.calls)
	}
}

func TestNoCredentialsIsReportedClearly(t *testing.T) {
	h, _ := setup(t)
	for _, c := range store.Credentials() {
		_ = store.DeleteCredential(c.ID)
	}

	rec := post(t, h, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`, true)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "no Bedrock credentials") {
		t.Errorf("body = %s", rec.Body.String())
	}
}

func TestModelsAndHealthNeedNoAuth(t *testing.T) {
	h, _ := setup(t)
	_ = store.SetModelMapping("my-alias", "anthropic.something")

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("models status = %d", rec.Code)
	}
	var list struct {
		Object string `json:"object"`
		Data   []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &list)
	if list.Object != "list" {
		t.Errorf("object = %q", list.Object)
	}
	found := false
	for _, m := range list.Data {
		if m.ID == "my-alias" {
			found = true
		}
	}
	if !found {
		t.Errorf("custom mapping missing from /v1/models: %+v", list.Data)
	}

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("health status = %d", rec.Code)
	}
}

func TestUsageIsRecordedAgainstTheKey(t *testing.T) {
	h, _ := setup(t)

	post(t, h, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`, true)

	keys := store.APIKeys()
	if len(keys) != 1 {
		t.Fatalf("keys = %+v", keys)
	}
	if keys[0].Requests != 1 || keys[0].InputTokens != 9 || keys[0].OutTokens != 2 {
		t.Errorf("usage not recorded: %+v", keys[0])
	}
}
