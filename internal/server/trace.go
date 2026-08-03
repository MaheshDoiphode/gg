package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"bedrock-simple/internal/bedrock"
	"bedrock-simple/internal/convert"
	"bedrock-simple/internal/logx"
)

// A stream that stops mid-answer looks identical to one still thinking, so the
// trace records where the bytes stopped rather than only that they did.

// stallWarnAfter is the silence that turns a slow turn into a suspicious one.
const stallWarnAfter = 30 * time.Second

var traceSeq atomic.Uint64

type trace struct {
	id     string
	path   string
	client string // model name the caller asked for
	model  string // model id sent upstream
	stream bool
	start  time.Time

	mu         sync.Mutex
	first      time.Time // first upstream event
	last       time.Time // most recent upstream event
	events     int
	textN      int
	textChars  int
	thinkN     int
	thinkChars int
	toolN      int
	toolChars  int
	tools      []string
	stop       string
	usage      *bedrock.TokenUsage
	longestGap time.Duration
	upstream   string
}

func newTrace(r *http.Request, client, model string, stream bool) *trace {
	return &trace{
		id:     fmt.Sprintf("r%d", traceSeq.Add(1)),
		path:   r.URL.Path,
		client: client,
		model:  model,
		stream: stream,
		start:  time.Now(),
	}
}

// request records the shape of the turn: how much context is going up, what
// budget is coming back, and which knobs are set.
func (t *trace) request(body *bedrock.ConverseRequest, effort string) {
	if !logx.DebugEnabled() {
		return
	}

	est := convert.EstimateRequestTokens(body)
	maxTok := "model default"
	if body.InferenceConfig != nil && body.InferenceConfig.MaxTokens != nil {
		maxTok = fmt.Sprint(*body.InferenceConfig.MaxTokens)
	}

	var toolNames []string
	if body.ToolConfig != nil {
		for _, tool := range body.ToolConfig.Tools {
			if tool.ToolSpec != nil {
				toolNames = append(toolNames, tool.ToolSpec.Name)
			}
		}
	}

	logx.Debugf("%s %s start client_model=%q model=%q stream=%v",
		t.id, t.path, t.client, t.model, t.stream)
	logx.Debugf("%s context: %d message(s), %d system block(s), ~%d input tokens, max_output=%s",
		t.id, len(body.Messages), len(body.System), est, maxTok)

	if effort != "" || body.ReasoningEffort != "" {
		logx.Debugf("%s reasoning effort=%q", t.id, firstNonEmpty(effort, body.ReasoningEffort))
	}
	if len(toolNames) > 0 {
		logx.Debugf("%s %d tool(s): %s", t.id, len(toolNames), strings.Join(toolNames, ", "))
	}
	if ic := body.InferenceConfig; ic != nil {
		if ic.Temperature != nil {
			logx.Debugf("%s temperature=%v", t.id, *ic.Temperature)
		}
		if ic.TopP != nil {
			logx.Debugf("%s top_p=%v", t.id, *ic.TopP)
		}
	}
}

// attempt records which upstream API is about to be called.
func (t *trace) attempt(api, region string, n int) {
	t.mu.Lock()
	t.upstream = api
	t.mu.Unlock()
	logx.Debugf("%s -> %s (%s) attempt %d", t.id, api, region, n)
}

// event folds one upstream event into the running totals and reports the
// first token, any long silence, and periodic progress.
func (t *trace) event(ev bedrock.StreamEvent) {
	if !logx.DebugEnabled() {
		return
	}
	now := time.Now()

	t.mu.Lock()
	if t.first.IsZero() {
		t.first = now
		defer logx.Debugf("%s first event after %s (%s)", t.id, since(t.start), ev.Type)
	} else if gap := now.Sub(t.last); gap > t.longestGap {
		t.longestGap = gap
	}
	t.last = now
	t.events++

	var progress string
	if d := ev.Delta; d != nil {
		switch {
		case d.ReasoningContent != nil:
			t.thinkN++
			t.thinkChars += len(d.ReasoningContent.Text)
		case d.ToolUse != nil:
			t.toolN++
			t.toolChars += len(d.ToolUse.Input)
		case d.Text != "":
			t.textN++
			t.textChars += len(d.Text)
		}
		// One line per 50 deltas keeps a long answer visible without flooding.
		if n := t.textN + t.thinkN + t.toolN; n%50 == 0 {
			progress = fmt.Sprintf("%s progress: %d text (%d ch), %d thinking (%d ch), %d tool (%d ch) at %s",
				t.id, t.textN, t.textChars, t.thinkN, t.thinkChars, t.toolN, t.toolChars, since(t.start))
		}
	}
	if ev.Start != nil && ev.Start.ToolUse != nil {
		t.tools = append(t.tools, ev.Start.ToolUse.Name)
		progress = fmt.Sprintf("%s tool call opened: %s", t.id, ev.Start.ToolUse.Name)
	}
	if ev.StopReason != "" {
		t.stop = ev.StopReason
	}
	if ev.Usage != nil {
		t.usage = ev.Usage
	}
	t.mu.Unlock()

	if progress != "" {
		logx.Debugf("%s", progress)
	}
}

// watch reports silence while a turn is in flight, so a stall is visible as it
// happens instead of only in the post-mortem. The returned stop is idempotent.
func (t *trace) watch(ctx context.Context) func() {
	if !logx.DebugEnabled() {
		return func() {}
	}
	done := make(chan struct{})
	finished := make(chan struct{})
	var once sync.Once

	go func() {
		defer close(finished)
		tick := time.NewTicker(stallWarnAfter)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-done:
				return
			case <-tick.C:
				t.mu.Lock()
				idleSince, events, waiting := t.last, t.events, t.first.IsZero()
				t.mu.Unlock()

				if waiting {
					logx.Debugf("%s still waiting for the first event after %s (model is thinking)",
						t.id, since(t.start))
					continue
				}
				if idle := time.Since(idleSince); idle >= stallWarnAfter {
					logx.Warnf("%s no upstream data for %s after %d event(s)",
						t.id, since(idleSince), events)
				}
			}
		}
	}()

	return func() {
		once.Do(func() { close(done) })
		<-finished
	}
}

// finish writes the post-mortem. err is nil on success.
func (t *trace) finish(err error) {
	if !logx.DebugEnabled() {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	// Claude Code fires prompt-suggestion and memory-extraction requests after
	// every turn and abandons them, so a cancellation is routine, not a fault.
	cancelled := errors.Is(err, context.Canceled)

	verdict := "ok"
	switch {
	case cancelled:
		verdict = "cancelled by the client"
	case err != nil:
		verdict = "FAILED: " + err.Error()
	}

	logx.Debugf("%s done in %s via %s - %s", t.id, since(t.start), orUnknown(t.upstream), verdict)
	logx.Debugf("%s output: %d event(s), %d text ch, %d thinking ch, %d tool ch, stop_reason=%s",
		t.id, t.events, t.textChars, t.thinkChars, t.toolChars, orUnknown(t.stop))

	if t.usage != nil {
		logx.Debugf("%s usage: %d in, %d out, %d total (cache read %d, write %d)",
			t.id, t.usage.InputTokens, t.usage.OutputTokens, t.usage.TotalTokens,
			t.usage.CacheReadInputTokens, t.usage.CacheWriteInputTokens)
	}
	if !t.first.IsZero() {
		logx.Debugf("%s timing: %s to first event, %s streaming, longest silence %s",
			t.id, since2(t.start, t.first), since2(t.first, t.last), t.longestGap.Round(time.Millisecond))
	}

	// The combinations that explain a truncated answer.
	switch {
	case cancelled:
		logx.Debugf("%s the client hung up after %s; background requests are routinely abandoned",
			t.id, since(t.start))
	case t.stop == "max_tokens" && t.textChars == 0 && t.thinkChars > 0:
		logx.Warnf("%s truncated: the whole max_tokens budget went to reasoning, no text was produced",
			t.id)
	case t.stop == "max_tokens":
		logx.Warnf("%s truncated: hit max_tokens after %d characters", t.id, t.textChars)
	case err == nil && t.textChars == 0 && t.toolChars == 0:
		logx.Warnf("%s empty answer with stop_reason=%s", t.id, orUnknown(t.stop))
	case err != nil && t.events > 0:
		logx.Warnf("%s stopped mid-answer after %d event(s) and %d characters",
			t.id, t.events, t.textChars+t.thinkChars)
	}
}

func since(t time.Time) string { return time.Since(t).Round(time.Millisecond).String() }

func since2(from, to time.Time) string { return to.Sub(from).Round(time.Millisecond).String() }

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
