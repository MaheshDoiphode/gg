# bedrock-simple

An AWS Bedrock proxy that speaks the **OpenAI** and **Anthropic** APIs.

One Go file tree, **zero third-party dependencies**, one `.exe`. No `pip install`,
no `npm install`, no Docker, no DynamoDB, no `aws configure`.

```
POST /v1/chat/completions    OpenAI compatible (streaming + tools + vision)
GET  /v1/models              OpenAI compatible
POST /v1/messages            Anthropic compatible (streaming + tools + vision)
GET  /health
```

## Why this exists

The reference `sample-bedrock-api-proxy` needs Python, ~40 packages, DynamoDB,
Cognito and an AWS CLI profile. On a locked-down machine none of that is
possible. This rewrite keeps the useful part and drops everything else:

| | sample-bedrock-api-proxy | bedrock-simple |
|---|---|---|
| Runtime | Python 3.12 + FastAPI + boto3 | one static binary |
| Dependencies | ~40 PyPI packages | **0** |
| State | 12 DynamoDB tables | one JSON file |
| Admin | Cognito + React portal | edit the JSON file |
| AWS auth | `aws configure` / IAM role | API key or access key you paste in |
| Model list | hardcoded, goes stale | fetched live from Bedrock |

## Build

Go 1.21+ is the only requirement, and only to build. Because `go.mod` has no
`require` block, this works with no network access at all.

```bash
go build -o bedrock-simple.exe .        # Windows
go build -o bedrock-simple .            # macOS / Linux
```

Copy the resulting binary anywhere. It has no runtime dependencies.

## Run

Pick **one** of the two credential styles, and supply it either through a
`.env` file (easiest) or through real environment variables.

### Using a .env file

Copy [.env.example](.env.example) to `.env`, put it next to the binary or in the
directory you run from, and fill in one section:

```ini
BEDROCK_API_KEY=ABSKQmVkcm9ja0FQSUtleS...
AWS_REGION=us-east-1
```

Then just run it:

```bash
./bedrock-simple
```

The file is read before anything else, so every variable in the table below
works there, including `CONFIG_PATH`. A variable already set in your real
environment always wins over the file.

Values are taken literally: no unescaping, no URL-decoding, and the split is on
the **first** `=` only, so a base64 secret ending in `==` survives intact.
Quotes are optional and stripped if present.

### Option A - Bedrock API key (simplest)

Create one in the AWS console under *Bedrock > API keys*. It is a bearer token;
no signing, no CLI, no profile.

```ini
BEDROCK_API_KEY=ABSKQmVkcm9ja0FQSUtleS...
AWS_REGION=us-east-1
```

### Option B - access key + secret key

SigV4 signing is implemented in-process, so you still do not need the AWS CLI.

```ini
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_REGION=us-east-1
```

`AWS_SESSION_TOKEN` is only needed for temporary credentials.

### Or skip .env entirely

Shell exports work exactly the same:

```bash
export BEDROCK_API_KEY=ABSK...     # macOS/Linux
set BEDROCK_API_KEY=ABSK...        # Windows cmd
$env:BEDROCK_API_KEY="ABSK..."     # PowerShell
```

And so does putting the credential straight into `data/config.json` under
`credentials` (see below) - that is the most permanent option, and it is the
only one that supports more than one credential.

Either way the first run creates `data/config.json`, mints a client API key and
prints it:

```
bedrock-simple 1.0.0  ->  http://127.0.0.1:8080

  OpenAI compatible      POST  /v1/chat/completions
                         GET   /v1/models
  Anthropic compatible   POST  /v1/messages
  Health                 GET   /health

  credential   env-bearer       bearer us-east-1    enabled
  env file     .env
  client key   sk-6ee1f617016...

  Only failures are logged below.
```

After that the terminal stays quiet. **Only failed requests are logged**, so
anything you see is something that needs attention.

## Use it

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-6ee1f617016..." \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}]}'
```

Any OpenAI client works by pointing `base_url` at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-6ee1f617016...")
```

```bash
# Claude Code / any Anthropic SDK
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=sk-6ee1f617016...
```

Supported in both directions: streaming, tool / function calling, multi-turn
tool results, inline images, extended thinking (`reasoning_effort` on the
OpenAI side, `thinking` on the Anthropic side), and token usage reporting.

## Models

Nothing is hardcoded. On startup and every 6 hours the proxy discovers models
from **both** Bedrock catalogues:

- `bedrock-runtime` via `ListFoundationModels` + `ListInferenceProfiles`
- `bedrock-mantle` via its OpenAI-style `GET /v1/models`

It filters out deprecated, non-text and unavailable models and builds the alias
table itself. `GET /v1/models` returns what your account can actually invoke.

Names resolve in this order:

1. an override in `modelMap` (see below)
2. an exact discovered id or alias — `global.anthropic.claude-sonnet-4-5-20250929-v1:0`, `claude-sonnet-4-5-20250929`, `claude-sonnet-4-5`
3. a punctuation-insensitive match — `claude-sonnet-4.5` finds `claude-sonnet-4-5`
4. otherwise the name is sent upstream unchanged, so raw ids and ARNs always work

When a short alias is ambiguous, a `global.` inference profile wins, then one in
your own region, then anything else — so you get cross-region routing by default.

## Thinking / reasoning effort

Append `-thinking-<level>` to any model name, where the level is `none`, `low`,
`medium` or `high`. A bare `-thinking` means medium.

```
xai.grok-4.3                  default effort
xai.grok-4.3-thinking-none    answer immediately
xai.grok-4.3-thinking-high    think hard first
```

This exists because editor integrations such as Copilot's custom endpoints have
no reasoning control — the only thing you choose is a model name. The suffix is
only treated as an effort selector when the full name is *not* itself a real
model, so genuine names like `moonshotai.kimi-k2-thinking` still work.

The level is translated per upstream:

| Upstream | Field |
|---|---|
| Mantle Responses | `reasoning: {"effort": "high"}` |
| Mantle Chat Completions | `reasoning_effort: "high"` |
| Converse (Anthropic) | `thinking: {"type": "enabled", "budget_tokens": N}` |

Native fields work too: Anthropic `thinking.budget_tokens` and OpenAI
`reasoning_effort` are both honoured. An explicit suffix beats the body.

Reasoning output comes back as an Anthropic `thinking` block or an OpenAI
`reasoning_content` field, but only where the provider exposes it. GLM-5 and
Kimi return their full reasoning; **Grok returns none** — it reports reasoning
token counts only, and asking for `reasoning.summary` still yields an empty
summary. That is an xAI decision, not a limitation of this proxy.

> **Reasoning models need output headroom.** They spend `max_tokens` on thinking
> before writing anything, so a small budget returns an empty answer with
> `stop_reason: max_tokens` rather than an error. The proxy only sends a limit
> when the client sets one; on Mantle, omitting it lets the model use its full
> output length. `maxTokensDefault` applies to Converse only.

## Configuration

Everything lives in `data/config.json`. Edit it and restart.

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "requireApiKey": true,
  "logLevel": "info",
  "defaultRegion": "us-east-1",
  "maxTokensDefault": 4096,
  "preferMantle": false,

  "credentials": [
    {
      "id": "a1b2c3d4",
      "name": "work",
      "enabled": true,
      "authMode": "bearer",
      "region": "us-east-1",
      "bearerKey": "ABSKQmVkcm9ja0..."
    },
    {
      "id": "e5f6a7b8",
      "name": "backup",
      "enabled": true,
      "authMode": "sigv4",
      "region": "eu-west-1",
      "accessKey": "AKIA...",
      "secretKey": "..."
    }
  ],

  "apiKeys": [
    { "id": "0f1e2d3c", "name": "default", "key": "sk-...", "enabled": true, "tokenLimit": 0 }
  ],

  "modelMap": {
    "gpt-4o": "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
  }
}
```

- **Multiple credentials** are used round-robin. A throttled or failing one is
  put in cooldown and the request is retried on the next one, up to 3 times.
  Retries stop as soon as any bytes have been streamed to the client.
- **`modelMap`** is how you make an OpenAI-only tool work — map `gpt-4o` to
  whatever you actually want it to hit.
- **`tokenLimit`** of `0` means unlimited. Usage counters are written back to
  the same file.
- **`requireApiKey: false`** disables client auth entirely. Only do that on
  `127.0.0.1`.

### Environment variables

All of these can go in `.env` instead of the shell.

| Variable | Effect |
|---|---|
| `BEDROCK_API_KEY` / `AWS_BEARER_TOKEN_BEDROCK` | seed a bearer credential on first run |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | seed a SigV4 credential on first run |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | region for the seeded credential |
| `MANTLE_REGION` | region for the Mantle endpoint, when it differs from Converse |
| `PROXY_API_KEY` | the client key, created if missing and printed at startup |
| `REQUIRE_API_KEY` | `false` disables client auth (localhost only) |
| `PREFER_MANTLE` | `true` resolves shared model names to the Mantle endpoint |
| `HOST`, `PORT` | bind address, overrides the config file |
| `CONFIG_PATH` | config file location, default `data/config.json` |
| `LOG_LEVEL` | `debug`, `info`, `warn`, `error` |

The credential variables only take effect on **first run**, when no credential
exists yet. After that `data/config.json` is the source of truth - edit it there
to rotate a key.

### About the Bedrock API key format

It is a **single opaque token**, not two values joined together. There is no
access-key-plus-secret-key concatenation. It is base64, so the value can contain
`+` and `/` and end with `=` padding - that is encoding, not structure.

| Type | Created by | Lifetime | Prefix |
|---|---|---|---|
| Long-term | `iam create-service-specific-credential --service-name bedrock.amazonaws.com`, or the console | until the expiry you choose | `ABSK` |
| Short-term | the `aws-bedrock-token-generator` library | up to 12 hours | `bedrock-api-key-` |

Both are sent as `Authorization: Bearer <token>`. The short-term kind is derived
from access keys - it wraps a presigned SigV4 request - which is why it expires.
This proxy does not validate or parse the prefix; whatever you provide is passed
through unchanged, so both work.

## How it works

```
OpenAI  /v1/chat/completions ─┐                 ┌─ Converse            (bedrock-runtime)
                              ├─► internal hub ─┤─ Chat Completions    (bedrock-mantle /v1)
Anthropic  /v1/messages ──────┘                 └─ Responses           (bedrock-mantle /openai/v1)
```

Both client dialects are translated to one internal hub format, and the hub is
rendered onto whichever upstream serves the model. That is why adding an
upstream needed one adapter pair rather than changes in every handler.

AWS exposes three inference APIs and they do not carry the same models:

| Upstream | Carries | Auth |
|---|---|---|
| `bedrock-runtime` Converse | Claude, Nova, Llama, Mistral, DeepSeek… | SigV4 or API key |
| `bedrock-mantle` `/v1/chat/completions` | GLM, Kimi, Qwen, MiniMax, gpt-oss… | API key |
| `bedrock-mantle` `/openai/v1/responses` | xAI Grok, GPT-5.x | API key |

**Which Mantle route a model accepts is not in its catalogue entry**, so the
proxy learns it: it tries Chat Completions and, on `isn't supported on this
route`, switches that model to the Responses API and caches the result. The
switch only happens before any bytes are flushed, so it cannot splice two
answers together.

Be aware that a Bedrock API key may be scoped to only one of these. A key whose
name decodes to `MantleApiKey-…` can list models but gets `Operation not
allowed` from Converse — set `preferMantle` so shared model names resolve to the
endpoint you can actually use.

Everything the AWS SDK would normally provide is implemented here in the
standard library:

| Concern | Where | Notes |
|---|---|---|
| SigV4 signing | [internal/awsauth/sigv4.go](internal/awsauth/sigv4.go) | verified against AWS's published `get-vanilla` test vector |
| Binary event stream | [internal/awsauth/eventstream.go](internal/awsauth/eventstream.go) | prelude + CRC32 validation, streamed not buffered |
| Converse client | [internal/bedrock/client.go](internal/bedrock/client.go) | runtime + control plane |
| Mantle clients | [internal/bedrock/mantle.go](internal/bedrock/mantle.go), [responses.go](internal/bedrock/responses.go) | OpenAI-shaped, SSE |
| Model discovery | [internal/bedrock/registry.go](internal/bedrock/registry.go) | alias derivation and ranking |
| Reasoning effort | [internal/bedrock/effort.go](internal/bedrock/effort.go) | suffix parsing, budget mapping |
| Translation | [internal/convert](internal/convert) | hub ↔ OpenAI, Anthropic, Responses |
| Storage | [internal/store/store.go](internal/store/store.go) | RWMutex + atomic temp-file rename |
| .env loading | [internal/dotenv/dotenv.go](internal/dotenv/dotenv.go) | splits on the first `=` so base64 padding survives |

One subtlety worth knowing about if you touch the signing code: Bedrock model
ids contain `:`, and SigV4 requires the path to be percent-encoded **twice** for
the canonical request but only once on the wire. That is covered by a test.

## Tests

```bash
go test ./...
```

The server tests run the full path — HTTP in, translation, a fake Bedrock that
emits real binary event-stream frames, translation back, SSE out.

## Limits

- Text and images in, text out. No embeddings, image generation or Bedrock
  Guardrails.
- Prompt caching is not requested (no `cachePoint` blocks are inserted).
- Single process, single JSON file. It is a personal/team proxy, not a
  horizontally scaled service.
- Remote image URLs are not fetched; inline `data:` URLs only.

## Not using SQLite — and why

The original ask was DynamoDB → SQLite. Every Go SQLite driver either needs cgo
and a C toolchain (`mattn/go-sqlite3`) or vendors a very large transpiled
library (`modernc.org/sqlite`) that has to be downloaded — both break the "zero
install, builds offline" requirement that motivated this rewrite.

The data here is a few dozen rows of config and counters with no queries beyond
key lookup, so a single JSON file with a RWMutex and atomic writes covers it,
exactly like Kiro-Go does. Same outcome: no external database, no server, one
file you can read and edit.
