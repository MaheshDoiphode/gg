#!/usr/bin/env bash
# Probes every Mantle model in a region with a minimal request and reports which
# route answers: chat completions, the responses API, or neither.
set -u
REGION="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY="$(grep '^BEDROCK_API_KEY=' "$ROOT/.env" | cut -d= -f2-)"
HOST="https://bedrock-mantle.${REGION}.api.aws"

ids=$(curl -s -m 30 "$HOST/v1/models" -H "Authorization: Bearer $KEY" \
  | sed 's/},{/}\n{/g' \
  | grep '"status":"available"' \
  | grep -o '"id":"[^"]*"' | sed 's/"id":"//;s/"//')

probe() {
  local model="$1"
  local chat resp

  chat=$(curl -s -m 60 "$HOST/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}")

  if echo "$chat" | grep -q '"choices"'; then
    echo "$REGION|$model|chat|OK"
    return
  fi

  # Some models live only behind the responses API, which must be streamed.
  resp=$(curl -s -m 90 -N "$HOST/openai/v1/responses" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"hi\"}]}],\"stream\":true,\"max_output_tokens\":2000}")

  if echo "$resp" | grep -q '"type":"response.completed"'; then
    echo "$REGION|$model|responses|OK"
    return
  fi

  local why
  why=$(echo "$chat" | grep -o '"message":"[^"]*"' | head -1 | cut -c13-72)
  [ -z "$why" ] && why=$(echo "$resp" | grep -o '"message":"[^"]*"' | head -1 | cut -c13-72)
  [ -z "$why" ] && why="no response"
  echo "$REGION|$model|-|FAIL: $why"
}

export -f probe
export REGION KEY HOST

echo "$ids" | xargs -P 8 -I{} bash -c 'probe "$@"' _ {}
