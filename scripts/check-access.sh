#!/usr/bin/env bash
# Reports agreement and authorization status for the models an access request
# was submitted for, then confirms with a real call whether they can be invoked.
set -u
REGION="${AWS_REGION:-us-east-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODELS=(
  anthropic.claude-sonnet-5
  anthropic.claude-opus-5
  anthropic.claude-opus-4-8
  anthropic.claude-opus-4-7
  anthropic.claude-haiku-4-5
  openai.gpt-5.6-sol
  openai.gpt-5.5
  openai.gpt-5.4
)

printf '%-28s %-12s %-16s %s\n' MODEL AGREEMENT AUTHORIZATION INVOKE
printf '%-28s %-12s %-16s %s\n' '----' '---------' '-------------' '------'

KEY="$(grep '^BEDROCK_API_KEY=' "$ROOT/.env" 2>/dev/null | cut -d= -f2-)"

for m in "${MODELS[@]}"; do
  read -r agreement auth <<<"$(aws bedrock get-foundation-model-availability \
    --model-id "$m" --region "$REGION" \
    --query '[agreementAvailability.status,authorizationStatus]' \
    --output text 2>/dev/null)"
  [ -z "${agreement:-}" ] && agreement="n/a" && auth="not-in-registry"

  invoke="skipped"
  if [ -n "$KEY" ]; then
    # Anthropic models answer on the messages route, OpenAI ones on responses.
    case "$m" in
      anthropic.*)
        body="{\"model\":\"$m\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
        out=$(curl -s -m 45 "https://bedrock-mantle.$REGION.api.aws/anthropic/v1/messages" \
          -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
          -H "anthropic-version: 2023-06-01" -d "$body")
        echo "$out" | grep -q '"content"' && invoke="WORKS" || invoke="denied"
        ;;
      *)
        body="{\"model\":\"$m\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"hi\"}]}],\"stream\":true,\"max_output_tokens\":512}"
        out=$(curl -s -m 60 -N "https://bedrock-mantle.$REGION.api.aws/openai/v1/responses" \
          -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "$body")
        echo "$out" | grep -q '"type":"response.completed"' && invoke="WORKS" || invoke="denied"
        ;;
    esac
  fi

  printf '%-28s %-12s %-16s %s\n' "$m" "$agreement" "$auth" "$invoke"
done
