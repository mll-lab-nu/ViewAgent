#!/bin/bash
ROOT="${VIEWSUITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate viewagent_thor
export VIEWSUITE_ROOT="$ROOT" fileroot="$ROOT"
# Key from an .env outside the repo; point ENV_FILE at yours.
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
export OPENROUTER_API_KEY=$(grep -oE '^OPENROUTER_API=.*' "$ENV_FILE" | cut -d= -f2- | tr -d "\"'")
# Only needed where outbound traffic must go through a proxy.
if [ -n "${EGRESS_PROXY:-}" ]; then
  export HTTPS_PROXY="$EGRESS_PROXY" HTTP_PROXY="$EGRESS_PROXY" NO_PROXY=localhost,127.0.0.1
fi
cd "$ROOT"
for m in gpt_5_4 gemini_3_1_pro grok_4_20_beta claude_opus_4_6; do
  echo "[eval_all] launching $m"
  nohup python -m vagen.evaluate.run_eval --config examples/evaluation/eval_all_openrouter_ai2thor/$m.yaml \
     fileroot="$ROOT" run.max_concurrent_jobs=8 backends.openai.max_concurrency=8 \
     > /tmp/eval_ai2thor_$m.log 2>&1 &
done
wait
echo "[eval_all] all 4 models done"
