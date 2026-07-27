#!/bin/bash
ROOT=/home/kangrui/projects/viewagent_ai2thor/ViewAgent
source ~/miniconda3/etc/profile.d/conda.sh; conda activate viewagent_thor
export VIEWSUITE_ROOT="$ROOT" fileroot="$ROOT"
export OPENROUTER_API_KEY=$(grep -oE '^OPENROUTER_API=.*' /home/kangrui/projects/viewagent_ai2thor/.env | cut -d= -f2- | tr -d "\"'")
export HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 NO_PROXY=localhost,127.0.0.1
cd "$ROOT"
for m in gpt_5_4 gemini_3_1_pro grok_4_20_beta claude_opus_4_6; do
  echo "[eval_all] launching $m"
  nohup python -m vagen.evaluate.run_eval --config examples/evaluation/eval_all_openrouter_ai2thor/$m.yaml \
     fileroot="$ROOT" run.max_concurrent_jobs=8 backends.openai.max_concurrency=8 \
     > /tmp/eval_ai2thor_$m.log 2>&1 &
done
wait
echo "[eval_all] all 4 models done"
