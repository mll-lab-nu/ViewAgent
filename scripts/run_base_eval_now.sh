#!/bin/bash
set -u
ROOT="${VIEWSUITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOGF="${BASE_EVAL_LOG:-$ROOT/../base_eval_now.log}"
exec >> "$LOGF" 2>&1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate viewagent_thor
export VIEWSUITE_ROOT="$ROOT"; cd "$ROOT"
log(){ echo "[$(date '+%F %T')] $*"; }
for un in vast2-render-tunnel vast3-render-tunnel; do systemctl --user is-active "$un" >/dev/null 2>&1 || systemctl --user restart "$un"; done
log "=== base eval (Qwen2.5-VL-7B-Instruct) on test n=252 ==="
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct" MODEL_NAME=qwen25vl7b_base \
CONFIG="$ROOT/examples/evaluation/eval_default_ai2thor.yaml" \
DP_SIZE=8 TP_SIZE=1 MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=30000 \
  bash "$ROOT/examples/evaluation/eval_sglang/eval_model.sh" \
    default_chat_config.max_tokens=2048 backends.sglang.max_concurrency=32 run.max_concurrent_jobs=32 \
  && log "base eval finished" || log "base eval FAILED"
pkill -9 -f "[s]glang" 2>/dev/null; sleep 5
python3 $ROOT/../build_ai2thor_table_full.py 2>&1 | tee -a "$LOGF" \
  > "$ROOT/examples/evaluation/eval_all_openrouter_ai2thor/RESULTS_full.md"
log "rebuilt RESULTS_full.md; DONE"
