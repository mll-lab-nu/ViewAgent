#!/bin/bash
# One-shot: eval the latest trained checkpoint on the CLEAN test set (n=252, 3 tasks)
# with 8-GPU sglang (data-parallel). Training must already be stopped.
set -u
ROOT="${VIEWSUITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOGF="${TEST_EVAL_LOG:-$ROOT/../test_eval_now.log}"
exec >> "$LOGF" 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate viewagent_thor
export VIEWSUITE_ROOT="$ROOT"; cd "$ROOT"
log(){ echo "[$(date '+%F %T')] $*"; }
CKROOT="$ROOT/GraphRL/exps/viewsuite/ai2thor_interactive_view_planning/iter_003/rl/verl_checkpoints"

CKPT=$(python3 - "$CKROOT" <<'PY'
import sys,glob,re
ck=sys.argv[1]; best=(-1,None)
for d in glob.glob(f"{ck}/global_step_*/actor/huggingface"):
    if glob.glob(f"{d}/*.safetensors"):
        m=re.search(r"global_step_(\d+)",d)
        if m and int(m.group(1))>best[0]: best=(int(m.group(1)),d)
print(best[1] or "")
PY
)
[ -z "$CKPT" ] && { log "ERROR: no full checkpoint found"; exit 1; }
log "=== test eval; checkpoint: $CKPT ==="

# verl exports a malformed Qwen2.5-VL config (nested text_config -> model_type
# resolves to 'qwen2_5_vl_text' -> sglang mrope 'Unimplemented'). Patch it so
# model_type == 'qwen2_5_vl' and dtype is bf16. Idempotent; backs up once.
python3 - "$CKPT" <<'PYEOF'
import json,sys,os,shutil
d=sys.argv[1]; f=os.path.join(d,"config.json")
c=json.load(open(f))
if "text_config" in c or c.get("dtype")=="float32":
    if not os.path.exists(f+".orig"): shutil.copy(f,f+".orig")
    c.pop("text_config",None); c["model_type"]="qwen2_5_vl"
    c["dtype"]="bfloat16"; c["torch_dtype"]="bfloat16"
    if isinstance(c.get("vision_config"),dict):
        c["vision_config"]["dtype"]="bfloat16"; c["vision_config"]["torch_dtype"]="bfloat16"
    json.dump(c,open(f,"w"),indent=2); print("[patch] config fixed")
else:
    print("[patch] config already clean")
PYEOF

# ensure GPUs free
for i in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "${u:-9999}" -lt 6000 ] && { log "GPUs free (${u}MiB)"; break; }
  log "waiting for GPUs (${u}MiB used)"; sleep 5
done

# render tunnels for IVP
for un in vast2-render-tunnel vast3-render-tunnel; do systemctl --user is-active "$un" >/dev/null 2>&1 || systemctl --user restart "$un"; done
for pt in 8766 8767; do curl -sf --max-time 8 "http://localhost:$pt/health" >/dev/null 2>&1 && log "render :$pt ok" || log "WARN render :$pt DOWN"; done

MODEL_PATH="$CKPT" MODEL_NAME=qwen25vl7b_trained \
CONFIG="$ROOT/examples/evaluation/eval_default_ai2thor.yaml" \
DP_SIZE=8 TP_SIZE=1 MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=30000 \
  bash "$ROOT/examples/evaluation/eval_sglang/eval_model.sh" \
    default_chat_config.max_tokens=2048 backends.sglang.max_concurrency=32 run.max_concurrent_jobs=32 \
  && log "eval finished" || log "eval FAILED"

pkill -9 -f "[s]glang" 2>/dev/null; sleep 5
log "=== TABLE (trained + frontier, test n=252) ==="
python3 "$ROOT/../build_ai2thor_table_full.py" 2>&1 | tee -a "$LOGF" \
  > "$ROOT/examples/evaluation/eval_all_openrouter_ai2thor/RESULTS_full.md"
log "wrote RESULTS_full.md; DONE"
