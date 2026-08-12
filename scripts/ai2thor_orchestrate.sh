#!/bin/bash
# Autonomous orchestrator for the AI2-THOR IVP experiment.
#   1) babysit the training run (heal tunnels + render servers, resume on failure)
#   2) when training completes, eval the trained model on the 3 tasks vs baseline
# Runs under systemd-run --user so it survives session teardown.
ROOT="${VIEWSUITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="${ORCHESTRATE_LOG:-$ROOT/../orchestrate.log}"
exec >> "$LOG" 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate viewagent_thor
export VIEWSUITE_ROOT="$ROOT"
cd "$ROOT"
log(){ echo "[$(date '+%F %T')] $*"; }

heal_tunnel(){ systemctl --user is-active "$1" >/dev/null 2>&1 || { log "tunnel $1 down -> restart"; systemctl --user restart "$1" 2>/dev/null; }; }

render_up(){ # $1 localport ; 3 tries before declaring down (tolerate load)
  for _ in 1 2 3; do curl -sf --max-time 10 "http://localhost:$1/health" >/dev/null 2>&1 && return 0; sleep 5; done
  return 1
}
restart_render(){ # $1 host  — clean reset: kill by PORT + bracket-safe controller kill (no self-match), start fresh
  ssh -o BatchMode=yes "$1" 'bash -lc "
    fuser -k -9 8766/tcp 2>/dev/null; sleep 1
    pkill -9 -f \"[t]hor-CloudRendering\" 2>/dev/null; sleep 3
    source /venv/main/bin/activate; cd /root/ViewAgent
    export VIEWSUITE_ROOT=/root/ViewAgent UNIFIED_MAX_INFLIGHT=256
    nohup python view_suite/ai2thor/service_http/service.py --max_workers=8 --port=8766 --platform=CloudRendering --agentMode=default --width=512 --height=512 --fieldOfView=90.0 --gpu_ids=0 > /root/ai2thor_service.log 2>&1 &
    for i in \$(seq 1 25); do sleep 3; curl -sf --max-time 5 http://127.0.0.1:8766/health >/dev/null 2>&1 && break; done
  "' 2>/dev/null
}
heal_render(){ # $1 host  $2 localport — restart only if actually down
  render_up "$2" && return
  log "render $1 (:$2) DOWN -> clean restart"; restart_render "$1"; sleep 5
}
heal_infra(){ heal_tunnel vast2-render-tunnel; heal_tunnel vast3-render-tunnel; heal_render vast2 8766; heal_render vast3 8767; }
# periodic recycle to clear the ai2thor controller leak (staggered so failover covers)
RECYCLE_SECS=10800; LAST_RECYCLE=0   # every 6h; 0 = compute at start
recycle_if_due(){
  local now; now=$(date +%s)
  [ "$LAST_RECYCLE" -eq 0 ] && { LAST_RECYCLE=$now; return; }
  if [ $((now-LAST_RECYCLE)) -ge "$RECYCLE_SECS" ]; then
    log "periodic render recycle (clear controller leak)"
    restart_render vast3; sleep 10; restart_render vast2   # staggered: other server covers via failover
    LAST_RECYCLE=$now
  fi
}

# ---------------- Phase 1: babysit training ----------------
LOGDIR="$ROOT/GraphRL/exps/viewsuite/ai2thor_interactive_view_planning"
rl_alive(){ pgrep -f "[m]ain_ppo" >/dev/null 2>&1 || pgrep -f "[r]aylet" >/dev/null 2>&1; }
log_age(){ local lf; lf=$(ls -t $LOGDIR/pipeline_*.log 2>/dev/null | head -1); [ -z "$lf" ] && { echo 999999; return; }; echo $(( $(date +%s) - $(date -r "$lf" +%s) )); }
clean_kill_train(){   # stop unit + purge all ray/sglang/verl leftovers (bracket-safe) + free ports
  systemctl --user stop ai2thor_ivp_train.service 2>/dev/null
  for pat in "[g]raphrl.main" "[m]ain_ppo" "[r]ay::" "[s]glang" "raylet" "gcs_server"; do pkill -9 -f "$pat" 2>/dev/null; done
  ray stop --force >/dev/null 2>&1 || true; sleep 5
  systemctl --user reset-failed ai2thor_ivp_train.service 2>/dev/null
}
start_train(){ systemd-run --user --unit=ai2thor_ivp_train --same-dir /bin/bash $ROOT/../run_ai2thor_full.sh; }

log "orchestrator started; babysitting training"
RESUMES=0; MAXRESUMES=8
while true; do
  heal_infra
  recycle_if_due
  LF=$(ls -t $LOGDIR/pipeline_*.log 2>/dev/null | head -1)
  if grep -q "PIPELINE COMPLETE" "$LF" 2>/dev/null; then log "training PIPELINE COMPLETE"; break; fi
  healthy=0
  if systemctl --user is-active ai2thor_ivp_train.service >/dev/null 2>&1; then
    age=$(log_age)
    if rl_alive || [ "$age" -le 150 ]; then
      healthy=1
      STEP=$(grep -oE "training/global_step:[0-9]+" $LOGDIR/pipeline_*.log 2>/dev/null | tail -1)
      log "training active ($STEP, log_age=${age}s); infra ok"
    else
      log "HUNG: unit active but RL subprocess dead (log_age=${age}s) -> clean kill + resume"
    fi
  fi
  if [ "$healthy" -eq 1 ]; then sleep 240; continue; fi
  if [ "$RESUMES" -lt "$MAXRESUMES" ]; then
    RESUMES=$((RESUMES+1)); log "training not healthy -> clean RESUME #$RESUMES"
    clean_kill_train; start_train; sleep 200
  else
    log "MAXRESUMES reached -> giving up babysit, proceeding to eval whatever exists"; break
  fi
done

# ---------------- Phase 2: eval the trained model ----------------
FINAL=$(ls -d $ROOT/GraphRL/exps/viewsuite/ai2thor_interactive_view_planning/iter_*/sft/sft_model 2>/dev/null | sort | tail -1)
if [ -z "$FINAL" ] || [ ! -f "$FINAL/config.json" ]; then
  log "no trained sft_model found; skipping eval. Looked under iter_*/sft/sft_model"; log "ORCHESTRATION COMPLETE (no eval)"; exit 0
fi
log "final trained model: $FINAL"
heal_infra
export fileroot="$ROOT"
log "launching final eval (3 tasks, n=331) on 8 GPUs"
MODEL_PATH="$FINAL" MODEL_NAME=qwen25vl7b_trained \
CONFIG="$ROOT/examples/evaluation/eval_default_ai2thor.yaml" \
DP_SIZE=8 TP_SIZE=1 MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=30000 \
  bash "$ROOT/examples/evaluation/eval_sglang/eval_model.sh" \
    default_chat_config.max_tokens=2048 backends.sglang.max_concurrency=32 run.max_concurrent_jobs=48 \
  >> "$LOG" 2>&1
log "final eval finished; results:"
python - <<PY >> "$LOG" 2>&1
import json
def rd(run,t):
    try: return json.load(open(f"$ROOT/rollouts/{run}/tag_{t}/summary.json"))["success_rate"]*100
    except Exception: return None
print("=================  AI2-THOR RESULTS (test, n=331/task)  =================")
print(f"{'task':6} {'base':>8} {'trained':>9}")
base={"path_to_view":36.9,"view_to_path":24.5,"interactive_view_planning":6.9}
for t,l in [("path_to_view","P2V"),("view_to_path","V2P"),("interactive_view_planning","IVP")]:
    tr=rd("qwen25vl7b_trained",t)
    print(f"{l:6} {base[t]:>7.1f}% {('%.1f%%'%tr) if tr is not None else 'NA':>9}")
print("========================================================================")
PY
log "ORCHESTRATION COMPLETE"
