#!/usr/bin/env bash
# =============================================================================
# CHEAP TEST HARNESS — for iterating on ideas fast instead of burning 20-30h/run.
#
#   * Qwen2.5-VL-**7B** by default (MODEL_SIZE=3B is NOT usable: measured 0% action-tag
#     rate with free_think and 20% with grounding, vs 100% for 7B — the 3B model emits
#     incoherent text on this multi-turn visual task, so cost is cut via STEPS not size)
#   * SFT every **61** RL steps — MUST match the baselines. 20 was too few:
#     IVP only emerges after ~120-180 cumulative RL steps (baseline was 0.0 at 61
#     steps and only hit 18% by ~180), so a 20x3=60-step run measures nothing.
#     Cost is saved by skipping iter3's 300 steps, not by starving the RL.
#   * **3** iterations (3 SFT rounds), no 300-step final iter
#   => 183 vs 483 RL steps ~= 40% of the full run, while staying in the regime
#      where IVP actually emerges.
#
# Knobs meant to be flipped from the command line (see examples below):
#   general_overrides.traj_to_sft.graph_builder.merge_tol.position/.angle
#       Idea-5 merge tolerance. 0.2/10 is the aggressive setting; user suspects it
#       hurts action->view accuracy, so 0.001/0.001 (OFF) is the control arm.
#   general_overrides.traj_to_sft.graph_builder.atomize.enabled
#
# Usage:
#   bash run_cheap3b.sh                       # grounding + merge_tol OFF (kitchen-sink v1)
#   TRAIN_CFG=train_turn_format.yaml bash run_cheap3b.sh   # old free-form format (ablation)
#   bash run_cheap3b.sh EXTRA_MERGE_TOL=on    # merge_tol 0.2/10
# =============================================================================
set -euo pipefail
: "${VIEWSUITE_ROOT:?VIEWSUITE_ROOT must be exported}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cheap3b}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PWD}/exps/viewsuite/${EXPERIMENT_NAME}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"

# merge_tol: OFF by default (control arm). Set MERGE_TOL=on for the 0.2/10 arm.
if [ "${MERGE_TOL:-off}" = "on" ]; then MT_POS=0.2; MT_ANG=10.0; else MT_POS=0.001; MT_ANG=0.001; fi

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "[fast] model=Qwen2.5-VL-${MODEL_SIZE:-7B} steps/iter=${STEPS_PER_ITER:-61} iters=3 merge_tol=${MT_POS}/${MT_ANG} train_cfg=${TRAIN_CFG:-train_grounding.yaml}"
echo "Logging to: ${LOG_FILE}"
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

python -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    project_name=viewsuite_graph_improve \
    experiment_name="${EXPERIMENT_NAME}" \
    initial_model_path=Qwen/Qwen2.5-VL-${MODEL_SIZE:-7B}-Instruct \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/${TRAIN_CFG:-train_grounding.yaml}" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val.yaml" \
    iterations=3 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    general_overrides.rl.hydra_overrides.trainer.nnodes=1 \
    general_overrides.rl.hydra_overrides.trainer.save_freq=10 \
    general_overrides.rl.hydra_overrides.trainer.test_freq=10 \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    iteration_overrides.iter0.rl.training_steps=${STEPS_PER_ITER:-61} \
    iteration_overrides.iter1.rl.training_steps=${STEPS_PER_ITER:-61} \
    iteration_overrides.iter2.rl.training_steps=${STEPS_PER_ITER:-61} \
    general_overrides.traj_to_sft.graph_builder.atomize.enabled=true \
    general_overrides.traj_to_sft.graph_builder.merge_tol.position=${MT_POS} \
    general_overrides.traj_to_sft.graph_builder.merge_tol.angle=${MT_ANG} \
    "$@" 2>&1 | tee "${LOG_FILE}"
