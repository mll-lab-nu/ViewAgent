#!/usr/bin/env bash
# =============================================================================
# Run the GraphRL pipeline for Habitat-GS Interactive View Planning (Qwen2.5-VL).
#
# Prereqs:
#   - VIEWSUITE_ROOT exported (repo root)
#   - Habitat-GS HTTP render service running; its URL in
#     $VIEWSUITE_ROOT/client_url_habitat_gs.txt  (see scripts/habitat_gs_http_service.sh)
#   - Habitat-GS IVP dataset generated + split into _train/_eval/_test under
#     $VIEWSUITE_ROOT/data/habitat_gs/
#
# Usage:
#   bash run.sh
#   bash run.sh iterations=5
# =============================================================================
set -euo pipefail
: "${VIEWSUITE_ROOT:?VIEWSUITE_ROOT must be exported}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# verl is vendored at GraphRL/VAGEN/verl -- and that is the *repository* root; the
# package itself is one level down. The repo root therefore has to be on PYTHONPATH or
# `import verl` finds nothing, which surfaces as
# "ModuleNotFoundError: No module named 'verl.experimental'" and reads like a version
# problem rather than a missing path. Set here so the script is self-contained.
GRAPHRL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${VIEWSUITE_ROOT}:${GRAPHRL_ROOT}:${GRAPHRL_ROOT}/VAGEN:${GRAPHRL_ROOT}/VAGEN/verl:${GRAPHRL_ROOT}/LLaMA-Factory/src${PYTHONPATH:+:${PYTHONPATH}}"

EXPERIMENT_DIR="${PWD}/exps/viewsuite/habitat_gs_interactive_view_planning"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"
echo "Using ${N_GPUS_PER_NODE} GPU(s) for RL and ${SFT_N_GPUS} GPU(s) for SFT"

export WANDB_MODE="${WANDB_MODE:-online}"   # wandb authed via ~/.netrc; set WANDB_MODE=offline to disable

python -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val.yaml" \
    iterations=4 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    general_overrides.rl.hydra_overrides.trainer.nnodes=1 \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    iteration_overrides.iter0.rl.training_steps=61 \
    iteration_overrides.iter1.rl.training_steps=61 \
    iteration_overrides.iter2.rl.training_steps=61 \
    +iteration_overrides.iter3.rl.hydra_overrides.trainer.log_image.enable=false \
    "$@" 2>&1 | tee "${LOG_FILE}"
