#!/usr/bin/env bash
# =============================================================================
# Short RL-ONLY smoke run for Habitat-GS Interactive View Planning.
# Proves the GraphRL RL loop (verl+sglang rollouts against the Habitat-GS render
# service) works end-to-end on Qwen2.5-VL. Skips traj_to_sft + SFT.
#
# Prereqs (same as run.sh): VIEWSUITE_ROOT exported, render service up with
# client_url_habitat_gs.txt, dataset generated + split under data/habitat_gs/.
#
# GPUs: defaults to 4 (set via CUDA_VISIBLE_DEVICES) so the render service GPU
# is left alone. Override N_GPUS_PER_NODE / CUDA_VISIBLE_DEVICES as needed.
#
# Usage:
#   bash run_smoke.sh
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

EXPERIMENT_DIR="${PWD}/exps/viewsuite/habitat_gs_ivp_smoke"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/smoke_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}  (GPUs=${CUDA_VISIBLE_DEVICES})"

export WANDB_MODE="${WANDB_MODE:-online}"   # wandb authed via ~/.netrc; set WANDB_MODE=offline to disable

python -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    experiment_name=habitat_gs_ivp_smoke \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val_smoke.yaml" \
    iterations=1 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    general_overrides.rl.hydra_overrides.trainer.nnodes=1 \
    general_overrides.rl.hydra_overrides.data.train_batch_size=8 \
    general_overrides.rl.hydra_overrides.actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    +general_overrides.rl.hydra_overrides.critic.ppo_mini_batch_size=8 \
    iteration_overrides.iter0.rl.training_steps=2 \
    +iteration_overrides.iter0.traj_to_sft=null \
    iteration_overrides.iter0.sft=null \
    "$@" 2>&1 | tee "${LOG_FILE}"
