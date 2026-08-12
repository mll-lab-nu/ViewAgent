#!/usr/bin/env bash
# =============================================================================
# Run GraphRL pipeline for ViewSuite Interactive View Planning (v2)
# =============================================================================
# Changes from v1:
#   - Uses pipeline.yaml with prefer_single_action knob
#
# Usage:
#   bash run_v2.sh
#   bash run_v2.sh iterations=5
#
# experiment_dir is computed by pipeline.yaml as
#   exps/viewsuite/viewsuite_interactive_view_planning_v2/
# resolved relative to the current working directory.
# =============================================================================

set -euo pipefail
: "${VIEWSUITE_ROOT:?VIEWSUITE_ROOT must be exported}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${PWD}/exps/viewsuite/viewsuite_interactive_view_planning"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"
echo "Using ${N_GPUS_PER_NODE} GPU(s) for RL and ${SFT_N_GPUS} GPU(s) for SFT"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

python -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    project_name=viewsuite_graph_improve \
    experiment_name=graph_atomize_ir5 \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train_turn_format.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val.yaml" \
    iterations=4 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    general_overrides.rl.hydra_overrides.trainer.nnodes=1 \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    iteration_overrides.iter0.rl.training_steps=61 \
    iteration_overrides.iter1.rl.training_steps=61 \
    iteration_overrides.iter2.rl.training_steps=61 \
    iteration_overrides.iter3.rl.training_steps=300 \
    general_overrides.traj_to_sft.graph_builder.atomize.enabled=true \
    general_overrides.traj_to_sft.graph_builder.merge_tol.position=0.2 \
    general_overrides.traj_to_sft.graph_builder.merge_tol.angle=10.0 \
    +iteration_overrides.iter3.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train.yaml" \
    +iteration_overrides.iter3.rl.hydra_overrides.huggingface_hub.repo_id=viewsuite_interactive_view_planning \
    +iteration_overrides.iter3.rl.hydra_overrides.trainer.log_image.enable=false \
    "$@" 2>&1 | tee "${LOG_FILE}"
