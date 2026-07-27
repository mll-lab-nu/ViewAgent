#!/bin/bash
# Full iterative RL<->SFT training run (persistent wrapper for systemd).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate viewagent_thor
cd /home/kangrui/projects/viewagent_ai2thor/ViewAgent/GraphRL
export VIEWSUITE_ROOT=/home/kangrui/projects/viewagent_ai2thor/ViewAgent
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=online
export N_GPUS_PER_NODE=8 SFT_N_GPUS=8
exec bash examples/viewsuite/ai2thor_interactive_view_planning/run.sh
