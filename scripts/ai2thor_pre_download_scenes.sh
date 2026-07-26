#!/bin/bash
# Pre-download AI2-THOR scene assets so the service / data_gen never hits a
# cold boot (first Controller.reset() downloads ~800 MB of CloudRendering
# binary + per-scene assets into ~/.ai2thor/).
#
# Usage:
#   ./ai2thor_pre_download_scenes.sh                       # all 120 FloorPlans, auto-detect GPU
#   ./ai2thor_pre_download_scenes.sh default               # 30-scene balanced subset
#   ./ai2thor_pre_download_scenes.sh kitchen 1             # kitchen only, GPU 1
#   ./ai2thor_pre_download_scenes.sh "FloorPlan1,FloorPlan5" 0
#
# Positional args:
#   SCENES (default: "all")  — see view_suite.ai2thor.scene_list.parse_subset
#   GPU    (default: auto)

set -euo pipefail

SCENES=${1:-all}
GPU=${2:-}

ARGS=(--scenes="${SCENES}")
if [ -n "${GPU}" ]; then
  ARGS+=(--gpu="${GPU}")
fi

echo "[pre-dl] scenes=${SCENES}  gpu=${GPU:-auto}"
exec python -m view_suite.ai2thor.pre_download_scenes "${ARGS[@]}"
