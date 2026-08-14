#!/bin/bash
# Supervisor for the Habitat-GS render service.
#
# Thin wrapper over scannet_http_service_loop.sh: the Habitat-GS backend is a backend of
# that service, not a second service, so it reuses the worker pool, GPU pinning, TLS and
# multipart protocol that are already hardened. All this adds is the two things that
# differ -- the scene root and the interpreter.
#
#   args: MAX_WORKERS GPU_IDS OMP_CAP PORT RESTART_INTERVAL
#
# The GS corpus has 129 scenes and the pool is sticky by scene, so more than ~17 workers
# per GPU buys nothing: every scene is already resident. Measured footprint is 0.8-2.3
# GiB per worker (it tracks gaussian-cloud size), so 25/GPU is comfortable on a 143 GiB
# card even at the worst case.
set -euo pipefail

: "${VIEWSUITE_ROOT:?set VIEWSUITE_ROOT to the repo root}"
cd "$VIEWSUITE_ROOT"

MAX_WORKERS=${1:-136}
GPU_IDS=${2:-0,1,2,3,4,5,6,7}
OMP_CAP=${3:-1}
PORT=${4:-8812}
RESTART_INTERVAL=${5:-86400}

# habitat-gs cannot share an interpreter with habitat-sim 0.3.3 (the ScanNet backend).
export PY_BIN=${PY_BIN:-$HOME/miniconda3/envs/habitat-gs/bin/python}
export SCANNET_ROOT=${HABITAT_GS_ROOT:-$VIEWSUITE_ROOT/data/gs_scenes}
export HABITAT_SIM_LOG=${HABITAT_SIM_LOG:-quiet}
export MAGNUM_LOG=${MAGNUM_LOG:-quiet}

exec bash scripts/scannet_http_service_loop.sh \
  "$MAX_WORKERS" "$GPU_IDS" "$OMP_CAP" "$PORT" "$RESTART_INTERVAL" habitat_gs
