#!/bin/bash
# Wrapper to start the ScanNet HTTP render service.
# Usage: ./scannet_http_service.sh [MAX_WORKERS] [OMP_CAP] [PORT] [GPU_IDS] [BACKEND]

set -euo pipefail
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

MAX_WORKERS=${1:-24}
OMP_CAP=${2:-4}
PORT=${3:-8767}
GPU_IDS=${4:-"0"}
BACKEND=${5:-open3d}  # open3d (mesh) | gsplat (3DGS) | habitat (mesh, multi-GPU)
SCANNET_ROOT=${SCANNET_ROOT:-data/scannet/scans}
# habitat lives in its own conda env (py3.9); the training env cannot import it.
PY_BIN=${PY_BIN:-python}
if [ "$BACKEND" = "habitat" ] && [ -x "$HOME/miniconda3/envs/habitat/bin/python3" ]; then
  PY_BIN="$HOME/miniconda3/envs/habitat/bin/python3"
fi
LOG_LEVEL=${LOG_LEVEL:-info}

export UNIFIED_MAX_INFLIGHT=${UNIFIED_MAX_INFLIGHT:-256}
export UNIFIED_ADMIT_TIMEOUT=${UNIFIED_ADMIT_TIMEOUT:-2.0}
export UNIFIED_RENDER_TIMEOUT=${UNIFIED_RENDER_TIMEOUT:-120.0}
export UNIFIED_GPU_BINDING_STRATEGY=${UNIFIED_GPU_BINDING_STRATEGY:-shared}

CORES=$(nproc)
PER_WORKER=$((CORES / MAX_WORKERS))
((PER_WORKER < 1)) && PER_WORKER=1
((PER_WORKER > OMP_CAP)) && PER_WORKER=$OMP_CAP

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       PYTORCH_NUM_THREADS=1 OMP_NUM_THREADS=$PER_WORKER

export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-$(id -u)}
mkdir -p "$XDG_RUNTIME_DIR"

CMD=("$PY_BIN" "${VIEWSUITE_ROOT}/view_suite/scannet/service_http/service.py"
     --scannet_root="$SCANNET_ROOT"
     --max_workers="$MAX_WORKERS"
     --port="$PORT"
     --log_level="$LOG_LEVEL"
     --backend="$BACKEND"
     --forced_render_size=None)
[ -n "$GPU_IDS" ] && CMD+=(--gpu_ids="$GPU_IDS")

exec "${CMD[@]}"
