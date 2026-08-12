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
# Bind address and TLS pair come from the environment, so the supervisor can serve
# HTTPS and bind "::" when the client is on another machine. Both default to the
# previous behaviour (IPv4 bind, plain HTTP). Set the bind deliberately: an IPv4-only
# listener is simply invisible to an IPv6 client, with no error on either side.
HOST=${HOST:-0.0.0.0}
SSL_KEYFILE=${SSL_KEYFILE:-}
SSL_CERTFILE=${SSL_CERTFILE:-}
# Anchor to VIEWSUITE_ROOT, not the CWD: setsid does not change directory, so a
# CWD-relative default resolves against wherever the supervisor happened to be started
# and the service then comes up healthy but fails on every scene.
SCANNET_ROOT=${SCANNET_ROOT:-${VIEWSUITE_ROOT}/data/scannet/scans}
# habitat lives in its own conda env (py3.9); the training env cannot import it.
PY_BIN=${PY_BIN:-python}
if [ "$BACKEND" = "habitat" ]; then
  HABITAT_PY=${HABITAT_PY:-${HOME:-/root}/miniconda3/envs/habitat/bin/python3}
  if [ ! -x "$HABITAT_PY" ]; then
    # Fail loudly. Falling through to the default interpreter would start a service
    # that binds the port, looks healthy, and returns HTTP 200 with zero images for
    # every request — habitat_sim is imported lazily inside the worker.
    echo "FATAL: BACKEND=habitat but no habitat interpreter at $HABITAT_PY" >&2
    echo "       create it: conda create -y -n habitat python=3.9 && \\" >&2
    echo "       conda install -y -n habitat habitat-sim headless -c conda-forge -c aihabitat" >&2
    exit 1
  fi
  PY_BIN="$HABITAT_PY"
fi
# The habitat env has habitat-sim but not the repo on its path, and service.py is run by
# absolute path so sys.path[0] is service_http/, not the repo root. Without this the
# service dies with ModuleNotFoundError: view_suite on every supervisor restart.
export PYTHONPATH="${VIEWSUITE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
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
     --host="$HOST"
     --forced_render_size=None)
[ -n "$GPU_IDS" ] && CMD+=(--gpu_ids="$GPU_IDS")
[ -n "$SSL_KEYFILE" ] && CMD+=(--ssl_keyfile="$SSL_KEYFILE")
[ -n "$SSL_CERTFILE" ] && CMD+=(--ssl_certfile="$SSL_CERTFILE")

exec "${CMD[@]}"
