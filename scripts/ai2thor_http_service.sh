#!/bin/bash
# Wrapper to start the AI2-THOR HTTP render service.
#
# Only Interactive View Planning (IVP / active-exploration) needs this service;
# P2V and V2P read pre-rendered images straight from the jsonl.
#
# Usage:
#   ./ai2thor_http_service.sh [MAX_WORKERS] [OMP_CAP] [PORT] [GPU_IDS]
#
# Examples:
#   ./ai2thor_http_service.sh 24 4 8765 "0,1,2,3"
#   ./ai2thor_http_service.sh 32 4 8765

set -euo pipefail

MAX_WORKERS=${1:-24}
OMP_CAP=${2:-4}
PORT=${3:-8765}
GPU_IDS=${4:-"0"}

# Resolve repo root (env var wins; else two levels up from this script).
VIEWSUITE_ROOT="${VIEWSUITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# Environment variables for concurrency control
export UNIFIED_MAX_INFLIGHT=${UNIFIED_MAX_INFLIGHT:-256}
export UNIFIED_ADMIT_TIMEOUT=${UNIFIED_ADMIT_TIMEOUT:-2.0}
export UNIFIED_RENDER_TIMEOUT=${UNIFIED_RENDER_TIMEOUT:-60.0}

# Optional: Set unified API key for auth
# export UNIFIED_API_KEY="your-secret-key-here"

# Compute optimal OMP threads per worker
CORES=$(nproc)
PER_WORKER=$((CORES / MAX_WORKERS))
((PER_WORKER < 1)) && PER_WORKER=1
((PER_WORKER > OMP_CAP)) && PER_WORKER=$OMP_CAP

# Set threading environment variables
export OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 \
       PYTORCH_NUM_THREADS=1 \
       OMP_NUM_THREADS=$PER_WORKER

# Headless rendering hint (prevents EGL warnings in CloudRendering backend)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-$(id -u)}
mkdir -p "$XDG_RUNTIME_DIR"

# Do NOT export CUDA_VISIBLE_DEVICES here: the handler distributes GPUs
# explicitly by passing gpu_device=N into each AI2-THOR Controller. Masking at
# the env-var level would renumber Unity's GPUs to 0..N-1 and desync them from
# the physical indices in --gpu_ids. Set CUDA_VISIBLE_DEVICES yourself BEFORE
# invoking this script if you need a hard mask (pass matching GPU_IDS).
:

# Display resource limits and configuration
echo "========================================="
echo "AI2-THOR HTTP Service Configuration"
echo "========================================="
echo "VIEWSUITE_ROOT:   $VIEWSUITE_ROOT"
echo "System Cores:     $CORES"
echo "Max Workers:      $MAX_WORKERS"
echo "OMP Threads:      $PER_WORKER per worker"
echo "Port:             $PORT"
echo "GPU IDs:          ${GPU_IDS:-'not set (CPU mode)'}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'not set'}"
echo "XDG_RUNTIME_DIR:  $XDG_RUNTIME_DIR"
echo "Max Inflight:     $UNIFIED_MAX_INFLIGHT"
echo "Admit Timeout:    $UNIFIED_ADMIT_TIMEOUT s"
echo "Render Timeout:   $UNIFIED_RENDER_TIMEOUT s"
echo "========================================="
echo ""

ulimit -a || true

# Build command
CMD=(python "${VIEWSUITE_ROOT}/view_suite/ai2thor/service_http/service.py"
  --max_workers="$MAX_WORKERS"
  --port="$PORT"
  --platform=CloudRendering
  --agentMode=default
  --width=512
  --height=512
  --fieldOfView=90.0)

# Add GPU IDs if specified
[ -n "$GPU_IDS" ] && CMD+=(--gpu_ids="$GPU_IDS")

echo ""
echo "Starting AI2-THOR service..."
exec "${CMD[@]}"
