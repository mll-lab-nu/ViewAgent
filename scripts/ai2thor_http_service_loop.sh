#!/bin/bash
# Simple foreground supervisor for the AI2-THOR HTTP render service.
# - Runs in the current terminal.
# - Restarts the service every T seconds.
# - Prints ONLY supervisor logs to the terminal.
# - Writes service outputs to:
#       $PWD/ai2thor_http_service_YYYYmmdd_HHMMSS/
#         supervisor.log
#         ai2thor_http_service_1.log
#         ai2thor_http_service_2.log
#         ...
#
# Usage (arg order matches scripts/scannet_http_service_loop.sh for consistency):
#   ./ai2thor_http_service_loop.sh [MAX_WORKERS] [GPU_IDS] [OMP_CAP] [PORT] [T]
#
# Examples:
#   ./ai2thor_http_service_loop.sh 16 "0,1,2,3" 1 8765 86400
#   ./ai2thor_http_service_loop.sh 16 "0"       1 8765

# No "set -euo pipefail" to avoid early exit on minor errors.
MAX_WORKERS=${1:-16}
GPU_IDS=${2:-""}
OMP_CAP=${3:-1}
PORT=${4:-8765}
T=${5:-86400}  # Restart interval in seconds (default: 24h; AI2THOR boot is expensive)

SCRIPT_DIR="$(dirname "$0")"
SERVICE_SCRIPT="${SCRIPT_DIR}/ai2thor_http_service.sh"

# Use current working directory for logs
BASE_DIR="$PWD"

# Create a run folder for this supervisor session
START_TS=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${BASE_DIR}/ai2thor_http_service_${START_TS}"
mkdir -p "$RUN_DIR"

SUPERVISOR_LOG="${RUN_DIR}/supervisor.log"

log() {
  echo "[$(date)] $*" | tee -a "$SUPERVISOR_LOG"
}

log "Supervisor started (interval=${T}s, MAX_WORKERS=${MAX_WORKERS}, OMP_CAP=${OMP_CAP}, PORT=${PORT}, GPU_IDS=${GPU_IDS:-'not set'})"
log "Run directory: ${RUN_DIR}"

SERVICE_COUNT=0
CURRENT_PGID=""

kill_group() {
  local sig="$1"   # e.g. -TERM or -KILL
  local pgid="$2"
  if [ -n "${pgid}" ] && kill -0 "-${pgid}" 2>/dev/null; then
    log "Sending ${sig} to process group PGID=${pgid}"
    kill "${sig}" "-${pgid}" 2>/dev/null || true
  fi
}

cleanup_and_exit() {
  # Ignore further INT/TERM during cleanup
  trap '' INT TERM

  log "Supervisor interrupted, cleaning up current service (if any)"
  if [ -n "${CURRENT_PGID}" ]; then
    kill_group -TERM "${CURRENT_PGID}"
    sleep 2
    kill_group -KILL "${CURRENT_PGID}"
  fi
  log "Supervisor exiting"
  exit 0
}

trap cleanup_and_exit INT TERM

while true; do
  SERVICE_COUNT=$((SERVICE_COUNT + 1))
  SERVICE_LOG="${RUN_DIR}/ai2thor_http_service_${SERVICE_COUNT}.log"

  log "====================================================="
  log "Starting service #${SERVICE_COUNT}"
  log "Service log: ${SERVICE_LOG}"
  log "====================================================="

  # Pass log filename to inner script (just for banner)
  export LOG="${SERVICE_LOG}"

  # Start service in a new process group
  setsid "${SERVICE_SCRIPT}" "${MAX_WORKERS}" "${OMP_CAP}" "${PORT}" "${GPU_IDS}" >> "${SERVICE_LOG}" 2>&1 &
  CHILD_PID=$!

  # Get PGID for process-group control
  PGID=$(ps -o pgid= -p "${CHILD_PID}" 2>/dev/null | tr -d ' ')
  if [ -z "${PGID}" ]; then
    PGID="${CHILD_PID}"
  fi
  CURRENT_PGID="${PGID}"

  log "Service #${SERVICE_COUNT} started: PID=${CHILD_PID} PGID=${PGID}"

  # Wait loop
  SECS=0
  while [ "${SECS}" -lt "${T}" ]; do
    if ! kill -0 "${CHILD_PID}" 2>/dev/null; then
      log "Service #${SERVICE_COUNT} exited early (runtime=${SECS}s)"
      break
    fi
    sleep 1
    SECS=$((SECS + 1))
  done

  # Restart when timeout hit
  if kill -0 "${CHILD_PID}" 2>/dev/null; then
    log "Timeout reached (${T}s). Restarting service #${SERVICE_COUNT}"
    kill_group -TERM "${PGID}"
    sleep 10
    if kill -0 "${CHILD_PID}" 2>/dev/null; then
      log "Service still alive after SIGTERM; sending SIGKILL"
      kill_group -KILL "${PGID}"
    fi
  fi

  CURRENT_PGID=""
done
