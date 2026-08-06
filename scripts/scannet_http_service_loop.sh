#!/bin/bash
# Foreground supervisor: restarts the ScanNet HTTP render service every T seconds.
# Logs go to $PWD/scannet_http_service_<TS>/{supervisor.log,scannet_http_service_N.log}

MAX_WORKERS=${1:-32}
GPU_IDS=${2:-""}
OMP_CAP=${3:-1}
PORT=${4:-8767}
T=${5:-10800}            # restart interval in seconds (default: 3h)
BACKEND=${6:-open3d}     # open3d (mesh) | gsplat (3DGS) | habitat (mesh, multi-GPU)

SCRIPT_DIR="$(dirname "$0")"
SERVICE_SCRIPT="${SCRIPT_DIR}/scannet_http_service.sh"

RUN_DIR="$PWD/scannet_http_service_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
SUPERVISOR_LOG="${RUN_DIR}/supervisor.log"
# Survives supervisor death so a later run can still clean up leaked worker groups.
# Per-user and per-port: a world-writable shared path in /tmp lets any local user
# pre-create the file (and inject a PGID), and makes the first user to touch a port
# permanently deny reaping to everyone else on the box.
umask 077
PGID_FILE="${PGID_FILE:-${XDG_RUNTIME_DIR:-/tmp}/scannet_render_pgids_$(id -u)_${PORT}.txt}"
mkdir -p "$(dirname "$PGID_FILE")" 2>/dev/null || true
# Only one supervisor per port may reap. Without this, restarting the loop without
# stopping the old one makes the new supervisor SIGKILL the running service, after which
# both fight over the port and reap each other every generation.
LOCK_FILE="${PGID_FILE}.lock"
exec 9>"$LOCK_FILE" 2>/dev/null || true
if ! flock -n 9 2>/dev/null; then
  echo "FATAL: another supervisor already owns port ${PORT} (lock: $LOCK_FILE)" >&2
  exit 1
fi

log() { echo "[$(date)] $*" | tee -a "$SUPERVISOR_LOG"; }

log "Supervisor started (interval=${T}s MAX_WORKERS=${MAX_WORKERS} OMP_CAP=${OMP_CAP} PORT=${PORT} GPU_IDS=${GPU_IDS:-unset} BACKEND=${BACKEND})"
log "Run directory: ${RUN_DIR}"

CURRENT_PGID=""

# Killing only the supervisor's child leaves the pool workers behind: they are spawned
# processes, so they survive as orphans and keep their GPU contexts (a habitat worker
# holds ~330 MiB of VRAM). Observed 151 leaked workers after one supervisor was killed.
# Workers share the service's process GROUP -- their cmdline is `multiprocessing.spawn`,
# not service.py, so pgrep on the service path never finds them -- so we record each
# generation's PGID and reap by group.
#
# Each record is "PGID STARTTIME". The start time (field 22 of /proc/<pid>/stat, in
# clock ticks since boot) pins the record to one specific process: PIDs are recycled, and
# a stale file left by a SIGKILLed supervisor would otherwise make a later run SIGKILL
# whatever unrelated process now owns that group id.
record_pgid() {
  local pg="$1" st actual
  case "$pg" in ''|*[!0-9]*) return 0 ;; esac
  [ "$pg" -gt 1 ] 2>/dev/null || return 0
  # Only record a genuine group LEADER. The PGID capture above falls back to the child
  # PID when `ps` misses a fast-exiting child, and persisting a PID as if it were a PGID
  # would later SIGKILL whatever group happens to carry that number.
  actual=$(awk '{print $5}' "/proc/$pg/stat" 2>/dev/null)
  [ "$actual" = "$pg" ] || { log "Not recording $pg: not a process-group leader"; return 0; }
  st=$(awk '{print $22}' "/proc/$pg/stat" 2>/dev/null)
  [ -n "$st" ] && echo "$pg $st" >> "$PGID_FILE" 2>/dev/null
  return 0
}

reap_orphans() {
  local pg st now_st reaped=0 mypg
  mypg=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
  while read -r pg st _; do
    # `kill -0 -$pg` is NOT a sufficient guard: it returns success for "0" (the caller's
    # own process group) and for "1" (kill(-1) == every process this user owns,
    # including the training job and this script). Both must be rejected explicitly.
    case "$pg" in ''|*[!0-9]*) continue ;; esac
    [ "$pg" -gt 1 ] 2>/dev/null || continue
    [ "$pg" = "$mypg" ] && continue
    now_st=$(awk '{print $22}' "/proc/$pg/stat" 2>/dev/null)
    [ -n "$now_st" ] || continue                 # group leader is gone; nothing to reap
    [ "$now_st" = "$st" ] || continue            # PID recycled -- not our process
    kill -0 "-$pg" 2>/dev/null || continue
    log "Reaping orphaned render process group $pg"
    kill -KILL "-$pg" 2>/dev/null || true
    reaped=1
  done < "$PGID_FILE" 2>/dev/null
  : > "$PGID_FILE" 2>/dev/null || log "WARNING: cannot write $PGID_FILE; orphans will leak"
  [ "$reaped" = 1 ] && sleep 2                   # only pay the wait when we killed something
  return 0
}

kill_group() {
  local sig="$1" pgid="$2"
  if [ -n "$pgid" ] && kill -0 "-$pgid" 2>/dev/null; then
    kill "$sig" "-$pgid" 2>/dev/null || true
  fi
}

cleanup_and_exit() {
  trap '' INT TERM
  if [ -n "$CURRENT_PGID" ]; then
    log "Cleaning up service PGID=${CURRENT_PGID}"
    kill_group -TERM "$CURRENT_PGID"
    sleep 2
    kill_group -KILL "$CURRENT_PGID"
  fi
  reap_orphans
  exit 0
}
trap cleanup_and_exit INT TERM

# A previous supervisor may have died without cleaning up; take its workers first.
reap_orphans

SERVICE_COUNT=0
while true; do
  SERVICE_COUNT=$((SERVICE_COUNT + 1))
  SERVICE_LOG="${RUN_DIR}/scannet_http_service_${SERVICE_COUNT}.log"
  export LOG="$SERVICE_LOG"

  setsid "$SERVICE_SCRIPT" "$MAX_WORKERS" "$OMP_CAP" "$PORT" "$GPU_IDS" "$BACKEND" \
      >> "$SERVICE_LOG" 2>&1 &
  CHILD_PID=$!
  PGID=$(ps -o pgid= -p "$CHILD_PID" 2>/dev/null | tr -d ' ')
  PGID=${PGID:-$CHILD_PID}
  CURRENT_PGID="$PGID"
  record_pgid "$PGID"
  log "Service #${SERVICE_COUNT} started PID=${CHILD_PID} PGID=${PGID} log=${SERVICE_LOG}"

  SECS=0
  while [ "$SECS" -lt "$T" ]; do
    if ! kill -0 "$CHILD_PID" 2>/dev/null; then
      log "Service #${SERVICE_COUNT} exited early (runtime=${SECS}s)"
      break
    fi
    sleep 1
    SECS=$((SECS + 1))
  done

  if kill -0 "$CHILD_PID" 2>/dev/null; then
    log "Timeout (${T}s), restarting service #${SERVICE_COUNT}"
    kill_group -TERM "$PGID"
    sleep 10
    kill -0 "$CHILD_PID" 2>/dev/null && kill_group -KILL "$PGID"
  fi

  reap_orphans
  CURRENT_PGID=""
done
