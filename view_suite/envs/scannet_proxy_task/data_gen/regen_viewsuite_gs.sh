#!/bin/bash
# Regenerate viewsuite_15k_gs_test/ by re-rendering with gsplat.
#
# Assumes:
#   - build_manifest.py has produced viewsuite_15k_gs_test_manifest.jsonl
#   - scripts/download_scannet_3dgs.sh has populated /root/projects/viewsuite/data/scannet_3dgs_mcmc/
#
# Usage:
#   ./regen_viewsuite_gs.sh
#   ./regen_viewsuite_gs.sh <SRC_ROOT> <OUT_ROOT> <GS_ROOT>
#   LIMIT=10 ./regen_viewsuite_gs.sh    # smoke-test: first 10 samples only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_ROOT="${1:-/root/projects/viewsuite/data/viewsuite_15k}"
OUT_ROOT="${2:-/root/projects/viewsuite/data/viewsuite_15k_gs_test}"
GS_ROOT="${3:-/root/projects/viewsuite/data/scannet_3dgs_mcmc}"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/viewsuite_15k_gs_test_manifest.jsonl}"
LIMIT="${LIMIT:-0}"

echo "[cfg] SRC_ROOT=$SRC_ROOT"
echo "[cfg] OUT_ROOT=$OUT_ROOT"
echo "[cfg] GS_ROOT=$GS_ROOT"
echo "[cfg] MANIFEST=$MANIFEST"
echo "[cfg] LIMIT=$LIMIT"

exec python "$SCRIPT_DIR/regen_viewsuite_gs.py" \
  --manifest "$MANIFEST" \
  --src_root "$SRC_ROOT" \
  --out_root "$OUT_ROOT" \
  --gs_root  "$GS_ROOT" \
  --limit    "$LIMIT"