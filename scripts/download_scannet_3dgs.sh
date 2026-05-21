#!/bin/bash
# Download pretrained 3DGS checkpoints for scenes referenced by the viewsuite
# test jsonls (built via build_manifest.py).
#
# Required env: HF_TOKEN (HuggingFace access token with access to
# GaussianWorld/scannet_mcmc_1.5M_3dgs).
#
# Usage:
#   export HF_TOKEN=hf_...
#   ./ViewSuite/scripts/download_scannet_3dgs.sh
#   ./ViewSuite/scripts/download_scannet_3dgs.sh <MANIFEST> <GS_ROOT>
#
# Defaults:
#   MANIFEST = ${VIEWSUITE_ROOT}/view_suite/envs/scannet_proxy_task/data_gen/viewsuite_15k_gs_test_manifest.jsonl
#   GS_ROOT  = ${VIEWSUITE_ROOT}/data/scannet_3dgs_mcmc

set -euo pipefail
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"  # .../ViewSuite

MANIFEST="${1:-$REPO_ROOT/view_suite/envs/scannet_proxy_task/data_gen/viewsuite_15k_gs_test_manifest.jsonl}"
GS_ROOT="${2:-${VIEWSUITE_ROOT}/data/scannet_3dgs_mcmc}"

: "${HF_TOKEN:?HF_TOKEN is not set. export HF_TOKEN=hf_...}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "[error] manifest not found: $MANIFEST" >&2
  echo "  Run build_manifest.py first:" >&2
  echo "  python -m view_suite.envs.scannet_proxy_task.data_gen.build_manifest" >&2
  exit 2
fi

echo "[cfg] MANIFEST=$MANIFEST"
echo "[cfg] GS_ROOT=$GS_ROOT"

exec python "$SCRIPT_DIR/download_scannet_3dgs.py" \
  --manifest "$MANIFEST" \
  --gs_root  "$GS_ROOT"
