#!/bin/bash
# Fetch the Habitat-GS 3DGS scene corpus (129 scenes, ~30 GB).
#
# Public, not gated -- unlike the ScanNet dataset, no token is needed. Each scene is a
# <scene>.gs.ply stage plus a Habitat <scene>.navmesh; both are required (the navmesh is
# what keeps sampled cameras on walkable space, which a gaussian reconstruction needs).
set -euo pipefail

: "${VIEWSUITE_ROOT:?set VIEWSUITE_ROOT to the repo root}"
DEST=${HABITAT_GS_ROOT:-$VIEWSUITE_ROOT/data/gs_scenes}
mkdir -p "$DEST"

# hf_transfer is absent from most base envs and fails loudly if requested.
export HF_HUB_ENABLE_HF_TRANSFER=0

python - "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
p = snapshot_download(
    "RukawaY/gs_scenes", repo_type="dataset", local_dir=dest, max_workers=8,
    allow_patterns=["train/**", "val/**", "*.scene_dataset_config.json", "README.md"],
)
print("downloaded to", p)
PY

echo
echo "train: $(ls "$DEST/train" 2>/dev/null | wc -l) scenes"
echo "val:   $(ls "$DEST/val" 2>/dev/null | wc -l) scenes"
