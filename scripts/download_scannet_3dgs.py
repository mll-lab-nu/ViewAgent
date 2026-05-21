"""
Download pretrained 3DGS checkpoints (GaussianWorld/scannet_mcmc_1.5M_3dgs)
for scenes referenced in a manifest jsonl.

Each scene on the HF repo has 3 files:
  scene{ID}/cfg.yml
  scene{ID}/ckpts/point_cloud_30000.ply   (~355 MB)
  scene{ID}/stats/val_step30000.json

After download, handler's resolve_scene_gs_ply expects:
  <gs_root>/<scene_id>/ckpts/point_cloud_30000.ply

Usage:
    export HF_TOKEN=hf_xxxx
    python ViewSuite/scripts/download_scannet_3dgs.py \
        --manifest ViewSuite/view_suite/envs/scannet_proxy_task/data_gen/viewsuite_15k_gs_test_manifest.jsonl \
        --gs_root  /root/projects/viewsuite/data/scannet_3dgs_mcmc

Or invoke with no args for defaults.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Sequence

from huggingface_hub import snapshot_download


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.normpath(os.path.join(
    _THIS_DIR, "..",
    "view_suite/envs/scannet_proxy_task/data_gen/viewsuite_15k_gs_test_manifest.jsonl",
))
DEFAULT_GS_ROOT = "/root/projects/viewsuite/data/scannet_3dgs_mcmc"
REPO_ID = "GaussianWorld/scannet_mcmc_1.5M_3dgs"


def _unique_scene_ids(manifest_path: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row["scene_id"]
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def _already_have(gs_root: str, scene_id: str) -> bool:
    ply = os.path.join(gs_root, scene_id, "ckpts", "point_cloud_30000.ply")
    return os.path.exists(ply) and os.path.getsize(ply) > 0


def main(
    manifest: str = DEFAULT_MANIFEST,
    gs_root: str = DEFAULT_GS_ROOT,
    scenes: Sequence[str] | None = None,
    skip_existing: bool = True,
    max_workers: int = 4,
) -> None:
    """
    Args:
        manifest: Path to manifest jsonl (from build_manifest.py).
        gs_root:  Destination root. Each scene goes to <gs_root>/<scene_id>/.
        scenes:   Optional explicit scene_id list. Overrides manifest.
        skip_existing: If True, skip scenes whose PLY already exists.
        max_workers: Parallel file downloads per scene (HF snapshot_download).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[error] HF_TOKEN env var is not set. export HF_TOKEN=hf_...")
        sys.exit(2)

    if scenes is not None:
        scene_ids = list(scenes)
    else:
        if not os.path.exists(manifest):
            print(f"[error] manifest not found: {manifest}")
            sys.exit(2)
        scene_ids = _unique_scene_ids(manifest)

    os.makedirs(gs_root, exist_ok=True)

    total = len(scene_ids)
    print(f"[cfg] repo={REPO_ID}")
    print(f"[cfg] gs_root={gs_root}")
    print(f"[cfg] scenes={total} (from {'args' if scenes else manifest})")
    print(f"[cfg] skip_existing={skip_existing}")

    t0 = time.time()
    n_skipped = n_downloaded = n_failed = 0

    for i, sid in enumerate(scene_ids, 1):
        if skip_existing and _already_have(gs_root, sid):
            n_skipped += 1
            print(f"[{i}/{total}] skip (exists): {sid}")
            continue

        try:
            t = time.time()
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=[f"{sid}/**"],
                local_dir=gs_root,
                token=token,
                max_workers=max_workers,
            )
            n_downloaded += 1
            dt = time.time() - t
            print(f"[{i}/{total}] ok: {sid}  ({dt:.1f}s)")
        except Exception as exc:
            n_failed += 1
            print(f"[{i}/{total}] FAIL: {sid}  err={exc}")

    elapsed = time.time() - t0
    print(
        f"\n[summary] ok={n_downloaded} skip={n_skipped} fail={n_failed}  "
        f"elapsed={elapsed:.1f}s"
    )
    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
