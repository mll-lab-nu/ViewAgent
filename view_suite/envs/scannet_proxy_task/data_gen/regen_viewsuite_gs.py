"""
Regenerate the viewsuite test dataset with gsplat-rendered images.

Input:  data/viewsuite_15k/ + the three test jsonls + a manifest (build_manifest.py)
Output: data/viewsuite_15k_gs_test/ with the same layout but:
  - the 3 jsonl files are copied as-is
  - per scene:      top_down_view.png         copied from source
  - per sample:     meta.json                 copied from source
  - per sample:     initial_view.png + option_*.png   re-rendered via gsplat
                    (camera K/pose read from meta.json, fixed 512x512)

Processing order: samples are sorted by scene_id so we load each scene's 3DGS
PLY once and render all its samples before moving on.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Sequence

import numpy as np
from PIL import Image

# Defaults
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(_THIS_DIR, "viewsuite_15k_gs_test_manifest.jsonl")
DEFAULT_SRC_ROOT = "/root/projects/viewsuite/data/viewsuite_15k"
DEFAULT_OUT_ROOT = "/root/projects/viewsuite/data/viewsuite_15k_gs_test"
DEFAULT_GS_ROOT = "/root/projects/viewsuite/data/scannet_3dgs_mcmc"

RENDER_WIDTH = 512
RENDER_HEIGHT = 512

# meta.json keys for each per-image view. Order matches image filenames.
SAMPLE_IMAGE_TO_META_KEY = {
    "initial_view.png": ("initial",),
    "option_000.png": ("options", 0),
    "option_001.png": ("options", 1),
    "option_002.png": ("options", 2),
    "option_003.png": ("options", 3),
}


def _iter_manifest(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_meta(sample_dir_abs: str) -> dict:
    meta_path = os.path.join(sample_dir_abs, "meta.json")
    with open(meta_path) as f:
        return json.load(f)


def _resolve_view(meta: dict, key: tuple):
    """Pull (K, c2w) for an image key from meta.json."""
    if key[0] == "initial":
        view = meta["initial"]
    elif key[0] == "options":
        view = meta["options"][key[1]]
    else:
        raise KeyError(key)
    K = np.asarray(view["intrinsics"], dtype=np.float64)
    c2w = np.asarray(view["pose_c2w"], dtype=np.float64)
    return K, c2w


def _copy_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _save_rgb_png(img_u8: np.ndarray, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    Image.fromarray(img_u8).save(dst)


def main(
    manifest: str = DEFAULT_MANIFEST,
    src_root: str = DEFAULT_SRC_ROOT,
    out_root: str = DEFAULT_OUT_ROOT,
    gs_root: str = DEFAULT_GS_ROOT,
    jsonls: Sequence[str] | None = None,
    limit: int = 0,
) -> None:
    """
    Args:
        manifest: path to manifest jsonl (from build_manifest.py)
        src_root: source viewsuite root (must contain scene*/sample_*/meta.json)
        out_root: destination for regenerated dataset
        gs_root:  root containing <scene_id>/ckpts/point_cloud_30000.ply
        jsonls:   jsonl filenames (basenames) to copy as-is into out_root.
                  Defaults to the three *_test.jsonl files.
        limit:    if > 0, only process the first `limit` samples (for smoke tests)
    """
    # Lazy import so --help doesn't require torch/gsplat
    from view_suite.scannet.render.gsplat_render import GaussianSplatRenderer
    from view_suite.scannet.utils.path_utils import resolve_scene_gs_ply

    if jsonls is None:
        jsonls = [
            "active_explore_test.jsonl",
            "inverse_dynamics_test.jsonl",
            "forward_dynamics_test.jsonl",
        ]

    if not os.path.exists(manifest):
        print(f"[error] manifest not found: {manifest}")
        sys.exit(2)

    os.makedirs(out_root, exist_ok=True)

    # 1) copy the 3 test jsonls
    for name in jsonls:
        src = os.path.join(src_root, name)
        dst = os.path.join(out_root, name)
        if not os.path.exists(src):
            print(f"[warn] jsonl missing: {src} (skipped)")
            continue
        _copy_file(src, dst)
        print(f"[jsonl-copy] {name}")

    # 2) iterate manifest, grouping by scene_id
    rows = list(_iter_manifest(manifest))
    rows.sort(key=lambda r: (r["scene_id"], r["sample_dir"]))
    if limit > 0:
        rows = rows[:limit]

    total = len(rows)
    n_scenes_loaded = 0
    n_samples_done = 0
    n_img_rendered = 0
    copied_topdowns: set[str] = set()

    renderer = None
    cur_scene: str | None = None

    t_all = time.time()
    for idx, row in enumerate(rows, 1):
        scene_id = row["scene_id"]
        sample_dir_rel = row["sample_dir"]
        sample_dir_src = os.path.join(src_root, sample_dir_rel)
        sample_dir_dst = os.path.join(out_root, sample_dir_rel)

        # Load scene if changed
        if scene_id != cur_scene:
            if renderer is not None:
                try:
                    renderer.release()
                except Exception:
                    pass
                renderer = None
            ply = resolve_scene_gs_ply(gs_root, scene_id)
            t0 = time.time()
            renderer = GaussianSplatRenderer(ply)
            print(
                f"[scene] {scene_id}  loaded in {time.time() - t0:.1f}s "
                f"({n_scenes_loaded + 1} scenes, row {idx}/{total})"
            )
            cur_scene = scene_id
            n_scenes_loaded += 1

            # Copy top_down once per scene
            td_src = os.path.join(src_root, scene_id, "top_down_view.png")
            td_dst = os.path.join(out_root, scene_id, "top_down_view.png")
            if os.path.exists(td_src) and scene_id not in copied_topdowns:
                _copy_file(td_src, td_dst)
                copied_topdowns.add(scene_id)

        # Copy meta.json
        meta_src = os.path.join(sample_dir_src, "meta.json")
        meta_dst = os.path.join(sample_dir_dst, "meta.json")
        if not os.path.exists(meta_src):
            print(f"[warn] missing meta.json: {meta_src} (skip sample)")
            continue
        _copy_file(meta_src, meta_dst)
        meta = _read_meta(sample_dir_src)

        # Render each image
        for img_name in row["images"]:
            key = SAMPLE_IMAGE_TO_META_KEY.get(img_name)
            if key is None:
                # Unknown image name; skip silently
                continue
            try:
                K, c2w = _resolve_view(meta, key)
            except (KeyError, IndexError) as exc:
                print(f"[warn] {sample_dir_rel}/{img_name}: meta lookup failed ({exc})")
                continue
            img_u8 = renderer.render_image_from_cam_param(
                K, c2w, width=RENDER_WIDTH, height=RENDER_HEIGHT
            )
            _save_rgb_png(img_u8, os.path.join(sample_dir_dst, img_name))
            n_img_rendered += 1

        n_samples_done += 1
        if n_samples_done % 50 == 0:
            dt = time.time() - t_all
            print(
                f"[progress] samples={n_samples_done}/{total}  "
                f"images={n_img_rendered}  elapsed={dt:.1f}s "
                f"({n_img_rendered / max(dt, 1e-3):.1f} img/s)"
            )

    if renderer is not None:
        try:
            renderer.release()
        except Exception:
            pass

    dt = time.time() - t_all
    print(
        f"\n[done] samples={n_samples_done}  scenes={n_scenes_loaded}  "
        f"images={n_img_rendered}  elapsed={dt:.1f}s  "
        f"out={out_root}"
    )


if __name__ == "__main__":
    import fire
    fire.Fire(main)