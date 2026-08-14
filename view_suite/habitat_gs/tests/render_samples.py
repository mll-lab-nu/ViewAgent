"""Render a contact sheet across scenes -- eyeball check for the GS corpus.

Not a pass/fail test (that is sanity_render.py). This exists because the failure modes
that matter here are visual: a point-cloud fallback looks like confetti, an
out-of-manifold viewpoint looks like smeared fog, and a night-lit scene looks nearly
black while being perfectly correct. No scalar separates those three.

    VIEWSUITE_ROOT=$PWD PYTHONPATH=$PWD \
      ~/miniconda3/envs/habitat-gs/bin/python \
      view_suite/habitat_gs/tests/render_samples.py --scenes=8 --views=4
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.getenv("VIEWSUITE_ROOT", os.getcwd()))

from view_suite.habitat_gs.habitat_gs_render import HabitatGSRenderer  # noqa: E402
from view_suite.habitat_gs.scene_list import (  # noqa: E402
    default_root, scene_navmesh, scene_ply, scenes_in_split,
)
from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator  # noqa: E402


def intrinsics(w: int, h: int, hfov_deg: float = 90.0) -> np.ndarray:
    f = 0.5 * w / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def contact_sheet(images, cols: int, pad: int = 4) -> Image.Image:
    if not images:
        raise ValueError("no images")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad,
                              rows * h + (rows + 1) * pad), (24, 24, 24))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eye_height", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/tmp/habitat_gs_samples")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    names = scenes_in_split(a.root, a.split)
    # Spread over the size range and over both id families (interior_* and sceneNN);
    # sampling the alphabetical head would only ever show one of them.
    picked = sorted(rng.sample(names, min(a.scenes, len(names))))
    K = intrinsics(a.size, a.size)

    for name in picked:
        ply = scene_ply(a.root, name, a.split)
        r = HabitatGSRenderer(ply, gpu_device_id=a.gpu, width=a.size, height=a.size,
                              navmesh_path=scene_navmesh(a.root, name, a.split))
        try:
            shots = []
            for _ in range(a.views):
                p = r.sample_navigable_point()
                if p is None:
                    p = np.zeros(3)
                vm = HabitatGSViewManipulator(
                    position=(p[0], p[1] + a.eye_height, p[2]),
                    yaw_deg=rng.choice(range(0, 360, 30)), pitch_deg=0.0)
                img = r.render_image_from_cam_param(K, vm.get_c2w(), a.size, a.size)
                shots.append(Image.fromarray(img))
            sheet = contact_sheet(shots, cols=a.views)
            out = os.path.join(a.out_dir, f"{name}.png")
            sheet.save(out)
            arr = np.asarray(shots[0])
            print(f"{name:32s} ply={os.path.getsize(ply)/1e6:5.0f}MB "
                  f"mean={arr.mean():6.1f} std={arr.std():5.1f} -> {out}")
        finally:
            r.close()
    print(f"\n{len(picked)} contact sheets in {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
