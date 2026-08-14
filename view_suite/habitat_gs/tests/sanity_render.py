"""First check on a new Habitat-GS box: does one scene actually render?

Run this BEFORE standing up the service or generating any data. A 3DGS stage that fails
to load still returns frames -- black, or confetti from the point-cloud fallback -- and
the service answers HTTP 200 either way, so a broken renderer looks exactly like a
working one until an IVP metric reads 0.000 for reasons nobody can find.

    VIEWSUITE_ROOT=$PWD PYTHONPATH=$PWD \
      ~/miniconda3/envs/habitat-gs/bin/python \
      view_suite/habitat_gs/tests/sanity_render.py --scenes=3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getenv("VIEWSUITE_ROOT", os.getcwd()))

from view_suite.habitat_gs.habitat_gs_render import HabitatGSRenderer  # noqa: E402
from view_suite.habitat_gs.scene_list import (  # noqa: E402
    default_root, scene_navmesh, scene_ply, scenes_in_split,
)
from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator  # noqa: E402


def intrinsics(width: int, height: int, hfov_deg: float = 90.0) -> np.ndarray:
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def check_one(root: str, scene_id: str, split: str, gpu: int, out_dir: str) -> bool:
    ply, navmesh = scene_ply(root, scene_id, split), scene_navmesh(root, scene_id, split)
    size_mb = os.path.getsize(ply) / 1e6
    r = HabitatGSRenderer(ply, gpu_device_id=gpu, width=512, height=512,
                          navmesh_path=navmesh)
    try:
        pf_loaded = r.pathfinder is not None and r.pathfinder.is_loaded
        start = r.sample_navigable_point()
        if start is None:
            print(f"  {scene_id}: NO NAVMESH -- falling back to the origin")
            start = np.zeros(3)
        # Eye height above the walkable surface, looking level.
        vm = HabitatGSViewManipulator(position=(start[0], start[1] + 1.5, start[2]),
                                      yaw_deg=0.0, pitch_deg=0.0)

        stds, means = [], []
        for i in range(4):                       # four yaws, so one bad wall cannot pass
            img = r.render_image_from_cam_param(intrinsics(512, 512), vm.get_c2w(), 512, 512)
            stds.append(float(np.asarray(img).std()))
            means.append(float(np.asarray(img).mean()))
            if out_dir:
                from PIL import Image
                os.makedirs(out_dir, exist_ok=True)
                Image.fromarray(img).save(os.path.join(out_dir, f"{scene_id}_{i}.png"))
            for _ in range(3):
                vm.step("e")                     # 3 x 30deg = 90deg between shots

        best_std, best_mean = max(stds), max(means)
        # A dead render is flat (std ~ 0). A loaded-but-wrong one is usually near-black.
        ok = best_std > 5.0 and best_mean > 5.0
        print(f"  {scene_id}: {'OK  ' if ok else 'FAIL'} "
              f"ply={size_mb:.0f}MB navmesh={'yes' if pf_loaded else 'NO'} "
              f"std={min(stds):.1f}..{best_std:.1f} mean={min(means):.1f}..{best_mean:.1f}")
        return ok
    finally:
        r.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", type=int, default=3, help="how many scenes to probe")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_dir", default="/tmp/habitat_gs_sanity")
    a = ap.parse_args()

    names = scenes_in_split(a.root, a.split)[: a.scenes]
    if not names:
        print(f"no scenes under {a.root}/{a.split}")
        return 2
    print(f"probing {len(names)} scene(s) from {a.root}/{a.split} on GPU {a.gpu}")
    results = [check_one(a.root, n, a.split, a.gpu, a.out_dir) for n in names]
    print(f"\n{sum(results)}/{len(results)} scenes rendered non-blank; "
          f"images in {a.out_dir}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
