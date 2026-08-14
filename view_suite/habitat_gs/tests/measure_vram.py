"""Measure the VRAM a resident Habitat-GS scene costs, per worker process.

The service runs one scene per worker process, so the number that decides how many
workers fit on a GPU is this process's total GPU footprint -- the EGL/CUDA context plus
the gaussian cloud -- not the marginal cost of a second scene in the same process.

Gaussian clouds are much heavier than the ScanNet meshes the existing sizing came from
(~350 MB of .gs.ply vs 42-580 MB of mesh, but the mesh only cost 248-401 MiB resident),
so the ScanNet rule of ~25 scenes/GPU cannot be assumed to carry over.

    VIEWSUITE_ROOT=$PWD PYTHONPATH=$PWD \
      ~/miniconda3/envs/habitat-gs/bin/python \
      view_suite/habitat_gs/tests/measure_vram.py --scenes=6 --gpu=0
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.getenv("VIEWSUITE_ROOT", os.getcwd()))

from view_suite.habitat_gs.scene_list import (  # noqa: E402
    default_root, scene_navmesh, scene_ply, scenes_in_split,
)


def gpu_mem_used(gpu: int) -> int:
    """MiB in use on one GPU.

    Whole-device, not per-process: habitat draws through EGL, and a graphics context
    does not appear in `nvidia-smi --query-compute-apps` at all (that table is CUDA
    compute only). Querying per-pid returns 0 for every worker and makes a renderer
    holding gigabytes look free. So this needs an otherwise-idle GPU to be meaningful
    -- check the baseline it prints.
    """
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, text=True, check=False,
    ).stdout.strip()
    return int(out.splitlines()[0]) if out else 0


CHILD = r"""
import os, sys, time
sys.path.insert(0, os.environ["VIEWSUITE_ROOT"])
from view_suite.habitat_gs.habitat_gs_render import HabitatGSRenderer
from view_suite.habitat_gs.scene_list import scene_ply, scene_navmesh
import numpy as np
root, scene, split, gpu = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
print("STAGE idle", flush=True); time.sleep(4)
r = HabitatGSRenderer(scene_ply(root, scene, split), gpu_device_id=gpu,
                      width=512, height=512,
                      navmesh_path=scene_navmesh(root, scene, split))
K = np.array([[256.0,0,256.0],[0,256.0,256.0],[0,0,1.0]])
p = r.sample_navigable_point()
if p is None: p = np.zeros(3)          # `or` on an ndarray raises; test for None
c2w = np.eye(4); c2w[:3,3] = p; c2w[1,3] += 1.5
r.render_image_from_cam_param(K, c2w, 512, 512)   # force the full pipeline to allocate
print("STAGE loaded", flush=True); time.sleep(6)
"""


def measure(root: str, scene: str, split: str, gpu: int):
    env = dict(os.environ, VIEWSUITE_ROOT=os.getenv("VIEWSUITE_ROOT", os.getcwd()),
               HABITAT_SIM_LOG="quiet", MAGNUM_LOG="quiet",
               CUDA_VISIBLE_DEVICES=str(gpu))
    baseline = gpu_mem_used(gpu)
    p = subprocess.Popen(
        [sys.executable, "-c", CHILD, root, scene, split, "0"],
        # stderr kept: discarding it once already turned a crashing child into a
        # confident "0 MiB per scene".
        stdout=subprocess.PIPE, stderr=None, text=True, env=env)
    idle_mib = loaded_mib = 0
    try:
        for line in p.stdout:
            if line.startswith("STAGE idle"):
                time.sleep(2)
                idle_mib = gpu_mem_used(gpu) - baseline
            elif line.startswith("STAGE loaded"):
                time.sleep(3)
                loaded_mib = gpu_mem_used(gpu) - baseline
                break
    finally:
        p.terminate()
        p.wait(timeout=30)
    return max(0, idle_mib), max(0, loaded_mib)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    names = scenes_in_split(a.root, a.split)
    # Span the size range rather than taking the first N alphabetically: per-scene cost
    # tracks cloud size, so a sample from one end of the range predicts nothing.
    sized = sorted(((os.path.getsize(scene_ply(a.root, n, a.split)) / 1e6, n) for n in names))
    step = max(1, len(sized) // a.scenes)
    picks = sized[::step][: a.scenes]

    print(f"{'scene':32s} {'ply MB':>8s} {'ctx MiB':>9s} {'total MiB':>10s} {'scene MiB':>10s}")
    rows = []
    for mb, n in picks:
        idle, loaded = measure(a.root, n, a.split, a.gpu)
        rows.append((mb, idle, loaded))
        print(f"{n:32s} {mb:8.0f} {idle:9d} {loaded:10d} {loaded - idle:10d}")

    if rows:
        worst = max(r[2] for r in rows)
        print(f"\nworst-case process footprint: {worst} MiB")
        for n_workers in (10, 15, 20, 25):
            print(f"  {n_workers:2d} workers/GPU -> {worst * n_workers / 1024:6.1f} GiB "
                  f"of 143 GiB {'OK' if worst * n_workers / 1024 < 120 else 'TOO MUCH'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
