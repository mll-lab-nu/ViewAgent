"""Pre-download AI2-THOR scene assets used by the proxy-task suite.

Opens a single Controller and resets through every scene in the requested
subset, letting AI2-THOR cache:
  - the CloudRendering binary (~800 MB, one-time per machine)
  - each scene's procedural assets (geometry, materials, textures)
to `~/.ai2thor/` so subsequent service / data_gen runs never hit a cold boot.

Usage:
    conda activate viewsuite
    python -m view_suite.ai2thor.pre_download_scenes
    python -m view_suite.ai2thor.pre_download_scenes --scenes=all
    python -m view_suite.ai2thor.pre_download_scenes --scenes=kitchen --gpu=1
    python -m view_suite.ai2thor.pre_download_scenes --scenes="FloorPlan1,FloorPlan305"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from typing import Sequence

from view_suite.ai2thor.scene_list import parse_subset


def _auto_gpu() -> int:
    """Pick the first visible GPU index (best-effort)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def run(
    scenes: str = "all",
    gpu: int | None = None,
    width: int = 256,
    height: int = 256,
    fov: float = 90.0,
    server_timeout: int = 300,
    server_start_timeout: int = 300,
) -> None:
    """Pre-cache AI2-THOR assets for a given scene subset."""
    if gpu is None:
        gpu = _auto_gpu()
        print(f"[pre-dl] auto-detected gpu={gpu}")

    scene_ids: Sequence[str] = parse_subset(scenes)
    print(f"[pre-dl] scenes={len(scene_ids)} (spec={scenes!r})")
    if not scene_ids:
        print("[pre-dl] nothing to do (empty scene list)")
        return

    # Bind the whole process tree to the selected GPU before importing ai2thor
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import ai2thor.controller
    from ai2thor.platform import CloudRendering

    print(f"[pre-dl] creating Controller on gpu={gpu} ...")
    ctrl = ai2thor.controller.Controller(
        platform=CloudRendering,
        agentMode="default",
        scene=scene_ids[0],
        width=width,
        height=height,
        fieldOfView=fov,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        gpu_device=gpu,
        server_timeout=server_timeout,
        server_start_timeout=server_start_timeout,
    )
    print("[pre-dl] controller ready")

    ok = fail = 0
    t_all = time.time()
    try:
        for i, sid in enumerate(scene_ids, 1):
            t0 = time.time()
            try:
                ctrl.reset(scene=sid)
                ok += 1
                print(f"  [{i}/{len(scene_ids)}] {sid}  ({time.time() - t0:.1f}s)")
            except Exception as exc:
                fail += 1
                print(f"  [{i}/{len(scene_ids)}] {sid}  FAIL: {exc}")
    finally:
        ctrl.stop()

    elapsed = time.time() - t_all
    print(
        f"\n[pre-dl] done: ok={ok} fail={fail}  elapsed={elapsed:.1f}s  "
        f"(cache: ~/.ai2thor/)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="all",
                    help="subset spec (see scene_list.parse_subset); "
                         "default 'all' = 120 iTHOR FloorPlans")
    ap.add_argument("--gpu", type=int, default=None, help="GPU index (default: auto)")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--fov", type=float, default=90.0)
    args = ap.parse_args()
    run(scenes=args.scenes, gpu=args.gpu, width=args.width, height=args.height, fov=args.fov)


if __name__ == "__main__":
    main()
