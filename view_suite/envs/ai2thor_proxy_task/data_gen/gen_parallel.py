"""Parallel AI2-THOR data generation across multiple GPUs.

Splits the scene list round-robin over N GPUs, runs one ``generate_data`` process
per GPU (each into its own shard dir), then merges the shards into ``out_root``:
scene image dirs are moved in (scene ids are disjoint across shards) and the
per-task jsonls are concatenated.

Usage:
  conda activate viewagent_thor
  python -m view_suite.envs.ai2thor_proxy_task.data_gen.gen_parallel \
      --out_root=$VIEWSUITE_ROOT/data/viewagent15k_ai2thor_full \
      --scenes=all --samples_per_scene=16 --n_gpus=8
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import List, Optional

from view_suite.ai2thor.scene_list import parse_subset

_TASK_JSONLS = ("path_to_view.jsonl", "view_to_path.jsonl", "interactive_view_planning.jsonl")


def run(
    out_root: str,
    scenes: str = "all",
    samples_per_scene: int = 16,
    n_gpus: int = 8,
    gpus: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    fov: float = 90.0,
    seed: int = 0,
    max_dominant_pixel_frac: float = 0.8,
    keep_shards: bool = False,
):
    scene_ids = parse_subset(scenes)
    if gpus is None:
        gpu_list: List[int] = list(range(n_gpus))
    elif isinstance(gpus, (list, tuple)):
        gpu_list = [int(g) for g in gpus]
    else:
        gpu_list = [int(g) for g in str(gpus).split(",") if str(g).strip() != ""]
    ng = len(gpu_list)
    chunks = [scene_ids[i::ng] for i in range(ng)]  # round-robin
    os.makedirs(out_root, exist_ok=True)
    print(f"[gen_parallel] {len(scene_ids)} scenes over {ng} GPUs "
          f"({samples_per_scene}/scene) -> {out_root}")

    procs = []
    for gpu, chunk in zip(gpu_list, chunks):
        if not chunk:
            continue
        shard = os.path.join(out_root, f"_shard{gpu}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [
            "python", "-m", "view_suite.envs.ai2thor_proxy_task.data_gen.generate_data",
            f"--out_root={shard}",
            f"--scenes={','.join(chunk)}",
            f"--samples_per_scene={samples_per_scene}",
            f"--width={width}", f"--height={height}", f"--fov={fov}",
            f"--seed={seed}", f"--max_dominant_pixel_frac={max_dominant_pixel_frac}",
            "--gpu_id=0",  # CUDA_VISIBLE_DEVICES already masks to this GPU
        ]
        log = open(os.path.join(out_root, f"_shard{gpu}.log"), "w")
        print(f"  GPU {gpu}: {len(chunk)} scenes -> {shard}")
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

    t0 = time.time()
    failed = []
    for gpu, p, log in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            failed.append(gpu)
            print(f"  [GPU {gpu}] FAILED rc={rc} (see _shard{gpu}.log)")
    if failed:
        raise RuntimeError(f"generation failed on GPUs {failed}")
    print(f"[gen_parallel] all shards done in {time.time()-t0:.1f}s; merging…")

    # Merge: concat jsonls, move scene dirs.
    for name in _TASK_JSONLS:
        with open(os.path.join(out_root, name), "w") as out:
            for gpu in gpu_list:
                sp = os.path.join(out_root, f"_shard{gpu}", name)
                if os.path.exists(sp):
                    out.write(open(sp).read())
    n_scene_dirs = 0
    for gpu in gpu_list:
        shard = os.path.join(out_root, f"_shard{gpu}")
        if not os.path.isdir(shard):
            continue
        for entry in os.listdir(shard):
            src = os.path.join(shard, entry)
            if entry.startswith("FloorPlan") and os.path.isdir(src):
                shutil.move(src, os.path.join(out_root, entry))
                n_scene_dirs += 1
        if not keep_shards:
            shutil.rmtree(shard, ignore_errors=True)

    counts = {n: sum(1 for _ in open(os.path.join(out_root, n))) for n in _TASK_JSONLS}
    print(f"[gen_parallel] merged {n_scene_dirs} scene dirs; rows: {counts}")
    return counts


if __name__ == "__main__":
    import fire
    fire.Fire(run)
