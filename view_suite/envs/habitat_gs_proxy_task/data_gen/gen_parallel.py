"""Generate the Habitat-GS proxy-task corpus, one worker process per GPU.

Resumable: a scene whose marker file exists is skipped, so an interrupted run picks up
where it stopped instead of re-rendering. Each scene writes its own shard; shards are
concatenated at the end. That is deliberate -- 129 scenes appending to three shared
JSONL files from 8 processes is a corrupted-line generator.

    VIEWSUITE_ROOT=$PWD PYTHONPATH=$PWD ~/miniconda3/envs/habitat-gs/bin/python \
      -m view_suite.envs.habitat_gs_proxy_task.data_gen.gen_parallel \
      --out_root=$VIEWSUITE_ROOT/data/viewagent15k_habitat_gs --scenes=all --samples_per_scene=24 --n_gpus=8
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from typing import List, Optional

import fire

from view_suite.envs.habitat_gs_proxy_task.data_gen.generate_data import (
    GenConfig, run_scene, write_jsonl,
)
from view_suite.habitat_gs.scene_list import default_root, parse_subset

TASK_FILES = ("path_to_view", "view_to_path", "interactive_view_planning")


def _worker(args) -> tuple:
    scene_id, gpu, cfg_kwargs = args
    cfg = GenConfig(**cfg_kwargs)
    shard_dir = os.path.join(cfg.out_root, "_shards")
    marker = os.path.join(shard_dir, f"{scene_id}.done")
    if os.path.exists(marker):
        return scene_id, -1, "skipped"
    os.makedirs(shard_dir, exist_ok=True)
    t0 = time.time()
    try:
        rows = run_scene(scene_id, cfg, gpu=gpu)
    except Exception as e:                      # one bad scene must not kill the run
        return scene_id, 0, f"ERROR {type(e).__name__}: {e}"
    n = len(rows["forward"])
    for key, base in zip(("forward", "inverse", "active_explore"), TASK_FILES):
        with open(os.path.join(shard_dir, f"{scene_id}.{base}.jsonl"), "w") as f:
            for r in rows[key]:
                f.write(json.dumps(r) + "\n")
    open(marker, "w").close()
    return scene_id, n, f"{time.time() - t0:.0f}s"


def run(
    out_root: str = "data/viewagent15k_habitat_gs",
    root: Optional[str] = None,
    scenes: str = "all",
    samples_per_scene: int = 24,
    n_gpus: int = 8,
    width: int = 512,
    height: int = 512,
    fov: float = 90.0,
    pitch_limit_deg: float = 60.0,
    eye_height_m: float = 1.5,
    seed: int = 0,
) -> None:
    root = root or default_root()
    scene_ids: List[str] = parse_subset(scenes, root=root)
    cfg_kwargs = dict(out_root=out_root, root=root, samples_per_scene=samples_per_scene,
                      width=width, height=height, fov=fov, seed=seed,
                      pitch_limit_deg=pitch_limit_deg, eye_height_m=eye_height_m)
    os.makedirs(out_root, exist_ok=True)

    jobs = [(s, i % max(1, n_gpus), cfg_kwargs) for i, s in enumerate(scene_ids)]
    print(f"{len(jobs)} scenes over {n_gpus} GPU(s), {samples_per_scene} samples each")

    # spawn, not fork: a forked process inherits the parent's (absent) GL state and
    # habitat's context handling does not survive it.
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(processes=n_gpus) as pool:
        for scene_id, n, note in pool.imap_unordered(_worker, jobs):
            done += 1
            print(f"[{done}/{len(jobs)}] {scene_id:28s} {n:4d} samples  {note}", flush=True)

    # Concatenate shards in a deterministic order.
    shard_dir = os.path.join(out_root, "_shards")
    for base in TASK_FILES:
        out = os.path.join(out_root, f"{base}.jsonl")
        with open(out, "w") as w:
            for s in sorted(scene_ids):
                p = os.path.join(shard_dir, f"{s}.{base}.jsonl")
                if os.path.exists(p):
                    with open(p) as r:
                        w.write(r.read())
        n = sum(1 for _ in open(out))
        print(f"{out}: {n} rows")


if __name__ == "__main__":
    fire.Fire(run)
