"""Split the generated JSONL into train / dev / test along scene boundaries.

Not the generic ratio splitter: this corpus ships its own train/val division and we keep
it, so the test split is exactly the upstream `val/` scenes and our numbers can be put
next to Habitat-GS's own navigation results without an asterisk. `train/` is then split
scene-disjointly into train and dev.

Scene-disjoint throughout, so dev and test measure transfer to unseen rooms rather than
to unseen viewpoints of rooms the model already trained on.

    VIEWSUITE_ROOT=$PWD PYTHONPATH=$PWD python -m \
      view_suite.envs.habitat_gs_proxy_task.data_gen.split_by_corpus \
      --data_root=$VIEWSUITE_ROOT/data/viewagent_habitat_gs
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import fire

from view_suite.habitat_gs.scene_list import DEFAULT_EVAL_FRACTION, default_root, task_splits

TASKS = ("path_to_view", "view_to_path", "interactive_view_planning")


def run(data_root: str, root: str = None, eval_fraction: float = DEFAULT_EVAL_FRACTION,
        seed: int = 0) -> None:
    root = root or default_root()
    splits = task_splits(root, eval_fraction=eval_fraction, seed=seed)
    # '_eval' is the suffix split_jsonl_by_scene writes and the training configs read.
    scene_to_split = {}
    for name, out_name in (("train", "train"), ("eval", "eval"), ("test", "test")):
        for s in splits[name]:
            scene_to_split[s] = out_name
    print(f"scenes: train={len(splits['train'])} eval={len(splits['eval'])} "
          f"test={len(splits['test'])}")

    for task in TASKS:
        src = os.path.join(data_root, f"{task}.jsonl")
        if not os.path.exists(src):
            print(f"  {task}: missing, skipped")
            continue
        buckets: Dict[str, List[str]] = {"train": [], "eval": [], "test": []}
        unknown = 0
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sid = json.loads(line)["scene_id"]
                where = scene_to_split.get(sid)
                if where is None:
                    unknown += 1
                    continue
                buckets[where].append(line)
        for where, rows in buckets.items():
            out = os.path.join(data_root, f"{task}_{where}.jsonl")
            with open(out, "w") as f:
                f.write("\n".join(rows) + ("\n" if rows else ""))
        counts = " ".join(f"{k}={len(v)}" for k, v in buckets.items())
        print(f"  {task}: {counts}" + (f"  (dropped {unknown} rows from unknown scenes)"
                                       if unknown else ""))


if __name__ == "__main__":
    fire.Fire(run)
