"""Filter out degenerate IVP samples where the initial view is ALREADY within
the success threshold of the target view (so the task is trivially solved at
step 0 — the agent can just answer immediately). Such samples inflate both
training reward and eval success and must be removed.

Degeneracy = pose(init) within (pos_thr, ang_thr) of pose(target), using the
same SE(3) + geodesic-angle math the IVP env uses to grade `answer(...)`.

Operates at the SAMPLE level (keyed by sample_id from the IVP jsonl) and applies
the same removal to all three task jsonls, then re-splits scene-disjoint.

Usage:
  python -m view_suite.envs.ai2thor_proxy_task.data_gen.filter_degenerate \
      --data_root=$VIEWSUITE_ROOT/data/viewagent_ai2thor
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np

from view_suite.scannet.utils.pose_utils import c2w_extrinsic_to_se3
from view_suite.envs.scannet_proxy_task.utils.gym_proxy_tool_utils import geodesic_angle_deg
from view_suite.envs.utils.split_jsonl_by_scene import split_jsonl_by_scene

_TASKS = ("path_to_view", "view_to_path", "interactive_view_planning")


def _pose_err(det) -> Tuple[float, float]:
    i = c2w_extrinsic_to_se3(np.array(det["init_view"]["c2w_extrinsics"]), degrees=True)
    t = c2w_extrinsic_to_se3(np.array(det["target_view"]["c2w_extrinsics"]), degrees=True)
    pos = float(np.linalg.norm(np.array(i[:3]) - np.array(t[:3])))
    ang = float(geodesic_angle_deg(np.array(i[3:]), np.array(t[3:])))
    return pos, ang


def run(
    data_root: str,
    pos_thr: float = 0.5,
    ang_thr: float = 30.0,
    ratios: Tuple[float, float, float] = (70, 15, 15),
    seed: int = 42,
    dry_run: bool = False,
):
    ivp = os.path.join(data_root, "interactive_view_planning.jsonl")
    rows = [json.loads(l) for l in open(ivp) if l.strip()]
    degenerate = set()
    for r in rows:
        # honor per-sample success_criteria if present, else the given thresholds
        sc = r.get("success_criteria") or {}
        pt = float(sc.get("pose_distance_threshold", pos_thr))
        at = float(sc.get("angle_threshold", ang_thr))
        pos, ang = _pose_err(r["image_detail"])
        if pos <= pt and ang <= at:
            degenerate.add(r["sample_id"])
    print(f"[degen] {len(degenerate)}/{len(rows)} IVP samples are degenerate "
          f"(init within {pos_thr}m/{ang_thr}deg of target) -> removing from all tasks")
    if dry_run:
        return {"degenerate": len(degenerate), "total": len(rows)}

    for stem in _TASKS:
        src = os.path.join(data_root, f"{stem}.jsonl")
        if not os.path.exists(src):
            print(f"[degen] skip missing {src}"); continue
        allrows = [json.loads(l) for l in open(src) if l.strip()]
        keep = [r for r in allrows if r.get("sample_id") not in degenerate]
        bak = os.path.join(data_root, f"{stem}.predegen.jsonl")
        if not os.path.exists(bak):
            os.rename(src, bak)
        with open(src, "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        stats = split_jsonl_by_scene(src, ratios=ratios, seed=seed)
        print(f"[degen] {stem}: {len(allrows)} -> {len(keep)} rows | split {json.dumps(stats['samples'])}")
    return {"degenerate": len(degenerate), "total": len(rows)}


if __name__ == "__main__":
    import fire
    fire.Fire(run)
