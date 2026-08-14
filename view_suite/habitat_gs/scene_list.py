"""Scene inventory for the Habitat-GS 3DGS corpus.

The corpus (https://huggingface.co/datasets/RukawaY/gs_scenes) ships its own split and
we keep it, rather than re-splitting all 129 scenes the way the AI2-THOR task does:
staying on the upstream boundary means our numbers can be put next to Habitat-GS's own
navigation results without an asterisk.

    train/  110 scenes   -> proxy-task train + eval (scene-disjoint, split here)
    val/     19 scenes   -> proxy-task test, untouched

Layout on disk, both splits identical:

    <root>/<split>/<scene>/<scene>.gs.ply
    <root>/<split>/<scene>/<scene>.navmesh

Scene ids are the directory names and come in two families -- `interior_XXXX_XXXXXX`
(InteriorGS) and `sceneNN` -- which are not interchangeable and must not be parsed for
meaning; treat them as opaque.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

SPLITS = ("train", "val")

# Fraction of the 110 train scenes held out as the proxy-task eval split. Scene-disjoint:
# a scene contributes to exactly one of train/eval, so eval measures transfer to unseen
# rooms rather than to unseen viewpoints of seen rooms.
DEFAULT_EVAL_FRACTION = 0.15


def default_root() -> str:
    """Corpus root, from HABITAT_GS_ROOT or the repo-relative default."""
    env = os.getenv("HABITAT_GS_ROOT")
    if env:
        return env
    return os.path.join(os.getenv("VIEWSUITE_ROOT", "."), "data", "gs_scenes")


def scene_dir(root: str, scene_id: str, split: Optional[str] = None) -> str:
    for s in ([split] if split else SPLITS):
        d = os.path.join(root, s, scene_id)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(
        f"scene {scene_id!r} not found under {root}/{{{','.join(SPLITS)}}}. "
        f"Did you run scripts/download_habitat_gs.sh?"
    )


def scene_ply(root: str, scene_id: str, split: Optional[str] = None) -> str:
    d = scene_dir(root, scene_id, split)
    return os.path.join(d, f"{scene_id}.gs.ply")


def scene_navmesh(root: str, scene_id: str, split: Optional[str] = None) -> str:
    d = scene_dir(root, scene_id, split)
    return os.path.join(d, f"{scene_id}.navmesh")


def scenes_in_split(root: str, split: str) -> List[str]:
    d = os.path.join(root, split)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        # Require both files: a scene missing its navmesh cannot be pose-sampled, and
        # an interrupted download leaves exactly that behind.
        if (os.path.exists(os.path.join(d, name, f"{name}.gs.ply"))
                and os.path.exists(os.path.join(d, name, f"{name}.navmesh"))):
            out.append(name)
    return out


def task_splits(root: str, eval_fraction: float = DEFAULT_EVAL_FRACTION,
                seed: int = 0) -> Dict[str, List[str]]:
    """{'train': [...], 'eval': [...], 'test': [...]} -- scene-disjoint.

    `test` is the corpus's own val/ split. `train` and `eval` partition the corpus's
    train/ split deterministically, so a regenerated dataset lands on the same
    boundary as the one before it.
    """
    import random

    train_pool = scenes_in_split(root, "train")
    test = scenes_in_split(root, "val")

    shuffled = list(train_pool)
    random.Random(seed).shuffle(shuffled)
    n_eval = int(round(len(shuffled) * float(eval_fraction)))
    return {
        "train": sorted(shuffled[n_eval:]),
        "eval": sorted(shuffled[:n_eval]),
        "test": sorted(test),
    }


def parse_subset(spec, root: Optional[str] = None) -> List[str]:
    """Resolve a --scenes argument.

    Accepts: 'all', 'train', 'val'/'test', 'eval', an int (first N of train), a
    comma-separated list of scene ids, or an actual sequence of ids.
    """
    root = root or default_root()
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "default")):
        return task_splits(root)["train"]
    if isinstance(spec, (list, tuple)):
        return list(spec)
    if isinstance(spec, int):
        return scenes_in_split(root, "train")[: max(0, spec)]

    s = str(spec).strip().lower()
    if s == "all":
        return scenes_in_split(root, "train") + scenes_in_split(root, "val")
    if s in ("train", "eval"):
        return task_splits(root)[s]
    if s in ("val", "test"):
        return scenes_in_split(root, "val")
    if s.isdigit():
        return scenes_in_split(root, "train")[: int(s)]
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def split_of(root: str, scene_id: str) -> Optional[str]:
    for s in SPLITS:
        if os.path.isdir(os.path.join(root, s, scene_id)):
            return s
    return None
