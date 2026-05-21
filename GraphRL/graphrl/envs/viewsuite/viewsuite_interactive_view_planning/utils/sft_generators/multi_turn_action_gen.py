"""Multi-turn action generation: step-by-step navigation."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from graphrl.traj_to_sft.utils.base_graph import BaseGraph

from .constants import _STEP_TRANSLATION, _STEP_ROTATION
from .prompts import _MULTI_TURN_PROMPTS
from .helpers import (
    _pick_prompt, _clean_action, _resolve_node_image, _resolve_top_down,
    _get_scene_id, _format_pose, _answer_pose,
    _path_action_counter, _select_balanced_paths,
    _get_scene_node_ids, _sample_paths_for_scene, _parse_sample_per_scene,
    DEFAULT_MAX_ACTION_RATIO,
)

logger = logging.getLogger(__name__)

DEFAULT_OVERSAMPLE = 10
DEFAULT_RATIO = (6, 2, 2)  # single : mix : multi


# ── path classifiers ─────────────────────────────────────────────────────

def _classify_path(path: List[Dict[str, Any]]) -> str:
    """Classify a path as 'single', 'mix', or 'multi'.

    - single: every edge has exactly 1 action
    - multi:  every edge has >1 action
    - mix:    some edges single, some multi
    """
    has_single = False
    has_multi = False
    for step in path:
        actions = [a.strip() for a in step["action"].split("|") if a.strip()]
        if len(actions) == 1:
            has_single = True
        else:
            has_multi = True
        if has_single and has_multi:
            return "mix"
    if has_multi:
        return "multi"
    return "single"


def _parse_ratio(ratio: str) -> Tuple[int, int, int]:
    """Parse 'A:B:C' string into (A, B, C) ints."""
    parts = [int(x.strip()) for x in ratio.split(":")]
    if len(parts) != 3 or any(p < 0 for p in parts):
        raise ValueError(f"Invalid single_mix_multi_ratio: {ratio!r}, expected 'A:B:C'")
    return (parts[0], parts[1], parts[2])


def _select_bucket(
    pool: List,
    target: int,
    balanced_sampling: bool,
    rng: random.Random,
) -> List:
    """Select *target* items from *pool*, optionally with balanced sampling."""
    if not pool or target <= 0:
        return []
    if balanced_sampling:
        return _select_balanced_paths(pool, target, DEFAULT_MAX_ACTION_RATIO, rng)
    else:
        rng.shuffle(pool)
        return [rec for _, rec in pool[:target]]


def generate_multi_turn_action_gen(
    graph: BaseGraph,
    images_dir: Path,
    output_dir: Path,
    min_path_len: int = 3,
    max_path_len: int = 5,
    sample_per_scene: int = 10,
    viewsuite_15k_dir: Optional[str] = None,
    rng: Optional[random.Random] = None,
    balanced_sampling: bool = True,
    oversample: int = DEFAULT_OVERSAMPLE,
    single_mix_multi_ratio: Optional[str] = "6:2:2",
) -> List[Dict[str, Any]]:
    """
    Multi-turn navigation: step-by-step from initial to target view.

    Sampling is done per scene.  Each scene gets ``sample_per_scene`` samples
    (random in range).  Within each scene the pipeline:
      1. Oversamples ``target * oversample`` candidate paths.
      2. If ``single_mix_multi_ratio`` is set (e.g. ``"6:2:2"``):
         classify paths → allocate by ratio → select within each bucket.
      3. ``balanced_sampling`` controls whether selection uses
         action-balanced scoring or pure random.

    Args:
        sample_per_scene: max samples per scene (int).
        oversample: oversample factor per scene.
        single_mix_multi_ratio: colon-separated ratio string, e.g. "6:2:2".
        balanced_sampling: if True, use action-balanced selection.

    Output: ShareGPT messages + images.
    """
    if rng is None:
        rng = random.Random()

    use_ratio = bool(single_mix_multi_ratio and single_mix_multi_ratio.strip())
    ratio = _parse_ratio(single_mix_multi_ratio) if use_ratio else None

    target = _parse_sample_per_scene(sample_per_scene)
    scene_nodes = _get_scene_node_ids(graph)
    copied: Set[str] = set()
    ds = "multi_turn_action_gen"
    all_records: List[Dict[str, Any]] = []

    for scene_id, node_ids in scene_nodes.items():
        n_candidates = target * oversample
        paths = _sample_paths_for_scene(
            graph, node_ids, min_path_len, max_path_len, n_candidates, rng,
        )

        if use_ratio:
            selected = _select_with_ratio_for_scene(
                graph, paths, ratio, target, balanced_sampling,
                images_dir, output_dir, ds, viewsuite_15k_dir, copied, rng,
                scene_id,
            )
        else:
            selected = _select_flat_for_scene(
                graph, paths, target, balanced_sampling,
                images_dir, output_dir, ds, viewsuite_15k_dir, copied, rng,
                scene_id,
            )

        all_records.extend(selected)

    rng.shuffle(all_records)
    return all_records


def _select_flat_for_scene(
    graph: BaseGraph,
    paths: List,
    num_samples: int,
    balanced_sampling: bool,
    images_dir: Path,
    output_dir: Path,
    ds: str,
    viewsuite_15k_dir: Optional[str],
    copied: Set[str],
    rng: random.Random,
    scene_id: str,
) -> List[Dict[str, Any]]:
    """Flat pool: no bucket splitting, for one scene."""
    candidates: List[Tuple] = []
    for path in paths:
        rec = _build_record(
            graph, path, images_dir, output_dir, ds,
            viewsuite_15k_dir, copied, rng,
        )
        if rec is not None:
            candidates.append((_path_action_counter(path), rec))

    logger.info(
        "[multi_turn_action_gen] scene %s flat: %d candidates (target %d)",
        scene_id, len(candidates), num_samples,
    )

    if balanced_sampling:
        return _select_balanced_paths(candidates, num_samples, DEFAULT_MAX_ACTION_RATIO, rng)
    else:
        rng.shuffle(candidates)
        return [rec for _, rec in candidates[:num_samples]]


def _select_with_ratio_for_scene(
    graph: BaseGraph,
    paths: List,
    ratio: Tuple[int, int, int],
    num_samples: int,
    balanced_sampling: bool,
    images_dir: Path,
    output_dir: Path,
    ds: str,
    viewsuite_15k_dir: Optional[str],
    copied: Set[str],
    rng: random.Random,
    scene_id: str,
) -> List[Dict[str, Any]]:
    """Bucket-based selection with ratio split, for one scene."""
    # ── classify ──
    buckets: Dict[str, List] = {"single": [], "mix": [], "multi": []}
    for p in paths:
        buckets[_classify_path(p)].append(p)

    # ── build records per bucket ──
    bucket_candidates: Dict[str, List[Tuple]] = {"single": [], "mix": [], "multi": []}
    for bname, bpaths in buckets.items():
        for path in bpaths:
            rec = _build_record(
                graph, path, images_dir, output_dir, ds,
                viewsuite_15k_dir, copied, rng,
            )
            if rec is not None:
                bucket_candidates[bname].append((_path_action_counter(path), rec))

    # ── ratio-based allocation with spill ──
    # Buckets whose ratio is exactly 0 are STRICT EXCLUSIONS — spillover
    # never lands in them, even if other buckets fall short. Without this
    # guard, ``ratio="1:0:0"`` (intent: "single-action paths only")
    # silently includes mix/multi paths to hit the target whenever single
    # is scarce, which is the observed bug. Buckets with non-zero ratio
    # still trade quota among themselves via the spill loop.
    ratio_sum = sum(ratio)
    bucket_keys = ["single", "mix", "multi"]
    excluded = {bname for bname, w in zip(bucket_keys, ratio) if w == 0}
    targets = {
        bname: int(num_samples * w / ratio_sum)
        for bname, w in zip(bucket_keys, ratio)
    }
    # Round-off goes to the first non-excluded bucket so the totals match.
    leftover = num_samples - sum(targets.values())
    for bname in bucket_keys:
        if bname not in excluded:
            targets[bname] += leftover
            break

    # Spill shortfalls to next NON-EXCLUDED bucket only.
    order = [bname for bname in bucket_keys if bname not in excluded]
    actual: Dict[str, int] = {bname: 0 for bname in bucket_keys}
    spillover = 0
    for bname in order:
        t = targets[bname] + spillover
        avail = len(bucket_candidates[bname])
        if avail >= t:
            actual[bname] = t
            spillover = 0
        else:
            actual[bname] = avail
            spillover = t - avail

    # Distribute remaining spillover backwards into NON-EXCLUDED buckets.
    if spillover > 0:
        for bname in reversed(order):
            avail = len(bucket_candidates[bname]) - actual[bname]
            take = min(avail, spillover)
            actual[bname] += take
            spillover -= take
            if spillover <= 0:
                break

    logger.info(
        "[multi_turn_action_gen] scene %s ratio: single=%d/%d mix=%d/%d multi=%d/%d (target %d)",
        scene_id,
        actual["single"], len(bucket_candidates["single"]),
        actual["mix"], len(bucket_candidates["mix"]),
        actual["multi"], len(bucket_candidates["multi"]),
        num_samples,
    )

    # ── select within each bucket ──
    selected: List[Dict[str, Any]] = []
    for bname in order:
        picked = _select_bucket(
            bucket_candidates[bname], actual[bname],
            balanced_sampling, rng,
        )
        selected.extend(picked)

    rng.shuffle(selected)
    return selected


def _build_record(
    graph: BaseGraph,
    path: List[Dict[str, Any]],
    images_dir: Path,
    output_dir: Path,
    ds: str,
    viewsuite_15k_dir: Optional[str],
    copied: Set[str],
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """Convert a single path into a ShareGPT record, or None on failure."""
    num_edges = len(path)
    num_turns = num_edges + 1  # includes final answer turn

    initial_id = path[0]["from_id"]
    target_id = path[-1]["to_id"]
    scene_id = _get_scene_id(graph, initial_id)

    # Verify all nodes have images stored in graph
    all_node_ids = [initial_id] + [step["to_id"] for step in path]
    if not all(graph.nodes[nid].get("image_paths") for nid in all_node_ids):
        return None

    messages: List[Dict[str, str]] = []
    images: List[str] = []

    # ── system ──
    messages.append({
        "role": "system",
        "content": _pick_prompt(_MULTI_TURN_PROMPTS, rng),
    })

    # ── turn 1: initial + target views ──
    initial_img = _resolve_node_image(
        graph, initial_id, images_dir, output_dir, ds, copied,
    )
    target_img = _resolve_node_image(
        graph, target_id, images_dir, output_dir, ds, copied,
    )
    if not initial_img or not target_img:
        return None

    images.extend([initial_img, target_img])
    initial_pose = _format_pose(path[0]["from_state"], label="Initial view")

    top_down = _resolve_top_down(
        scene_id, viewsuite_15k_dir, output_dir, ds, copied,
    )
    if top_down:
        images.append(top_down)
        first_user = (
            f"You're in the scene {scene_id}.\n"
            f"Please study the initial view <image>, the target view <image>, and the top-down view <image>.\n"
            f"You start from the initial view. Move toward the target view using actions.\n"
            f"{initial_pose}\n"
            f"Success thresholds: position error <= {_STEP_TRANSLATION}m, rotation error <= {_STEP_ROTATION}\u00b0\n"
            f"Step 1/{num_turns}"
        )
    else:
        first_user = (
            f"You're in the scene {scene_id}.\n"
            f"Please study the initial view <image> and the target view <image>.\n"
            f"You start from the initial view. Move toward the target view using actions.\n"
            f"{initial_pose}\n"
            f"Success thresholds: position error <= {_STEP_TRANSLATION}m, rotation error <= {_STEP_ROTATION}\u00b0\n"
            f"Step 1/{num_turns}"
        )

    messages.append({"role": "user", "content": first_user})
    messages.append({
        "role": "assistant",
        "content": f"<action>{_clean_action(path[0]['action'])}</action>",
    })

    # ── middle turns ──
    for i in range(1, num_edges):
        current_id = path[i]["from_id"]
        step_num = i + 1

        current_img = _resolve_node_image(
            graph, current_id, images_dir, output_dir, ds, copied,
        )
        if not current_img:
            return None

        images.append(current_img)
        current_pose = _format_pose(path[i]["from_state"])
        messages.append({
            "role": "user",
            "content": (
                f"format: ok\n{current_pose}\n<image>\nStep {step_num}/{num_turns}"
            ),
        })
        messages.append({
            "role": "assistant",
            "content": f"<action>{_clean_action(path[i]['action'])}</action>",
        })

    # ── final turn: target reached ──
    target_img_final = _resolve_node_image(
        graph, target_id, images_dir, output_dir, ds, copied,
    )
    if not target_img_final:
        return None

    images.append(target_img_final)
    target_pose = _format_pose(path[-1]["to_state"])
    messages.append({
        "role": "user",
        "content": (
            f"format: ok\n{target_pose}\n<image>\nStep {num_turns}/{num_turns}"
        ),
    })
    messages.append({
        "role": "assistant",
        "content": f"<action>{_answer_pose(graph, target_id)}</action>",
    })

    return {"messages": messages, "images": images}
