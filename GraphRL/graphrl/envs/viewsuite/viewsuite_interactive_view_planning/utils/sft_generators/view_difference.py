"""View difference: predict number of navigation actions between two views."""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from graphrl.traj_to_sft.utils.base_graph import BaseGraph

from .prompts import _VIEW_DIFFERENCE_PROMPTS
from .helpers import (
    _pick_prompt, _resolve_node_image, _resolve_top_down,
    _get_scene_id, _parse_individual_actions,
    _get_scene_node_ids, _sample_paths_for_scene, _parse_sample_per_scene,
)

logger = logging.getLogger(__name__)

_OVERSAMPLE_FACTOR = 5


def generate_view_difference(
    graph: BaseGraph,
    images_dir: Path,
    output_dir: Path,
    min_path_len: int = 2,
    max_path_len: int = 5,
    sample_per_scene: int = 15,
    viewsuite_15k_dir: Optional[str] = None,
    rng: Optional[random.Random] = None,
    balanced_sampling: bool = True,
) -> List[Dict[str, Any]]:
    """
    Per scene: given two views sampled from a path, predict the number of
    individual actions between them (numerical fill-in-the-blank).

    When *balanced_sampling* is True, oversamples candidates and then
    draws evenly across different action-count buckets within each scene.

    Args:
        sample_per_scene: max samples per scene (int).

    Output: ShareGPT messages + images.
    """
    if rng is None:
        rng = random.Random()

    target = _parse_sample_per_scene(sample_per_scene)
    scene_nodes = _get_scene_node_ids(graph)
    copied: Set[str] = set()
    ds = "view_difference"
    all_records: List[Dict[str, Any]] = []

    for scene_id, node_ids in scene_nodes.items():
        n_candidates = target * _OVERSAMPLE_FACTOR if balanced_sampling else target
        paths = _sample_paths_for_scene(
            graph, node_ids, min_path_len, max_path_len, n_candidates, rng,
        )

        # ── build candidate records ──
        candidates: List[tuple] = []  # (num_steps, record_dict)

        for path in paths:
            num_edges = len(path)
            if num_edges < 1:
                continue

            # Pick two distinct node positions along the path (start < end)
            all_node_ids = [path[0]["from_id"]] + [step["to_id"] for step in path]
            start_idx = rng.randint(0, num_edges - 1)
            end_idx = rng.randint(start_idx + 1, num_edges)

            from_id = all_node_ids[start_idx]
            to_id = all_node_ids[end_idx]
            # Count individual atomic actions across the sub-path edges
            sub_path = path[start_idx:end_idx]
            num_steps = sum(len(_parse_individual_actions(step["action"])) for step in sub_path)

            from_img = _resolve_node_image(
                graph, from_id, images_dir, output_dir, ds, copied,
            )
            to_img = _resolve_node_image(
                graph, to_id, images_dir, output_dir, ds, copied,
            )
            if not from_img or not to_img:
                continue

            images: List[str] = [from_img, to_img]

            top_down = _resolve_top_down(
                scene_id, viewsuite_15k_dir, output_dir, ds, copied,
            )
            if top_down:
                images.append(top_down)
                user_content = (
                    f"Given view 1 <image>, view 2 <image>, "
                    f"and a top-down reference <image>, "
                    f"how many navigation actions are needed to go from view 1 to view 2?"
                )
            else:
                user_content = (
                    f"Given view 1 <image> and view 2 <image>, "
                    f"how many navigation actions are needed to go from view 1 to view 2?"
                )

            record = {
                "messages": [
                    {"role": "system", "content": _pick_prompt(_VIEW_DIFFERENCE_PROMPTS, rng)},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": f"<action>answer({num_steps})</action>"},
                ],
                "images": images,
            }
            candidates.append((num_steps, record))

        # ── balanced sampling across action-count buckets ──
        records: List[Dict[str, Any]] = []
        if not balanced_sampling:
            records = [rec for _, rec in candidates[:target]]
        else:
            buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for ns, rec in candidates:
                buckets[ns].append(rec)

            for recs in buckets.values():
                rng.shuffle(recs)

            sorted_keys = sorted(buckets.keys())
            per_bucket = max(1, target // len(sorted_keys)) if sorted_keys else 0
            for k in sorted_keys:
                records.extend(buckets[k][:per_bucket])
                buckets[k] = buckets[k][per_bucket:]
            if len(records) < target:
                leftovers = []
                for k in sorted_keys:
                    leftovers.extend(buckets[k])
                rng.shuffle(leftovers)
                records.extend(leftovers[: target - len(records)])

            records = records[:target]

        logger.info(
            "[view_difference] scene %s: %d candidates → %d selected (target %d)",
            scene_id, len(candidates), len(records), target,
        )
        all_records.extend(records)

    rng.shuffle(all_records)
    return all_records
