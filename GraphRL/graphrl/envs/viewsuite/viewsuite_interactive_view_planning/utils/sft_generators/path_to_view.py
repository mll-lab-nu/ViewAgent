"""Forward dynamics: given initial view + actions, pick resulting view (MCQ)."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from graphrl.traj_to_sft.utils.base_graph import BaseGraph

from .constants import _LABELS, _STEP_TRANSLATION, _STEP_ROTATION
from .prompts import _FORWARD_DYNAMICS_PROMPTS
from .helpers import (
    _pick_prompt, _concat_actions, _resolve_node_image, _resolve_top_down,
    _get_scene_id, _path_action_counter, _select_balanced_paths,
    _get_scene_node_ids, _sample_paths_for_scene, _parse_sample_per_scene,
    OVERSAMPLE_FACTOR, DEFAULT_MAX_ACTION_RATIO,
)

logger = logging.getLogger(__name__)


def generate_path_to_view(
    graph: BaseGraph,
    images_dir: Path,
    output_dir: Path,
    min_path_len: int = 1,
    max_path_len: int = 3,
    sample_per_scene: int = 15,
    viewsuite_15k_dir: Optional[str] = None,
    rng: Optional[random.Random] = None,
    balanced_sampling: bool = True,
) -> List[Dict[str, Any]]:
    """
    Sample paths per scene, show initial view + actions, pick correct resulting view (MCQ).

    Negatives are drawn from the same scene.

    Args:
        sample_per_scene: max samples per scene (int).

    Output: ShareGPT messages + images.
    """
    if rng is None:
        rng = random.Random()

    if graph.num_nodes < 4:
        logger.warning("[path_to_view] Graph has fewer than 4 nodes; skipping.")
        return []

    target = _parse_sample_per_scene(sample_per_scene)
    scene_nodes = _get_scene_node_ids(graph)
    copied: Set[str] = set()
    all_records: List[Dict[str, Any]] = []

    for scene_id, node_ids in scene_nodes.items():
        if len(node_ids) < 4:
            logger.info("[path_to_view] scene %s has < 4 nodes; skipping.", scene_id)
            continue

        n_candidates = target * OVERSAMPLE_FACTOR if balanced_sampling else target
        paths = _sample_paths_for_scene(
            graph, node_ids, min_path_len, max_path_len, n_candidates, rng,
        )

        candidates: List[Tuple] = []
        for path in paths:
            from_id = path[0]["from_id"]
            to_id = path[-1]["to_id"]

            # 3 negatives from the same scene, each with at least one image
            negatives = graph.get_random_nodes(
                3,
                exclude_ids={from_id, to_id},
                rng=rng,
                filter_fn=lambda attrs, _sid=scene_id: (
                    attrs.get("extra", {}).get("scene_id") == _sid
                    and len(attrs.get("image_paths", [])) > 0
                ),
            )
            if len(negatives) < 3:
                continue

            from_img = _resolve_node_image(
                graph, from_id, images_dir, output_dir, "path_to_view", copied,
            )
            correct_img = _resolve_node_image(
                graph, to_id, images_dir, output_dir, "path_to_view", copied,
            )
            if not from_img or not correct_img:
                continue

            neg_imgs: List[str] = []
            skip = False
            for neg in negatives:
                neg_img = _resolve_node_image(
                    graph, neg["id"], images_dir, output_dir, "path_to_view", copied,
                )
                if not neg_img:
                    skip = True
                    break
                neg_imgs.append(neg_img)
            if skip:
                continue

            # Build options: 1 correct + 3 negatives, randomly placed
            correct_idx = rng.randint(0, 3)
            option_imgs = list(neg_imgs)
            option_imgs.insert(correct_idx, correct_img)
            correct_label = _LABELS[correct_idx]

            action_str = "[" + _concat_actions(path) + "]"

            images: List[str] = [from_img]
            top_down = _resolve_top_down(
                scene_id, viewsuite_15k_dir, output_dir, "path_to_view", copied,
            )
            if top_down:
                images.append(top_down)
                user_prefix = (
                    f"Given the initial view <image> and a top-down reference <image>, "
                    f"after you execute the following action sequence "
                    f"(translation step = {_STEP_TRANSLATION} m; "
                    f"rotation step = {_STEP_ROTATION} degrees per step):\n"
                )
            else:
                user_prefix = (
                    f"Given the initial view <image>, "
                    f"after you execute the following action sequence "
                    f"(translation step = {_STEP_TRANSLATION} m; "
                    f"rotation step = {_STEP_ROTATION} degrees per step):\n"
                )

            images.extend(option_imgs)

            user_content = (
                f"{user_prefix}"
                f"{action_str}\n"
                f"which of the following images corresponds to the result?\n"
                f"A. <image>\nB. <image>\nC. <image>\nD. <image>"
            )

            record = {
                "messages": [
                    {"role": "system", "content": _pick_prompt(_FORWARD_DYNAMICS_PROMPTS, rng)},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": f"<action>answer({correct_label})</action>"},
                ],
                "images": images,
            }
            candidates.append((_path_action_counter(path), record))

        if balanced_sampling:
            selected = _select_balanced_paths(candidates, target, DEFAULT_MAX_ACTION_RATIO, rng)
        else:
            rng.shuffle(candidates)
            selected = [rec for _, rec in candidates[:target]]

        logger.info(
            "[path_to_view] scene %s: %d candidates → %d selected (target %d)",
            scene_id, len(candidates), len(selected), target,
        )
        all_records.extend(selected)

    rng.shuffle(all_records)
    return all_records
