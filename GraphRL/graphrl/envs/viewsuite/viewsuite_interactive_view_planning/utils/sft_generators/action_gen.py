"""Action generation: predict action sequence from initial + target views."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from graphrl.traj_to_sft.utils.base_graph import BaseGraph

from .constants import _STEP_TRANSLATION, _STEP_ROTATION
from .prompts import _ACTION_GEN_PROMPTS
from .helpers import (
    _pick_prompt, _concat_actions, _resolve_node_image, _resolve_top_down,
    _get_scene_id, _path_action_counter, _select_balanced_paths,
    _get_scene_node_ids, _sample_paths_for_scene, _parse_sample_per_scene,
    OVERSAMPLE_FACTOR, DEFAULT_MAX_ACTION_RATIO,
)

logger = logging.getLogger(__name__)


def generate_action_gen(
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
    Sample paths per scene, show initial + target views, predict action sequence.

    When *balanced_sampling* is True, oversamples paths and selects a
    subset with balanced action distribution within each scene.

    Args:
        sample_per_scene: max samples per scene (int).

    Output: ShareGPT messages + images.
    """
    if rng is None:
        rng = random.Random()

    target = _parse_sample_per_scene(sample_per_scene)
    scene_nodes = _get_scene_node_ids(graph)
    copied: Set[str] = set()
    all_records: List[Dict[str, Any]] = []

    for scene_id, node_ids in scene_nodes.items():
        n_candidates = target * OVERSAMPLE_FACTOR if balanced_sampling else target
        paths = _sample_paths_for_scene(
            graph, node_ids, min_path_len, max_path_len, n_candidates, rng,
        )

        candidates: List[Tuple] = []
        for path in paths:
            from_id = path[0]["from_id"]
            to_id = path[-1]["to_id"]

            from_img = _resolve_node_image(
                graph, from_id, images_dir, output_dir, "action_gen", copied,
            )
            to_img = _resolve_node_image(
                graph, to_id, images_dir, output_dir, "action_gen", copied,
            )
            if not from_img or not to_img:
                continue

            action_str = _concat_actions(path)
            images: List[str] = [from_img, to_img]

            top_down = _resolve_top_down(
                scene_id, viewsuite_15k_dir, output_dir, "action_gen", copied,
            )
            if top_down:
                images.append(top_down)
                user_content = (
                    f"Given the initial view <image>, target view <image>, "
                    f"and a top-down reference <image>, "
                    f"what action sequence should be executed to navigate from "
                    f"the initial view to the target view? "
                    f"(translation step = {_STEP_TRANSLATION} m; "
                    f"rotation step = {_STEP_ROTATION} degrees per step)"
                )
            else:
                user_content = (
                    f"Given the initial view <image> and target view <image>, "
                    f"what action sequence should be executed to navigate from "
                    f"the initial view to the target view? "
                    f"(translation step = {_STEP_TRANSLATION} m; "
                    f"rotation step = {_STEP_ROTATION} degrees per step)"
                )

            record = {
                "messages": [
                    {"role": "system", "content": _pick_prompt(_ACTION_GEN_PROMPTS, rng)},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": f"<action>{action_str}</action>"},
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
            "[action_gen] scene %s: %d candidates → %d selected (target %d)",
            scene_id, len(candidates), len(selected), target,
        )
        all_records.extend(selected)

    rng.shuffle(all_records)
    return all_records
