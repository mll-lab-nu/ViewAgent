"""Mixed multi-turn navigation: per-turn MCQ/fill-in with configurable probability."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from graphrl.traj_to_sft.utils.base_graph import BaseGraph

from .constants import _LABELS
from .prompts import _MULTI_TURN_MIX_PROMPTS
from .helpers import (
    _pick_prompt, _clean_action, _resolve_node_image, _resolve_top_down,
    _get_scene_id, _format_pose, _answer_pose,
    _parse_individual_actions, _generate_negative_action_seqs,
    _path_action_counter, _select_balanced_paths,
    _get_scene_node_ids, _sample_paths_for_scene, _parse_sample_per_scene,
    OVERSAMPLE_FACTOR, DEFAULT_MAX_ACTION_RATIO,
)

logger = logging.getLogger(__name__)


def _make_turn_mcq(
    gt_action: str,
    rng: random.Random,
) -> tuple:
    """Convert a single-turn fill-in action into an MCQ turn.

    Returns (question_suffix, assistant_content) where *question_suffix* is
    the A/B/C/D option block to append to the user message.
    Returns (None, None) if negatives cannot be generated.
    """
    gt_actions = _parse_individual_actions(gt_action)
    if not gt_actions:
        return None, None

    negatives = _generate_negative_action_seqs(gt_actions, 3, rng)
    if len(negatives) < 3:
        return None, None

    correct_idx = rng.randint(0, 3)
    options = [" | ".join(neg) for neg in negatives]
    options.insert(correct_idx, _clean_action(gt_action))
    correct_label = _LABELS[correct_idx]

    suffix = (
        f"\nWhich action should be taken next?\n"
        f"A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}"
    )
    answer = f"<action>answer({correct_label})</action>"
    return suffix, answer


def generate_multi_turn_action_gen_mix(
    graph: BaseGraph,
    images_dir: Path,
    output_dir: Path,
    min_path_len: int = 3,
    max_path_len: int = 5,
    sample_per_scene: int = 10,
    mcq_prob: float = 0.5,
    viewsuite_15k_dir: Optional[str] = None,
    rng: Optional[random.Random] = None,
    balanced_sampling: bool = True,
) -> List[Dict[str, Any]]:
    """
    Mixed multi-turn navigation per scene: each non-final turn is independently
    converted to MCQ with probability *mcq_prob*, otherwise kept as
    free-form fill-in.  The final turn (pose answer) is always free-form.

    Args:
        sample_per_scene: max samples per scene (int).
        mcq_prob: probability of MCQ per turn.

    Output: ShareGPT messages + images.
    """
    if rng is None:
        rng = random.Random()

    target = _parse_sample_per_scene(sample_per_scene)
    scene_nodes = _get_scene_node_ids(graph)
    copied: Set[str] = set()
    ds = "multi_turn_action_gen_mix"
    all_records: List[Dict[str, Any]] = []

    for scene_id, node_ids in scene_nodes.items():
        n_candidates = target * OVERSAMPLE_FACTOR if balanced_sampling else target
        paths = _sample_paths_for_scene(
            graph, node_ids, min_path_len, max_path_len, n_candidates, rng,
        )

        candidates: List[Tuple] = []
        for path in paths:
            num_edges = len(path)
            num_turns = num_edges + 1  # includes final answer turn

            initial_id = path[0]["from_id"]
            target_id = path[-1]["to_id"]

            # Verify all nodes have images
            all_node_ids = [initial_id] + [step["to_id"] for step in path]
            if not all(graph.nodes[nid].get("image_paths") for nid in all_node_ids):
                continue

            messages: List[Dict[str, str]] = []
            images: List[str] = []

            # ── system ──
            messages.append({
                "role": "system",
                "content": _pick_prompt(_MULTI_TURN_MIX_PROMPTS, rng),
            })

            # ── turn 1: initial + target views ──
            initial_img = _resolve_node_image(
                graph, initial_id, images_dir, output_dir, ds, copied,
            )
            target_img = _resolve_node_image(
                graph, target_id, images_dir, output_dir, ds, copied,
            )
            if not initial_img or not target_img:
                continue

            images.extend([initial_img, target_img])
            initial_pose = _format_pose(path[0]["from_state"])

            top_down = _resolve_top_down(
                scene_id, viewsuite_15k_dir, output_dir, ds, copied,
            )
            if top_down:
                images.append(top_down)
                first_user = (
                    f"Navigate from the initial view <image> to the target view <image>. "
                    f"Top-down reference: <image>\n"
                    f"{initial_pose}\n"
                    f"Step 1/{num_turns}"
                )
            else:
                first_user = (
                    f"Navigate from the initial view <image> to the target view <image>.\n"
                    f"{initial_pose}\n"
                    f"Step 1/{num_turns}"
                )

            # Decide MCQ for turn 1
            is_mcq = rng.random() < mcq_prob
            if is_mcq:
                suffix, mcq_answer = _make_turn_mcq(path[0]["action"], rng)
                if suffix is None:
                    is_mcq = False  # fallback to fill-in

            if is_mcq:
                messages.append({"role": "user", "content": first_user + suffix})
                messages.append({"role": "assistant", "content": mcq_answer})
            else:
                messages.append({"role": "user", "content": first_user})
                messages.append({
                    "role": "assistant",
                    "content": f"<action>{_clean_action(path[0]['action'])}</action>",
                })

            # ── middle turns ──
            skip = False
            for i in range(1, num_edges):
                current_id = path[i]["from_id"]
                step_num = i + 1

                current_img = _resolve_node_image(
                    graph, current_id, images_dir, output_dir, ds, copied,
                )
                if not current_img:
                    skip = True
                    break

                images.append(current_img)
                current_pose = _format_pose(path[i]["from_state"])
                user_text = (
                    f"format: ok\n{current_pose}\n<image>\nStep {step_num}/{num_turns}"
                )

                is_mcq = rng.random() < mcq_prob
                if is_mcq:
                    suffix, mcq_answer = _make_turn_mcq(path[i]["action"], rng)
                    if suffix is None:
                        is_mcq = False

                if is_mcq:
                    messages.append({"role": "user", "content": user_text + suffix})
                    messages.append({"role": "assistant", "content": mcq_answer})
                else:
                    messages.append({"role": "user", "content": user_text})
                    messages.append({
                        "role": "assistant",
                        "content": f"<action>{_clean_action(path[i]['action'])}</action>",
                    })

            if skip:
                continue

            # ── final turn: target reached (always free-form pose) ──
            target_img_final = _resolve_node_image(
                graph, target_id, images_dir, output_dir, ds, copied,
            )
            if not target_img_final:
                continue

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

            record = {"messages": messages, "images": images}
            candidates.append((_path_action_counter(path), record))

        if balanced_sampling:
            selected = _select_balanced_paths(candidates, target, DEFAULT_MAX_ACTION_RATIO, rng)
        else:
            rng.shuffle(candidates)
            selected = [rec for _, rec in candidates[:target]]

        logger.info(
            "[multi_turn_action_gen_mix] scene %s: %d candidates → %d selected (target %d)",
            scene_id, len(candidates), len(selected), target,
        )
        all_records.extend(selected)

    rng.shuffle(all_records)
    return all_records
