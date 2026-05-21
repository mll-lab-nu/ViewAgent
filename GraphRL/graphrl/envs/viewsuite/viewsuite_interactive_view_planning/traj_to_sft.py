"""
ViewSuite Interactive View Planning TrajToSFT phase.

Builds a graph from VAGEN rollouts via ``InteractiveViewPlanningGraphBuilder`` and
generates seven LLaMA-Factory datasets:

  action_gen, path_to_view, multi_turn_action_gen,
  multi_turn_action_gen_mcq, multi_turn_action_gen_mix,
  view_difference, view_difference_mcq

All datasets use ShareGPT message format with image references.

Pipeline.yaml::

    traj_to_sft:
      module: graphrl.envs.viewsuite.viewsuite_interactive_view_planning.InteractiveViewPlanningTrajToSFT
      generators: [action_gen, path_to_view, ...]
      viewsuite_15k_dir: ${oc.env:HOME}/projects/viewsuite/data/viewsuite_15k
      action_gen: { min_path_len: 1, max_path_len: 3, sample_per_scene: 15, ... }
      ...
      seed: 42

      # graph_builder config (passed to InteractiveViewPlanningGraphBuilder)
      graph_builder:
        num_workers: 4
        filter: { void_threshold: 0.7, std_threshold: 10.0 }
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from graphrl import TrajToSFTGraphBase
from graphrl.traj_to_sft.utils.base_graph import BaseGraph
from graphrl.traj_to_sft.utils.graph_builder import VagenGraphBuilder
from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.interactive_view_planning_graph_builder import (
    InteractiveViewPlanningGraphBuilder,
)

logger = logging.getLogger(__name__)

_ALL_GENERATORS = [
    "action_gen",
    "path_to_view",
    "multi_turn_action_gen",
    "multi_turn_action_gen_mcq",
    "multi_turn_action_gen_mix",
    "view_difference",
    "view_difference_mcq",
]


def _prune_multi_action_edges(graph: BaseGraph) -> BaseGraph:
    """Return a deep-copied :class:`BaseGraph` with all multi-action edges
    removed. The ON-DISK graph is untouched — pruning is in-memory.

    A multi-action edge is one whose ``obs_str`` (the action label,
    e.g. ``"turn_left | move_forward"``) contains a ``|``. The graph
    builder produces these for transitions that the agent took as a
    single batched action; they're useful for compact graph storage
    but harmful when the SFT generator wants single-primitive turns.
    """
    pruned = BaseGraph()
    pruned._g = graph._g.copy()
    n_before = pruned._g.number_of_edges()
    to_drop = [
        (u, v, k)
        for u, v, k, data in pruned._g.edges(keys=True, data=True)
        if "|" in (data.get("obs_str") or "")
    ]
    for u, v, k in to_drop:
        pruned._g.remove_edge(u, v, k)
    n_after = pruned._g.number_of_edges()
    logger.info(
        "[traj_to_sft] prune_multi_action_edges: dropped %d/%d multi-action edges "
        "(%d single-action edges remain)",
        n_before - n_after, n_before, n_after,
    )
    return pruned

_SHAREGPT_FMT = {
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    },
}


class InteractiveViewPlanningTrajToSFT(TrajToSFTGraphBase):
    """Convert ViewSuite Interactive View Planning VAGEN rollouts → 7 datasets."""

    name = "TrajToSFT(viewsuite_interactive_view_planning)"

    def graph_builder_class(self) -> Type[VagenGraphBuilder]:
        return InteractiveViewPlanningGraphBuilder

    # ── dataset-config hook (RandomActionTrajToSFT overrides this) ────────

    def _dataset_cfg(self) -> Dict[str, Any]:
        """Return the dict used for dataset-generation knobs.

        Default: ``self.config``. Subclasses (e.g. RandomActionTrajToSFT) can
        override to point at a sub-block such as ``self.config["sft"]``.
        """
        return self.config

    # ── dataset generation ────────────────────────────────────────────────

    def generate_datasets(
        self,
        graph: BaseGraph,
        images_dir: Path,
    ) -> Dict[str, Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.utils.sft_generators import (
            generate_action_gen,
            generate_path_to_view,
            generate_multi_turn_action_gen,
            generate_multi_turn_action_gen_mcq,
            generate_multi_turn_action_gen_mix,
            generate_view_difference,
            generate_view_difference_mcq,
        )

        output_dir = self.paths.sft_data
        cfg = self._dataset_cfg()

        # ``prune_multi_action_edges: true`` (default False) drops every
        # edge whose ``obs_str`` (the action label) contains '|', i.e.
        # multi-action transitions like ``turn_left | move_forward``.
        # Path sampling on the pruned graph then naturally yields ONLY
        # single-action paths, so per-turn assistant outputs are
        # guaranteed to be a single primitive — much friendlier to MCQ
        # reasoning annotation, where the distractor is the OPPOSITE
        # primitive and a multi-action GT has no defensible distractor.
        # The on-disk graph dir is NOT modified — the prune is in-memory
        # for this run only. Stats are logged so callers can see how
        # many edges (and therefore how much path diversity) was
        # discarded.
        if cfg.get("prune_multi_action_edges"):
            graph = _prune_multi_action_edges(graph)
        enabled: List[str] = cfg.get("generators", _ALL_GENERATORS)
        action_cfg: Dict[str, Any] = cfg.get("action_gen", {})
        fwd_cfg: Dict[str, Any] = cfg.get("path_to_view", {})
        multi_cfg: Dict[str, Any] = cfg.get("multi_turn_action_gen", {})
        multi_mcq_cfg: Dict[str, Any] = cfg.get("multi_turn_action_gen_mcq", {})
        multi_mix_cfg: Dict[str, Any] = cfg.get("multi_turn_action_gen_mix", {})
        vd_cfg: Dict[str, Any] = cfg.get("view_difference", {})
        vd_mcq_cfg: Dict[str, Any] = cfg.get("view_difference_mcq", {})
        seed: int = cfg.get("seed", 42)
        viewsuite_15k_dir: Optional[str] = cfg.get("viewsuite_15k_dir")

        master_rng = random.Random(seed)

        def child_rng() -> random.Random:
            return random.Random(master_rng.randint(0, 2 ** 32 - 1))

        result: Dict[str, Any] = {}

        if "action_gen" in enabled:
            records = generate_action_gen(
                graph,
                images_dir=images_dir,
                output_dir=output_dir,
                min_path_len=action_cfg.get("min_path_len", 1),
                max_path_len=action_cfg.get("max_path_len", 3),
                sample_per_scene=action_cfg.get("sample_per_scene", 15),
                viewsuite_15k_dir=viewsuite_15k_dir,
                rng=child_rng(),
                balanced_sampling=action_cfg.get("balanced_sampling", True),
            )
            result["action_gen"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] action_gen: %d", self.name, len(records))

        if "path_to_view" in enabled:
            records = generate_path_to_view(
                graph,
                images_dir=images_dir,
                output_dir=output_dir,
                min_path_len=fwd_cfg.get("min_path_len", 1),
                max_path_len=fwd_cfg.get("max_path_len", 3),
                sample_per_scene=fwd_cfg.get("sample_per_scene", 15),
                viewsuite_15k_dir=viewsuite_15k_dir,
                rng=child_rng(),
                balanced_sampling=fwd_cfg.get("balanced_sampling", True),
            )
            result["path_to_view"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] path_to_view: %d", self.name, len(records))

        if "multi_turn_action_gen" in enabled:
            records = generate_multi_turn_action_gen(
                graph,
                images_dir=images_dir,
                output_dir=output_dir,
                min_path_len=multi_cfg.get("min_path_len", 3),
                max_path_len=multi_cfg.get("max_path_len", 5),
                sample_per_scene=multi_cfg.get("sample_per_scene", 10),
                viewsuite_15k_dir=viewsuite_15k_dir,
                rng=child_rng(),
                balanced_sampling=multi_cfg.get("balanced_sampling", True),
                oversample=int(multi_cfg.get("oversample", 10)),
                single_mix_multi_ratio=multi_cfg.get("single_mix_multi_ratio", "6:2:2"),
            )
            result["multi_turn_action_gen"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] multi_turn_action_gen: %d", self.name, len(records))

        if "multi_turn_action_gen_mcq" in enabled:
            records = generate_multi_turn_action_gen_mcq(
                graph,
                images_dir=images_dir,
                output_dir=output_dir,
                min_path_len=multi_mcq_cfg.get("min_path_len", 1),
                max_path_len=multi_mcq_cfg.get("max_path_len", 3),
                sample_per_scene=multi_mcq_cfg.get("sample_per_scene", 15),
                viewsuite_15k_dir=viewsuite_15k_dir,
                rng=child_rng(),
                balanced_sampling=multi_mcq_cfg.get("balanced_sampling", True),
            )
            result["multi_turn_action_gen_mcq"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] multi_turn_action_gen_mcq: %d", self.name, len(records))

        if "multi_turn_action_gen_mix" in enabled:
            records = generate_multi_turn_action_gen_mix(
                graph,
                images_dir=images_dir,
                output_dir=output_dir,
                min_path_len=multi_mix_cfg.get("min_path_len", 3),
                max_path_len=multi_mix_cfg.get("max_path_len", 5),
                sample_per_scene=multi_mix_cfg.get("sample_per_scene", 10),
                mcq_prob=multi_mix_cfg.get("mcq_prob", 0.5),
                viewsuite_15k_dir=viewsuite_15k_dir,
                rng=child_rng(),
                balanced_sampling=multi_mix_cfg.get("balanced_sampling", True),
            )
            result["multi_turn_action_gen_mix"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] multi_turn_action_gen_mix: %d", self.name, len(records))

        if "view_difference" in enabled:
            records = generate_view_difference(
                graph,
                images_dir=images_dir,
                output_dir=output_dir,
                min_path_len=vd_cfg.get("min_path_len", 2),
                max_path_len=vd_cfg.get("max_path_len", 5),
                sample_per_scene=vd_cfg.get("sample_per_scene", 15),
                viewsuite_15k_dir=viewsuite_15k_dir,
                rng=child_rng(),
                balanced_sampling=vd_cfg.get("balanced_sampling", True),
            )
            result["view_difference"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] view_difference: %d", self.name, len(records))

        if "view_difference_mcq" in enabled:
            records = generate_view_difference_mcq(
                graph,
                images_dir=images_dir,
                output_dir=output_dir,
                min_path_len=vd_mcq_cfg.get("min_path_len", 2),
                max_path_len=vd_mcq_cfg.get("max_path_len", 5),
                sample_per_scene=vd_mcq_cfg.get("sample_per_scene", 15),
                viewsuite_15k_dir=viewsuite_15k_dir,
                rng=child_rng(),
                balanced_sampling=vd_mcq_cfg.get("balanced_sampling", True),
            )
            result["view_difference_mcq"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] view_difference_mcq: %d", self.name, len(records))

        return result
