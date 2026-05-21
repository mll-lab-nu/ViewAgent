"""
Random-action TrajToSFT phase.

Differs from the standard active-explore pipeline in WHERE the rollouts come
from: instead of consuming the just-finished RL rollouts, this phase invokes
``vagen.evaluate.run_eval`` with the random-navigation backend on the training
split each iteration to collect *fresh* random-action trajectories. Once the
graph is built from those, the dataset-generation logic is identical to
``InteractiveViewPlanningTrajToSFT`` (subclassed for that reason).

Per-iteration cache layout::

    iter_XXX/random_sft_stage/
        random_dump/      # raw VAGEN eval dump (PNG + JSONL per scene)
        rollouts/         # converted to VAGEN rollout layout
        graph/            # built graph (graph.json + images/)

If ``reuse_cached: true`` (default) and ``graph/graph.json`` already exists,
the eval+convert+graph stages are skipped.

Pipeline.yaml::

    traj_to_sft:
      module: graphrl.envs.viewsuite.viewsuite_random_sft_rl.RandomActionTrajToSFT
      eval_config: <abs path to collect_random_train.yaml>
      seed_offset_per_iter: 10000
      base_seed: 0
      reuse_cached: true
      cleanup_dump: true
      graph_builder:
        num_workers: 4
        filter: { void_threshold: 0.7, std_threshold: 10.0 }
      sft:
        # Same knobs as InteractiveViewPlanningTrajToSFT — see that class.
        generators: [multi_turn_action_gen, view_difference, ...]
        viewsuite_15k_dir: ...
        seed: 42
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.traj_to_sft import InteractiveViewPlanningTrajToSFT
from graphrl.envs.viewsuite.viewsuite_random_sft_rl.converter import convert_dump_to_vagen_rollouts

logger = logging.getLogger(__name__)

_ITER_DIR_RE = re.compile(r"iter_(\d+)")


def _infer_iter_num(p: Path) -> int:
    for part in p.resolve().parts[::-1]:
        m = _ITER_DIR_RE.fullmatch(part)
        if m:
            return int(m.group(1))
    return 0


class RandomActionTrajToSFT(InteractiveViewPlanningTrajToSFT):
    """Collect fresh random-action rollouts → build graph → generate SFT data."""

    name = "TrajToSFT(viewsuite_random_sft_rl)"

    # ── dataset config lives under ``sft:`` sub-block ─────────────────────

    def _dataset_cfg(self) -> Dict[str, Any]:
        return self.config.get("sft", {}) or {}

    # ── override the graph build to run random eval + convert first ───────

    def _build_or_load_graph(self) -> Path:
        cfg = self.config

        eval_config_path = cfg.get("eval_config")
        if not eval_config_path:
            raise ValueError(
                f"{self.__class__.__name__} requires 'eval_config' "
                "(path to a vagen.evaluate.run_eval YAML for the training split)"
            )
        eval_config = Path(eval_config_path).expanduser().resolve()
        if not eval_config.is_file():
            raise FileNotFoundError(f"eval_config not found: {eval_config}")

        # Per-iter caches under random_sft_stage/
        stage_dir = self.paths.base_dir / "random_sft_stage"
        dump_dir = stage_dir / "random_dump"
        rollouts_dir = stage_dir / "rollouts"
        graph_dir = stage_dir / "graph"
        for d in (stage_dir, dump_dir, rollouts_dir, graph_dir):
            d.mkdir(parents=True, exist_ok=True)
        graph_json = graph_dir / "graph.json"

        iter_num = _infer_iter_num(self.paths.base_dir)
        seed_offset_per_iter = int(cfg.get("seed_offset_per_iter", 10_000))
        base_seed = int(cfg.get("base_seed", 0)) + iter_num * seed_offset_per_iter
        extra_overrides: List[str] = list(cfg.get("eval_overrides", []) or [])
        reuse_cached = bool(cfg.get("reuse_cached", True))

        if reuse_cached and graph_json.is_file():
            logger.info(
                "[%s] reusing cached graph at %s (reuse_cached=true)",
                self.name, graph_json,
            )
            return graph_dir

        # ── 1. Run random-eval subprocess ───────────────────────────────
        self._run_eval_random(eval_config, dump_dir, base_seed, extra_overrides)

        # ── 2. Convert eval dump → VAGEN rollout layout ─────────────────
        tag_filter = self._resolve_dump_tag(eval_config)
        n_rollouts, n_images = convert_dump_to_vagen_rollouts(
            dump_root=dump_dir,
            output_rollout_dir=rollouts_dir,
            tag_filter=tag_filter,
            step_idx=0,
        )
        if n_rollouts == 0:
            raise RuntimeError(
                f"[{self.name}] no rollouts converted from {dump_dir}; "
                "check that the eval subprocess actually dumped episodes."
            )
        logger.info(
            "[%s] converted %d rollouts (%d images) → %s",
            self.name, n_rollouts, n_images, rollouts_dir,
        )

        # ── 3. Build graph (uses inherited graph_builder_class()) ───────
        builder = self.graph_builder_class()(cfg.get("graph_builder", {}) or {})
        jsonl_files = sorted(rollouts_dir.glob("*.jsonl"))
        builder.convert_files(jsonl_files, rollouts_dir, graph_dir)
        if not graph_json.is_file():
            raise RuntimeError(f"[{self.name}] graph builder did not produce {graph_json}")

        # Optional cleanup of intermediate stage data (graph itself preserved).
        if bool(cfg.get("cleanup_dump", False)):
            for d in (dump_dir, rollouts_dir):
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
            logger.info("[%s] cleaned up %s, %s", self.name, dump_dir, rollouts_dir)

        return graph_dir

    # ── helpers (unchanged from old impl) ─────────────────────────────────

    @staticmethod
    def _run_eval_random(
        eval_config: Path,
        dump_dir: Path,
        base_seed: int,
        extra_overrides: List[str],
    ) -> None:
        """Invoke vagen.evaluate.run_eval as a subprocess (own asyncio loop)."""
        cmd = [
            sys.executable, "-m",
            "graphrl.envs.viewsuite.viewsuite_random_sft_rl._run_eval_subprocess",
            "--config", str(eval_config),
            f"experiment.dump_dir={dump_dir}",
            f"run.base_seed={base_seed}",
        ]
        cmd.extend(extra_overrides)
        logger.info("[random_sft] launching eval: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    @staticmethod
    def _resolve_dump_tag(eval_config: Path) -> Optional[str]:
        """Read the first env's ``tag_id`` from the eval YAML, used to filter dumps."""
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(eval_config)
            envs = cfg.get("envs") or []
            if envs:
                tag = envs[0].get("tag_id")
                if tag is None:
                    return None
                return str(tag)
        except Exception as exc:
            logger.warning("Failed to read tag_id from %s: %s", eval_config, exc)
        return None
