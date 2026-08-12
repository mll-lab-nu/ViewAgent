"""
TrajToSFT phase: convert VAGEN rollout output → LLaMA-Factory dataset.

This is the **only user-extension point** in the framework. Subclass
``TrajToSFTModule`` and override ``run()``.

I/O contract (both ends fixed by the mono-backend choice):

  Inputs (always present at ``self.paths.*``):
      ``rollout_data``  -- VAGEN's ``rollout_data/`` directory
                           (JSONL files + image subdirs, may be missing if RL
                           was skipped this iter — handle gracefully)
      ``rl_model``      -- ``iter_XXX/rl_model/`` (HF model dir, may be missing
                           if RL was skipped)
      ``base_dir``      -- ``iter_XXX/`` (use for env-specific scratch dirs)

  Output (must be produced by ``run()``):
      ``sft_data``      -- ``iter_XXX/sft_data/`` populated with
                           ``dataset_info.json`` + per-dataset .json files,
                           ready for LLaMA-Factory to consume.

Subclass example (graph-based)::

    class MyTrajToSFT(TrajToSFTModule):
        def run(self):
            from my_pkg.graph_builder import MyGraphBuilder
            builder = MyGraphBuilder(self.config.get("graph_builder", {}))
            jsonls = sorted(self.paths.rollout_data.glob("*.jsonl"))
            graph_dir = self.paths.base_dir / "_graph"
            builder.convert_files(jsonls, self.paths.rollout_data, graph_dir)
            self.write_dataset_info(self._build_records(graph_dir))

The base class ships a couple of helpers (``write_dataset_info``,
``read_rollouts``) that subclasses may use; they're not required.
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from graphrl.state import ModuleOutput, ModuleState

logger = logging.getLogger(__name__)


@dataclass
class TrajToSFTPaths:
    """Typed paths handed to every TrajToSFTModule by the controller."""

    base_dir: Path        # iter_XXX/
    rollout_data: Path    # iter_XXX/rl/rollout_data/
    rl_model: Path        # iter_XXX/rl_model/
    sft_data: Path        # iter_XXX/sft_data/   (output target)


class TrajToSFTModule:
    """Concrete base class. Subclass and override ``run()``.

    The base class handles lifecycle bookkeeping; subclasses only need to
    implement ``run()``. ``run()`` is invoked synchronously from ``launch()``
    and may use the GPU — the controller has already killed the prior
    RL/SFT module before scheduling this phase, so the GPU is free.
    """

    name = "TrajToSFT"

    def __init__(
        self,
        config: Dict[str, Any],
        paths: TrajToSFTPaths,
    ):
        self.config = config
        self.paths = paths
        self._state = ModuleState.IDLE

    @property
    def state(self) -> ModuleState:
        return self._state

    # ── lifecycle (synchronous; GPU is available if the subclass wants it) ──

    def launch(self) -> None:
        """Run the conversion synchronously."""
        self.paths.sft_data.mkdir(parents=True, exist_ok=True)
        self._state = ModuleState.LAUNCHED
        try:
            self.run()
        except Exception:
            self._state = ModuleState.FAILED
            raise
        # Touch the phase-done marker only after run() completes successfully —
        # which for reasoning subclasses includes the reasoning post-step.
        # Resume detection uses this rather than ``dataset_info.json`` so a
        # crashed reasoning step doesn't trick the next run into skipping it.
        (self.paths.sft_data / ".phase_done").write_text("done\n", encoding="utf-8")
        self._state = ModuleState.DONE
        self._log("Conversion complete")

    def is_done(self) -> bool:
        return (self.paths.sft_data / "dataset_info.json").exists()

    def kill(self) -> None:
        self._state = ModuleState.TERMINATED

    def is_already_complete(self) -> bool:
        return self.is_done()

    def get_output(self) -> ModuleOutput:
        if self.is_done():
            return ModuleOutput(data_paths={"sft_data": str(self.paths.sft_data)})
        return ModuleOutput()

    # ── subclass interface ────────────────────────────────────────────────

    def run(self) -> None:
        """Override in subclass.

        Read from ``self.paths.rollout_data`` / ``self.paths.rl_model``,
        write into ``self.paths.sft_data`` (must include ``dataset_info.json``).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run()."
        )

    # ── helpers (optional) ────────────────────────────────────────────────

    def read_rollouts(self) -> Iterator[Dict[str, Any]]:
        """Yield rollout records from VAGEN's ``rollout_data/*.jsonl`` files."""
        jsonl_files = sorted(self.paths.rollout_data.glob("*.jsonl"))
        for path in jsonl_files:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)

    def write_dataset_info(
        self,
        datasets: Dict[str, Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]],
    ) -> None:
        """Write LLaMA-Factory ``dataset_info.json`` + per-dataset ``.json`` files.

        ``datasets`` is a mapping ``name → (records, fmt_override)``:

          * ``records``      -- list of dicts (Alpaca or ShareGPT format)
          * ``fmt_override`` -- ``None`` to auto-detect, or a dict to merge
                                into the dataset_info entry (set
                                ``{"formatting": "sharegpt", "columns": {...}}``
                                for multi-turn ShareGPT datasets).

        Empty datasets are skipped with a warning. Raises if every dataset is
        empty.
        """
        out_dir = self.paths.sft_data
        out_dir.mkdir(parents=True, exist_ok=True)

        info: Dict[str, Any] = {}
        for name, (records, fmt_override) in datasets.items():
            if not records:
                logger.warning("[%s] dataset %r is empty — skipping", self.name, name)
                continue
            (out_dir / f"{name}.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            entry: Dict[str, Any] = {"file_name": f"{name}.json"}
            if fmt_override:
                entry.update(fmt_override)
            elif records and "conversations" in records[0]:
                entry["formatting"] = "sharegpt"
                entry["columns"] = {"messages": "conversations", "system": "system"}
            info[name] = entry

        # ── anti-overfit: mix in general instruction/VQA data ───────────────
        # Task-only SFT collapses auxiliary abilities fast (lit: aux task 81%->16.5%,
        # most of it within the first 1000 steps — matches our P2V/V2P -> 0). Mixing
        # even ~6% general data removes the collapse with NO measurable loss of
        # specialization, so we register a general set alongside the task sets.
        # Config:  general_mix: {file: /abs/path.json, ratio: 0.15, formatting: sharegpt}
        gm = (self.config.get("general_mix") or {}) if hasattr(self, "config") else {}
        gm_file = gm.get("file")
        if gm_file and os.path.exists(gm_file):
            import shutil as _sh
            dst = out_dir / "general_mix.json"
            try:
                _sh.copyfile(gm_file, dst)
                n_task = sum(len(v) for v in datasets.values()) if isinstance(datasets, dict) else 0
                ratio = float(gm.get("ratio", 0.15))
                entry = {"file_name": dst.name}
                if gm.get("formatting", "sharegpt") == "sharegpt":
                    entry["formatting"] = "sharegpt"
                    entry["columns"] = {"messages": "conversations", "system": "system"}
                if n_task and ratio > 0:
                    # cap general samples at ratio/(1-ratio) of the task data
                    entry["num_samples"] = max(1, int(n_task * ratio / max(1e-6, 1 - ratio)))
                info["general_mix"] = entry
                logger.info("[%s] general_mix registered: %s (ratio=%.2f, cap=%s)",
                            self.name, dst.name, ratio, entry.get("num_samples"))
            except Exception as e:
                logger.warning("[%s] general_mix skipped: %s", self.name, e)
        elif gm_file:
            logger.warning("[%s] general_mix file not found, skipping: %s", self.name, gm_file)

        if not info:
            raise RuntimeError(f"[{self.name}] no non-empty datasets produced")

        (out_dir / "dataset_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("[%s] wrote %d dataset(s) → %s", self.name, len(info), out_dir)

    def _log(self, msg: str) -> None:
        logger.info("[%s] %s", self.name, msg)


# ── dispatcher (called by controller) ─────────────────────────────────────


def load_traj_to_sft_class(spec: str):
    """Resolve a dotted-path class name to the actual class.

    Example::

        cls = load_traj_to_sft_class("graphrl.envs.viewsuite.viewsuite_interactive_view_planning.InteractiveViewPlanningTrajToSFT")
        instance = cls(config, paths)
    """
    if "." not in spec:
        raise ValueError(
            f"traj_to_sft.module must be a dotted Python path "
            f"(e.g. 'graphrl.envs.my_env.MyTrajToSFT'), got {spec!r}"
        )
    import importlib

    module_name, _, class_name = spec.rpartition(".")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"{spec}: class {class_name!r} not found in {module_name}")
    if not isinstance(cls, type) or not issubclass(cls, TrajToSFTModule):
        raise TypeError(
            f"{spec}: must be a subclass of graphrl.traj_to_sft.TrajToSFTModule"
        )
    return cls
