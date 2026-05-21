"""
Point Cloud by Turn Analysis — main entry point.

Computes how many unique mesh vertices an agent observes cumulatively
across turns in the ScanNet proxy-tool environment.

Supports checkpointing: if interrupted, re-run with the same arguments
and it will resume from where it left off (per-scene granularity).

Usage:
    # Full pipeline (parse → visibility → aggregate → plot)
    # Output defaults to <rollout_dir>/coverage_analysis/
    python -m view_suite.analysis.scannet_point_by_turn.main run \
        --rollout_dir /path/to/rollouts \
        --scannet_dir /path/to/scannet

    # Re-plot from previously saved results
    python -m view_suite.analysis.scannet_point_by_turn.main plot_only \
        --result_json ./output/results.json \
        --output_dir  ./output_replot
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import fire

from view_suite.scannet.utils.path_utils import resolve_scene_ply
from view_suite.envs.utils.scannet_utils import default_intrinsics

from view_suite.analysis.proxy_analysis.pointcloud_coverage.trajectory_parser import (
    TrajectoryInfo,
    TrajectoryParser,
)
from view_suite.analysis.proxy_analysis.pointcloud_coverage.scene_processor import process_scene
from view_suite.analysis.proxy_analysis.pointcloud_coverage.aggregator import ResultAggregator
from view_suite.analysis.proxy_analysis.pointcloud_coverage.plotter import Plotter


# Default JSONL path (matches the evaluation config)
_DEFAULT_JSONL = "/root/projects/viewsuite/data/viewsuite_15k/interactive_view_planning_test_filter.jsonl"

# Checkpoint file name (one JSON object per line, each line = one trajectory result)
_CHECKPOINT_FILE = "checkpoint.jsonl"


class PointCloudByTurnAnalyzer:
    """
    CLI interface for analysing point-cloud coverage over agent turns.

    Methods exposed via fire.Fire():
      - run():       Full pipeline — parse, compute, aggregate, plot.
      - plot_only(): Re-generate plots from a saved results.json.
    """

    def run(
        self,
        rollout_dir: str,
        scannet_dir: str,
        output_dir: Optional[str] = None,
        jsonl_path: Optional[str] = None,
        n_workers: int = 8,
        width: int = 512,
        height: int = 512,
    ) -> None:
        """
        Full analysis pipeline.

        Args:
            rollout_dir: Directory containing trajectory folders
                         (e.g. .../tag_ae_example_loose_no_example).
            scannet_dir: Root of the ScanNet data tree (contains scans/ sub-dir).
            output_dir:  Where to write results.json, summary.json, plots.
                         Defaults to <rollout_dir>/coverage_analysis/.
            jsonl_path:  Path to the evaluation JSONL. If None, auto-detected
                         from each trajectory's env_config.jsonl_path.
            n_workers:   Number of parallel worker processes.
            width:       Image width for visibility rendering.
            height:      Image height for visibility rendering.
        """
        t0 = time.time()
        scannet_scans = str(Path(scannet_dir) / "scans")
        output_dir_p = Path(output_dir) if output_dir else Path(rollout_dir) / "coverage_analysis"
        output_dir_p.mkdir(parents=True, exist_ok=True)

        # Use default_intrinsics (fix_intrinsics=True in eval config).
        # default_intrinsics() returns 4x4; we need the 3x3 upper-left.
        K4 = default_intrinsics()
        K3 = K4[:3, :3].copy()

        # ----- Phase 1: Parse trajectories -----
        print("=" * 60)
        print("Phase 1: Parsing trajectories ...")
        parser = TrajectoryParser(jsonl_path)  # None → auto-detect from each traj's env_config
        trajectories = parser.parse_all(rollout_dir)
        print(f"  → {len(trajectories)} trajectories parsed")

        # ----- Phase 2: Group by scene -----
        print("=" * 60)
        print("Phase 2: Grouping by scene ...")
        scene_groups = _group_by_scene(trajectories)
        print(f"  → {len(scene_groups)} unique scenes")
        for sid, trajs in sorted(scene_groups.items()):
            print(f"    {sid}: {len(trajs)} trajectories")

        # ----- Phase 3: Load checkpoint (if resuming) -----
        checkpoint_path = output_dir_p / _CHECKPOINT_FILE
        cached_results, done_scenes = _load_checkpoint(checkpoint_path)
        if cached_results:
            print(f"  → Checkpoint: {len(cached_results)} trajectories from "
                  f"{len(done_scenes)} scenes already done, resuming ...")

        # ----- Phase 4: Resolve mesh paths & filter to remaining scenes -----
        remaining_args: List[Tuple] = []
        total_remaining_trajs = 0
        for sid, trajs in scene_groups.items():
            if sid in done_scenes:
                continue
            mesh_path = resolve_scene_ply(scannet_scans, sid)
            traj_dicts = [_traj_to_dict(t) for t in trajs]
            remaining_args.append((sid, mesh_path, traj_dicts, K3, width, height))
            total_remaining_trajs += len(trajs)

        # ----- Phase 5: Parallel visibility computation with checkpointing -----
        print("=" * 60)
        if not remaining_args:
            print("All scenes already processed (checkpoint). Skipping computation.")
            all_results = cached_results
        else:
            print(f"Phase 3: Computing visibility — "
                  f"{len(remaining_args)} scenes, {total_remaining_trajs} trajectories "
                  f"({n_workers} workers) ...")

            all_results = list(cached_results)
            n_pool = min(n_workers, len(remaining_args))
            scenes_done = 0
            t_compute = time.time()

            with mp.Pool(processes=n_pool) as pool:
                # imap_unordered yields results as each scene completes,
                # allowing incremental checkpointing.
                for scene_results in pool.imap_unordered(
                    _process_scene_wrapper, remaining_args
                ):
                    all_results.extend(scene_results)
                    _append_checkpoint(scene_results, checkpoint_path)
                    scenes_done += 1
                    sid = scene_results[0]["scene_id"] if scene_results else "?"
                    elapsed = time.time() - t_compute
                    print(f"  → scene {scenes_done}/{len(remaining_args)} ({sid}): "
                          f"+{len(scene_results)} trajs, "
                          f"{len(all_results)}/{len(trajectories)} total "
                          f"[{elapsed:.0f}s elapsed]")

        # Sort by traj_id for reproducibility
        all_results.sort(key=lambda r: r["traj_id"])

        # ----- Phase 6: Aggregate and save -----
        print("=" * 60)
        print("Phase 4: Aggregating results ...")
        summary = ResultAggregator.aggregate(all_results)

        config_info = {
            "rollout_dir": str(rollout_dir),
            "scannet_dir": str(scannet_dir),
            "jsonl_path": str(jsonl_path) if jsonl_path else "auto-detected",
            "width": width,
            "height": height,
            "n_workers": n_workers,
        }
        ResultAggregator.save_json(all_results, summary, output_dir_p, config=config_info)
        ResultAggregator.save_csv(all_results, output_dir_p)

        # ----- Phase 7: Plot -----
        print("=" * 60)
        print("Phase 5: Generating plots ...")
        Plotter.plot_all(all_results, summary, output_dir_p)

        # Clean up checkpoint now that final outputs are written
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"Removed checkpoint (final results saved).")

        elapsed = time.time() - t0
        print("=" * 60)
        print(f"Done in {elapsed:.1f}s.  Results → {output_dir_p}")

    def run_all(
        self,
        rollouts_dir: str = "/root/projects/viewsuite/rollouts",
        scannet_dir: str = "/root/projects/viewsuite/data/scannet",
        output_dir: Optional[str] = None,
        n_workers: int = 8,
        width: int = 512,
        height: int = 512,
        models: Optional[str] = None,
        task: str = "tag_interactive_view_planning",
    ) -> None:
        """
        Run coverage analysis on all models and generate combined comparison.

        Args:
            rollouts_dir: Root directory with one sub-dir per model.
            scannet_dir:  Root of the ScanNet data tree.
            output_dir:   Where to write all results.
                          Defaults to <rollouts_dir>_pointcloud_coverage.
            n_workers:    Number of parallel worker processes per model.
            width:        Image width for visibility rendering.
            height:       Image height for visibility rendering.
            models:       Comma-separated model names. Default: auto-discover.
            task:         Task subdirectory to analyze.
        """
        from view_suite.analysis.proxy_analysis.pointcloud_coverage.compare import compare as compare_fn

        rollouts_path = Path(rollouts_dir)
        out_root = Path(output_dir) if output_dir else rollouts_path.parent / (rollouts_path.name + "_pointcloud_coverage")
        out_root.mkdir(parents=True, exist_ok=True)

        # Discover models
        if models:
            model_list = [m.strip() for m in models.split(",")]
        else:
            model_list = sorted([
                d.name for d in rollouts_path.iterdir()
                if d.is_dir() and (d / task).is_dir()
            ])

        print(f"Rollouts dir: {rollouts_path}")
        print(f"Output dir:   {out_root}")
        print(f"Task:         {task}")
        print(f"Models ({len(model_list)}): {model_list}")
        print()

        # Run each model
        coverage_dirs = []
        labels = []
        _MODEL_DISPLAY = {
            "claude_opus_4_6": "Claude Opus 4.6",
            "gemini_3_1_pro": "Gemini 3.1 Pro",
            "gemini_3_pro": "Gemini 3 Pro",
            "glm_4_6v": "GLM-4.6V",
            "glm_4_6v_refined": "GLM-4.6V (refined)",
            "gpt_5_1": "GPT-5.1",
            "gpt_5_4": "GPT-5.4",
            "gpt_5_4_pro": "GPT-5.4 Pro",
            "grok_4_20_beta": "Grok 4.20 Beta",
            "qwen2_5_vl_72b": "Qwen2.5-VL-72B",
            "qwen3_5_397b": "Qwen3.5-397B",
            "qwen3_vl_32b": "Qwen3-VL-32B",
            "qwen_25_vl_7b": "Qwen2.5-VL-7B",
            "qwen_25_vl_7b_trained": "Qwen2.5-VL-7B (trained)",
        }

        for model in model_list:
            rollout_dir = str(rollouts_path / model / task)
            model_out = str(out_root / model)

            print(f"\n{'='*60}")
            print(f"Model: {model}")
            print(f"{'='*60}")

            try:
                self.run(
                    rollout_dir=rollout_dir,
                    scannet_dir=scannet_dir,
                    output_dir=model_out,
                    n_workers=n_workers,
                    width=width,
                    height=height,
                )
                coverage_dirs.append(model_out)
                labels.append(_MODEL_DISPLAY.get(model, model))
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Generate combined comparison plots
        if len(coverage_dirs) >= 2:
            print(f"\n{'='*60}")
            print("Generating combined comparison plots...")
            print(f"{'='*60}")

            compare_out = str(out_root / "combined")
            compare_fn(
                *coverage_dirs,
                output_dir=compare_out,
                labels=",".join(labels),
            )
        else:
            print("Not enough models for comparison plots.")

        print(f"\nAll results saved to {out_root}")

    def plot_only(
        self,
        result_json: str,
        output_dir: str = "./output_replot",
    ) -> None:
        """
        Re-generate all plots from a previously saved results.json.

        Args:
            result_json: Path to the results.json produced by run().
            output_dir:  Where to write the new plots.
        """
        with open(result_json, "r") as f:
            payload = json.load(f)

        trajectory_results = payload["trajectories"]
        summary = ResultAggregator.aggregate(trajectory_results)

        output_dir_p = Path(output_dir)
        output_dir_p.mkdir(parents=True, exist_ok=True)

        # Save updated summary
        with open(output_dir_p / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        Plotter.plot_all(trajectory_results, summary, output_dir_p)
        print(f"Plots → {output_dir_p}")

    def split_by_success(
        self,
        rollout_dir: str,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Split existing coverage results by success/fail and generate
        separate plots + comparison plots.

        Reads results.json from <rollout_dir>/coverage_analysis/ and
        metrics.json from each trajectory to determine success/fail.

        Outputs three subdirectories:
          - coverage_analysis_success/  — plots for successful trajectories
          - coverage_analysis_fail/     — plots for failed trajectories
          - coverage_analysis_compare/  — success vs fail comparison plots

        Args:
            rollout_dir: Directory containing trajectory folders (same as run()).
            output_dir:  Base output directory. Defaults to <rollout_dir>.
                         _success, _fail, _compare subdirs are created here.
        """
        from view_suite.analysis.proxy_analysis.pointcloud_coverage.compare import compare as compare_fn

        rollout_path = Path(rollout_dir)
        base_output = Path(output_dir) if output_dir else rollout_path

        # Load existing results
        results_path = rollout_path / "coverage_analysis" / "results.json"
        if not results_path.exists():
            print(f"ERROR: {results_path} not found. Run 'run' first.")
            return

        with open(results_path) as f:
            payload = json.load(f)
        all_results = payload["trajectories"]

        # Read success/fail from each trajectory's metrics.json
        success_results = []
        fail_results = []
        missing = 0

        for r in all_results:
            metrics_path = rollout_path / r["traj_id"] / "metrics.json"
            if not metrics_path.exists():
                missing += 1
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            if metrics.get("success", False):
                success_results.append(r)
            else:
                fail_results.append(r)

        print(f"Total: {len(all_results)} trajectories")
        print(f"  Success: {len(success_results)}")
        print(f"  Fail:    {len(fail_results)}")
        if missing:
            print(f"  Missing metrics: {missing}")

        # Generate plots for success
        out_success = base_output / "coverage_analysis_success"
        out_success.mkdir(parents=True, exist_ok=True)
        if success_results:
            summary_s = ResultAggregator.aggregate(success_results)
            ResultAggregator.save_json(success_results, summary_s, out_success)
            Plotter.plot_all(success_results, summary_s, out_success)
            print(f"\nSuccess plots → {out_success}")
        else:
            print("\nNo successful trajectories — skipping success plots.")

        # Generate plots for fail
        out_fail = base_output / "coverage_analysis_fail"
        out_fail.mkdir(parents=True, exist_ok=True)
        if fail_results:
            summary_f = ResultAggregator.aggregate(fail_results)
            ResultAggregator.save_json(fail_results, summary_f, out_fail)
            Plotter.plot_all(fail_results, summary_f, out_fail)
            print(f"Fail plots → {out_fail}")
        else:
            print("No failed trajectories — skipping fail plots.")

        # Generate comparison plots (success vs fail on same figure)
        if success_results and fail_results:
            out_compare = base_output / "coverage_analysis_compare"
            out_compare.mkdir(parents=True, exist_ok=True)
            _plot_success_fail_comparison(
                summary_s, summary_f,
                len(success_results), len(fail_results),
                out_compare,
            )
            print(f"Compare plots → {out_compare}")
        else:
            print("Cannot generate comparison (need both success and fail).")


# ---------------------------------------------------------------------------
# Success vs Fail comparison plots
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COMPARE_FIG_SIZE = (10, 6)
_COMPARE_DPI = 150
_COLOR_SUCCESS = "#10B981"
_COLOR_FAIL = "#EF4444"


def _extract_compare_series(stats_by_turn):
    """Extract turns, means, stds, counts from a summary sub-dict."""
    turns = sorted(int(k) for k in stats_by_turn.keys())
    means = [stats_by_turn[str(t)]["mean"] for t in turns]
    stds = [stats_by_turn[str(t)]["std"] for t in turns]
    counts = [stats_by_turn[str(t)]["count"] for t in turns]
    return turns, means, stds, counts


def _filter_valid(turns, means, stds, counts, min_frac=0.01):
    """Filter out turns with very few trajectories."""
    if not counts:
        return turns, means, stds
    max_count = max(counts)
    valid = [j for j, c in enumerate(counts) if c >= max_count * min_frac]
    return (
        [turns[j] for j in valid],
        [means[j] for j in valid],
        [stds[j] for j in valid],
    )


def _set_int_xticks(ax, turns):
    if turns:
        ax.set_xticks(turns)
        ax.set_xticklabels([str(t) for t in turns])


def _plot_one_compare(
    summary_s, summary_f, n_success, n_fail, output_dir,
    key, ylabel, title, filename, marker="o-", is_bar=False,
):
    """Plot a single success vs fail comparison chart."""
    if key not in summary_s and key not in summary_f:
        return

    fig, ax = plt.subplots(figsize=_COMPARE_FIG_SIZE)
    all_turns = set()

    for summary, color, label_prefix, n in [
        (summary_s, _COLOR_SUCCESS, "Success", n_success),
        (summary_f, _COLOR_FAIL, "Fail", n_fail),
    ]:
        if key not in summary or not summary[key]:
            continue
        turns, means, stds, counts = _extract_compare_series(summary[key])
        turns, means, stds = _filter_valid(turns, means, stds, counts)
        all_turns.update(turns)
        label = f"{label_prefix} (n={n})"

        if is_bar:
            offset = -0.2 if label_prefix == "Success" else 0.2
            positions = np.array(turns) + offset
            ax.bar(positions, means, width=0.35, color=color, alpha=0.7, label=label)
            ax.errorbar(positions, means, yerr=stds, fmt="none", ecolor="gray",
                        capsize=2, linewidth=0.8)
        else:
            ax.plot(turns, means, marker, color=color, linewidth=2, markersize=4, label=label)
            ax.fill_between(
                turns,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.12, color=color,
            )

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _set_int_xticks(ax, sorted(all_turns))

    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=_COMPARE_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_success_fail_comparison(
    summary_s, summary_f, n_success, n_fail, output_dir,
):
    """Generate all success vs fail comparison plots."""
    output_dir = Path(output_dir)

    _plot_one_compare(
        summary_s, summary_f, n_success, n_fail, output_dir,
        "cumulative", "Cumulative visible vertices",
        "Cumulative Coverage — Success vs Fail", "compare_cumulative.png",
    )
    _plot_one_compare(
        summary_s, summary_f, n_success, n_fail, output_dir,
        "increment", "New vertices this turn",
        "Per-Turn Increment — Success vs Fail", "compare_increment.png",
        is_bar=True,
    )
    _plot_one_compare(
        summary_s, summary_f, n_success, n_fail, output_dir,
        "coverage_ratio", "Coverage ratio (visible / total)",
        "Coverage Ratio — Success vs Fail", "compare_coverage_ratio.png",
        marker="s-",
    )
    _plot_one_compare(
        summary_s, summary_f, n_success, n_fail, output_dir,
        "target_intersection", "Target intersection vertices",
        "Cumulative Target Intersection — Success vs Fail",
        "compare_target_cumulative.png",
    )
    _plot_one_compare(
        summary_s, summary_f, n_success, n_fail, output_dir,
        "target_intersection_inc", "New target-intersection vertices this turn",
        "Per-Turn Target Intersection Increment — Success vs Fail",
        "compare_target_increment.png", is_bar=True,
    )
    _plot_one_compare(
        summary_s, summary_f, n_success, n_fail, output_dir,
        "target_intersection_ratio",
        "Target intersection ratio (|cum ∩ target| / |target|)",
        "Target Intersection Ratio — Success vs Fail",
        "compare_target_intersection_ratio.png", marker="D-",
    )


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(
    checkpoint_path: Path,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Load previously saved trajectory results from checkpoint.

    Returns:
        (results, done_scene_ids) — results list and set of scene_ids
        that are fully processed.
    """
    results: List[Dict[str, Any]] = []
    done_scenes: Set[str] = set()

    if not checkpoint_path.exists():
        return results, done_scenes

    with open(checkpoint_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                results.append(r)
                done_scenes.add(r["scene_id"])

    return results, done_scenes


def _append_checkpoint(
    scene_results: List[Dict[str, Any]],
    checkpoint_path: Path,
) -> None:
    """Append one scene's trajectory results to the checkpoint file."""
    with open(checkpoint_path, "a") as f:
        for r in scene_results:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------

def _process_scene_wrapper(args: Tuple) -> List[Dict[str, Any]]:
    """Unpack tuple args for imap_unordered (which takes single-arg callables)."""
    return process_scene(*args)


def _group_by_scene(
    trajectories: List[TrajectoryInfo],
) -> Dict[str, List[TrajectoryInfo]]:
    """Group trajectories by scene_id."""
    groups: Dict[str, List[TrajectoryInfo]] = defaultdict(list)
    for t in trajectories:
        groups[t.scene_id].append(t)
    return dict(groups)


def _traj_to_dict(t: TrajectoryInfo) -> Dict[str, Any]:
    """Convert a TrajectoryInfo to a plain dict safe for multiprocessing pickling."""
    return {
        "traj_id": t.traj_id,
        "jsonl_idx": t.jsonl_idx,
        "sample_id": t.sample_id,
        "init_view_c2w": t.init_view_c2w.tolist(),
        "top_down_view_c2w": t.top_down_view_c2w.tolist(),
        "target_view_c2w": t.target_view_c2w.tolist(),
        "turn_c2ws": [c.tolist() for c in t.turn_c2ws],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fire.Fire(PointCloudByTurnAnalyzer)
