#!/usr/bin/env python3
"""
Performance Summary Analysis — main entry point.

Scans a rollouts directory (one sub-dir per model, each containing
tag_path_to_view / tag_view_to_path / tag_interactive_view_planning)
and produces:

1. Per-model performance CSV
2. Combined summary CSV + Markdown table
3. Success-by-action-length CSV + Markdown table
4. GLM-style format-error refinement (optional)

Usage:
    python -m view_suite.proxy_analysis.performance_summary.main run \
        --rollouts_dir /root/projects/viewsuite/rollouts \
        --output_dir  /root/projects/viewsuite/rollouts_performance_summary

    python -m view_suite.proxy_analysis.performance_summary.main refine_glm \
        --rollouts_dir /root/projects/viewsuite/rollouts \
        --model glm_4_6v
"""

from __future__ import annotations

import json
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fire

from .metrics_reader import (
    read_all_metrics,
    compute_success_rate,
    compute_detailed_success_rates,
    compute_adaptive_success_rate,
    compute_success_by_action_len,
)
from .table_writer import (
    write_combined_csv,
    write_combined_md,
    write_action_len_csv,
    write_action_len_md,
    write_per_model_csv,
)

# Default thresholds matching the evaluation config
DEFAULT_SUCCESS_THRESHOLDS = [(0.25, 15), (0.5, 30), (1.0, 30), (1.0, 60)]
DEFAULT_TOL_PER_ACTION_LEN = "0.25,15;2:0.5,30;3-5:1,30;1,60"
DEFAULT_ACTION_LEN_INTERVALS = [2, 5, 8]

# Tasks
TASKS = ["tag_path_to_view", "tag_view_to_path", "tag_interactive_view_planning"]
TASK_SHORT = {
    "tag_path_to_view": "Path2View",
    "tag_view_to_path": "View2Path",
    "tag_interactive_view_planning": "Interactive View Planning",
}

# Nice model names
MODEL_DISPLAY = {
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


def _threshold_key(m: float, d: float) -> str:
    """Build key like 'success_0p25m15degree'."""
    def fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.6g}".replace(".", "p").replace("-", "neg")
    return f"success_{fmt(m)}m{fmt(d)}degree"


def _threshold_label(m: float, d: float) -> str:
    """Build human-readable label like '0.25m/15°'."""
    def fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:g}"
    return f"{fmt(m)}m/{fmt(d)}°"


def _discover_models(rollouts_dir: Path) -> List[str]:
    """Find all model directories that contain at least one task sub-dir."""
    models = []
    for d in sorted(rollouts_dir.iterdir()):
        if not d.is_dir():
            continue
        # Must have at least one task dir
        if any((d / t).is_dir() for t in TASKS):
            models.append(d.name)
    return models


class PerformanceSummary:
    """CLI interface for performance summary analysis."""

    def run(
        self,
        rollouts_dir: str = "/root/projects/viewsuite/rollouts",
        output_dir: Optional[str] = None,
        viewsuite_data_path: str = "/root/projects/viewsuite/data/viewsuite_15k",
        jsonl_dir: Optional[str] = None,
        thresholds: str = "0.25,15;0.5,30;1,30;1,60",
        tol_per_action_len: str = DEFAULT_TOL_PER_ACTION_LEN,
        action_len_intervals: str = "2,5,8",
        models: Optional[str] = None,
    ) -> None:
        """
        Full analysis pipeline.

        Args:
            rollouts_dir: Root directory with one sub-dir per model.
            output_dir: Where to write results. Default: <rollouts_dir>_performance_summary
            viewsuite_data_path: ViewSuite data root (for gt_action_len lookup).
            jsonl_dir: Directory containing JSONL files. Default: <viewsuite_data_path>
            thresholds: Semicolon-separated (meters,degrees) pairs for detailed success.
            tol_per_action_len: Adaptive threshold spec.
            action_len_intervals: Comma-separated interval boundaries for action-length analysis.
            models: Comma-separated model names to process. Default: all discovered models.
        """
        rollouts_path = Path(rollouts_dir)
        out_path = Path(output_dir) if output_dir else rollouts_path.parent / (rollouts_path.name + "_performance_summary")
        out_path.mkdir(parents=True, exist_ok=True)
        data_path = Path(viewsuite_data_path)
        jdir = Path(jsonl_dir) if jsonl_dir else data_path

        # Parse thresholds
        thresh_list = _parse_thresholds(thresholds)
        intervals = [int(x.strip()) for x in action_len_intervals.split(",")]

        # Discover models
        if models:
            model_list = [m.strip() for m in models.split(",")]
        else:
            model_list = _discover_models(rollouts_path)

        print(f"Rollouts dir: {rollouts_path}")
        print(f"Output dir:   {out_path}")
        print(f"Models ({len(model_list)}): {model_list}")
        print(f"Thresholds: {thresh_list}")
        print(f"Adaptive tol: {tol_per_action_len}")
        print(f"Action-len intervals: {intervals}")
        print()

        # JSONL paths for each task
        jsonl_map = {
            "tag_path_to_view": jdir / "path_to_view_test_filter.jsonl",
            "tag_view_to_path": jdir / "view_to_path_test_filter.jsonl",
            "tag_interactive_view_planning": jdir / "interactive_view_planning_test_filter.jsonl",
        }

        # Collect all results
        all_model_results: Dict[str, Dict[str, Any]] = {}
        all_action_len_results: Dict[str, Dict[str, Any]] = {}

        for model in model_list:
            print(f"{'='*60}")
            print(f"Model: {model}")
            print(f"{'='*60}")

            model_result: Dict[str, Any] = {"model": model, "display_name": MODEL_DISPLAY.get(model, model)}
            model_action_len: Dict[str, Any] = {"model": model}

            for task in TASKS:
                task_dir = rollouts_path / model / task
                if not task_dir.is_dir():
                    print(f"  {task}: MISSING")
                    continue

                metrics_list = read_all_metrics(task_dir)
                n = len(metrics_list)
                print(f"  {task}: {n} rollouts")

                # Basic success rate (from summary.json or computed)
                sr = compute_success_rate(metrics_list)
                model_result[f"{task}_success"] = sr
                model_result[f"{task}_n"] = n

                # Action-length breakdown (for all tasks)
                jsonl_path = jsonl_map.get(task)
                al_results = compute_success_by_action_len(
                    metrics_list, intervals, data_path, jsonl_path
                )
                model_action_len[task] = al_results

                # For active exploration: detailed thresholds + adaptive
                if task == "tag_interactive_view_planning":
                    detailed = compute_detailed_success_rates(
                        metrics_list, thresh_list, data_path, jsonl_path
                    )
                    for (m_tol, d_tol), rate in detailed.items():
                        key = _threshold_key(m_tol, d_tol)
                        model_result[key] = rate
                        label = _threshold_label(m_tol, d_tol)
                        print(f"    {label}: {rate:.4f}")

                    adaptive = compute_adaptive_success_rate(
                        metrics_list, tol_per_action_len, data_path, jsonl_path
                    )
                    model_result["adaptive_success"] = adaptive
                    print(f"    adaptive: {adaptive:.4f}")

                    # Avg across thresholds
                    thresh_rates = [detailed[k] for k in detailed]
                    avg_thresh = sum(thresh_rates) / len(thresh_rates) if thresh_rates else 0.0
                    model_result["ae_avg_success"] = avg_thresh
                    print(f"    avg_thresh: {avg_thresh:.4f}")

                else:
                    print(f"    success: {sr:.4f}")

            # Overall score = avg(forward, inverse, ae_avg_success)
            parts = []
            if "tag_path_to_view_success" in model_result:
                parts.append(model_result["tag_path_to_view_success"])
            if "tag_view_to_path_success" in model_result:
                parts.append(model_result["tag_view_to_path_success"])
            if "ae_avg_success" in model_result:
                parts.append(model_result["ae_avg_success"])
            model_result["overall_score"] = sum(parts) / len(parts) if parts else 0.0
            print(f"  Overall score: {model_result['overall_score']:.4f}")

            all_model_results[model] = model_result
            # Store action-len results if any task has data
            if any(model_action_len.get(t) for t in TASKS):
                all_action_len_results[model] = model_action_len

            # Per-model CSV
            model_out = out_path / model
            model_out.mkdir(parents=True, exist_ok=True)
            write_per_model_csv(model_result, thresh_list, model_out / "performance.csv")

        # Combined tables
        print(f"\n{'='*60}")
        print("Writing combined tables...")
        print(f"{'='*60}")

        write_combined_csv(all_model_results, thresh_list, out_path / "combined_performance.csv")
        write_combined_md(all_model_results, thresh_list, out_path / "combined_performance.md")

        if all_action_len_results:
            # One table per task
            for task in TASKS:
                task_short = TASK_SHORT[task].lower().replace(" ", "_")
                task_data = {
                    model: {"model": data["model"], task: data.get(task, {})}
                    for model, data in all_action_len_results.items()
                    if data.get(task)
                }
                if task_data:
                    write_action_len_csv(
                        task_data, intervals,
                        out_path / f"success_by_action_len_{task_short}.csv",
                        task_key=task,
                    )
                    write_action_len_md(
                        task_data, intervals,
                        out_path / f"success_by_action_len_{task_short}.md",
                        task_key=task,
                        title=f"Success Rate by Action Sequence Length — {TASK_SHORT[task]}",
                    )

        # Save raw JSON for downstream use
        with open(out_path / "results.json", "w") as f:
            json.dump(all_model_results, f, indent=2)

        print(f"\nAll results saved to {out_path}")

    def refine_glm(
        self,
        rollouts_dir: str = "/root/projects/viewsuite/rollouts",
        model: str = "glm_4_6v",
        viewsuite_data_path: str = "/root/projects/viewsuite/data/viewsuite_15k",
    ) -> None:
        """
        Fix GLM format errors by re-parsing bare-letter answers from transcripts.

        Creates <model>_refined directory with corrected metrics.json files.
        """
        from .glm_refine import refine_model
        refine_model(
            rollouts_dir=Path(rollouts_dir),
            model=model,
            data_path=Path(viewsuite_data_path),
        )


def _parse_thresholds(spec: str) -> List[Tuple[float, float]]:
    """Parse "0.25,15;0.5,30" -> [(0.25, 15.0), (0.5, 30.0)]"""
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(",")
        out.append((float(a.strip()), float(b.strip())))
    return sorted(set(out))


if __name__ == "__main__":
    fire.Fire(PerformanceSummary)
