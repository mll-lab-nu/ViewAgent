#!/usr/bin/env python3
"""
Easy/Hard performance analysis based on unified view distance.

Unified view distance:  d = sqrt((d_pos / 0.5)^2 + (d_rot / 30)^2)
  where d_pos = Euclidean distance between init and target camera centers (meters)
        d_rot = geodesic angle between init and target rotations (degrees)

Threshold: d < 3 => Easy, d >= 3 => Hard

Outputs:
  1. Performance table (Easy / Hard / All) for each task, sorted by Overall
  2. Unified view distance distribution histogram for all three tasks
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────

DEFAULT_ROLLOUTS_DIR = Path("/root/projects/viewsuite/rollouts_all")
DATA_DIR = Path("/root/projects/viewsuite/data/viewsuite_15k")

EASY_THRESHOLD = 3.0  # unified view distance threshold
STEP_TRANSLATION_M = 0.5
STEP_ROTATION_DEG = 30.0

TASKS = ["tag_path_to_view", "tag_view_to_path", "tag_interactive_view_planning"]
TASK_DISPLAY = {
    "tag_path_to_view": "Path2View",
    "tag_view_to_path": "View2Path",
    "tag_interactive_view_planning": "Interactive View Planning",
}
TASK_JSONL = {
    "tag_path_to_view": "path_to_view_test_filter.jsonl",
    "tag_view_to_path": "view_to_path_test_filter.jsonl",
    "tag_interactive_view_planning": "interactive_view_planning_test_filter.jsonl",
}

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
    "kimi_k2_5": "Kimi K2.5",
    "random_response": "Random Response",
}

# Models to skip in the table
SKIP_MODELS = {"glm_4_6v", "qwen_25_vl_7b_trained"}


# ── Pose utilities ──────────────────────────────────────────────────────────

def c2w_to_se3(c2w: list) -> np.ndarray:
    """Convert 4x4 c2w matrix (list of lists) to SE(3) [tx,ty,tz,rx,ry,rz] in degrees."""
    M = np.array(c2w, dtype=np.float64)
    R_c2w = M[:3, :3]
    t_c2w = M[:3, 3]
    eul = R.from_matrix(R_c2w).as_euler('xyz', degrees=True)
    return np.concatenate([t_c2w, eul])


def geodesic_angle_deg(euler_a: np.ndarray, euler_b: np.ndarray) -> float:
    """Geodesic angle between two rotations (Euler XYZ degrees). Returns [0, 180]."""
    Ra = R.from_euler('xyz', np.asarray(euler_a, dtype=np.float64), degrees=True).as_matrix()
    Rb = R.from_euler('xyz', np.asarray(euler_b, dtype=np.float64), degrees=True).as_matrix()
    Rrel = Ra @ Rb.T
    cos_theta = (np.trace(Rrel) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def unified_view_distance(init_c2w: list, target_c2w: list) -> float:
    """Compute unified view distance between two c2w poses."""
    init_se3 = c2w_to_se3(init_c2w)
    target_se3 = c2w_to_se3(target_c2w)
    d_pos = np.linalg.norm(init_se3[:3] - target_se3[:3])
    d_rot = geodesic_angle_deg(init_se3[3:], target_se3[3:])
    return float(np.sqrt((d_pos / STEP_TRANSLATION_M) ** 2 + (d_rot / STEP_ROTATION_DEG) ** 2))


# ── Data loading ────────────────────────────────────────────────────────────

def load_jsonl_index(jsonl_path: Path) -> Dict[str, dict]:
    """Build sample_id -> JSONL entry index."""
    idx = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = item.get("sample_id", "")
            if sid:
                idx[sid] = item
    return idx


def compute_view_distances(jsonl_index: Dict[str, dict]) -> Dict[str, float]:
    """Compute unified view distance for each sample_id from JSONL data."""
    distances = {}
    for sid, item in jsonl_index.items():
        img_detail = item.get("image_detail", {})
        init_view = img_detail.get("init_view", {})
        target_view = img_detail.get("target_view", {})
        init_c2w = init_view.get("c2w_extrinsics")
        target_c2w = target_view.get("c2w_extrinsics")
        if init_c2w is None or target_c2w is None:
            continue
        try:
            distances[sid] = unified_view_distance(init_c2w, target_c2w)
        except Exception as e:
            print(f"  Warning: failed to compute distance for {sid}: {e}")
    return distances


def read_rollout_metrics(task_dir: Path) -> Dict[str, dict]:
    """Read metrics from rollout dirs, keyed by sample_id."""
    result = {}
    for rollout_dir in task_dir.iterdir():
        if not rollout_dir.is_dir():
            continue
        metrics_path = rollout_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path) as f:
                m = json.load(f)
        except Exception:
            continue
        sid = m.get("sample_id", "")
        if not sid:
            infos = m.get("infos", [])
            for info in infos:
                if isinstance(info, dict) and info.get("sample_id"):
                    sid = str(info["sample_id"])
                    break
        if sid:
            result[sid] = m
    return result


AE_POS_THRESHOLD = 0.5   # meters
AE_ANG_THRESHOLD = 30.0  # degrees


def is_success_ae(m: dict) -> bool:
    """Check success for active exploration using 0.5m/30deg threshold."""
    pos = m.get("pos_err_m")
    ang = m.get("ang_err_deg")
    if pos is None or ang is None:
        infos = m.get("infos", [])
        if infos and isinstance(infos[-1], dict):
            pos = infos[-1].get("pos_err_m")
            ang = infos[-1].get("ang_err_deg")
    if pos is not None and ang is not None:
        return pos <= AE_POS_THRESHOLD + 1e-9 and ang <= AE_ANG_THRESHOLD + 1e-9
    return False


def get_success(m: dict, task: str) -> bool:
    """Determine success for a rollout."""
    if task == "tag_interactive_view_planning":
        return is_success_ae(m)
    return bool(m.get("success", False))


# ── Main analysis ───────────────────────────────────────────────────────────

def discover_models(rollouts_dir: Path) -> List[str]:
    models = []
    for d in sorted(rollouts_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name in SKIP_MODELS:
            continue
        if any((d / t).is_dir() for t in TASKS):
            models.append(d.name)
    return models


def run_analysis(rollouts_dir: str = None):
    ROLLOUTS_DIR = Path(rollouts_dir) if rollouts_dir else DEFAULT_ROLLOUTS_DIR
    OUTPUT_DIR = ROLLOUTS_DIR.parent / (ROLLOUTS_DIR.name + "_easy_hard_analysis")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Rollouts dir: {ROLLOUTS_DIR}")
    print(f"Output dir:   {OUTPUT_DIR}")

    # Load JSONL indices and compute view distances per task
    print("Loading JSONL data and computing view distances...")
    task_distances: Dict[str, Dict[str, float]] = {}
    task_jsonl_indices: Dict[str, Dict[str, dict]] = {}
    for task in TASKS:
        jsonl_path = DATA_DIR / TASK_JSONL[task]
        idx = load_jsonl_index(jsonl_path)
        task_jsonl_indices[task] = idx
        distances = compute_view_distances(idx)
        task_distances[task] = distances
        n_easy = sum(1 for d in distances.values() if d < EASY_THRESHOLD)
        n_hard = sum(1 for d in distances.values() if d >= EASY_THRESHOLD)
        print(f"  {TASK_DISPLAY[task]}: {len(distances)} samples, "
              f"{n_easy} easy, {n_hard} hard")

    # Discover models
    models = discover_models(ROLLOUTS_DIR)
    print(f"\nModels ({len(models)}): {[MODEL_DISPLAY.get(m, m) for m in models]}\n")

    # Compute per-model, per-task results
    # results[model][task] = {"easy": rate, "hard": rate, "all": rate}
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model in models:
        results[model] = {}
        for task in TASKS:
            task_dir = ROLLOUTS_DIR / model / task
            if not task_dir.is_dir():
                continue
            metrics_by_sid = read_rollout_metrics(task_dir)
            distances = task_distances[task]

            easy_total, easy_success = 0, 0
            hard_total, hard_success = 0, 0

            for sid, m in metrics_by_sid.items():
                if sid not in distances:
                    continue
                d = distances[sid]
                success = get_success(m, task)
                if d < EASY_THRESHOLD:
                    easy_total += 1
                    if success:
                        easy_success += 1
                else:
                    hard_total += 1
                    if success:
                        hard_success += 1

            all_total = easy_total + hard_total
            results[model][task] = {
                "easy": (easy_success / easy_total * 100) if easy_total > 0 else 0.0,
                "hard": (hard_success / hard_total * 100) if hard_total > 0 else 0.0,
                "all": ((easy_success + hard_success) / all_total * 100) if all_total > 0 else 0.0,
                "easy_n": easy_total,
                "hard_n": hard_total,
                "easy_s": easy_success,
                "hard_s": hard_success,
            }

    # Compute overall score for each model
    overall_scores = {}
    for model in models:
        task_alls = [results[model][t]["all"] for t in TASKS if t in results[model]]
        overall_scores[model] = sum(task_alls) / len(task_alls) if task_alls else 0.0

    # Sort models by overall score descending
    sorted_models = sorted(models, key=lambda m: overall_scores[m], reverse=True)

    # ── Write table ─────────────────────────────────────────────────────────

    # Build markdown table
    lines = []
    header = "| Model | " + " | ".join(
        f"{TASK_DISPLAY[t]} Easy | {TASK_DISPLAY[t]} Hard | {TASK_DISPLAY[t]} All"
        for t in TASKS
    ) + " | Overall |"
    sep = "|" + "|".join(["---"] * (1 + 3 * len(TASKS) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for model in sorted_models:
        display = MODEL_DISPLAY.get(model, model)
        row = f"| {display} "
        for task in TASKS:
            r = results[model].get(task, {"easy": 0, "hard": 0, "all": 0})
            row += f"| {r['easy']:.1f} | {r['hard']:.1f} | **{r['all']:.1f}** "
        row += f"| **{overall_scores[model]:.1f}** |"
        lines.append(row)

    table_md = "\n".join(lines)
    print(table_md)
    print()

    with open(OUTPUT_DIR / "easy_hard_performance.md", "w") as f:
        f.write("# Easy/Hard Performance by Unified View Distance\n\n")
        f.write(f"Unified view distance: d = sqrt((d_pos/{STEP_TRANSLATION_M})² + (d_rot/{STEP_ROTATION_DEG})²)\n\n")
        f.write(f"Easy: d < {EASY_THRESHOLD}, Hard: d ≥ {EASY_THRESHOLD}\n\n")
        # Show sample counts
        for task in TASKS:
            distances = task_distances[task]
            n_easy = sum(1 for d in distances.values() if d < EASY_THRESHOLD)
            n_hard = sum(1 for d in distances.values() if d >= EASY_THRESHOLD)
            f.write(f"- {TASK_DISPLAY[task]}: {n_easy} easy, {n_hard} hard ({len(distances)} total)\n")
        f.write("\n")
        f.write(table_md + "\n")

    # Also write CSV
    with open(OUTPUT_DIR / "easy_hard_performance.csv", "w") as f:
        cols = ["Model"]
        for task in TASKS:
            tn = TASK_DISPLAY[task]
            cols.extend([f"{tn} Easy", f"{tn} Hard", f"{tn} All"])
        cols.append("Overall")
        f.write(",".join(cols) + "\n")
        for model in sorted_models:
            display = MODEL_DISPLAY.get(model, model)
            row = [display]
            for task in TASKS:
                r = results[model].get(task, {"easy": 0, "hard": 0, "all": 0})
                row.extend([f"{r['easy']:.1f}", f"{r['hard']:.1f}", f"{r['all']:.1f}"])
            row.append(f"{overall_scores[model]:.1f}")
            f.write(",".join(row) + "\n")

    # Also write LaTeX-style table
    _write_latex_table(sorted_models, results, overall_scores, task_distances, OUTPUT_DIR)

    # ── Write distribution plot (single figure, all tasks share the same samples) ──

    dists_by_sid = task_distances[TASKS[0]]  # same for all tasks
    dists = list(dists_by_sid.values())
    n_easy = sum(1 for d in dists if d < EASY_THRESHOLD)
    n_hard = sum(1 for d in dists if d >= EASY_THRESHOLD)

    # Write distribution CSV
    with open(OUTPUT_DIR / "view_distance_distribution.csv", "w") as f:
        f.write("sample_id,unified_view_distance,difficulty\n")
        for sid in sorted(dists_by_sid.keys()):
            d = dists_by_sid[sid]
            label = "easy" if d < EASY_THRESHOLD else "hard"
            f.write(f"{sid},{d:.4f},{label}\n")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dists, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(x=EASY_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"$d = {EASY_THRESHOLD:.0f}$")
    ax.set_xlim(left=1)
    ax.set_xlabel("Unified View Distance  $d$", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Unified View Distance Distribution", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "view_distance_distribution.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "view_distance_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Distribution plot saved to {OUTPUT_DIR / 'view_distance_distribution.png'}")

    # Save raw JSON results
    raw = {
        "config": {
            "easy_threshold": EASY_THRESHOLD,
            "step_translation_m": STEP_TRANSLATION_M,
            "step_rotation_deg": STEP_ROTATION_DEG,
        },
        "task_sample_counts": {
            task: {
                "total": len(task_distances[task]),
                "easy": sum(1 for d in task_distances[task].values() if d < EASY_THRESHOLD),
                "hard": sum(1 for d in task_distances[task].values() if d >= EASY_THRESHOLD),
            }
            for task in TASKS
        },
        "results": {
            MODEL_DISPLAY.get(m, m): {
                TASK_DISPLAY[t]: results[m][t]
                for t in TASKS if t in results[m]
            }
            for m in sorted_models
        },
        "overall_scores": {MODEL_DISPLAY.get(m, m): overall_scores[m] for m in sorted_models},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}")


def _write_latex_table(
    sorted_models: List[str],
    results: Dict[str, Dict[str, Dict[str, float]]],
    overall_scores: Dict[str, float],
    task_distances: Dict[str, Dict[str, float]],
    OUTPUT_DIR: Path,
):
    """Write a LaTeX-formatted table similar to the reference image."""
    lines = []
    lines.append(r"\begin{tabular}{l" + "ccc" * len(TASKS) + "c}")
    lines.append(r"\toprule")

    # Header row 1: task names spanning 3 columns each
    header1 = " & "
    for task in TASKS:
        header1 += r"\multicolumn{3}{c}{" + TASK_DISPLAY[task] + "} & "
    header1 = header1.rstrip("& ") + r" & \\"
    # Actually let me redo this more carefully
    parts = ["Model"]
    for task in TASKS:
        parts.append(r"\multicolumn{3}{c}{" + TASK_DISPLAY[task] + "}")
    parts.append("Overall")
    lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")

    # Header row 2: Easy/Hard/All
    parts2 = [""]
    for _ in TASKS:
        parts2.extend(["Easy", "Hard", "All"])
    parts2.append("")
    lines.append(" & ".join(parts2) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for model in sorted_models:
        display = MODEL_DISPLAY.get(model, model)
        parts = [display]
        for task in TASKS:
            r = results[model].get(task, {"easy": 0, "hard": 0, "all": 0})
            parts.extend([f"{r['easy']:.1f}", f"{r['hard']:.1f}", f"{r['all']:.1f}"])
        parts.append(f"\\textbf{{{overall_scores[model]:.1f}}}")
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(OUTPUT_DIR / "easy_hard_performance.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import fire
    fire.Fire(run_analysis)
