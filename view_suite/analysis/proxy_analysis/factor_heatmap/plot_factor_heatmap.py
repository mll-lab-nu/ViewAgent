#!/usr/bin/env python3
"""Compute per-sample geometric factors and correlate with per-model success.

Produces a 3-panel heatmap (one per task) of Spearman rho between each
(factor, model) pair, saved as PDF and PNG.

Usage:
    python plot_factor_heatmap.py              # compute from rollouts, dump cache, plot
    python plot_factor_heatmap.py --from-cache # plot from cached factor_heatmap.json only
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JSONL_DIR = Path("/root/projects/viewsuite/data/viewsuite_15k")
ROLLOUT_DIR = Path("/root/projects/viewsuite/data/rollouts/rollouts_all_new")
COVERAGE_DIR = Path("/root/projects/viewsuite/data/rollouts/rollouts_pointcloud_coverage")
PAPER_ROOT = Path("/root/projects/viewsuite/69f16f2122f12968eeb43bf1")
FIG_DIR = PAPER_ROOT / "sections" / "3_evaluation" / "figures"
PAPER_DATA_DIR = PAPER_ROOT / "data"
CACHE_PATH = PAPER_DATA_DIR / "factor_heatmap.json"

TASKS = {
    "tag_path_to_view": {
        "jsonl": JSONL_DIR / "path_to_view_test_filter.jsonl",
        "name": "Path-to-View (P2V)",
        "suffix": "p2v",
    },
    "tag_view_to_path": {
        "jsonl": JSONL_DIR / "view_to_path_test_filter.jsonl",
        "name": "View-to-Path (V2P)",
        "suffix": "v2p",
    },
    "tag_interactive_view_planning": {
        "jsonl": JSONL_DIR / "interactive_view_planning_test_filter.jsonl",
        "name": "Interactive View Planning (IVP)",
        "suffix": "ivp",
    },
}

MODELS = {
    "gpt_5_4_pro": "GPT-5.4 Pro",
    "gemini_3_1_pro": "Gemini 3.1 Pro",
    "gpt_5_4": "GPT-5.4",
    "grok_4_20_beta": "Grok 4.20 Beta",
    "gpt_5_1": "GPT-5.1",
    "claude_opus_4_6": "Claude Opus 4.6",
    "gemini_3_pro": "Gemini 3 Pro",
}

FACTOR_NAMES_A = ["pos_dist", "rot_dist", "unified_dist", "horiz_dist", "height_diff"]
FACTOR_NAMES_B = ["vis_init_norm", "vis_target_norm", "vis_iou"]
FACTOR_NAMES_C = [
    "forward_alignment",
    "target_bearing",
    "target_elevation",
    "orientation_agreement",
]
ALL_FACTORS = FACTOR_NAMES_A + FACTOR_NAMES_B + FACTOR_NAMES_C
GROUP_BOUNDARIES = [len(FACTOR_NAMES_A), len(FACTOR_NAMES_A) + len(FACTOR_NAMES_B)]

# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------


def _extract_pose(c2w):
    """Return (R, t) from a 4x4 c2w matrix (nested list)."""
    m = np.array(c2w, dtype=np.float64)
    R = m[:3, :3]
    t = m[:3, 3]
    return R, t


def compute_factors(entry):
    """Return a dict of factor values for a single JSONL entry."""
    R_init, t_init = _extract_pose(entry["image_detail"]["init_view"]["c2w_extrinsics"])
    R_tgt, t_tgt = _extract_pose(entry["image_detail"]["target_view"]["c2w_extrinsics"])

    # Group A: Geometric Distance
    delta = t_tgt - t_init
    pos_dist = np.linalg.norm(delta)
    # Geodesic rotation distance
    R_rel = R_init.T @ R_tgt
    cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    rot_dist = np.degrees(np.arccos(cos_angle))
    unified_dist = math.sqrt((pos_dist / 0.5) ** 2 + (rot_dist / 30.0) ** 2)
    horiz_dist = np.linalg.norm(delta[:2])
    height_diff = abs(delta[2])

    # Group C: Directional Geometry
    f_init = -R_init[:, 2]  # forward = -z in camera frame
    f_tgt = -R_tgt[:, 2]

    if pos_dist > 1e-9:
        d_hat = delta / pos_dist
        fwd_align = float(np.dot(f_init, d_hat))
    else:
        fwd_align = 0.0

    target_bearing = np.degrees(np.arccos(np.clip(fwd_align, -1.0, 1.0)))
    delta_xy = np.linalg.norm(delta[:2])
    target_elevation = np.degrees(np.arctan2(delta[2], delta_xy))
    orientation_agreement = float(np.dot(f_init, f_tgt))

    return {
        "pos_dist": float(pos_dist),
        "rot_dist": float(rot_dist),
        "unified_dist": float(unified_dist),
        "horiz_dist": float(horiz_dist),
        "height_diff": float(height_diff),
        "forward_alignment": fwd_align,
        "target_bearing": float(target_bearing),
        "target_elevation": float(target_elevation),
        "orientation_agreement": orientation_agreement,
    }


def load_factors(jsonl_path):
    """Return {sample_id: {factor_name: value}}."""
    factors = {}
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            sid = entry["sample_id"]
            factors[sid] = compute_factors(entry)
    return factors


# ---------------------------------------------------------------------------
# Visual overlap from pointcloud coverage (only for CN / interactive_view_planning)
# ---------------------------------------------------------------------------


def load_coverage(model_key):
    """Return {sample_id: {vis_init_norm, vis_target_norm, vis_iou}} from coverage data."""
    results_path = COVERAGE_DIR / model_key / "results.json"
    if not results_path.exists():
        return {}
    with open(results_path) as f:
        data = json.load(f)
    coverage = {}
    for traj in data["trajectories"]:
        sid = traj["sample_id"]
        init_visible = traj["cumulative_counts"][0]  # vertices visible from init
        target_visible = traj["target_vertices"]
        intersection = traj["target_intersections"][0]  # init ∩ target
        union = init_visible + target_visible - intersection
        coverage[sid] = {
            "vis_init_norm": intersection / init_visible if init_visible > 0 else 0.0,
            "vis_target_norm": intersection / target_visible if target_visible > 0 else 0.0,
            "vis_iou": intersection / union if union > 0 else 0.0,
        }
    return coverage


# ---------------------------------------------------------------------------
# Rollout loading
# ---------------------------------------------------------------------------


def load_success(model_key, task_tag):
    """Return {sample_id: bool} for a model/task."""
    task_dir = ROLLOUT_DIR / model_key / task_tag
    results = {}
    if not task_dir.exists():
        return results
    for rollout_id in os.listdir(task_dir):
        metrics_path = task_dir / rollout_id / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        sid = m["infos"][0]["sample_id"]
        results[sid] = bool(m["success"])
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_combined_heatmap(all_rho, all_model_labels, factor_labels, titles, out_prefix):
    """Plot 3 heatmaps side by side sharing y-axis, with one colorbar."""
    n_tasks = len(all_rho)
    n_factors = len(factor_labels)

    fig, axes = plt.subplots(1, n_tasks, figsize=(36, 8),
                             gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.05})
    norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    cmap = plt.get_cmap("RdBu_r")

    for idx, (ax, rho_matrix, model_labels, title) in enumerate(
            zip(axes, all_rho, all_model_labels, titles)):
        n_models = len(model_labels)
        im = ax.imshow(rho_matrix, aspect="auto", cmap=cmap, norm=norm)

        # Cell grid lines
        for i in range(n_models + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.6)
        for j in range(n_factors + 1):
            ax.axvline(j - 0.5, color="gray", linewidth=0.5, alpha=0.6)

        ax.set_xticks(range(n_factors))
        ax.set_xticklabels(factor_labels, rotation=40, ha="right", fontsize=22)

        if idx == 0:
            ax.set_yticks(range(n_models))
            ax.set_yticklabels(model_labels, fontsize=24)
        else:
            ax.set_yticks(range(n_models))
            ax.set_yticklabels([])

        for bnd in GROUP_BOUNDARIES:
            ax.axvline(bnd - 0.5, color="black", linewidth=2)

        ax.set_title(title, fontsize=28, pad=12)

    cbar = fig.colorbar(im, ax=axes.tolist(), label="Spearman ρ",
                        shrink=0.8, pad=0.015)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Spearman ρ", fontsize=24)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_prefix}.pdf / .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def compute_rho_from_rollouts():
    """Read jsonl + rollouts, return (all_rho_reordered, global_labels, all_titles)."""
    all_rho = []
    all_model_labels = []
    all_titles = []

    for task_tag, task_cfg in TASKS.items():
        print(f"\n=== {task_cfg['name']} ({task_tag}) ===")
        factors = load_factors(task_cfg["jsonl"])
        print(f"  Loaded factors for {len(factors)} samples")

        model_success = {}
        model_coverage = {}
        model_rate = {}
        for mk in MODELS:
            succ = load_success(mk, task_tag)
            model_success[mk] = succ
            model_coverage[mk] = load_coverage(mk)
            model_rate[mk] = sum(succ.values()) / len(succ) if succ else 0.0
            print(f"  {MODELS[mk]}: {len(succ)} rollouts, "
                  f"success rate = {model_rate[mk]:.1%}")

        sorted_models = sorted(MODELS.keys(), key=lambda m: -model_rate[m])
        rho_matrix = np.full((len(sorted_models), len(ALL_FACTORS)), np.nan)

        for i, mk in enumerate(sorted_models):
            succ = model_success[mk]
            cov = model_coverage[mk]
            common_sids = sorted(set(factors.keys()) & set(succ.keys()))
            if len(common_sids) < 10:
                continue
            factor_arrays = {fn: [] for fn in ALL_FACTORS}
            success_arr = []
            for sid in common_sids:
                for fn in FACTOR_NAMES_A + FACTOR_NAMES_C:
                    factor_arrays[fn].append(factors[sid][fn])
                if sid in cov:
                    for fn in FACTOR_NAMES_B:
                        factor_arrays[fn].append(cov[sid][fn])
                else:
                    for fn in FACTOR_NAMES_B:
                        factor_arrays[fn].append(float("nan"))
                success_arr.append(float(succ[sid]))

            success_arr = np.array(success_arr)
            for j, fn in enumerate(ALL_FACTORS):
                fa = np.array(factor_arrays[fn], dtype=np.float64)
                valid = ~np.isnan(fa)
                fa_v = fa[valid]
                sa_v = success_arr[valid]
                if len(fa_v) < 10 or np.std(fa_v) < 1e-12 or np.std(sa_v) < 1e-12:
                    rho_matrix[i, j] = 0.0
                else:
                    rho, _ = spearmanr(fa_v, sa_v)
                    rho_matrix[i, j] = rho

        all_rho.append(rho_matrix)
        all_model_labels.append([MODELS[mk] for mk in sorted_models])
        all_titles.append(task_cfg['name'])

    # Use consistent model order across all tasks (sort by mean rate)
    mean_rate = {}
    for mk in MODELS:
        rates = []
        for task_tag in TASKS:
            succ = load_success(mk, task_tag)
            if succ:
                rates.append(sum(succ.values()) / len(succ))
        mean_rate[mk] = np.mean(rates) if rates else 0.0
    global_order = sorted(MODELS.keys(), key=lambda m: -mean_rate[m])
    global_labels = [MODELS[mk] for mk in global_order]

    all_rho_reordered = []
    for task_idx, _ in enumerate(TASKS.items()):
        old_labels = all_model_labels[task_idx]
        old_rho = all_rho[task_idx]
        new_rho = np.full((len(global_order), len(ALL_FACTORS)), np.nan)
        for new_i, mk in enumerate(global_order):
            display = MODELS[mk]
            if display in old_labels:
                old_i = old_labels.index(display)
                new_rho[new_i] = old_rho[old_i]
        all_rho_reordered.append(new_rho)

    return all_rho_reordered, global_labels, all_titles


def _nan_to_none(arr):
    """Convert numpy array (possibly with NaN) to nested list with None for NaN."""
    return [[None if np.isnan(v) else float(v) for v in row] for row in arr]


def _none_to_nan(rows):
    return np.array([[float("nan") if v is None else v for v in row] for row in rows],
                    dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip rollout reading; plot from cached factor_heatmap.json.")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        print(f"Loading cached data from {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        all_rho_reordered = [_none_to_nan(m) for m in cache["rho_matrices"]]
        global_labels = cache["model_labels"]
        all_titles = cache["task_titles"]
    else:
        all_rho_reordered, global_labels, all_titles = compute_rho_from_rollouts()
        dump = {
            "factors": ALL_FACTORS,
            "task_titles": all_titles,
            "model_labels": global_labels,
            "rho_matrices": [_nan_to_none(m) for m in all_rho_reordered],
        }
        with open(CACHE_PATH, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"Saved data dump to {CACHE_PATH}")

    out_prefix = str(FIG_DIR / "factor_heatmap")
    plot_combined_heatmap(all_rho_reordered, [global_labels]*3, ALL_FACTORS, all_titles, out_prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
