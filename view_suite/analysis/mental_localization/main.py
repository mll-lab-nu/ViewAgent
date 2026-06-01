#!/usr/bin/env python3
"""
Mental-localization analysis for Interactive View Planning (IVP).

Question
--------
In IVP, an agent does **not** need to physically reproduce the target view: in
principle it can take a few informative moves, build a local coordinate frame,
infer the spatial relation between observed views and the target, and *submit*
the target pose without ever revisiting it.  Whether current VLMs actually do
this is an empirical question.

For every IVP rollout we ask: did the agent ever *observe* a camera view that is
itself within the success threshold of the target (i.e. it "saw" the answer), or
did it submit a correct pose *without* ever visiting a view that close (genuine
mental localization)?

Method
------
A rollout counts as **successful** using the benchmark's own success flag
(``metrics.json: success``, equivalent to position error <= 0.5 m AND rotation
error <= 30 deg between the *submitted* pose and the target).

For each successful rollout we replay the trajectory and reuse the exact
criterion that the ``no_submit`` IVP variant applies as an *auto-success* check
(see ``interactive_view_planning.py``): a view is "reached" if its camera pose
satisfies

    pos_err_m <= pos_threshold_m + 1e-9   AND   ang_err_deg <= ang_threshold_deg + 1e-9

where ``pos_err_m`` is the Euclidean translation distance and ``ang_err_deg`` is
the geodesic angle between Euler-XYZ rotations (identical to ``geodesic_angle_deg``).
The set of observed views is the initial view plus every post-action
``Current camera`` observation, excluding the terminal answer turn.

We report, per model:

    inferred-without-visiting (%) = (# successful rollouts that never reached a
                                     view within threshold) / (# successful rollouts)

A high value means successes come from mental localization; a low value means
successes are coupled to view matching (the agent had to fly into the target
view before it could answer).

Usage
-----
    python -m view_suite.analysis.mental_localization.main \
        --rollouts_dir /home/kangrui/projects/viewsuite/data/rollouts/rollouts_all_new \
        --output_dir   /home/kangrui/projects/viewsuite/data/rollouts/rollouts_all_new_mental_localization \
        --models gpt_5_4,gpt_5_4_pro,gemini_3_1_pro,claude_opus_4_6,grok_4_20_beta
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import fire


# IVP task directory name inside each model folder.
IVP_TAG = "tag_active_exploration"

# Fallback thresholds (the benchmark's main IVP success criterion).
DEFAULT_POS_THRESHOLD_M = 0.5
DEFAULT_ANG_THRESHOLD_DEG = 30.0

MODEL_DISPLAY = {
    "claude_opus_4_6": "Claude Opus 4.6",
    "gemini_3_1_pro": "Gemini 3.1 Pro",
    "gemini_3_pro": "Gemini 3 Pro",
    "gpt_5_1": "GPT-5.1",
    "gpt_5_4": "GPT-5.4",
    "gpt_5_4_pro": "GPT-5.4 Pro",
    "grok_4_20_beta": "Grok 4.20 Beta",
}


# ---------------------------------------------------------------------------
# Pose error (mirrors view_suite.envs...gym_proxy_tool_utils.geodesic_angle_deg)
# ---------------------------------------------------------------------------

def geodesic_angle_deg(euler_a_deg: np.ndarray, euler_b_deg: np.ndarray) -> float:
    """Geodesic angle (deg, in [0,180]) between two Euler-XYZ rotations."""
    Ra = R.from_euler("xyz", np.asarray(euler_a_deg, dtype=np.float64), degrees=True).as_matrix()
    Rb = R.from_euler("xyz", np.asarray(euler_b_deg, dtype=np.float64), degrees=True).as_matrix()
    Rrel = Ra @ Rb.T
    cos_theta = float(np.clip((np.trace(Rrel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def pose_errors(pose: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """(pos_err_m, ang_err_deg) between two 6-DoF SE(3) poses [tx,ty,tz,rx,ry,rz]."""
    pos_err = float(np.linalg.norm(pose[:3] - target[:3]))
    ang_err = geodesic_angle_deg(pose[3:], target[3:])
    return pos_err, ang_err


# ---------------------------------------------------------------------------
# Parsing messages.json
# ---------------------------------------------------------------------------

def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


_SE3_BODY = (
    r"\[tx=([-\d.]+),\s*ty=([-\d.]+),\s*tz=([-\d.]+),\s*"
    r"rx=([-\d.]+)\xb0?,\s*ry=([-\d.]+)\xb0?,\s*rz=([-\d.]+)\xb0?\]"
)
_CURRENT_CAMERA_RE = re.compile(r"Current camera[^\n]*\n" + _SE3_BODY)
_INITIAL_VIEW_RE = re.compile(r"Initial view camera[^\n]*\n" + _SE3_BODY)
_GT_RE = re.compile(r"gt=" + _SE3_BODY)


def _se3_from_match(m: re.Match) -> np.ndarray:
    return np.array([float(m.group(i)) for i in range(1, 7)], dtype=np.float64)


def parse_transcript(transcript: str) -> Optional[Dict[str, Any]]:
    """
    Extract the target pose and the sequence of *observed* camera poses from a
    rollout ``transcript.txt``.

    ``transcript.txt`` is used instead of ``messages.json`` because it is complete
    and uniform across all models (some models, e.g. GPT-5.4 Pro, ship
    ``messages.json`` files whose final answer-feedback content blocks are empty).

    The observed views are the ``Initial view camera`` pose plus every
    ``Current camera`` pose that appears *before* the terminal ``[answer]`` line.
    The ``[answer]`` line itself carries the ground-truth ``gt=[...]`` pose; any
    ``Current camera`` block printed after it (the unchanged post-answer pose) is
    ignored.

    Returns dict with:
        target:   np.ndarray (6,)  -- ground-truth target pose
        observed: List[np.ndarray] -- observed camera poses (pre-answer)
    or None if the target pose cannot be recovered.
    """
    # Truncate at the answer line so post-answer "Current camera" blocks are dropped.
    answer_idx = transcript.find("[answer]")
    pre = transcript if answer_idx < 0 else transcript[:answer_idx]

    observed: List[np.ndarray] = []
    init = _INITIAL_VIEW_RE.search(pre)
    if init is not None:
        observed.append(_se3_from_match(init))
    for cur in _CURRENT_CAMERA_RE.finditer(pre):
        observed.append(_se3_from_match(cur))

    gt = _GT_RE.search(transcript)
    if gt is None:
        return None
    target = _se3_from_match(gt)
    return {"target": target, "observed": observed}


# ---------------------------------------------------------------------------
# Per-rollout / per-model analysis
# ---------------------------------------------------------------------------

def _thresholds(metrics: Dict[str, Any]) -> Tuple[float, float]:
    """Per-sample success thresholds, falling back to the benchmark default."""
    for src in (metrics, *(metrics.get("infos") or [])):
        if isinstance(src, dict):
            pt = src.get("pos_threshold_m")
            at = src.get("ang_threshold_deg")
            if pt is not None and at is not None:
                return float(pt), float(at)
    return DEFAULT_POS_THRESHOLD_M, DEFAULT_ANG_THRESHOLD_DEG


def analyze_rollout(traj_dir: Path) -> Optional[Dict[str, Any]]:
    metrics = json.loads((traj_dir / "metrics.json").read_text())
    transcript = (traj_dir / "transcript.txt").read_text()

    parsed = parse_transcript(transcript)
    if parsed is None:
        return None

    pos_thr, ang_thr = _thresholds(metrics)
    target = parsed["target"]

    # Closest approach over all observed views, and whether any view was "reached".
    best_pos = float("inf")
    best_ang = float("inf")
    reached = False
    for pose in parsed["observed"]:
        pe, ae = pose_errors(pose, target)
        if pe < best_pos:
            best_pos = pe
        if ae < best_ang:
            best_ang = ae
        if pe <= pos_thr + 1e-9 and ae <= ang_thr + 1e-9:
            reached = True

    return {
        "traj_id": traj_dir.name,
        "success": bool(metrics.get("success", False)),
        "reached": reached,
        "n_observed": len(parsed["observed"]),
        "min_pos_err_m": best_pos,
        "min_ang_err_deg": best_ang,
        "pos_threshold_m": pos_thr,
        "ang_threshold_deg": ang_thr,
    }


def analyze_model(model_dir: Path) -> Dict[str, Any]:
    task_dir = model_dir / IVP_TAG
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for traj_dir in sorted(task_dir.iterdir()):
        if not traj_dir.is_dir():
            continue
        try:
            r = analyze_rollout(traj_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] {traj_dir.name}: {e}")
            r = None
        if r is None:
            skipped += 1
            continue
        rows.append(r)

    success_rows = [r for r in rows if r["success"]]
    n_success = len(success_rows)
    n_reached = sum(1 for r in success_rows if r["reached"])
    n_inferred = n_success - n_reached

    # Sanity counterfactual: rollouts that reached a view but still failed.
    n_reached_all = sum(1 for r in rows if r["reached"])
    n_reached_but_failed = sum(1 for r in rows if r["reached"] and not r["success"])

    return {
        "n_rollouts": len(rows),
        "n_skipped": skipped,
        "n_success": n_success,
        "n_success_reached": n_reached,
        "n_success_inferred": n_inferred,
        "inferred_rate": (n_inferred / n_success) if n_success else 0.0,
        "reached_rate_among_success": (n_reached / n_success) if n_success else 0.0,
        "n_reached_all": n_reached_all,
        "n_reached_but_failed": n_reached_but_failed,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Output: LaTeX table
# ---------------------------------------------------------------------------

def write_latex_table(results: Dict[str, Dict[str, Any]], order: List[str], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Visual re-encounter vs.\ mental localization in IVP. Each model's successful",
        r"IVP rollouts, split by whether the agent ever observed a view within the success",
        r"threshold ($0.5$m / $30^\circ$) of the target before answering.}",
        r"\label{tab:mental_localization}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Successful rollouts} \\",
        r"\cmidrule(lr){3-4}",
        r"Model & \#Success & Visited target view & Inferred (no visit) \\",
        r"\midrule",
    ]
    for model in order:
        res = results.get(model)
        if not res:
            continue
        name = MODEL_DISPLAY.get(model, model)
        n = res["n_success"]
        reached_pct = 100 * res["reached_rate_among_success"]
        inferred_pct = 100 * res["inferred_rate"]
        lines.append(
            f"{name} & {n} & "
            f"{res['n_success_reached']} ({reached_pct:.1f}\\%) & "
            f"{res['n_success_inferred']} ({inferred_pct:.1f}\\%) \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(
    rollouts_dir: str,
    output_dir: Optional[str] = None,
    models: Any = "gpt_5_4,gpt_5_4_pro,gemini_3_1_pro,claude_opus_4_6,grok_4_20_beta",
) -> None:
    rollouts_path = Path(rollouts_dir)
    out_path = Path(output_dir) if output_dir else rollouts_path.parent / (
        rollouts_path.name + "_mental_localization"
    )
    out_path.mkdir(parents=True, exist_ok=True)

    if isinstance(models, (list, tuple)):
        model_list = [str(m).strip() for m in models if str(m).strip()]
    else:
        model_list = [m.strip() for m in str(models).split(",") if m.strip()]

    results: Dict[str, Dict[str, Any]] = {}
    print(f"{'Model':18s} {'#succ':>6s} {'reached':>8s} {'inferred':>9s} {'inferred%':>10s}")
    print("-" * 56)
    for model in model_list:
        model_dir = rollouts_path / model
        if not (model_dir / IVP_TAG).is_dir():
            print(f"{model:18s}  MISSING {IVP_TAG}")
            continue
        res = analyze_model(model_dir)
        results[model] = res
        print(
            f"{model:18s} {res['n_success']:6d} {res['n_success_reached']:8d} "
            f"{res['n_success_inferred']:9d} {100 * res['inferred_rate']:9.1f}%"
        )

    # Persist: raw JSON (rows dropped for the compact summary), full JSON, LaTeX.
    summary = {
        m: {k: v for k, v in res.items() if k != "rows"} for m, res in results.items()
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_path / "results_full.json").write_text(json.dumps(results, indent=2))
    write_latex_table(results, model_list, out_path / "mental_localization_table.tex")

    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    fire.Fire({"run": run})
