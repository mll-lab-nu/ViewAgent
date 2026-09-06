"""Compare two generated corpora: what a sampling change costs, not just what it saves.

Written for the pitch-limit question, but the shape is general. Yield alone is the wrong
metric: any restriction on where the camera may look raises the fraction of views that
survive screening, because the views being screened out are the ones the restriction
removes. The question is whether the surviving distribution is meaningfully poorer.

So this reports both sides:
  - yield: views rejected, samples dropped
  - cost: the pitch and action-mix of the samples that SURVIVED in each corpus

If the surviving distributions match, the restriction only avoided wasted rendering. If
the stricter corpus has visibly less vertical variation among its keepers, that is a
real reduction in what the benchmark tests, and it should be paid for knowingly.

    ~/miniconda3/envs/habitat-gs/bin/python -m \
      view_suite.envs.habitat_gs_proxy_task.data_gen.compare_corpora \
      --a=data/viewagent_habitat_gs --b=data/habitat_gs_p40 --label_a=pitch60 --label_b=pitch40
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Dict, List

import fire
import numpy as np

_CV_TO_GL_3 = np.diag([1.0, -1.0, -1.0])


def _pitch_deg(c2w) -> float:
    R = np.asarray(c2w, dtype=np.float64)[:3, :3] @ _CV_TO_GL_3
    fwd = R @ np.array([0.0, 0.0, -1.0])
    return math.degrees(math.asin(float(np.clip(fwd[1], -1.0, 1.0))))


def _load(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _verdict_stats(root: str) -> Dict:
    p = os.path.join(root, "_verdicts.jsonl")
    if not os.path.exists(p):
        return {}
    c = Counter(json.loads(ln)["verdict"] for ln in open(p) if ln.strip())
    t = sum(c.values())
    return {"judged": t, "filtered": c["FILTER"], "error": c["ERROR"],
            "filter_pct": 100.0 * c["FILTER"] / max(1, t)}


def _corpus_stats(root: str) -> Dict:
    kept = _load(os.path.join(root, "path_to_view.jsonl"))
    pre = _load(os.path.join(root, "path_to_view.jsonl.prefilter")) or kept

    pitches, look_frac, seq_len = [], [], []
    for r in kept:
        for k, v in r["image_detail"].items():
            if k.startswith("view_") and v.get("c2w_extrinsics"):
                pitches.append(abs(_pitch_deg(v["c2w_extrinsics"])))
        names = r["meta"]["gt_action_seq_names"]
        look_frac.append(sum(1 for n in names if n in ("look_up", "look_down")) / max(1, len(names)))
        seq_len.append(len(names))

    pitches = np.array(pitches) if pitches else np.array([0.0])
    return {
        "generated": len(pre), "kept": len(kept),
        "drop_pct": 100.0 * (1 - len(kept) / max(1, len(pre))),
        "pitch_mean": float(pitches.mean()),
        "pitch_p90": float(np.percentile(pitches, 90)),
        "pitch_gt30_pct": float(100.0 * (pitches > 30).mean()),
        "pitch_eq0_pct": float(100.0 * (pitches < 1e-6).mean()),
        "look_frac_mean": float(np.mean(look_frac)) if look_frac else 0.0,
        "seq_len_mean": float(np.mean(seq_len)) if seq_len else 0.0,
        "verdicts": _verdict_stats(root),
    }


def run(a: str, b: str, label_a: str = "A", label_b: str = "B") -> None:
    sa, sb = _corpus_stats(a), _corpus_stats(b)

    def row(name, ka, kb, fmt="{:.1f}"):
        va, vb = ka, kb
        print(f"  {name:34s} {fmt.format(va):>10s} {fmt.format(vb):>10s}")

    print(f"\n{'':36s} {label_a:>10s} {label_b:>10s}")
    print("  " + "-" * 56)
    print("  YIELD")
    va, vb = sa.get("verdicts", {}), sb.get("verdicts", {})
    if va and vb:
        row("views judged", va["judged"], vb["judged"], "{:.0f}")
        row("views rejected %", va["filter_pct"], vb["filter_pct"])
    row("samples generated", sa["generated"], sb["generated"], "{:.0f}")
    row("samples kept", sa["kept"], sb["kept"], "{:.0f}")
    row("samples dropped %", sa["drop_pct"], sb["drop_pct"])

    print("  COST -- distribution among the samples that SURVIVED")
    row("mean |pitch| of kept views", sa["pitch_mean"], sb["pitch_mean"])
    row("p90 |pitch|", sa["pitch_p90"], sb["pitch_p90"])
    row("kept views with |pitch| > 30 %", sa["pitch_gt30_pct"], sb["pitch_gt30_pct"])
    row("kept views level (pitch = 0) %", sa["pitch_eq0_pct"], sb["pitch_eq0_pct"])
    row("look actions per GT sequence", sa["look_frac_mean"], sb["look_frac_mean"], "{:.3f}")
    row("GT sequence length", sa["seq_len_mean"], sb["seq_len_mean"], "{:.2f}")
    print()


if __name__ == "__main__":
    fire.Fire(run)
