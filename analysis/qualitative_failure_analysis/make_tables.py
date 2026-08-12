#!/usr/bin/env python3
"""Build the three failure-category tables.
IVP: from ivp_table.json (full data-driven partition, see categorize.py).
P2V/V2P: from mcq_labels/*.json (LLM-coded sample) + mcq_meta.json (exact format rate)."""
import json, os
from collections import Counter
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["gpt_5_4", "gpt_5_4_pro", "gemini_3_1_pro", "claude_opus_4_6", "grok_4_20_beta"]
# Trained Qwen (proposed method) has no MCQ reasoning to code (format collapse + no CoT),
# so it is added to the IVP table only.
IVP_MODELS = MODELS + ["qwen_25_vl_7b_trained"]

# ---- MCQ tables ----
meta = json.load(open(os.path.join(BASE, "mcq_meta.json")))
CATS = ["GEO_MISGROUND", "DIR_SIGN", "MAG_COUNT", "SEM", "GUESS"]
disp = {"GEO_MISGROUND": "Coherent geometry, wrong visual grounding",
        "DIR_SIGN": "Direction / sign error", "MAG_COUNT": "Magnitude / count error",
        "SEM": "Semantic-only matching", "GUESS": "Low-confidence guess"}
for task in ["P2V", "V2P"]:
    print(f"\n### {task} (% of each model's failures)")
    print("| Category | " + " | ".join(MODELS) + " |")
    cols = {}
    for m in MODELS:
        lab = json.load(open(os.path.join(BASE, "mcq_labels", f"{task}_{m}.json")))
        c = Counter(lab.values()); n = len(lab)
        fr = meta[f"{task}/{m}"]["format_rate"] / 100.0
        cols[m] = {cat: round(100 * c.get(cat, 0) / n * (1 - fr), 1) for cat in CATS}
        cols[m]["FORMAT"] = round(100 * fr, 1)
    for cat in CATS:
        print(f"| {disp[cat]} | " + " | ".join(f"{cols[m][cat]}" for m in MODELS) + " |")
    print("| Format / parse error | " + " | ".join(f"{cols[m]['FORMAT']}" for m in MODELS) + " |")

# ---- IVP table ----
ivp = json.load(open(os.path.join(BASE, "ivp_table.json")))
order = [("no_answer", "No valid answer (ran out / unparseable)"),
         ("snap_answer", "Snap-answer, no exploration (turn 0)"),
         ("orientation_flip", "Orientation flip (ang err >= 150 deg)"),
         ("both_off", "Both position & rotation off"),
         ("position_off_only", "Position off only (rot within 30 deg)"),
         ("rotation_off_only", "Rotation off only (pos within 0.5 m)")]
print("\n### IVP (% of each model's failures)")
print("| Category | " + " | ".join(IVP_MODELS) + " |")
for k, label in order:
    print(f"| {label} | " + " | ".join(f"{ivp[m][k]}" for m in IVP_MODELS) + " |")
print("| (n failures) | " + " | ".join(f"{ivp[m]['n']}" for m in IVP_MODELS) + " |")

# Trained-Qwen IVP success by horizon (where the method breaks down)
import glob
recs = [json.load(open(os.path.join(BASE, "analysis_dump.json")))]  # noqa
dump = recs[0]
q = dump["ivp_records"]["qwen_25_vl_7b_trained"]
buckets = {"Short (1-2)": (0, 2), "Medium (3-5)": (3, 5), "Long (6+)": (6, 99)}
print("\n### Trained-Qwen IVP success by GT-path length")
print("| horizon | n | success |")
for name, (lo, hi) in buckets.items():
    sub = [r for r in q if lo <= (r["gt_len"] or 0) <= hi]
    sc = sum(r["success"] for r in sub)
    print(f"| {name} | {len(sub)} | {100*sc/max(len(sub),1):.1f}% |")
