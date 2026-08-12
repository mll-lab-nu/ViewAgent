#!/usr/bin/env python3
"""Build mutually-exclusive failure-category partitions.
- IVP: fully programmatic from pose/behavior signals.
- P2V/V2P: measure format-error rate exactly; write random samples of non-format
  failures (with the model's reasoning) for LLM coding."""
import json, os, random
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(BASE, "analysis_dump.json")))
MODELS = ["gpt_5_4", "gpt_5_4_pro", "gemini_3_1_pro", "claude_opus_4_6", "grok_4_20_beta", "qwen_25_vl_7b_trained"]
random.seed(0)

# ---------------- IVP partition ----------------
IVP_CATS = ["no_answer", "snap_answer", "orientation_flip",
            "both_off", "position_off_only", "rotation_off_only"]

def ivp_cat(r):
    if not r["answered"] or r["pos"] is None or r["ang"] is None:
        return "no_answer"
    if r["ans_turn"] == 0:
        return "snap_answer"
    if r["ang"] >= 150:
        return "orientation_flip"
    if r["pos"] > 0.5 and r["ang"] > 30:
        return "both_off"
    if r["pos"] > 0.5 and r["ang"] <= 30:
        return "position_off_only"
    if r["pos"] <= 0.5 and r["ang"] > 30:
        return "rotation_off_only"
    return "both_off"  # safety

ivp_table = {}
for m in MODELS:
    fails = [r for r in d["ivp_records"][m] if not r["success"]]
    c = Counter(ivp_cat(r) for r in fails)
    n = len(fails)
    ivp_table[m] = {"n": n, **{k: round(100*c[k]/n, 1) for k in IVP_CATS}}

print("=== IVP failure partition (% of each model's failures) ===")
print(f"{'category':20s}" + "".join(f"{m[:12]:>13s}" for m in MODELS))
for k in IVP_CATS:
    print(f"{k:20s}" + "".join(f"{ivp_table[m][k]:12.1f}%" for m in MODELS))
print(f"{'(n failures)':20s}" + "".join(f"{ivp_table[m]['n']:12d} " for m in MODELS))
json.dump(ivp_table, open(os.path.join(BASE, "ivp_table.json"), "w"), indent=1)

# ---------------- MCQ: format rate + sample for coding ----------------
mcq_meta = {}
os.makedirs(os.path.join(BASE, "mcq_samples"), exist_ok=True)
N = 60
for task in ["P2V", "V2P"]:
    for m in MODELS:
        s = d["mcq_fail_samples"].get(f"{m}/{task}", [])
        nfail = len(s)
        nfmt = sum(1 for x in s if x["fmt_bad"])
        withreason = [x for x in s if not x["fmt_bad"] and (x["reason"] or x["raw"])]
        samp = random.sample(withreason, min(N, len(withreason)))
        # keep reasoning (prefer <think>, else raw text before action)
        out = []
        for x in samp:
            txt = x["reason"] or x["raw"]
            out.append({"id": x["rollout"], "chose": x["parsed"], "reason": txt[:700]})
        json.dump(out, open(os.path.join(BASE, "mcq_samples", f"{task}_{m}.json"), "w"), indent=1)
        mcq_meta[f"{task}/{m}"] = {"n_fail": nfail, "n_format_err": nfmt,
                                   "format_rate": round(100*nfmt/max(nfail,1), 1),
                                   "n_sampled": len(samp)}
json.dump(mcq_meta, open(os.path.join(BASE, "mcq_meta.json"), "w"), indent=1)
print("\n=== MCQ format-error rate (% of failures) & sample sizes ===")
for k, v in mcq_meta.items():
    print(f"{k:24s} fails={v['n_fail']:3d} fmt_err={v['n_format_err']:2d} ({v['format_rate']:.1f}%) sampled={v['n_sampled']}")
print("\nwrote mcq_samples/*.json, ivp_table.json, mcq_meta.json")
