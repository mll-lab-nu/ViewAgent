#!/usr/bin/env python3
"""Aggregate failure statistics + sample failing traces for the 5 frontier models
across the three ViewSuite tasks (P2V=forward_dynamics, V2P=inverse_dynamics,
IVP=active_exploration)."""
import json, os, re, glob, random, statistics as st
from collections import Counter, defaultdict

BASE = "/home/kangrui/projects/viewagent/rebuttal_experiments/07_qualitative_failure_analysis/extracted/rollouts_all_new"
MODELS = ["gpt_5_4", "gpt_5_4_pro", "gemini_3_1_pro", "claude_opus_4_6", "grok_4_20_beta", "qwen_25_vl_7b_trained"]
TAGS = {"tag_forward_dynamics": "P2V", "tag_inverse_dynamics": "V2P", "tag_active_exploration": "IVP"}
random.seed(0)

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def strip_think(t):
    m = re.search(r"<think>(.*?)</think>", t or "", re.S)
    return m.group(1).strip() if m else ""

def get_action(t):
    m = re.search(r"<action>(.*?)</action>", t or "", re.S)
    return m.group(1).strip() if m else None

summary = {}
mcq_fail_samples = defaultdict(list)   # (model,task) -> list of dicts
ivp_records = defaultdict(list)        # (model) -> list of per-rollout dicts

for model in MODELS:
    for tag, task in TAGS.items():
        d = os.path.join(BASE, model, tag)
        rolls = sorted(glob.glob(os.path.join(d, "2*")))
        n = 0; nsucc = 0; nfail = 0
        parse_err_rollouts = 0
        ans_dist = Counter()
        # IVP accumulators
        pos_errs = []; ang_errs = []; nturns_list = []
        for r in rolls:
            m = load(os.path.join(r, "metrics.json"))
            if m is None:
                continue
            n += 1
            succ = bool(m.get("success"))
            if succ:
                nsucc += 1
            else:
                nfail += 1
            infos = m.get("infos", [])
            atexts = load(os.path.join(r, "assistant_texts.json")) or []
            if task in ("P2V", "V2P"):
                info = infos[1] if len(infos) > 1 else (infos[0] if infos else {})
                parsed = info.get("parsed_answer")
                raw = info.get("raw_response", atexts[0] if atexts else "")
                fmt_bad = not parsed
                if fmt_bad:
                    parse_err_rollouts += 1
                ans_dist[parsed or "PARSE_ERR"] += 1
                if not succ:
                    # extract action sequence / options from transcript
                    tr = ""
                    try:
                        tr = open(os.path.join(r, "transcript.txt")).read()
                    except Exception:
                        pass
                    useq = re.search(r"\[([^\]]*)\]", tr)
                    mcq_fail_samples[(model, task)].append({
                        "rollout": os.path.basename(r),
                        "sample_id": info.get("sample_id") or m.get("sample_id"),
                        "parsed": parsed,
                        "reason": strip_think(raw),
                        "raw": raw[:400],
                        "fmt_bad": fmt_bad,
                        "dir": r,
                    })
            else:  # IVP
                # final info holds pos/ang err
                fin = None
                for info in reversed(infos):
                    if "pos_err_m" in info:
                        fin = info; break
                pos = fin.get("pos_err_m") if fin else m.get("pos_err_m")
                ang = fin.get("ang_err_deg") if fin else m.get("ang_err_deg")
                nturns = m.get("num_turns")
                # count format-error turns
                fmt_errs = sum(1 for info in infos if isinstance(info, dict) and "parse error" in str(info.get("error", "")))
                answered = any("answer(" in (get_action(t) or "") for t in atexts)
                # turn index of first answer
                ans_turn = None
                for i, t in enumerate(atexts):
                    if "answer(" in (get_action(t) or ""):
                        ans_turn = i; break
                gt_len = m.get("gt_action_len") or (fin.get("gt_action_len") if fin else None) \
                         or (infos[0].get("gt_action_len") if infos else None)
                if pos is not None:
                    pos_errs.append(pos)
                if ang is not None:
                    ang_errs.append(ang)
                if nturns is not None:
                    nturns_list.append(nturns)
                ivp_records[model].append({
                    "rollout": os.path.basename(r), "dir": r,
                    "sample_id": m.get("sample_id"),
                    "success": succ, "pos": pos, "ang": ang,
                    "nturns": nturns, "fmt_errs": fmt_errs,
                    "answered": answered, "ans_turn": ans_turn,
                    "gt_len": gt_len,
                    "s_1m30": bool((fin or {}).get("success_1m30degree")),
                    "s_1m60": bool((fin or {}).get("success_1m60degree")),
                    "s_2m60": bool((fin or {}).get("success_2m60degree")),
                    "s_3m90": bool((fin or {}).get("success_3m90degree")),
                    "actions": [get_action(t) for t in atexts],
                    "reasons": [strip_think(t) for t in atexts],
                })
        rec = {"n": n, "success": nsucc, "fail": nfail,
               "success_rate": round(100*nsucc/max(n,1), 1),
               "parse_err_rollouts": parse_err_rollouts}
        if task in ("P2V", "V2P"):
            rec["ans_dist"] = dict(ans_dist)
        summary[f"{model}/{task}"] = rec

# ---- print summary table ----
print("MODEL/TASK                         N  succ  fail  succ%  parseErr")
for model in MODELS:
    for task in ["P2V", "V2P", "IVP"]:
        k = f"{model}/{task}"
        r = summary[k]
        print(f"{k:32s} {r['n']:4d} {r['success']:5d} {r['fail']:5d} {r['success_rate']:6.1f}  {r['parse_err_rollouts']:4d}")

# ---- IVP error decomposition ----
print("\n=== IVP FAILURE DECOMPOSITION (per model) ===")
POS_THR, ANG_THR = 0.5, 30.0
for model in MODELS:
    recs = [r for r in ivp_records[model] if not r["success"]]
    allr = ivp_records[model]
    if not allr:
        continue
    pos_ok_ang_bad = sum(1 for r in recs if r["pos"] is not None and r["ang"] is not None and r["pos"] <= POS_THR and r["ang"] > ANG_THR)
    ang_ok_pos_bad = sum(1 for r in recs if r["pos"] is not None and r["ang"] is not None and r["ang"] <= ANG_THR and r["pos"] > POS_THR)
    both_bad = sum(1 for r in recs if r["pos"] is not None and r["ang"] is not None and r["pos"] > POS_THR and r["ang"] > ANG_THR)
    nearmiss = sum(1 for r in recs if r.get("s_1m60"))  # within 1m/60deg but failed strict
    gross = sum(1 for r in recs if not r.get("s_3m90"))  # not even within 3m/90deg
    premature = sum(1 for r in recs if r["ans_turn"] is not None and r["ans_turn"] <= 1)
    noanswer = sum(1 for r in recs if not r["answered"])
    heavy_fmt = sum(1 for r in recs if (r["fmt_errs"] or 0) >= 3)
    valid_pos = [r["pos"] for r in recs if r["pos"] is not None]
    valid_ang = [r["ang"] for r in recs if r["ang"] is not None]
    print(f"\n-- {model} -- fails={len(recs)}/{len(allr)}")
    print(f"   pos-only-ok(rot fail): {pos_ok_ang_bad}   rot-only-ok(pos fail): {ang_ok_pos_bad}   both fail: {both_bad}")
    print(f"   near-miss(<=1m/60deg): {nearmiss}   gross(>3m or >90deg): {gross}")
    print(f"   premature-answer(turn<=1): {premature}   never-answered: {noanswer}   heavy-format-errs(>=3): {heavy_fmt}")
    if valid_pos:
        print(f"   pos_err  median={st.median(valid_pos):.2f}  mean={st.mean(valid_pos):.2f}")
    if valid_ang:
        print(f"   ang_err  median={st.median(valid_ang):.1f}  mean={st.mean(valid_ang):.1f}")

# ---- save everything ----
out = {
    "summary": summary,
    "ivp_records": {m: ivp_records[m] for m in MODELS},
    "mcq_fail_samples": {f"{m}/{t}": v for (m, t), v in mcq_fail_samples.items()},
}
json.dump(out, open(os.path.join(os.path.dirname(BASE), "..", "analysis_dump.json"), "w"))
print("\nsaved analysis_dump.json")
