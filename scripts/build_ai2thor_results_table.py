#!/usr/bin/env python3
"""Build the frontier-model AI2-THOR results table (Short/Long/All per task + Overall).

Short = GT turn-length <= 2, Long = > 2 (turn-length = len(gt_action_seq)).
Reads per-episode metrics.json under rollouts/ai2thor_<model>/tag_<task>/.
"""
import json, glob, os, sys

ROOT = os.environ.get("VIEWSUITE_ROOT", ".")
TASKS = [("path_to_view", "P2V"), ("view_to_path", "V2P"), ("interactive_view_planning", "IVP")]
MODELS = [("ai2thor_gpt_5_4", "GPT-5.4"), ("ai2thor_gemini_3_1_pro", "Gemini-3.1-Pro"),
          ("ai2thor_grok_4_20_beta", "Grok-4.20"), ("ai2thor_claude_opus_4_6", "Claude-Opus-4.6")]

# sample_id -> GT turn length (shared across tasks)
gtlen = {}
for line in open(f"{ROOT}/data/ai2thor/interactive_view_planning_test.jsonl"):
    r = json.loads(line)
    gtlen[r["sample_id"]] = len(r["meta"]["gt_action_seq_letters"])

def rate(succ):
    return 100.0 * sum(succ) / len(succ) if succ else float("nan")

def model_task_stats(model_dir, task):
    d = f"{ROOT}/rollouts/{model_dir}/tag_{task}"
    short, long_, all_ = [], [], []
    for mf in glob.glob(f"{d}/*/metrics.json"):
        try: m = json.load(open(mf))
        except Exception: continue
        sid = (m.get("infos") or [{}])[0].get("sample_id")
        s = bool(m.get("success"))
        all_.append(s)
        L = gtlen.get(sid)
        if L is None: continue
        (short if L <= 2 else long_).append(s)
    return short, long_, all_

def fmt(x): return "  -  " if x != x else f"{x:4.1f}"

rows = []
for md, ml in MODELS:
    cells = {}; task_all = []
    for tk, tl in TASKS:
        sh, lo, al = model_task_stats(md, tk)
        cells[tk] = (rate(sh), rate(lo), rate(al), len(al))
        if al: task_all.append(rate(al))
    overall = sum(task_all)/len(task_all) if task_all else float("nan")
    rows.append((ml, cells, overall))

# ---- markdown table ----
hdr1 = f"| {'Model':16} | {'P2V Short':>9} {'Long':>5} {'All':>5} | {'V2P Short':>9} {'Long':>5} {'All':>5} | {'IVP Short':>9} {'Long':>5} {'All':>5} | {'Overall':>7} |"
print(hdr1); print("|" + "-"*(len(hdr1)-2) + "|")
for ml, cells, overall in rows:
    def c(tk): s,l,a,n = cells[tk]; return f"{fmt(s):>9} {fmt(l):>5} {fmt(a):>5}"
    print(f"| {ml:16} | {c('path_to_view')} | {c('view_to_path')} | {c('interactive_view_planning')} | {fmt(overall):>7} |")

# counts (coverage) for transparency
print("\nEpisodes evaluated (All n):")
for ml, cells, _ in rows:
    print(f"  {ml:16} " + "  ".join(f"{tl}={cells[tk][3]}" for tk,tl in TASKS))
# short/long population sizes
ns = sum(1 for v in gtlen.values() if v<=2); nl = sum(1 for v in gtlen.values() if v>2)
print(f"\nTest-set turn split: short(<=2)={ns}  long(>2)={nl}  total={len(gtlen)}")
