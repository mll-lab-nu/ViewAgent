#!/usr/bin/env python3
"""Full AI2-THOR test-set table: base + trained Qwen + 4 frontier models.
Short = GT turn-length <= 2, Long = > 2 (len(gt_action_seq)). Reads per-episode
metrics.json under rollouts/<model>/tag_<task>/.
"""
import json, glob, os

ROOT = os.environ.get("VIEWSUITE_ROOT", ".")
TASKS = [("path_to_view", "P2V"), ("view_to_path", "V2P"), ("interactive_view_planning", "IVP")]
# (rollout_dir, label). Trained/base first for the rebuttal delta, then frontier.
MODELS = [
    ("qwen25vl7b_base",     "Qwen2.5-VL-7B (base)"),
    ("qwen25vl7b_trained",  "Qwen2.5-VL-7B (trained, ours)"),
    ("ai2thor_gpt_5_4",     "GPT-5.4 (zero-shot)"),
    ("ai2thor_gemini_3_1_pro", "Gemini-3.1-Pro (zero-shot)"),
    ("ai2thor_grok_4_20_beta", "Grok-4.20 (zero-shot)"),
    ("ai2thor_claude_opus_4_6", "Claude-Opus-4.6 (zero-shot)"),
]

gtlen = {}
for line in open(f"{ROOT}/data/ai2thor/interactive_view_planning_test.jsonl"):
    r = json.loads(line)
    gtlen[r["sample_id"]] = len(r["meta"]["gt_action_seq_letters"])

def rate(s): return 100.0 * sum(s) / len(s) if s else float("nan")

def stats(md, task):
    short, long_, all_ = [], [], []
    for mf in glob.glob(f"{ROOT}/rollouts/{md}/tag_{task}/*/metrics.json"):
        try: m = json.load(open(mf))
        except Exception: continue
        sid = (m.get("infos") or [{}])[0].get("sample_id")
        s = bool(m.get("success")); all_.append(s)
        L = gtlen.get(sid)
        if L is None: continue
        (short if L <= 2 else long_).append(s)
    return short, long_, all_

def fmt(x): return "  -  " if x != x else f"{x:4.1f}"

rows = []
for md, ml in MODELS:
    cells = {}; task_all = []
    present = False
    for tk, tl in TASKS:
        sh, lo, al = stats(md, tk)
        if al: present = True
        cells[tk] = (rate(sh), rate(lo), rate(al), len(al))
        if al: task_all.append(rate(al))
    if not present:  # model not evaluated yet -> skip
        continue
    overall = sum(task_all)/len(task_all) if task_all else float("nan")
    rows.append((ml, cells, overall))

hdr = (f"| {'Model':30} | {'P2V S/L/All':>18} | {'V2P S/L/All':>18} "
       f"| {'IVP S/L/All':>18} | {'Overall':>7} |")
print(hdr); print("|" + "-"*(len(hdr)-2) + "|")
for ml, cells, overall in rows:
    def c(tk):
        s,l,a,_ = cells[tk]; return f"{fmt(s)}/{fmt(l)}/{fmt(a)}"
    print(f"| {ml:30} | {c('path_to_view'):>18} | {c('view_to_path'):>18} "
          f"| {c('interactive_view_planning'):>18} | {fmt(overall):>7} |")

print("\nEpisodes evaluated (All n):")
for ml, cells, _ in rows:
    print(f"  {ml:30} " + "  ".join(f"{tl}={cells[tk][3]}" for tk,tl in TASKS))
ns = sum(1 for v in gtlen.values() if v<=2); nl = sum(1 for v in gtlen.values() if v>2)
print(f"\nTest-set turn split: short(<=2)={ns}  long(>2)={nl}  total={len(gtlen)}")
