# Qualitative Failure Analysis

Qualitative failure analysis of 5 frontier VLMs (GPT-5.4, GPT-5.4-Pro,
Gemini-3.1-Pro, Claude-Opus-4.6, Grok-4.20-Beta) **and the proposed trained
Qwen2.5-VL-7B (view-graph distilled)** on the three ViewSuite tasks
(P2V / V2P / IVP), for rebuttal weakness [w6] / reviewer BKvY.

**Read `qualitative_failure_analysis.md` first** (full report). The lightweight
derived artifacts (`analysis_dump.json`, `ivp_table.json`, `mcq_meta.json`,
`mcq_samples/`, `mcq_labels/`) are committed so the tables reproduce via
`python3 make_tables.py` **without** re-downloading the raw rollouts. The raw
rollouts themselves (54 GB) are **not** committed — see *Data* below. The scripts
contain absolute paths from the original run environment
(`.../rebuttal_experiments/07_qualitative_failure_analysis/extracted/...`); edit
`BASE` in `analyze.py` if you re-extract the rollouts elsewhere.

## Deliverables
- **`qualitative_failure_analysis.md`** — full report: setup, the three
  failure-category tables (§2), cross-cutting themes, per-task taxonomy,
  per-model signatures, takeaways.
- **`failure_category_tables.md`** — the three summary tables only (per task,
  per model, % of failures), for quick reference / pasting into the appendix.

## Reproducibility (run order)
1. `analyze.py` → `analysis_dump.json` — aggregate all rollouts' `metrics.json`:
   success rates, IVP pose-error records, sampled MCQ failures with reasoning.
2. `categorize.py` → `ivp_table.json`, `mcq_meta.json`, `mcq_samples/*.json` —
   IVP full partition; exact MCQ format-error rates; random samples for coding.
3. MCQ samples were LLM-coded into `mcq_labels/*.json` (rubric in
   `qualitative_failure_analysis.md` §2).
4. `make_tables.py` — prints the three tables from the labels + IVP partition.

## Data (not committed)
- `rollouts_all_new.tar.gz` (54 GB) — source rollouts, download from HF
  `JamesK2W/viewsuite-rollouts`.
- `extracted/rollouts_all_new/<model>/tag_{forward_dynamics,inverse_dynamics,active_exploration}/`
  — extracted rollouts for the 5 frontier models **plus the proposed
  `qwen_25_vl_7b_trained` (view-graph-distilled) model** (transcripts, per-turn
  outputs, images, metrics). `forward_dynamics`=P2V, `inverse_dynamics`=V2P,
  `active_exploration`=IVP.
- The trained Qwen is analyzed on **IVP** (its target task) in
  `qualitative_failure_analysis.md` §6; it exhibits MCQ format collapse and emits
  no reasoning, so it is not coded into the P2V/V2P reasoning tables.

Note: the 54 GB tarball and `extracted/` can be deleted once the analysis is
final; all numbers derive from `analysis_dump.json` + `mcq_labels/` + `ivp_table.json`.
