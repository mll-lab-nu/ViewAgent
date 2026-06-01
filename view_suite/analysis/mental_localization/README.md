# Mental-localization analysis (IVP)

Quantifies whether Interactive View Planning (IVP) **successes** come from
*mental localization* (inferring the target pose without ever seeing it) or from
*visual re-encounter* (flying the camera into the target view, then answering).

## Question

In IVP an agent submits a 6-DoF target pose. In principle it need not reproduce
the target view: it could establish a local frame from a few moves and infer the
target pose. We measure how often that actually happens.

## Definitions

- **Success**: the benchmark's own flag (`metrics.json: success`), i.e. the
  *submitted* pose is within `0.5 m` position **and** `30°` rotation of the
  target. This matches the headline IVP numbers in the main results table.
- **Reached / visited the target view**: at some point during interaction, an
  *observed* camera pose satisfies the **same** threshold against the target
  (position **and** rotation, on a single pose). This is exactly the
  auto-success criterion of the `no_submit` IVP variant in
  `view_suite/envs/scannet_proxy_task/interactive_view_planning.py`.
- **Observed poses**: the initial view plus every post-action `Current camera`
  pose, excluding the terminal answer turn.
- **Inferred (no visit)**: a successful rollout where no observed pose ever
  reached the target view → genuine mental localization.

Per model we report
`inferred (%) = #(success & never reached) / #success`.

## Data source

Poses and the ground-truth target are parsed from each rollout's
`transcript.txt` (uniform across models; some `messages.json` files ship empty
answer-feedback content blocks, e.g. GPT-5.4 Pro). The position/rotation error
math mirrors `gym_proxy_tool_utils.geodesic_angle_deg` and the env's pose-error
computation; the recorded `pos_err`/`ang_err` were cross-checked to reproduce.

## Usage

```bash
python -m view_suite.analysis.mental_localization.main run \
    --rollouts_dir /path/to/rollouts_all_new \
    --output_dir   /path/to/rollouts_all_new_mental_localization \
    --models gpt_5_4_pro,gemini_3_1_pro,gpt_5_4,grok_4_20_beta,claude_opus_4_6
```

Outputs in `--output_dir`:
- `summary.json` — per-model counts and rates.
- `results_full.json` — adds per-rollout rows (success, reached, min errors).
- `mental_localization_table.tex` — paper-ready LaTeX table
  (`\label{tab:mental_localization}`).

## Result (rollouts_all_new)

≥90% of IVP successes are coupled to a visual re-encounter of the target view
(up to 99.1% for Gemini 3.1 Pro); inference-without-visiting is at most ~10%.
Today's VLMs succeed by view matching, not mental localization.
