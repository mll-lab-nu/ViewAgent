# Performance Summary

Computes and aggregates performance metrics across all models and tasks in a rollouts directory.

## What it does

Given a rollouts directory structured as:
```
rollouts/
  model_a/
    tag_path_to_view/   (metrics.json per rollout)
    tag_view_to_path/
    tag_interactive_view_planning/
  model_b/
    ...
```

It produces:
1. **Per-model `performance.csv`** — all metrics for a single model
2. **`combined_performance.csv` / `.md`** — leaderboard table with all models, sorted by overall score
3. **`success_by_action_len.csv` / `.md`** — active exploration success rate broken down by ground-truth action sequence length
4. **`results.json`** — raw JSON for downstream use

### Metrics computed

| Metric | Description |
|--------|-------------|
| Path2View Success | Standard success rate from `tag_path_to_view` |
| View2Path Success | Standard success rate from `tag_view_to_path` |
| AE `Xm/Y°` | Active exploration success at position threshold X meters, angle threshold Y degrees |
| AE Adaptive | Adaptive success using action-length-dependent thresholds |
| AE Avg | Average across all AE threshold columns |
| Overall Score | Average of Forward Dyn., Inverse Dyn., and AE Avg |

### Threshold tiers (default)

The default `tol_per_action_len="0.25,15;2:0.5,30;3-5:1,30;1,60"` maps:
- action_len == 1: (0.25m, 15°)
- action_len == 2: (0.5m, 30°)
- action_len 3–5: (1m, 30°)
- action_len > 5: (1m, 60°)

## Usage

```bash
# Full analysis (auto-discovers all models)
python -m view_suite.analysis.proxy_analysis.performance_summary.main run \
    --rollouts_dir /path/to/rollouts \
    --viewsuite_data_path /path/to/viewsuite_15k

# Specify output directory
python -m view_suite.analysis.proxy_analysis.performance_summary.main run \
    --rollouts_dir /path/to/rollouts \
    --output_dir /path/to/output

# Only specific models
python -m view_suite.analysis.proxy_analysis.performance_summary.main run \
    --rollouts_dir /path/to/rollouts \
    --models "gpt_5_4,gemini_3_1_pro"

# Custom thresholds and intervals
python -m view_suite.analysis.proxy_analysis.performance_summary.main run \
    --rollouts_dir /path/to/rollouts \
    --thresholds "0.5,30;1,60;2,90" \
    --action_len_intervals "2,5,8"

# GLM format-error refinement
python -m view_suite.analysis.proxy_analysis.performance_summary.main refine_glm \
    --rollouts_dir /path/to/rollouts \
    --model glm_4_6v
```

## Design

```
performance_summary/
  main.py           # Fire CLI entry point (run, refine_glm)
  metrics_reader.py # Read metrics.json, compute success rates at various thresholds,
                    #   adaptive success, success-by-action-length
  table_writer.py   # Write CSV and Markdown tables (per-model, combined, action-length)
  glm_refine.py     # Re-parse GLM bare-letter answers and correct metrics
```

- **`main.py`**: Orchestrates the pipeline. Discovers models, iterates tasks, collects results, delegates to `metrics_reader` for computation and `table_writer` for output.
- **`metrics_reader.py`**: Stateless functions that take a list of metrics dicts and return aggregated rates. Supports JSONL index and meta.json fallback for ground truth lookup.
- **`table_writer.py`**: Pure formatting — takes result dicts and writes CSV/Markdown files.
- **`glm_refine.py`**: Copies a model directory, re-parses assistant responses using regex patterns, looks up ground truth from JSONL, and updates metrics.json with corrected success flags.
