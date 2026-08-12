# ViewSuite failure-category tables (per task, per model)

Each cell = **% of that model's failures** on that task in that category; columns sum to ~100%.
IVP is a full data-driven partition of all failures; P2V/V2P are reasoning-pattern codes over a
random sample of 60 failing traces per model, with the format-error row measured over all failures.
See `qualitative_failure_analysis.md` §2 for the coding caveat and interpretation.

## Table 1 — P2V (Path-to-View)

| Category | GPT-5.4 | GPT-5.4-Pro | Gemini-3.1-Pro | Claude-Opus-4.6 | Grok-4.20-Beta |
|---|---:|---:|---:|---:|---:|
| Coherent geometry, wrong visual grounding | 94.3 | 99.2 | 88.5 | 92.9 | 93.3 |
| Direction / sign error | 0.0 | 0.0 | 1.6 | 1.6 | 3.3 |
| Magnitude / count error | 0.0 | 0.0 | 4.7 | 0.0 | 1.7 |
| Semantic-only matching | 0.0 | 0.0 | 0.0 | 0.0 | 1.7 |
| Low-confidence guess | 5.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Format / parse error | 0.7 | 0.8 | 5.2 | 5.5 | 0.0 |

## Table 2 — V2P (View-to-Path)

| Category | GPT-5.4 | GPT-5.4-Pro | Gemini-3.1-Pro | Claude-Opus-4.6 | Grok-4.20-Beta |
|---|---:|---:|---:|---:|---:|
| Coherent geometry, wrong visual grounding | 74.8 | 94.2 | 86.9 | 80.1 | 90.0 |
| Direction / sign error | 1.7 | 0.0 | 0.0 | 0.0 | 6.7 |
| Magnitude / count error | 21.6 | 5.0 | 0.0 | 0.0 | 0.0 |
| Semantic-only matching | 0.0 | 0.0 | 1.6 | 18.0 | 3.3 |
| Low-confidence guess | 1.7 | 0.0 | 6.3 | 0.0 | 0.0 |
| Format / parse error | 0.3 | 0.8 | 5.2 | 1.9 | 0.0 |

## Table 3 — IVP (Interactive View Planning)

| Category | GPT-5.4 | GPT-5.4-Pro | Gemini-3.1-Pro | Claude-Opus-4.6 | Grok-4.20-Beta | Qwen-7B (ours) |
|---|---:|---:|---:|---:|---:|---:|
| No valid answer (ran out of turns / unparseable) | 0.0 | 0.7 | 1.0 | 0.0 | **31.4** | 0.4 |
| Snap-answer, no exploration (answers at turn 0) | 2.5 | 0.0 | 0.2 | 0.0 | **24.0** | 0.0 |
| Orientation flip (final angular error ≥ 150°) | 7.2 | 9.6 | 10.8 | **15.4** | 4.1 | 7.6 |
| Both position & rotation off | 49.3 | 47.8 | 50.6 | **55.8** | 21.5 | 49.1 |
| Position off only (rotation within 30°) | 37.8 | 39.4 | 35.3 | 25.2 | 17.2 | 40.4 |
| Rotation off only (position within 0.5 m) | 3.2 | 2.5 | 2.2 | 3.6 | 1.8 | 2.5 |
| *(n failures)* | *442* | *406* | *417* | *473* | *488* | *277* |

### Trained-Qwen IVP success by target horizon (where the method breaks down)

| GT path length (≈ target distance) | n | success |
|---|---:|---:|
| Short (1–2 actions) | 50 | 80.0% |
| Medium (3–5 actions) | 196 | 61.2% |
| Long (6+ actions) | 284 | 32.7% |

*Note: the trained Qwen is analyzed on IVP (its target task); on P2V/V2P it emits action
tokens instead of `answer(A/B/C/D)` (format collapse) and no reasoning, so it is not coded
into Tables 1–2. See `qualitative_failure_analysis.md` §6. Regenerate with `python3 make_tables.py`.*
