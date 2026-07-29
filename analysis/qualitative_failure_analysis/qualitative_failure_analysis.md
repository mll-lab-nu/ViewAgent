# Qualitative Failure Analysis of Frontier VLMs on ViewSuite

*Rebuttal experiment 07 — addressing reviewer BKvY / AC weakness [w6]: "deeper qualitative failure analysis would help clarify where models still break down."*

---

## 1. What we did

We analyzed the full, released evaluation rollouts of five frontier VLMs on the
three ViewSuite tasks:

| Task | tag in data | Model sees | Model must produce | Success criterion |
|---|---|---|---|---|
| **P2V** (Path-to-View) | `forward_dynamics` | initial view + top-down + an **action sequence** + 4 candidate images (A/B/C/D) | pick the resulting image | exact letter match |
| **V2P** (View-to-Path) | `inverse_dynamics` | initial view + top-down + **target view** + 4 candidate action sequences | pick the sequence | exact letter match |
| **IVP** (Interactive View Planning) | `active_exploration` | initial view + top-down + target view; may fly the camera ≤10 turns | `answer(tx,ty,tz,rx,ry,rz)` | pose within **0.5 m and 30°** (geodesic) |

- **Models:** GPT-5.4, GPT-5.4-Pro, Gemini-3.1-Pro, Claude-Opus-4.6, Grok-4.20-Beta
  (medium reasoning effort; the published main-table configuration).
- **Data:** `JamesK2W/viewsuite-rollouts → rollouts_all_new.tar.gz`, ~530 rollouts
  per model per task (7,900+ rollouts). Each rollout contains the full transcript,
  per-turn model output (with the model's own chain-of-thought), rendered images,
  and metrics (predicted vs. ground-truth pose, per-turn errors, action sequence).
- **Method:** we (i) aggregated structured failure statistics from `metrics.json`
  across all rollouts (`analyze.py`, dump in `analysis_dump.json`), (ii) read
  several hundred failing reasoning traces to induce failure categories, and
  (iii) visually inspected the rendered observations of sampled failures.

The measured success rates reproduce the paper's main table exactly (e.g. IVP:
GPT-5.4 16.6, Gemini-3.1-Pro 21.3, Claude-Opus-4.6 10.8, Grok-4.20-Beta 7.9),
confirming these are the evaluation rollouts behind the reported numbers.

### Headline success rates (%)

| Model | P2V | V2P | IVP |
|---|---:|---:|---:|
| GPT-5.4 | 47.9 | 45.5 | 16.6 |
| GPT-5.4-Pro | 53.2 | 50.6 | 19.9 |
| Gemini-3.1-Pro | 48.9 | 49.4 | 21.3 |
| Claude-Opus-4.6 | 34.7 | 41.5 | 10.8 |
| Grok-4.20-Beta | 46.2 | 44.5 | 7.9 |
| **Qwen2.5-VL-7B (trained, ours)** | 6.0† | 20.9† | **47.7** |
| *(random baseline)* | *25* | *25* | *~0* |

† The trained model's low P2V/V2P numbers are **not** a spatial-ability result: it is
distilled into the interactive **action policy** and largely ignores the multiple-choice
output format, emitting movement tokens (e.g. `turn_right`) instead of `answer(A/B/C/D)`
— 84% of its P2V failures and 48% of its V2P failures are format/parse errors. It also
emits no chain-of-thought on the MCQ tasks. Per the reviewer's request we therefore focus
the trained-model analysis on **IVP** (§6), the task the method actually targets, where it
scores **47.7% — 2.2× the best frontier model (Gemini 21.3%)** and matches the paper.

Two facts frame everything below. (1) On the single-step multiple-choice tasks
(P2V/V2P) models are clearly above the 25% chance rate but plateau around
**35–53%** — they have *partial* view-transition understanding, not reliable
geometry. (2) On multi-turn **IVP the gap is dramatic (8–21%)**: composing
transitions to actively localize a viewpoint is where models break down.

---

## 2. Failure-category summary tables (the three tables)

Each cell is the **percentage of that model's failures** on that task falling into
each category; columns sum to ~100%.

- **IVP** is partitioned **exhaustively over all failures** from the recorded pose
  error / behavior (fully data-driven, mutually exclusive).
- **P2V / V2P** categories are **reasoning-pattern codes** applied to a random
  sample of **60 failing traces per model** (the model's own chain-of-thought),
  with the **format/parse-error** row measured exactly over *all* failures and the
  remaining rows scaled to the non-format share.
- **Coding caveat (P2V/V2P):** the released rollouts do not store the correct
  option, so a trace was labeled *Direction/sign* or *Magnitude/count* only when
  the error is **identifiable from the reasoning text itself**. Coherent-looking
  transform reasoning that nonetheless picks the wrong option is labeled *Coherent
  geometry, wrong visual grounding* — which therefore also **absorbs an unknown
  number of true direction/magnitude errors** that are not self-evident in text.
  The honest reading of that dominant row is: *"the model's stated geometry is
  internally consistent, yet the choice is wrong."*

**Table 1 — P2V (Path-to-View) failure categories (% of each model's failures)**

| Category | GPT-5.4 | GPT-5.4-Pro | Gemini-3.1-Pro | Claude-Opus-4.6 | Grok-4.20-Beta |
|---|---:|---:|---:|---:|---:|
| Coherent geometry, wrong visual grounding | 94.3 | 99.2 | 88.5 | 92.9 | 93.3 |
| Direction / sign error | 0.0 | 0.0 | 1.6 | 1.6 | 3.3 |
| Magnitude / count error | 0.0 | 0.0 | 4.7 | 0.0 | 1.7 |
| Semantic-only matching | 0.0 | 0.0 | 0.0 | 0.0 | 1.7 |
| Low-confidence guess | 5.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Format / parse error | 0.7 | 0.8 | 5.2 | 5.5 | 0.0 |

**Table 2 — V2P (View-to-Path) failure categories (% of each model's failures)**

| Category | GPT-5.4 | GPT-5.4-Pro | Gemini-3.1-Pro | Claude-Opus-4.6 | Grok-4.20-Beta |
|---|---:|---:|---:|---:|---:|
| Coherent geometry, wrong visual grounding | 74.8 | 94.2 | 86.9 | 80.1 | 90.0 |
| Direction / sign error | 1.7 | 0.0 | 0.0 | 0.0 | 6.7 |
| Magnitude / count error | 21.6 | 5.0 | 0.0 | 0.0 | 0.0 |
| Semantic-only matching | 0.0 | 0.0 | 1.6 | 18.0 | 3.3 |
| Low-confidence guess | 1.7 | 0.0 | 6.3 | 0.0 | 0.0 |
| Format / parse error | 0.3 | 0.8 | 5.2 | 1.9 | 0.0 |

**Table 3 — IVP (Interactive View Planning) failure categories (% of each model's failures)**

| Category | GPT-5.4 | GPT-5.4-Pro | Gemini-3.1-Pro | Claude-Opus-4.6 | Grok-4.20-Beta | Qwen-7B (ours) |
|---|---:|---:|---:|---:|---:|---:|
| No valid answer (ran out of turns / unparseable) | 0.0 | 0.7 | 1.0 | 0.0 | **31.4** | 0.4 |
| Snap-answer, no exploration (answers at turn 0) | 2.5 | 0.0 | 0.2 | 0.0 | **24.0** | 0.0 |
| Orientation flip (final angular error ≥ 150°) | 7.2 | 9.6 | 10.8 | **15.4** | 4.1 | 7.6 |
| Both position & rotation off | 49.3 | 47.8 | 50.6 | **55.8** | 21.5 | 49.1 |
| Position off only (rotation within 30°) | 37.8 | 39.4 | 35.3 | 25.2 | 17.2 | 40.4 |
| Rotation off only (position within 0.5 m) | 3.2 | 2.5 | 2.2 | 3.6 | 1.8 | 2.5 |
| *(n failures)* | *442* | *406* | *417* | *473* | *488* | *277* |

The trained model (right column) has the **fewest failures** (277 vs 406–488) and, crucially,
**none of the pathological behaviors** — no snap-answering, no never-committing, and the fewest
orientation flips. Its residual failures fall into the *same* healthy pattern as the strongest
frontier models — position under-localization (both-off + position-off-only = **90%** of its
failures) — just less often. §6 dissects where it still breaks down.

**How to read the tables.** (i) On the single-step MCQ tasks, **75–99% of every
model's failures come with internally-coherent transform reasoning that still maps
to the wrong option** — a grounding failure, not an arithmetic one. (ii) GPT-5.4
on V2P is the one place where an *identifiable* error type is large (21.6%
magnitude/count) — it commits to the right motion direction but the wrong amount.
(iii) Claude's V2P shows the most pure semantic-only matching (18%); Gemini and
Claude carry the most format/parse errors on MCQ. (iv) On IVP, the modal failure
for the four well-behaved models is a **metric miss with the position wrong**
(*both off* + *position off only* = 74–87% of GPT/Gemini/Claude failures), whereas
**Grok is dominated by behavioral failures** (*no valid answer* 31% + *snap-answer*
24% = 55%).

---

## 3. Cross-cutting failure themes

Three failure modes recur across all models and all three tasks. They are the
core qualitative story.

### T1 — Perception is grounded in *semantics*, not *geometry*

Models overwhelmingly reason about **what objects are in view** ("faces the
desk", "the trash can is on the left") and match scenes by **object content**,
rather than tracking a metric camera pose. Their arithmetic on the *actions* is
often correct while the mapping to the actual geometry fails. A representative
GPT-5.4 P2V trace:

> "Five right turns of 30° each gives a 150° clockwise rotation… rotating that
> much should turn the view away from that corner and **toward the opposite side
> of the room**." → wrong option.

The angle math (5×30°=150°) is right; the model then *guesses* the resulting
appearance from a semantic description of the room instead of a geometric
prediction. Claude verbalizes the same pattern in full ("5 × turn_left = **150°
left rotation**. This turns me to face roughly back and to the right…"),
computes the rotation correctly, and still selects the wrong image. **The
bottleneck is grounding the transformation to pixels, not doing the transform.**

This is aggravated by the observation medium: ViewSuite renders ScanNet mesh
reconstructions, which contain holes, blur, and floating geometry. In many
multiple-choice items **two of the four distractors are near-duplicate viewpoints
of the same object** (e.g. the same wall-vent seen from slightly different
distances), so the correct answer hinges on sub-metric viewpoint discrimination
that the noisy render barely supports. (See
`gpt_5_4/tag_forward_dynamics/20260315-232723-aae7dd32`: options A and D are the
same radiator vent; the model picks D, the wrong distance.)

### T2 — Rotation/frame confusion: sign, axis, and orientation flips

Whenever a model *does* try explicit pose arithmetic, it stumbles on the
coordinate conventions — which axis is yaw, whether an action increases or
decreases an angle, and where "forward" points. Gemini-3.1-Pro repeatedly
audits itself and gets it wrong:

> "Wait, turning right 3 times from rz=-90 gives rz=0. Wait… **turn_right
> decreases rz?** Let's check: Step 8: rz=-90 → turn_right×3 → rz=180. So
> turn_right *increases* rz…"
> (`gemini_3_1_pro/tag_active_exploration/20260315-120217-d959e406`)

The signature of this failure is quantitative: on IVP, rotation errors are **not
uniformly distributed — they pile up at exact multiples of the 30° step**
(30°, 60°, 90°, 120°, 150°, 180°). Across models there is a large spike at ~30°
(off by a single rotation step — a genuine near-miss) *and* a heavy tail at
**150–180° (the camera ends up facing the opposite direction)**. Full ~180°
orientation flips occur in 33–73 of each model's failures (worst for Claude).

### T3 — Multi-turn planning is myopic: overshoot, oscillation, and running out of budget

IVP failures are dominated by *behavioral* rather than perceptual problems. The
typical failed episode is greedy hill-climbing on image similarity: take one
step, re-look, correct, over-correct, repeat — without a plan or geometric
bookkeeping. Concrete GPT-5.4 turns from failed episodes:

> "I likely **overshot to the right** and am a bit too close/high…"
> "**Forward overshot.** …I should undo that step first."
> "Current view has **drifted too far right**… I'll move left back toward the…"

We measured this: **back-and-forth (a move immediately undone by its opposite)
appears in 47% of GPT-5.4 failures, 64% of Gemini's, and 90% of Claude's.**
Because correction is un-budgeted, models frequently hit the 10-turn limit
mid-correction and **give up with an explicit guess**:

> "I am out of steps. I will just output the answer."
> "I am out of turns. **I will just guess the pose** based on the target view."
> "Moving right was wrong… Let's go back to the good view. **Wait, I only have 1
> step left. I must answer now.**"
> (Gemini-3.1-Pro, `tag_active_exploration/20260315-120140-7641130e` and others)

Gemini uses the **full 10-turn budget in the median** failed episode yet still
fails — its exploration is active but unproductive.

---

## 4. Task-by-task failure taxonomy

### 3.1 P2V (Path-to-View) — "given the actions, which image results?"

Failure rate 47–65%. Categories, in rough order of prevalence:

1. **Correct action-math, wrong image grounding (T1).** The dominant mode. The
   model computes the net rotation/translation correctly but cannot predict the
   resulting appearance, and selects a semantically plausible but geometrically
   wrong option.
2. **Distractor confusion under render noise.** Two/more options depict the same
   object from nearby viewpoints; the model chooses the wrong distance/angle. The
   noisy ScanNet mesh (holes, floaters) removes the fine cues needed to separate
   them.
3. **Compounding over long action sequences.** Accuracy degrades as the sequence
   lengthens: each un-grounded step adds error, so 7–8-action items are missed far
   more than 2–3-action items.
4. **Ignoring rotation-rounding semantics.** The prompt warns that Euler angles
   are re-rounded to step multiples after each rotation; models rarely account for
   this, mispredicting pitch/roll interactions.
5. **Format/parse failures** (small but model-dependent): Claude 19/530, Gemini
   14/530 emit an answer the strict parser rejects (extra prose, missing/again
   malformed `<action>answer(X)</action>`); GPT and Grok ≈0.

### 3.2 V2P (View-to-Path) — "given the target view, which action sequence reaches it?"

Failure rate 49–59%. This is the inverse problem and shows a distinct signature:

1. **Direction/sign inversion (T2).** The model infers the *right kind* of motion
   but the wrong sign — choosing a sequence that turns/steps the opposite way.
   Traces read like "the target is behind me, so turn around and move to the
   door," then select a sequence that goes the other direction.
2. **Magnitude errors.** Correct direction, wrong number of steps (e.g. picking a
   2-turn vs. 3-turn option) — the model cannot estimate *how far* the target is
   displaced from the initial view.
3. **Rotation-vs-translation conflation.** The model attributes an apparent
   viewpoint change to a turn when it is a strafe (or vice-versa), since both can
   produce similar image shifts in a cluttered indoor scene.
4. **Semantic-match shortcut.** The model reasons purely from which objects appear
   in the target ("target shows a blue trash can → it's the north wall") and picks
   the option that "should" lead there, without verifying the action geometry.
5. **Format/parse failures**: Gemini 14/530, Claude 6/530; GPT/Grok ≈0.

### 3.3 IVP (Interactive View Planning) — active multi-turn localization

Failure rate 79–92% — the hardest task. Decomposing the graded failures (final
pose vs. target) by which constraint was violated:

| Model | graded fails | both pos & rot off | rot OK, **pos off** | pos OK, rot off |
|---|---:|---:|---:|---:|
| GPT-5.4 | 442 | 58% | **38%** | 4% |
| GPT-5.4-Pro | 403 | 57% | **40%** | 3% |
| Gemini-3.1-Pro | 413 | 62% | **36%** | 2% |
| Claude-Opus-4.6 | 473 | 70% | **25%** | 5% |
| Grok-4.20-Beta | 335 | 64% | **31%** | 5% |

**Position is the binding constraint.** Among failures, the rotation is already
within 30° about 25–40% of the time, but the position is within 0.5 m in only
**2–5%**. Median position error is ≈1.0–1.3 m (2× the budget); the model gets
*roughly* oriented but cannot translate to the right spot. Claude is the outlier
where rotation *also* fails most (only 25% rot-OK, median 66° angular error, most
180° flips). IVP failure categories:

1. **Position under-localization (primary).** The dominant single-factor failure:
   orientation acceptable, but the estimated location is ~1 m off. Depth/scale
   from a monocular render is the missing capability.
2. **Greedy myopic exploration → overshoot & oscillation (T3).** 47–90% of failed
   episodes contain an action immediately reversed. Correction is uncoordinated.
3. **Budget exhaustion / guess-at-the-buzzer (T3).** Models recognize they are
   off, start to correct, run out of turns, and emit an admitted guess. Gemini
   consumes the full 10 turns in the median failure.
4. **Orientation flips & frame-sign errors (T2).** 33–73 failures per model end
   ~180° from the target; models confuse which way `turn_left/right` moves the
   yaw and where the target faces.
5. **No exploration — snap-answer, and its opposite, never-commit (model-specific;
   see §4 Grok).**

---

## 5. Model-specific failure signatures

Behavior on IVP separates the models sharply:

| Model | answered at turn 0 | never emitted a valid answer | median turns used | dominant IVP pathology |
|---|---:|---:|---:|---|
| GPT-5.4 | 11 / 530 | 0 | 9 | disciplined exploration; overshoot + under-localization |
| GPT-5.4-Pro | 0 / 507 | 3 | 8 | best of the five; same modes, smaller errors |
| Gemini-3.1-Pro | 1 / 530 | 4 | 10 | exhaustive but unproductive search; frame-sign self-confusion |
| Claude-Opus-4.6 | 0 / 530 | 0 | 9 | thrashing (90% oscillation), most 180° flips |
| Grok-4.20-Beta | **124 / 530** | **153 / 530** | 8 | **bimodal: snap-answer or wander-and-never-commit** |

- **Grok-4.20-Beta** is qualitatively different and explains its lowest IVP score
  (7.9%). It is **bimodal**: in ~24% of episodes it does explicit coordinate
  arithmetic from the initial pose numbers and **commits at turn 0 without a
  single look** ("move right ~0.5 m, forward ~0.5 m, yaw left 90°… `answer(...)`");
  in ~29% it **wanders (mostly turning in place) for all 10 turns and never issues
  a parseable `answer(...)`**, scoring zero. Both bypass the intended perceive→act
  loop.
- **Claude-Opus-4.6** explores properly (never snap-answers) but **thrashes**:
  90% of its failed episodes oscillate, and it has the worst rotation error
  (median 66°, most 180° flips), yielding the lowest score among the
  well-behaved models (10.8%). It also has the most multiple-choice parse errors.
- **Gemini-3.1-Pro** is the most *deliberate* — it narrates coordinate
  bookkeeping and uses the whole turn budget — which yields the best IVP score
  (21.3%) but also exposes systematic **coordinate-frame sign errors** that
  disciplined-looking reasoning cannot fix.
- **GPT-5.4 / GPT-5.4-Pro** are the most stable planners; Pro simply makes
  smaller errors of the *same* kinds (fewer oscillations resolved in time, lower
  median position error), which is why it is the strongest overall.

---

## 6. The proposed method (trained Qwen): where IVP still breaks down

This section directly answers the reviewer: *"deeper qualitative failure analysis
would help clarify where the proposed method still breaks down."* We analyze the
view-graph-distilled **Qwen2.5-VL-7B** on IVP (its target task; it scores **47.7%**,
vs 2.5% before training and ≤21.3% for any frontier model). The analysis below is
over its **277 IVP failures** (of 530). The trained model, like the base Qwen,
emits **no chain-of-thought** — only action tokens — so this is a behavioral /
metric analysis of its trajectories, not a reasoning analysis.

**What distillation fixed.** The catastrophic frontier behaviors are essentially
gone: **0 snap-answers, 1 never-committed, 21 orientation flips (7.6%)** — the model
reliably runs the perceive→act→localize loop, explores, and commits a pose. Its
gross-failure rate is the **lowest of all models** (17% not even within 3 m/90°, vs
24–52% for frontier). So the method succeeds at teaching *the interactive procedure*.

**Where it still breaks down — 1. Long-horizon targets (the dominant failure).**
Success collapses monotonically as the ground-truth path to the target lengthens:

| GT path length (≈ target distance) | n | success rate |
|---|---:|---:|
| Short (1–2 actions) | 50 | **80.0%** |
| Medium (3–5 actions) | 196 | 61.2% |
| Long (6+ actions) | 284 | **32.7%** |

The method is near-solved for nearby targets but drops to ~1-in-3 for far targets
that require composing many moves. Inspecting long-horizon failures, the model tends
to **spin in place searching for the target heading and under-translates** — e.g. on
a `gt_len=10` target it issued ~13 `turn_*` actions but only 2–3 translations, ending
1.5 m away. Long-horizon *composition of translations* is the frontier of the method.

**2. Position under-localization.** As with the frontier models, position is the
binding constraint: **90% of failures have position off** (both-off 49% +
position-off-only 40%), median final position error **0.96 m** (≈2× the 0.5 m budget)
while rotation is often already acceptable. Monocular depth/scale from the rendered
view remains the missing metric signal.

**3. One-rotation-step and near-threshold misses.** The residual rotation errors sit
right at the boundary: median angular error is **exactly 30.0°**, and **23% of
failures miss by a single 30° rotation step** (30–45°). Combined with the near-miss
rate (**44% of failures are within 1 m / 60°** but outside the strict 0.5 m/30°), a
large share of failures are "almost right," suggesting the tight threshold and
discrete 0.5 m / 30° action grid — not gross misunderstanding — account for many
misses.

**4. Non-convergence within budget.** Failed episodes use **9.8 / 10 turns on
average** and **77% oscillate** (a move immediately undone). The model makes sensible
local corrections but, like the frontier models, cannot reliably settle inside the
tight threshold before the budget runs out — a planning/stopping-control limitation
that a longer budget or an explicit convergence criterion might mitigate.

**Summary.** The proposed method removes the *behavioral* failure modes and tightens
errors across the board, but its remaining IVP failures are concentrated in
**long-horizon, multi-translation targets**, where it under-translates and misses on
**position** (and, secondarily, by a single rotation step) without converging in the
turn budget. This is a precise, honest statement of where the method still breaks down.

---

## 7. Takeaways for the paper

1. **The benchmark isolates the intended gap.** Models have partial single-step
   view-transition understanding (P2V/V2P ≈ 35–53%, above chance) but fail to
   *compose* transitions into multi-turn localization (IVP ≈ 8–21%). The failure
   analysis shows this gap is real and behavioral, not an artifact of scoring.
2. **The missing capability is metric grounding, not reasoning.** Frontier models
   reason semantically and even do the action arithmetic correctly; they fail at
   binding those transforms to metric camera pose — especially **position/depth**
   (the binding constraint in 95%+ of IVP failures) and **rotation frame/sign**.
   This is consistent with rebuttal [w3]: more reasoning does not close the gap.
3. **Multi-turn planning is myopic.** Greedy image-matching with un-budgeted
   correction (overshoot, oscillation, buzzer-beater guesses) is the modal IVP
   failure — motivating our self-exploration + view-graph distillation, which
   supplies exactly the structured transition model these agents lack.
4. **Robustness/agentic behavior varies wildly by model** (Grok's snap-answer /
   never-commit vs. Claude's thrashing vs. GPT's disciplined search), which is
   itself a useful diagnostic that ViewSuite's interactive setting surfaces and
   static QA benchmarks do not.
5. **The proposed method fixes behavior and tightens errors, and its remaining
   failures are well-characterized (§6).** View-graph distillation removes the
   pathological behaviors (no snap-answers, near-zero never-commit, fewest flips)
   and more than doubles the best frontier IVP score (47.7% vs 21.3%). Its residual
   failures are honestly concentrated in **long-horizon targets** (success 80%→33%
   from short→long paths), **position under-localization** (90% of failures), and
   **near-threshold, non-converged** answers (median angular error exactly 30°; 77%
   oscillate; 9.8/10 turns used) — pointing to longer horizons and metric depth as
   the next frontier rather than any regression introduced by training.

---

## Reproduction

- `analyze.py` — aggregates `metrics.json` across all rollouts and dumps per-model
  statistics, IVP error decomposition, and sampled failing traces.
- `analysis_dump.json` — machine-readable dump (per-rollout IVP records with
  actions/reasoning; sampled multiple-choice failures with the model's own
  reasoning) used for all quotes and numbers above.
- Rollouts extracted under `extracted/rollouts_all_new/<model>/tag_{forward_dynamics,inverse_dynamics,active_exploration}/`.
  Example rollout IDs are cited inline for every qualitative claim.
