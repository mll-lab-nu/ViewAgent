# Habitat-GS proxy task

ViewSuite's three view-reasoning tasks on 3D Gaussian-Splatting scenes. Same task
definitions and the same metric as the ScanNet and AI2-THOR versions; a third kind of
world. ScanNet is scanned meshes, AI2-THOR is a synthetic simulator, this is
photorealistic radiance-field reconstruction — and it is the only one of the three that
is not exclusively indoor.

| task | short | what the agent is given → asked for |
|---|---|---|
| Path-to-View | P2V | initial view + an action sequence → pick the resulting view (MCQ) |
| View-to-Path | V2P | initial view + target view + top-down reference → pick the action sequence (MCQ) |
| Interactive View Planning | IVP | a goal and a live camera → *act*, over up to 10 turns, to bring the target into view |

P2V and V2P are single-turn and read pre-rendered images from the JSONL, so they need no
service at eval time. **IVP renders every turn** and needs the render service below.

## What differs from the other two worlds

**The action space is Habitat's, not ours.** Habitat splits a yaw-only *body* from a
pitching *sensor*, so turning is ground-parallel and forward motion is horizontal
however far the camera is tilted; there is no roll action. The ScanNet manipulator yaws
about the camera's own +Y instead, which tilts once pitched. Neither is more correct —
this one follows the simulator it runs on, which also makes the two action spaces a
clean A/B for the same eval.

`HabitatGSViewManipulator` drives **both** data generation and the IVP env. Worth saying
explicitly: the AI2-THOR generator and its env use different manipulators and disagree
silently once pitch is non-zero.

**The translation step is per-scene.** The corpus spans an order of magnitude of scale —
`interior_*` rooms have a ~23 m navmesh diagonal, the `sceneNN` scenes (many outdoor) a
~90 m median and up to 538 m. A fixed 0.5 m step crosses a room and does nothing at all
in a plaza. It is derived from the navmesh diagonal, clamped to [0.25, 8] m, and written
into the prompt as before, so the model is told what a step means.

**Cameras are tied to the navmesh, and every view is screened.** A gaussian
reconstruction is faithful only near its training views; elsewhere it renders smeared
streaks and floating blobs — busy, detailed, and unrecognisable, with no error anywhere.
Positions are snapped to walkable space at eye height, and then every view is judged.

**Step size and pitch limit live in the data, not in the env.** Both are written into
every row and adopted on reset. They have to be: the ground truth was constructed under
them, so an env running different values is running a different task while every render
still looks correct. This is the failure mode the AI2-THOR pair has, and it bit here
too — before the fix, 265 of 288 test samples were told a 0.5 m step the data did not
use.

One thing to know before tuning the pitch limit: pitch snaps to multiples of the
rotation step, so at the default 30° step a limit of 40 and a limit of 45 both cap at
30. Only multiples of the step are reachable.

## Data

129 scenes: 110 train (55 `interior_*` + 55 `sceneNN`) and 19 val, the corpus's own
split. We keep that boundary — `val/` becomes the task test split, and the proxy-task
eval split is carved scene-disjointly out of `train/` — so results sit next to
Habitat-GS's own navigation numbers without an asterisk.

```bash
export VIEWSUITE_ROOT=$(pwd)

# 1. scenes (~30 GB, public, no token)
bash scripts/download_habitat_gs.sh

# 2. generate. Resumable: a scene with a .done marker is skipped.
~/miniconda3/envs/habitat-gs/bin/python -m \
  view_suite.envs.habitat_gs_proxy_task.data_gen.gen_parallel \
  --out_root=$VIEWSUITE_ROOT/data/habitat_gs --scenes=all --samples_per_scene=24 --n_gpus=8

# 3. screen every view with a VLM judge. --backend=openrouter needs OPENROUTER_API;
#    --backend=cli shells out to whatever VIEW_JUDGE_CMD names, for sites where a
#    vision-capable CLI is easier to come by than an API key.
python -m view_suite.envs.habitat_gs_proxy_task.data_gen.filter_low_semantic \
  --data_root=$VIEWSUITE_ROOT/data/habitat_gs \
  --workers=48 --review_dir=$VIEWSUITE_ROOT/gs_filter_review

# 4. split along scene boundaries (val/ becomes test; train/ splits into train+dev)
python -m view_suite.envs.habitat_gs_proxy_task.data_gen.split_by_corpus \
  --data_root=$VIEWSUITE_ROOT/data/habitat_gs
```

Pin the judge with `--judge_model=<name>` on any run whose provenance matters. Left
unset you get whatever the backend currently defaults to, which can change without
notice; the run is recorded either way in `_filter_provenance.json`.

### What came out

| | scenes | samples per task |
|---|---|---|
| train | 94 | 1329 |
| eval | 16 | 257 |
| test | 19 (the corpus's own `val/`) | 288 |

3096 samples generated, 1874 kept: the screen judged 15480 views and rejected 2326
(15.0%), which drops 39.5% of samples because a sample needs all five of its views.
Hand-checking 12 rejected and 12 kept: no false positives, and 2 of the 12 kept were
marginal (a sky-and-grass frame with no landmark; a frame with a black hole where
gaussians are missing), so it under-filters slightly at the margin.

### Why the VLM pass is not optional here

The generator already rejects a view that is flat, near-black, or dominated by one
colour. Those statistics are not enough, and the corpus shows exactly why: after a
`look_down` outdoors the camera sees nothing but cobblestone, which has healthy
variance, ordinary brightness and no dominant colour. It passes every scalar test and is
impossible to tell apart from the next cobblestone view. Off-manifold smear behaves the
same way — it is busy, not blank.

So the judge is asked one question: **is this view basically recognisable** — could a
reader say roughly where the camera is and tell this view from a nearby one? That single
criterion covers all three failure modes. The rubric names them, and names outdoor
landmarks as well as indoor furniture, because half of these scenes are outdoors.

The top-down reference is exempt: it is a map, rendered far off-manifold by
construction, and judging it would filter every sample.

## Render service (IVP only)

Habitat-GS is a *backend* of the existing render service, not a second service — same
worker pool, GPU pinning, TLS and multipart protocol.

```bash
export VIEWSUITE_ROOT=$(pwd)
bash scripts/habitat_gs_http_service_loop.sh          # 136 workers, 8 GPUs, :8812
echo "http://<host>:8812" > client_url_habitat_gs.txt
```

It needs the separate `habitat-gs` conda env; that build and habitat-sim 0.3.3 (the
ScanNet backend) cannot share an interpreter. See `../../habitat_gs/habitat_gs_render.py`
for the build, and run `view_suite/habitat_gs/tests/sanity_render.py` on any new machine
**before** anything else: a stage that fails to load still returns frames.

Measured, 8×H200, 512×512, otherwise idle:

| | |
|---|---|
| VRAM per worker | 0.8–2.3 GiB, tracking gaussian-cloud size (.gs.ply is 70–862 MB) |
| all 110 train scenes resident | 121 GiB total, ~1.1 GiB/scene |
| throughput, 32 scenes / 64 clients | 124 images/s, 0 failures, P95 4.2 s |
| throughput, 110 scenes / 160 clients | 84 images/s, 0 failures, P95 11.4 s |
| cold | ~21 images/s — scene load dominates, far slower than a ScanNet mesh |

Roughly half the ScanNet habitat backend's 178 images/s; gaussian rasterisation is
heavier than a mesh. Sizing: the pool is sticky by scene and there are only 129 scenes,
so more than ~17 workers/GPU buys nothing.

**Reap the pool workers when you stop it.** Killing the supervisor and the service
leaves the worker processes orphaned and still holding their scenes; one run here left
23 of them alive for ten hours.

```bash
pkill -TERM -f "scannet_http_service_loo[p].sh"
pkill -TERM -f "servic[e]_http/service.py"
pkill -9 -f "envs/habitat-g[s]/bin/python -c from multiprocessing"   # the orphans
```

Careful with that last pattern: written without the bracket trick it also matches the
shell you are typing it in, and kills your own command mid-way.

**Two separate things make GPU accounting confusing here, and it is easy to blame the
wrong one.** A habitat worker renders through EGL, and a graphics context does not
appear in `nvidia-smi --query-compute-apps` — so a per-pid query returns 0 MiB for a
worker holding gigabytes, and `measure_vram.py` has to take a whole-device delta on an
idle GPU instead. Separately, this box is *shared*: an unrelated tenant held ~37 GiB on
every one of the eight cards for much of this work. That one is perfectly visible — in
`nvidia-smi`'s own Processes table, which is worth actually reading before concluding
you have leaked something. Size against free memory, not against 143 GiB.

## Evaluation

```bash
# serve the model (needs ninja on PATH for flashinfer's JIT, and CUDA_HOME set)
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.8 \
  python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --port 30000 --mem-fraction-static 0.6 --chat-template qwen2-vl

python -m vagen.evaluate.run_eval \
  --config examples/evaluation/eval_default_habitat_gs.yaml fileroot=$VIEWSUITE_ROOT
```

`eval_random_habitat_gs.yaml` runs a random-navigation IVP baseline with no model
attached — a cheap check that the service and plumbing work before spending a real eval.

### Training

```bash
cd GraphRL                                  # the pipeline uses a relative vagen_dir
export VIEWSUITE_ROOT=$(cd .. && pwd)
CUDA_VISIBLE_DEVICES=0,2,3,4 \
  bash examples/viewsuite/habitat_gs_interactive_view_planning/run_smoke.sh   # RL only
bash examples/viewsuite/habitat_gs_interactive_view_planning/run.sh           # 4 iterations
```

Verified: the smoke reaches `PIPELINE COMPLETE` with two real PPO steps (grad_norm 60.2,
pg_loss 0.252, advantages spanning -3.91..3.42), rollouts served by the render service
with zero failures. Put RL and the service on disjoint GPUs — the scripts assume RL on
0,2,3,4, so start the service with `1,5,6,7`.

Two things that will otherwise cost time. The tracebacks after `PIPELINE COMPLETE` — a
killed DataLoader worker and a wandb atexit hook — are teardown noise, not a failed run;
read the wandb run summary above them.

And **read the whole metric key**. IVP is `val-aux/ae/...`; the smoke logs
`val-aux/path_to_view/...` and `val-aux/view_to_path/...` right beside it. Matching on
`traj_success/mean@1` alone picks up whichever came first, which is how a P2V number got
read as an IVP one here and turned into a six-fold discrepancy that did not exist. The
actual figures line up:

| | training val (n=16) | eval harness |
|---|---|---|
| IVP | `val-aux/ae` 0.1875 | 0.062 on the same 16 episodes, 0.059 on all 288 |
| P2V | `val-aux/path_to_view` 0.375 | 0.361 on 288 |
| V2P | `val-aux/view_to_path` 0.375 | 0.299 on 288 |

Three successes against one out of sixteen is where IVP sits; that is sampling noise at
this n, not a broken metric. `examples/evaluation/eval_ivp_valsmoke_replica.yaml`
reproduces the training validation config through the eval harness if it needs checking
again.

### Baseline

Qwen2.5-VL-7B-Instruct, zero-shot, on the full 288-sample test split. AI2-THOR's numbers
for the same base model are alongside; they are a different corpus and different scenes,
so read them as a sanity check on scale, not as a comparison.

| task | Habitat-GS | AI2-THOR |
|---|---|---|
| Path-to-View | 36.1 | 40.9 |
| View-to-Path | 29.9 | 27.4 |
| **Interactive View Planning** | **5.9** | 4.0 |

Chance is 25% on the two multiple-choice tasks, so V2P at 29.9 is barely above it. The
planning gap the benchmark exists to measure reproduces here: the same model that is
above chance on the single-turn tasks collapses to 5.9% when it has to plan.

One warning from producing these numbers. The first IVP run scored **0/288**, with 287
episodes running out of turns. Nothing was broken — the model was calling `answer(...)`
every time, with keyword arguments, and the parser rejected all of them. The action
description said only "answer(...): submit your final answer"; AI2-THOR's spells out the
signature and adds "All arguments must be positional plain numbers", which is clearly a
sentence someone else earned the same way. A prompt defect and a genuinely incapable
model produce the identical metric. Before believing a zero, read a transcript.
