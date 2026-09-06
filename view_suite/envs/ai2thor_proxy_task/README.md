# AI2-THOR proxy task

ViewSuite's three view-reasoning tasks on AI2-THOR instead of ScanNet. Same task
definitions, same metric, a different world: ScanNet is a fixed set of scanned meshes,
AI2-THOR is a live simulator, so the data is generated rather than collected and the
scene split can be made disjoint by construction.

| task | short | what the agent is given → asked for |
|---|---|---|
| Path-to-View | P2V | initial view + an action sequence → pick the resulting view (MCQ) |
| View-to-Path | V2P | initial view + target view + top-down reference → pick the action sequence (MCQ) |
| Interactive View Planning | IVP | a goal and a live camera → *act*, over up to 10 turns, to bring the target into view |

P2V and V2P are single-turn and need no simulator at eval time. **IVP is the multi-turn
one and renders every turn**, so it needs the render service below; it is also the task
the view graph is meant to help with.

## Layout

```
view_suite/envs/ai2thor_proxy_task/
    path_to_view.py view_to_path.py interactive_view_planning.py   the three envs
    gym_proxy_tool.py                                              multi-turn engine
    data_gen/                                                      generation + filtering
view_suite/ai2thor/
    service_http/          HTTP render service (one process per scene, GPU-pinned)
    gym_ai2thor_render_env.py  view_manipulator.py  pose_utils.py  scene_list.py
```

## Data

Generated from the simulator, not downloaded from a corpus:

```bash
export VIEWSUITE_ROOT=$(pwd)

# 120 iTHOR scenes x 24 samples, 8 GPUs, resumable
python -m view_suite.envs.ai2thor_proxy_task.data_gen.gen_parallel \
    --out_root=$VIEWSUITE_ROOT/data/viewagent15k_ai2thor_full --scenes=all --samples_per_scene=24 --n_gpus=8

# drop low-semantic views (blank walls and floors) with a VLM judge.
# Needs OPENROUTER_API; set EGRESS_PROXY if outbound traffic requires a proxy.
python -m view_suite.envs.ai2thor_proxy_task.data_gen.filter_low_semantic \
    --data_root=$VIEWSUITE_ROOT/data/viewagent15k_ai2thor_full --backend=openrouter \
    --model=qwen/qwen3.7-plus --workers=24 --review_dir=$VIEWSUITE_ROOT/filter_review

# scene-disjoint train/eval/test split
python -c "from view_suite.envs.utils.split_jsonl_by_scene import split_jsonl_by_scene as s; \
[s(f'data/viewagent15k_ai2thor/{t}.jsonl', ratios=(70,15,15)) for t in \
 ['path_to_view','view_to_path','interactive_view_planning']]"
```

The filter removes ~26% of samples. The released split is 120 scenes → 2,880 samples per
task → **train 1,468 / eval 325 / test 331**, over 84/18/18 disjoint scenes.
`bash scripts/download_ai2thor.sh` fetches the prepared copy instead of regenerating.

## Render service (IVP only)

```bash
bash scripts/ai2thor_http_service_loop.sh          # on a machine with a graphics stack
echo "http://<host>:<port>" > client_url_ai2thor.txt
```

Semicolon-separate several URLs to spread scenes across servers. IVP rollout throughput
is gated by how many render servers you give it, so this is the knob that decides how
long training takes.

## Evaluation

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
CONFIG=$VIEWSUITE_ROOT/examples/evaluation/eval_default_ai2thor.yaml \
  bash examples/evaluation/eval_sglang/eval_model.sh
```

`eval_random_ai2thor.yaml` runs a random-navigation IVP baseline with no model attached —
a quick check that the service and plumbing work before spending a real eval on them.

## Training

```bash
cd GraphRL                              # the pipeline uses a relative vagen_dir
export VIEWSUITE_ROOT=$(cd .. && pwd)
bash examples/viewsuite/ai2thor_interactive_view_planning/run_smoke.sh   # RL-only smoke
bash examples/viewsuite/ai2thor_interactive_view_planning/run.sh         # 4 iterations
```

Each iteration is RL (IVP rollouts rendered live) → traj_to_sft (build the view graph
from those rollouts, distil supervision) → SFT → the next iteration's starting model.

## Results

Test split, 252 episodes per task after filtering to episodes all models completed.
"Short" is a ground-truth path of ≤2 turns, "Long" is >2; success rate in %.

| Model | P2V S/L/All | V2P S/L/All | IVP S/L/All | Overall |
|---|---|---|---|---|
| Qwen2.5-VL-7B (base) | 28.9 / 43.0 / 40.9 | 34.2 / 26.2 / 27.4 | 10.5 / 2.8 / 4.0 | 24.1 |
| **Qwen2.5-VL-7B (trained)** | 28.9 / 24.3 / 25.0 | 63.2 / 37.4 / 41.3 | **73.7 / 58.9 / 61.1** | 42.5 |
| GPT-5.4 (zero-shot) | 65.8 / 75.2 / 73.8 | 94.7 / 65.0 / 69.4 | 47.4 / 34.1 / 36.1 | **59.8** |
| Gemini-3.1-Pro (zero-shot) | 76.3 / 77.1 / 77.0 | 89.5 / 65.4 / 69.0 | 21.1 / 15.9 / 16.7 | 54.2 |
| Grok-4.20 (zero-shot) | 73.7 / 72.0 / 72.2 | 63.2 / 73.4 / 71.8 | 15.8 / 8.9 / 9.9 | 51.3 |
| Claude-Opus-4.6 (zero-shot) | 39.5 / 57.9 / 55.2 | 65.8 / 55.1 / 56.7 | 28.9 / 15.0 / 17.1 | 43.0 |

Turn split: short (≤2) = 38, long (>2) = 214.

Reading these honestly:

- **The planning gap reproduces outside ScanNet.** Every frontier model is strong on the
  single-turn tasks and weak on IVP — Gemini-3.1-Pro is the best model overall on P2V
  (77.0) and lands at 16.7 on IVP. Whatever IVP measures, it is not what these models are
  already good at.
- **Training closes it on IVP and only on IVP.** 4.0 → 61.1 takes the 7B model past every
  frontier model on that task, by a wide margin. It does not make it a better model in
  general: P2V goes *down* (40.9 → 25.0), and Overall stays below GPT-5.4. The training
  signal is IVP rollouts, so this is the expected shape, but it should be quoted as
  "better at interactive view planning", not "better".
- **Overall is a blunt average** of three tasks with different difficulty; it flatters
  models that are even, and hides the thing being measured. Prefer the per-task columns.

Regenerate with `scripts/build_ai2thor_results_table.py` (base + trained + frontier) or
`examples/evaluation/eval_all_openrouter_ai2thor/build_ai2thor_table_full.py`.
