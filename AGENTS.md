# AI2-THOR Environment for ViewAgent/ViewSuite — Agent Notes

This branch (`feat/ai2thor-env`) adds an **AI2-THOR** environment to ViewSuite,
mirroring the existing ScanNet proxy-task suite. It provides the same three
view-planning tasks — **P2V** (Path-to-View), **V2P** (View-to-Path), **IVP**
(Interactive View Planning) — sourced from AI2-THOR iTHOR scenes instead of
ScanNet, plus data generation, a VLM data filter, an HTTP render service, eval
configs, and a GraphRL iterative RL↔SFT training setup.

> This doc is the practical guide to what was added and how to run it. Task
> mapping to the ViewSuite paper terms: **P2V = forward dynamics**,
> **V2P = inverse dynamics**, **IVP = active exploration**.

---

## 1. What was added

```
view_suite/ai2thor/                      # renderer + HTTP render service (thin over ScanNet base)
  ai2thor_unified_renderer.py            #   client-only renderer (talks to the HTTP service)
  gym_ai2thor_render_env.py              #   ViewBaseEnv wrapper around the renderer
  gym_ai2thor_tool_env.py               #   camera-control action vocabulary
  view_manipulator.py                    #   Unity-pose helper (data_gen only)
  pose_utils.py  scene_list.py           #   OpenCV<->Unity pose math; 120 iTHOR scene catalog
  pre_download_scenes.py                 #   warm the ~/.ai2thor asset cache
  service_http/{service,handler}.py      #   FastAPI render service (CloudRendering controllers)
  tests/test_ai2thor_http_stress.py

view_suite/envs/ai2thor_proxy_task/
  path_to_view.py                        # Ai2ThorPath2View  (P2V, single-turn MCQ)
  view_to_path.py                        # Ai2ThorView2Path  (V2P, single-turn MCQ)
  interactive_view_planning.py           # Ai2ThorInteractiveViewPlanning (IVP, multi-turn)
  gym_proxy_tool.py                      # IVP engine (GymAi2thorToolEnv-based)
  data_gen/
    generate_data.py                     # crash-resilient single-GPU generator
    gen_parallel.py                      # multi-GPU parallel generation + merge
    filter_low_semantic.py               # VLM/pixel low-content view filter

scripts/
  ai2thor_http_service.sh                # start one render service
  ai2thor_http_service_loop.sh           # supervised restart loop
  ai2thor_pre_download_scenes.sh         # prewarm scenes
  download_ai2thor.sh                    # pull + extract dataset from HF

examples/evaluation/
  eval_default_ai2thor.yaml              # 3-task eval (test split), any backend
  eval_random_ai2thor.yaml              # random-nav IVP baseline (no model)

GraphRL/examples/viewsuite/ai2thor_interactive_view_planning/
  pipeline.yaml train.yaml val.yaml val_smoke.yaml
  run.sh          # full iterative RL<->SFT (Qwen2.5-VL, 8 GPU)
  run_smoke.sh    # short RL-only smoke

# env registration
GraphRL/graphrl/configs/vagen_configs/env_registry.yaml   # + Ai2Thor{Path2View,View2Path,InteractiveViewPlanning}
GraphRL/VAGEN/vagen/configs/env_registry.yaml             # (same, for the eval harness)

# fixes
view_suite/envs/utils/parse_utils.py                      # eval_mode format instruction: explicit <think>/<action> tags
GraphRL/graphrl/envs/viewsuite/viewsuite_interactive_view_planning/interactive_view_planning_graph_builder.py
                                                          # recognize AI2-THOR "FloorPlanN" scene ids
```

The AI2-THOR env is a **thin layer** over the shared ScanNet base classes
(`gym/`, `envs/base`, `envs/utils`, `scannet/view_manipulator`, `service_http/`,
`gym_proxy_no_tool`, `gym_proxy_tool_*` utils), so most logic is reused.

---

## 2. Environment setup

- **conda env `viewagent_thor`** (cloned from `viewagent`; editable installs
  re-pointed to *this* checkout). ai2thor 5.0.0, torch 2.8, transformers 4.57.1,
  sglang 0.5.3.post3, verl/vagen/graphrl/LLaMA-Factory. 8× B200.
- **Vulkan** (ai2thor CloudRendering needs `vulkaninfo`): installed via
  `conda install -c conda-forge vulkan-tools` (env-local, no sudo).
- **Two cloned-env fixes** (needed for the SFT phase):
  1. `GraphRL/LLaMA-Factory/src/llamafactory/data/` was missing (LLaMA-Factory's
     own `.gitignore` excludes `data/`, so it never entered the ViewAgent tree).
     Restore it (e.g. from a fresh LLaMA-Factory clone or a sibling checkout).
  2. `trl` must be **==0.24.0** (LLaMA-Factory requires `<=0.24.0`; verl/vagen
     import fine under it — the `graphrl==0.26.2` pip pin is metadata-only).
- **wandb**: authed via `~/.netrc` → **https://meta.wandb.io** (project
  `viewsuite`, entity `kangrui`). Scripts default `WANDB_MODE=online`.

---

## 3. Render service (only IVP needs it)

P2V/V2P read pre-rendered images from the jsonl; **only IVP** renders on the fly.
AI2-THOR CloudRendering needs an NVIDIA **graphics/Vulkan** container (not
compute-only). It runs on vast.ai boxes; the training box reaches them via SSH
tunnels through the corp forward proxy.

```bash
# on a render box (graphics-capable):
python view_suite/ai2thor/service_http/service.py --max_workers=8 --port=8766 \
    --platform=CloudRendering --width=512 --height=512 --fieldOfView=90 --gpu_ids=0

# on the training box: tunnel + point the client at it
ssh -N -L 8766:localhost:8766 <renderbox> &         # (via ProxyCommand if behind a proxy)
echo "http://localhost:8766" > client_url_ai2thor.txt
# multiple servers: semicolon-separated -> scene-routed load balancing
echo "http://localhost:8766;http://localhost:8767" > client_url_ai2thor.txt
```

Compute-only container? enable Vulkan by dropping in the matching NVIDIA driver
graphics libs (incl. `libnvidia-gpucomp`) and keeping **one** vulkan ICD
(`/etc/vulkan/icd.d/nvidia_icd.json`); a duplicate ICD double-enumerates the GPU
and trips ai2thor's CUDA↔Vulkan device-index assertion.

---

## 4. Data pipeline

No external dataset — data is generated from the live simulator.

```bash
export VIEWSUITE_ROOT=$(pwd)
# generate (120 iTHOR scenes x 24 samples, 8 GPUs, crash-resilient):
python -m view_suite.envs.ai2thor_proxy_task.data_gen.gen_parallel \
    --out_root=$VIEWSUITE_ROOT/data/ai2thor_full --scenes=all --samples_per_scene=24 --n_gpus=8

# filter low-semantic views (blank walls/floors) via a VLM (OpenRouter qwen3.7-plus):
python -m view_suite.envs.ai2thor_proxy_task.data_gen.filter_low_semantic \
    --data_root=$VIEWSUITE_ROOT/data/ai2thor_full --backend=openrouter --model=qwen/qwen3.7-plus \
    --workers=24 --review_dir=$VIEWSUITE_ROOT/filter_review   # OPENROUTER_API from .env, via egress-proxy

# scene-disjoint split -> _train/_eval/_test:
python -c "from view_suite.envs.utils.split_jsonl_by_scene import split_jsonl_by_scene as s; \
[s(f'data/ai2thor/{t}.jsonl',ratios=(70,15,15)) for t in ['path_to_view','view_to_path','interactive_view_planning']]"
```

**Current dataset** (`data/ai2thor`, on HF `datasets/JamesK2W/viewagent-ai2thor`):
120 iTHOR scenes → 2,880 samples/task → VLM-filtered (−26%) → **train 1,468 /
eval 325 / test 331** per task, scene-disjoint (84/18/18 scenes).
`bash scripts/download_ai2thor.sh` pulls + extracts it.

---

## 5. Evaluation

```bash
export VIEWSUITE_ROOT=$(pwd)
# open model via sglang (boots server, runs the 3-task eval on the test split):
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
CONFIG=$VIEWSUITE_ROOT/examples/evaluation/eval_default_ai2thor.yaml \
  bash examples/evaluation/eval_sglang/eval_model.sh
```

IVP needs the render service (`client_url_ai2thor.txt`). `eval_random_ai2thor.yaml`
runs a no-model random-nav IVP baseline (good service/plumbing check).

**Base Qwen2.5-VL-7B baseline** (clean test, n=331/task): **P2V 36.9% · V2P 24.5%
· IVP 6.9%**, 0/331 render crashes. (This is the eval-before-train reference.)

---

## 6. Training (GraphRL iterative RL↔SFT)

```bash
cd GraphRL                              # pipeline uses relative vagen_dir="VAGEN"
export VIEWSUITE_ROOT=$(cd .. && pwd)   # render service must be reachable
# short RL-only smoke:
bash examples/viewsuite/ai2thor_interactive_view_planning/run_smoke.sh
# full iterative run (4 iters RL<->SFT view-graph distillation, 8 GPU, wandb):
bash examples/viewsuite/ai2thor_interactive_view_planning/run.sh
```

The loop = **RL** (verl+sglang, IVP rollouts rendered live via the render
service) → **traj_to_sft** (build a view graph from rollouts, distill 7 SFT
datasets) → **LLaMA-Factory SFT** → next iter's start model. All wandb-logged
(`meta.wandb.io/kangrui/viewsuite`, run names `ai2thor_interactive_view_planning_*`).
Outputs under `GraphRL/exps/viewsuite/ai2thor_interactive_view_planning/`.

Validated end-to-end (RL + traj_to_sft + SFT all run and produce `sft_model`).
For a long unattended run, launch under `systemd-run --user` so it survives
session teardown. Throughput is gated by render-server count (IVP rollouts) — add
more render servers to `client_url_ai2thor.txt` to speed it up.

---

## 7. Status / notes

- Code committed on `feat/ai2thor-env`. Not pushed (no GitHub creds on the box):
  `with-proxy git push -u origin feat/ai2thor-env`.
- Dataset uploaded: `datasets/JamesK2W/viewagent-ai2thor/ai2thor.tar.gz` (4.0 GB).
- `data/`, `rollouts/`, `exps/`, `filter_review*/`, `outputs/` are gitignored.
- The `llamafactory/data` restore and `trl==0.24.0` are **env** fixes (not in git;
  see §2) — needed on any fresh env for the SFT phase.
