<h1 align="center">
  <img src="assets/viewsuite_logo.png" alt="ViewSuite logo" height="34" style="vertical-align: middle; margin-right: 6px;">
  Planning with the Views
</h1>

<!-- Badges -->
<div align="center">

[![Paper](https://img.shields.io/badge/📄-Paper-b31b1b.svg)](https://viewsuite.github.io/viewsuite_paper.pdf)
[![Homepage](https://img.shields.io/badge/🏠-Homepage-blue.svg)](https://viewsuite.github.io/)
[![Models](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/collections/MLL-Lab/viewsuite-models)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-orange.svg)](https://huggingface.co/collections/MLL-Lab/viewsuite-datasets)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <a href="https://openreview.net/profile?id=~Kangrui_Wang2">Kangrui Wang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en">Linjie Li</a><sup>2</sup>,
  <a href="https://zyang-ur.github.io/">Zhengyuan Yang</a><sup>3</sup>,
  <a href="https://github.com/shiqichen17">Shiqi Chen</a><sup>4</sup>,
  <a href="https://zihanwang314.github.io/">Zihan Wang</a><sup>1</sup>,
  <a href="https://profiles.stanford.edu/fei-fei-li">Li Fei-Fei</a><sup>5</sup>,
  <a href="https://jiajunwu.com/">Jiajun Wu</a><sup>5</sup>,
  <a href="https://geometry.stanford.edu/member/guibas/">Leonidas Guibas</a><sup>5</sup>,
  <a href="https://www.microsoft.com/en-us/research/people/lijuanw/">Lijuan Wang</a><sup>3</sup>,
  <a href="https://limanling.github.io/">Manling Li</a><sup>1</sup>
</p>
<p align="center">
  <sup>1</sup>Northwestern University&nbsp;&nbsp;
  <sup>2</sup>University of Washington&nbsp;&nbsp;
  <sup>3</sup>Microsoft&nbsp;&nbsp;
  <sup>4</sup>University of Oxford&nbsp;&nbsp;
  <sup>5</sup>Stanford University
</p>

<p align="center">
  <img src="assets/environment_overview.png" alt="ViewSuite environment overview" width="90%">
</p>

---

## 📢 Updates

- **[2026-05-20]** We release the ViewSuite codebase, benchmark, and the iterative self-exploration training framework, along with the [dataset](https://huggingface.co/collections/MLL-Lab/viewsuite-datasets) and [trained checkpoints](https://huggingface.co/collections/MLL-Lab/viewsuite-models) on HuggingFace.
- **[Coming soon]** Paper on arXiv.

## 🌟 Overview

Can VLMs predict how each camera move changes the view, and plan many such moves ahead? We call this capability **view planning**, and decompose it into two coupled abilities: (1) **understanding** how a single action transforms the view, and (2) **composing** many such transformations across multi-turn plans to identify a target view.

**ViewSuite** is a 3D point-cloud environment and benchmark suite for view planning, built on ~300 real [ScanNet](http://www.scan-net.org/) indoor scenes (~55K view pairs, ~165K task instances). It probes view planning through three diagnostic tasks:

- **Path-to-View (P2V)** — predict the resulting view from an action sequence *(tests understanding)*.
- **View-to-Path (V2P)** — infer the action sequence between two views *(tests understanding)*.
- **Interactive View Planning (IVP)** — plan view changes over multiple turns and submit a 6-DoF estimate of the target *(tests multi-turn leveraging)*.

Across 13 frontier VLMs, a critical **planning gap** emerges: models possess basic view-action knowledge (~50–70% on short-horizon P2V/V2P) but fail to compose it across multi-turn plans (below 21% on IVP). To close this gap, we propose an **iterative training framework** that alternates *self-exploration* with *view graph distillation*. The key insight is that all exploration trajectories, regardless of outcome, collectively form a view graph; distilling it into diverse supervised tasks reshapes the policy distribution and overcomes the sparse rewards that stall pure RL. This improves Qwen2.5-VL-7B from **2.5% → 47.8%** on Interactive View Planning, surpassing GPT-5.4 Pro (18.5%) and Gemini 3.1 Pro (21.4%).

For more details, see our [paper](https://viewsuite.github.io/viewsuite_paper.pdf) and [project homepage](https://viewsuite.github.io/).

## 📦 Repository Structure

```
ViewSuite/
├── view_suite/      # Core ViewSuite environment & Python package
├── GraphRL/         # Iterative RL–SFT training framework
├── examples/        # Evaluation configs (API models, sglang, baselines)
├── scripts/         # Install, render-service, and data-download scripts
├── visualizer/      # Trajectory & view visualization tools
└── setup.py
```

> All commands below assume you are at the **ViewSuite repo root** (`cd /path/to/ViewSuite`).
> Where applicable, `${VIEWSUITE_ROOT}` is the absolute path to that root and is auto-exported by the install scripts.

---

## ⚙️ 1. Installation

### Full install (training + evaluation + service)

Used on the machine that runs RL/SFT training and the eval harness.

```bash
# Clone the repository
git clone https://github.com/mll-lab-nu/ViewSuite.git
cd ViewSuite

# Create env (Python 3.12)
conda create -n viewsuite python=3.12 -y
conda activate viewsuite

# Install ViewSuite + GraphRL + VAGEN + verl + LLaMA-Factory + sglang
bash scripts/install.sh
```

### Service-only install (render service host)

For the machine that hosts the ScanNet HTTP render service. Lighter — no RL stack.

```bash
conda create -n viewsuite python=3.12 -y
conda activate viewsuite

bash scripts/install_service.sh
```

---

## 📥 2. Download Data

### Service side — ScanNet scans + meshes (large)

Lives on the render-service machine. Downloads from the public dataset repo [`MLL-Lab/viewsuite`](https://huggingface.co/datasets/MLL-Lab/viewsuite).

```bash
bash scripts/download_scannet.sh
# downloads scannet.tar.gz into data/
```

### Local — ViewSuite tasks (jsonl + small assets)

Lives on the training/eval machine (the one talking to the render service).

```bash
bash scripts/download_viewsuite_all.sh
# downloads viewsuite_15k.tar.gz + mindcube.tar.gz into data/
```

After both, you should have:

```
data/
├── scannet/scans/...
└── viewsuite_15k/
    ├── interactive_view_planning_test.jsonl   # Interactive View Planning (IVP)
    ├── path_to_view_test.jsonl                # Path-to-View (P2V)
    ├── view_to_path_test.jsonl                # View-to-Path (V2P)
    └── ...
```

### Trained checkpoints (optional)

Download the released Qwen2.5-VL-7B checkpoints — used as starting points or eval targets.

```bash
bash scripts/download_model.sh
# downloads into model/qwen25-ivp/{viewsuite-all-qwen25vl7b,viewsuite-ivp-qwen25vl7b}/
```

---

## 🖥️ 3. Start the ScanNet Render Service

The service exposes an HTTP render endpoint that gym environments call to render camera views from ScanNet scenes. Run it on a GPU box, after the ScanNet data is downloaded (Step 2).

> **Who needs this?** Only **Interactive View Planning (IVP)** — and Gaussian-Splat–rendered evaluation — renders views on the fly and needs this service. **Path-to-View (P2V) and View-to-Path (V2P) do not**: they are single-turn tasks that read pre-rendered images straight from the jsonl, so they run and evaluate without ever starting the service.

**Mesh backend (open3d, recommended, full splits available)** — renders directly from the ScanNet meshes (Step 2); no extra download needed.

```bash
# Required: tells the service where to find data/scannet/...
export VIEWSUITE_ROOT="$(pwd)"

#   args: MAX_WORKERS=32 GPU_IDS=0 OMP_CAP=1 PORT=8767 T=10800 BACKEND=open3d
bash scripts/scannet_http_service_loop.sh 32 0 1 8767 10800 open3d
```

**3D-Gaussian-Splatting backend (gsplat, only test split available)** — renders from pretrained per-scene 3DGS reconstructions of the ScanNet scenes ([`GaussianWorld/scannet_mcmc_1.5M_3dgs`](https://huggingface.co/datasets/GaussianWorld/scannet_mcmc_1.5M_3dgs), from the [SceneSplat-7K](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k) project). Download those first into `data/scannet_3dgs_mcmc/`:

```bash
export VIEWSUITE_ROOT="$(pwd)"
export HF_TOKEN=hf_xxx               # huggingface_hub token
bash scripts/download_scannet_3dgs.sh
```

Then start the service with the gsplat backend (same args; `BACKEND` defaults to gsplat):

```bash
export VIEWSUITE_ROOT="$(pwd)"
bash scripts/scannet_http_service_loop_gs.sh 32 0 1 8767
```

The supervisor restarts the worker every `T` seconds (default 3h). Logs land under `./scannet_http_service_<TS>/`.

To run it in the background and persist its URL:

```bash
export VIEWSUITE_ROOT="$(pwd)"
nohup bash scripts/scannet_http_service_loop.sh 32 0 1 8767 \
  > scannet_http_service_loop.log 2>&1 &
echo "$!" > scannet_http_service_loop.pid
echo "http://0.0.0.0:8767" > client_url.txt   # consumed by env configs
```

**Choosing `MAX_WORKERS`** (the first arg). Each worker keeps a ScanNet scene resident in GPU memory, so the worker count is bounded by **both GPU VRAM and CPU core count** (see [`scripts/scannet_http_service_loop.sh`](scripts/scannet_http_service_loop.sh)). `32` is a safe default for a 24–48 GB GPU on a ~32-core host. On a large card with many cores — e.g. an RTX 6000 Pro (Blackwell) on a 64-core box — try `64`. If you hit GPU OOM or CPU thrashing, lower it.

---

## 🎮 4. Try the Environments

Each task is a self-contained gym environment you can **play interactively from the keyboard** — a quick way to get a feel for the tasks and to confirm your data, install, and (for IVP) render service are wired up. Run from the repo root, or with `VIEWSUITE_ROOT` exported. Every observed/rendered image is saved to a folder so you can look at it.

**P2V and V2P — no render service needed.** Single-turn multiple-choice tasks that read pre-rendered images from the jsonl, so they work as soon as the data (Step 2) is in place. The demo prints a question, saves its images, and you answer with `A` / `B` / `C` / `D`:

```bash
export VIEWSUITE_ROOT="$(pwd)"
python view_suite/envs/scannet_proxy_task/path_to_view.py   # images -> tests/p2v_play/
python view_suite/envs/scannet_proxy_task/view_to_path.py   # images -> tests/v2p_play/
```

**IVP — requires a running render service** (Step 3). You fly the camera around a ScanNet scene with the keyboard; each newly rendered view is saved to `tests/ivp_play/`. The demo runs in **action-only + no-submit** mode — you start on the initial view and just navigate; the episode auto-succeeds once you reach the target.

```
w/s  move forward / backward    q/e  turn left / right
a/d  move left / right          r/f  look up / down
y/h  move up / down             t/g  rotate ccw / cw
```

```bash
export VIEWSUITE_ROOT="$(pwd)"
# Service URL: --client_url, else client_url.txt, else http://0.0.0.0:8767
python view_suite/envs/scannet_proxy_task/interactive_view_planning.py
python view_suite/envs/scannet_proxy_task/interactive_view_planning.py \
  --client_url=http://0.0.0.0:8767
```

Type movement keys in any combination (e.g. `wwd`), or `quit` to exit. If the IVP demo cannot connect, the render service is unreachable — recheck Step 3 and `client_url.txt`.

---

## 📊 5. Evaluation

Both eval suites read `data/viewsuite_15k/*.jsonl`. **IVP** evaluation additionally needs the render service (Step 3) and `client_url.txt`; **P2V/V2P** do not.

### 5a. Closed-source / API models — `examples/evaluation/eval_scannet_proxy_task`

Configs already exist for each model (`claude_opus_4_6.yaml`, `gpt_5_4.yaml`, `gemini_3_pro.yaml`, ...).

```bash
export VIEWSUITE_ROOT="$(pwd)"
export fileroot="$(pwd)"

# Run all models (set the API keys for the models you intend to run):
export OPENROUTER_API_KEY=...        # Claude / GPT-5 family via OpenRouter
bash examples/evaluation/eval_scannet_proxy_task/eval_all.sh

# Or run a single model:
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

python -m vagen.evaluate.run_eval \
  --config examples/evaluation/eval_scannet_proxy_task/claude_opus_4_6.yaml \
  fileroot="$(pwd)"
```

Rollouts are dumped to `${fileroot}/rollouts/<model_name>/tag_<task>/...`.

### 5b. Open-source / custom models via sglang — `examples/evaluation/eval_sglang`

```bash
export VIEWSUITE_ROOT="$(pwd)"

# Minimal: any sglang-supported VLM, default 3-task YAML
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
  bash examples/evaluation/eval_sglang/eval_model.sh

# RTX 6000 / Blackwell + local checkpoint + IVP-only YAML
MODEL_PATH=/path/to/local/checkpoint \
MODEL_NAME=my_ckpt \
CONFIG=examples/evaluation/eval_sglang/interactive_view_planning_only.yaml \
DUMP_DIR="$(pwd)/rollouts/my_ckpt" \
CUDA_VISIBLE_DEVICES=1 \
SGLANG_EXTRA_ARGS="--attention-backend=flashinfer --mm-attention-backend=triton_attn" \
  bash examples/evaluation/eval_sglang/eval_model.sh
```

More examples are in [`examples/evaluation/eval_sglang/README.md`](examples/evaluation/eval_sglang/README.md).

---

## 🏋️ 6. Iterative RL–SFT Training

Trains the Qwen-VL agent on Interactive View Planning, alternating self-exploration (RL) with view graph distillation (SFT).

<p align="center">
  <img src="assets/iterative_training.png" alt="Iterative training pipeline (self-exploration + view graph distillation)" width="95%">
</p>

```bash
export VIEWSUITE_ROOT="$(pwd)"
export WANDB_API_KEY=your_wandb_key            # or `export WANDB_MODE=offline`
export HF_TOKEN=hf_xxx                         # for checkpoint upload, optional
cd GraphRL
# The render service must be reachable; the default expects http://0.0.0.0:8767
# (see client_url.txt produced in Step 3).

# Default: 8 GPUs per node for both RL and SFT.
bash examples/viewsuite/viewsuite_interactive_view_planning/run.sh

# Override GPU count or any pipeline knob:
N_GPUS_PER_NODE=8 SFT_N_GPUS=8 \
  bash examples/viewsuite/viewsuite_interactive_view_planning/run.sh \
  iterations=5
```

Outputs land under `exps/viewsuite/viewsuite_interactive_view_planning/`.

### Scaling the render service for training

IVP rollouts hit the render service constantly, and **switching scenes is expensive** — each switch reloads a ScanNet point cloud into GPU memory. To keep the trainer fed, run **multiple render services in parallel** and list all of them in `client_url.txt`, one URL per line:

```
http://10.0.0.1:8767
http://10.0.0.2:8767
http://10.0.0.3:8767
http://10.0.0.4:8767
```

`interactive_view_planning.py` talks to the service over HTTP and distributes its environments across every URL listed in `client_url.txt`. More services means more scenes stay resident at once, so workers reload point clouds far less often. For our training runs we ran one service per machine on **4× RTX 4090 boxes with 32 workers each**, which comfortably serves **~128 parallel environments**. Scale the number of services and `MAX_WORKERS` to your hardware (see the `MAX_WORKERS` note in Step 3).

---

## 📝 Citation

If you find ViewSuite useful in your research, please consider citing our paper:

```bibtex
@article{wang2026viewsuite,
  title   = {ViewSuite: View Planning via Scene Self-Exploration},
  author  = {Wang, Kangrui and Li, Linjie and Yang, Zhengyuan and Chen, Shiqi and
             Wang, Zihan and Fei-Fei, Li and Wu, Jiajun and Guibas, Leonidas and
             Wang, Lijuan and Li, Manling},
  year    = {2026}
}
```

## 🙏 Acknowledgements

ViewSuite is built on [ScanNet](http://www.scan-net.org/) for real 3D indoor scenes, and our training and evaluation framework draws on [VAGEN](https://github.com/RAGEN-AI/VAGEN), [verl](https://github.com/volcengine/verl), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and [sglang](https://github.com/sgl-project/sglang). The higher-fidelity Gaussian-Splatting renders use pretrained per-scene ScanNet 3DGS reconstructions from [SceneSplat-7K](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k) ([SceneSplat](https://arxiv.org/abs/2503.18052), ICCV 2025). We thank the authors of these projects for open-sourcing their work.

## 📄 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT). Note that ScanNet data and any third-party models are subject to their own licenses and terms of use.
