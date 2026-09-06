# Habitat-Sim render backend with real multi-GPU isolation

## Why

Open3D renders through Filament → **EGL**, and EGL enumerates devices independently of
CUDA. `CUDA_VISIBLE_DEVICES` therefore has no effect on which GPU Open3D draws on — every
render worker lands on EGL device 0. That is why multi-GPU rendering never worked and why
we could only ever use single-GPU boxes (4090/5090 on vast.ai), which became the
throughput bottleneck for RL rollouts.

Habitat-Sim calls `eglQueryDevicesEXT` + `eglGetPlatformDisplayEXT` to bind an **explicit**
device, exposed as `gpu_device_id`.

## What this adds

- `view_suite/scannet/render/habitat_render.py` — `HabitatRenderer`.
- `backend="habitat"` in the HTTP service (`SUPPORTED_BACKENDS`) and, partially, in
  `UnifiedRenderer` — see Limitations.
- `BACKEND=habitat` in the launcher scripts, plus orphan reaping in the supervisor.
- Two load-test harnesses (`test_habitat_multigpu.py`, `test_habitat_stress.py`), and a
  correctness fix to the existing `test_scannet_http_stress.py`.

Requires a separate env — habitat-sim cannot coexist with the training env — and it needs
the service deps and the repo on its path, not just habitat-sim:

```
conda create -y -n habitat python=3.9
conda install -y -n habitat habitat-sim headless -c conda-forge -c aihabitat
conda run -n habitat pip install uvicorn fastapi fire httpx python-multipart plyfile
```

The launcher selects that interpreter when `BACKEND=habitat`, exports
`PYTHONPATH=$VIEWSUITE_ROOT`, and **fails loudly** if the interpreter is missing rather
than silently serving zero images.

## Non-obvious things this had to get right

Each was a real failure, not a hypothetical:

**GPU index space.** `_bind_process_to_gpu` sets `CUDA_VISIBLE_DEVICES` to the worker's
GPU, so inside the process that device is renumbered to 0 — but habitat matches
`gpu_device_id` against *CUDA* devices. Passing the physical id aborted every worker
except GPU 0's with `unable to find CUDA device N among 17 EGL devices`, surfacing as an
endless `BrokenProcessPool` respawn loop with all traffic collapsed onto one GPU.

**One current GL context per process.** Constructing a second `Simulator` steals the
context from the first, so with N resident scenes only the last can draw, and `close()`
on a non-current simulator aborts the process. Both paths now call `acquire_gl_context()`.

**Return type.** `HabitatRenderer` originally returned a PIL Image while `MeshRenderer`
returns an ndarray. The service calls `.astype(np.uint8)`, so every habitat render raised
`AttributeError` in the worker, `handle()`'s catch-all turned it into HTTP 200 with an
error payload and **zero images**, and the harness counted it as success.

**There is no in-place FOV change in habitat-sim 0.3.3.** Setting `spec.hfov` and calling
`sensor.set_projection_params(spec)` is accepted and does nothing — a renderer built at
90° and asked for 60° returns a bit-identical image. An earlier version of this branch
used that path *and* updated its cached `_hfov`, so every subsequent request
short-circuited and the whole stream silently rendered at the construction FOV. FOV
changes now rebuild.

**Non-square pixels.** Habitat's projection is always square-pixel; ScanNet's intrinsics
are not (`default_intrinsics()` is fx 462.07 / fy 617.31). Using fx alone squashed every
frame vertically by 1.34× and showed ~30% too much vertical content. Now renders at
height `H·fx/fy` so the implied vertical FOV matches fy, then resamples (target vfov
45.05°, achieved 45.02°).

**Coordinates and lighting.** ScanNet is Z-up and habitat loads the mesh unrotated, so a
Z-up→Y-up rotation renders pure black. `override_scene_light_defaults=True` is required or
the `.ply` loads on a flat vertex-colour path that ignores lights. `clear_color` is white
to match `MeshRenderer`.

**Orphan reaping, safely.** Workers are multiprocessing-spawn children whose cmdline is
`multiprocessing.spawn`, so pgrep on the service path never finds them; killing only the
supervisor's child left 151 orphans holding VRAM. They do share the service's process
group, so the supervisor reaps by PGID — but `kill -0 -$pg` is not a sufficient guard: it
returns success for `0` (the caller's own group) and `1` (`kill(-1)`, every process the
user owns). Reaping now requires a numeric PGID > 1, not our own, still a live group
leader, with a matching `/proc` start time so a recycled PID is not mistaken for ours. The
record lives under `XDG_RUNTIME_DIR` per-user with `umask 077`, and an `flock` stops a
second supervisor on the same port from killing the first one's service.

**The harness scored errors as successes.** `test_scannet_http_stress.py` counted any
non-raising response as success, and `handle()` returns HTTP 200 + `meta={"error": ...}`
with zero images for internal failures. That is what hid the return-type bug. It now
requires `len(images) == len(tasks)`. It also benchmarked a hardcoded identity pose —
measured 1% non-background, a 1.9 KiB PNG, 11 ms, versus 46 ms and 88 KiB for a real
interior view — and randomised fx per task, which under the current (correct) code means
a full rebuild per frame. Both fixed: fixed intrinsics, in-room poses.

## Measurements

8×B200, real ScanNet meshes, 640×480, on the code as it stands. The box was shared with an
unrelated vLLM job throughout, so throughput is a lower bound.

| | |
|---|---|
| VRAM, 244 MB mesh | **322 MiB** marginal per scene, **433 MiB** for the first scene in a fresh process |
| Render, fixed intrinsics | **~25 ms** |
| Render, intrinsics changing per frame | **~2 s** — that is a scene rebuild, not a render |
| HTTP, 32 scenes / 64 clients, warm | **159 images/s**, median 1.63 s, p95 2.82 s, 0 failures |
| HTTP, same, cold | 66 images/s, p95 15.2 s (32 first-time scene loads) |

Per-GPU isolation verified through the service path: with `--gpu_ids=6,7`, worker 0 grew
GPU 6 by 433 MiB and worker 1 grew GPU 7 by 432 MiB, with no cross-talk.

Per-scene VRAM is **not** a constant — it tracked mesh size from 248 MiB (220 MB ply) to
401 MiB (386 MB ply), and the corpus ranges 42–580 MB. Do not extrapolate one number.

## Limitations

- **Not a drop-in `MeshRenderer` replacement.** It implements the same
  `render_image_from_cam_param` but does not subclass `BaseRenderer`, lacks `.mesh`,
  `.renderer`, `_scale_K_with_letterbox` (used by `analysis/.../visibility.py`), and does
  not accept `ssaa`/`taa_samples`/`jitter`/`downsample`. Default size is 512 vs 300.
- **cx/cy are ignored.** Habitat's projection is symmetric, so an off-centre principal
  point is silently dropped. Harmless for the Ks in use today (MeshRenderer's letterbox
  re-centres to within a sub-pixel of centre), wrong for a cropped K.
- **Residency is bounded by `max_workers`**, not by VRAM: the pool caches one scene per
  process and LRU-evicts beyond that. Measured re-request of a resident scene 0.04 s vs
  2.04 s for an evicted one. With 286 scenes and 64 workers, most requests miss. Sizing
  the pool for the working set matters more than the per-scene VRAM figure suggests.
- **`UnifiedRenderer` is only half-wired**: `render_tasks()` has no habitat branch and
  falls through to the HTTP client, and `RenderConfig.gpu_device_id` is unreachable
  because `__init__` never accepts it, so that path is pinned to GPU 0.
- **`_WorkerPool.warm()` calls `render(scene, [])`**, so with no tasks the sensor is built
  at the 300×300 fallback and the first real request pays a rebuild.
- **Visual distribution differs from Open3D**, so existing models and results do not
  transfer and must be re-run. Shading is a directional rig, not Open3D's `defaultLit`
  PBR: habitat saturates 0.7–1.9% of interior pixels where Open3D saturates none, and the
  white background is 255 flat versus Open3D's ~235 with dither. Structurally
  `lighting=False` matches Open3D better (NCC 0.917 vs 0.878 over 7 poses); for mean
  brightness `lighting=True` is closer.
- Pre-existing worker-pool issues this PR does not touch: LRU eviction updates parent-side
  bookkeeping without unloading the evicted worker's renderer; the `BrokenProcessPool`
  respawn path can load one scene into two processes and can drop an executor reference;
  `shutdown(wait=True)` blocks the event loop while holding the pool lock.

## Test plan

```
export VIEWSUITE_ROOT=$PWD PYTHONPATH=$PWD
HP=~/miniconda3/envs/habitat/bin/python3

# in-process, multi-GPU
$HP view_suite/scannet/tests/test_habitat_multigpu.py \
    --ply data/viewagent_scannet/scans/scene0011_00/scene0011_00_vh_clean.ply \
    --gpus 0,1,2,3,4,5,6,7 --scenes_per_gpu 4 --requests_per_gpu 100

# over HTTP, via the supervisor
bash scripts/scannet_http_service_loop.sh 64 0,1,2,3,4,5,6,7 1 8801 10800 habitat &
$HP view_suite/scannet/tests/test_scannet_http_stress.py --url=http://localhost:8801 \
    --scene_folder_path=data/viewagent_scannet --num_scenes=32 --num_clients=64 \
    --requests_per_client=5 --num_tasks_per_request=5
```

Expect `Success rate: 100%` — the harness now fails a request that returns fewer images
than tasks, which is what a broken renderer looks like.
