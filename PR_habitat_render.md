# Habitat-Sim render backend with real multi-GPU isolation

## Why

Open3D renders through Filament → **EGL**, and EGL enumerates devices independently of
CUDA. `CUDA_VISIBLE_DEVICES` therefore has no effect on which GPU Open3D draws on — every
render worker lands on EGL device 0. That is why multi-GPU rendering never worked and why
we could only ever use single-GPU boxes (4090/5090 on vast.ai), which became the
throughput bottleneck for RL rollouts.

Habitat-Sim calls `eglQueryDevicesEXT` + `eglGetPlatformDisplayEXT` to bind an **explicit**
device, exposed as `gpu_device_id`. Verified empirically: requesting `gpu_device_id=6`
grows GPU 6's memory and no other GPU's.

## What this adds

- `view_suite/scannet/render/habitat_render.py` — `HabitatRenderer`, a drop-in
  `MeshRenderer` replacement (same `render_image_from_cam_param(K, c2w, w, h)` with
  OpenCV-style extrinsics).
- `backend="habitat"` in `UnifiedRenderer` and in the HTTP service
  (`SUPPORTED_BACKENDS`), keeping the existing one-process-per-scene worker pool.
- `BACKEND=habitat` in the launcher scripts, plus orphan reaping in the supervisor.
- Two load-test harnesses (`test_habitat_multigpu.py`, `test_habitat_stress.py`).

Requires a separate conda env, since habitat-sim cannot coexist with the training env:

```
conda create -y -n habitat python=3.9
conda install -y -n habitat habitat-sim headless -c conda-forge -c aihabitat
```

The launcher selects it automatically when `BACKEND=habitat`.

## Non-obvious things this had to get right

Each of these was a real failure, not a hypothetical:

**GPU index space.** `_bind_process_to_gpu` sets `CUDA_VISIBLE_DEVICES` to the worker's
GPU, so inside the process that device is renumbered to 0 — but habitat matches
`gpu_device_id` against *CUDA* devices. Passing the physical id aborted every worker
except GPU 0's with `unable to find CUDA device N among 17 EGL devices`, surfacing as an
endless `BrokenProcessPool` respawn loop with all traffic collapsed onto one GPU.

**One current GL context per process.** Constructing a second `Simulator` silently steals
the context from the first, so with N resident scenes only the last one can draw, and
`close()` on a non-current simulator aborts the process outright. Both paths now call
`acquire_gl_context()` first.

**Return type.** `HabitatRenderer` originally returned a PIL Image while `MeshRenderer`
returns an ndarray. The service calls `.astype(np.uint8)`, so every habitat render raised
`AttributeError` in the worker, `handle()`'s catch-all turned it into HTTP 200 with an
error payload and zero images, and the load harness counted it as a success. Caught by
code review; the harness had been printing `Images/sec: 0.00` all along.

**Sensor sizing.** Habitat bakes resolution into the sensor framebuffer (a rebuild costs
2.2-3.0 s on a ~230 MB mesh). Building at the
class default and letting the first render discover the real size forced an immediate
`close()` + rebuild, i.e. every worker loaded its scene twice. FOV, unlike resolution, is
now retargeted in place via `set_projection_params` — otherwise a caller varying focal
length pays a full scene reload per frame.

**Coordinates and lighting.** ScanNet is Z-up and Habitat loads the mesh without
reorienting it, so applying a Z-up→Y-up rotation renders pure black — the pose must pass
through unrotated. `override_scene_light_defaults=True` is required or the `.ply` loads on
a flat vertex-colour path that ignores lights entirely. `clear_color` is set to white to
match `MeshRenderer`, whose black-vs-white background alone made renders look ~5× darker.

**Scene switching in `UnifiedRenderer`.** `set_scene()` dropped `_ply` and `_mesh` but not
`_habitat`, so the habitat backend silently kept rendering the *previous* scene. `close()`
had the same gap. Explicit `close()` is required rather than just dropping the reference:
habitat's `Simulator` sits in a reference cycle via its sensors' back-pointer, so its
~330 MiB survives until a cyclic gc runs, which in a render worker may be never.

**Orphan reaping.** Pool workers are multiprocessing-spawn children whose cmdline is
`multiprocessing.spawn`, not `service.py`, so pgrep on the service path never finds them.
Killing only the supervisor's child left 151 orphaned workers holding GPU memory. The
supervisor now records each generation's PGID and kills by process group.

## Measurements

8×B200, real ScanNet meshes (~230 MB `vh_clean.ply`), 640×480.

| | |
|---|---|
| VRAM per resident scene | **~322 MiB** (drift-corrected against a control GPU; an independent per-PID measurement gave ~334 MiB + ~80 MiB fixed context, agreeing) |
| Scene load | 0.6–0.8 s, once; resident scenes make switching free |
| Single render | ~41 ms, unchanged when focal length varies |
| HTTP warm, 16 scenes / 32 clients | **213 images/s**, median 0.49 s, p95 1.10 s, 0 failures |

Per-scene VRAM means all 286 ViewSuite scenes fit resident across 8 GPUs with room left
for training.

## Caveats

- **Visual distribution differs from Open3D.** Shading is approximated with a directional
  rig rather than Open3D's `defaultLit` PBR, so existing models and results do not
  transfer and must be re-run. Comparison renders are in `render_compare/`.
- Throughput above was measured on a box shared with an unrelated vLLM job; it is a lower
  bound.
- Code review flagged worker-pool issues not addressed here, all pre-existing and
  orthogonal to this backend: LRU eviction updates parent-side bookkeeping without
  unloading the evicted worker's renderer; the `BrokenProcessPool` respawn path has a
  window where one scene can be loaded into two processes, and can drop an executor
  reference so its worker leaks; `slot.current_scene` is never reset, so "prefer idle"
  degrades to pure LRU; and `shutdown(wait=True)` blocks the event loop while holding the
  pool lock. Worth a follow-up PR.
- Only `tasks[0]`'s size is used to build the sensor, so a batch mixing `task["size"]`
  values still pays a rebuild per differing task. Fine for current callers, which use a
  fixed size.

## Test plan

```
# in-process, multi-GPU
python view_suite/scannet/tests/test_habitat_multigpu.py --ply <mesh.ply> \
    --gpus 0,1,2,3,4,5,6,7 --scenes_per_gpu 20 --requests_per_gpu 200

# over HTTP
BACKEND=habitat SCANNET_ROOT=data/scannet/scans \
    ./scripts/scannet_http_service_loop.sh 64 0,1,2,3,4,5,6,7 1 8796 10800 habitat
python view_suite/scannet/tests/test_scannet_http_stress.py --url=http://localhost:8796 \
    --scene_folder_path=data/scannet --num_scenes=16 --num_clients=32 \
    --requests_per_client=4 --num_tasks_per_request=5
```

Check `Images/sec` is non-zero — that is what the return-type bug suppressed.
