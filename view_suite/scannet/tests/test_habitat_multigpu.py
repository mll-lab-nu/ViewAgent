"""Multi-GPU Habitat stress test — the shape the real service will run in.

Same worker-pool idea as the open3d HTTP service (handler.py): render work happens in
separate PROCESSES, each pinned to one GPU. Two differences from open3d, both the point
of the migration:
  * the pin actually works (habitat gpu_device_id -> explicit EGL device; open3d's
    Filament/EGL path ignores CUDA_VISIBLE_DEVICES and always lands on device 0)
  * a worker can keep MANY scenes resident instead of one, so scene switching — the
    dominant cost in the old service — disappears.

Processes, not threads: habitat keeps ONE current GL context per process, so parallel
draws inside a process serialise no matter what — concurrency has to come from processes.
(Multiple resident scenes per process are still fine: each render calls
acquire_gl_context() first, which is what stops the "GL::Context::current(): no current
context" abort that killed the first version of this test.)

Usage (the 8x20 config):
    python test_habitat_multigpu.py --ply <mesh.ply> --gpus 0,1,2,3,4,5,6,7 \
        --scenes_per_gpu 20 --requests_per_gpu 200
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
import time

import numpy as np


def gpu_used_mib(gpu: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu)], capture_output=True, text=True).stdout.strip()
    try:
        return int(out.splitlines()[0])
    except Exception:
        return -1


def random_c2w(rng: np.random.Generator) -> np.ndarray:
    """Random horizontal camera pose inside a typical ScanNet room (Z-up)."""
    yaw = rng.uniform(0, 2 * np.pi)
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = right, down, fwd
    c2w[:3, 3] = [rng.uniform(2, 7), rng.uniform(2, 8), 1.5]
    return c2w


def worker(gpu: int, ply: str, n_scenes: int, n_req: int, w: int, h: int, q, barrier=None) -> None:
    """One process = one GPU = n_scenes resident scenes, serving n_req renders."""
    from view_suite.scannet.render.habitat_render import HabitatRenderer

    base = gpu_used_mib(gpu)
    t0 = time.perf_counter()
    sims = []
    try:
        for _ in range(n_scenes):
            sims.append(HabitatRenderer(ply, gpu_device_id=gpu, width=w, height=h))
    except Exception as e:                       # OOM or EGL failure -> report partial
        q.put({"gpu": gpu, "error": f"{type(e).__name__}: {str(e)[:160]}",
               "loaded": len(sims)})
        q.close(); q.join_thread()
        import os; os._exit(0)
    load_s = time.perf_counter() - t0
    after = gpu_used_mib(gpu)

    K = np.array([[525., 0, w / 2], [0, 525., h / 2], [0, 0, 1.]])
    rng = np.random.default_rng(gpu)
    # warm up (first draw per sim pays one-off GL setup), then wait for everyone
    sims[0].render_image_from_cam_param(K, random_c2w(rng), w, h)
    if barrier is not None:
        barrier.wait()
    lat = []
    errs = 0
    t0 = time.perf_counter()
    for i in range(n_req):
        s = time.perf_counter()
        try:
            # round-robin across resident scenes: every request is a "scene switch"
            # that costs nothing because the scene is already loaded
            sims[i % len(sims)].render_image_from_cam_param(K, random_c2w(rng), w, h)
            lat.append(time.perf_counter() - s)
        except Exception:
            errs += 1
    wall = time.perf_counter() - t0

    arr = np.array(lat) if lat else np.array([0.0])
    q.put({"gpu": gpu, "loaded": len(sims), "load_s": load_s,
           "vram": after - base, "per_scene": (after - base) / max(1, len(sims)),
           "n": len(lat), "wall": wall, "errors": errs,
           "mean": arr.mean(), "p50": float(np.percentile(arr, 50)),
           "p95": float(np.percentile(arr, 95))})
    q.close(); q.join_thread()          # make sure the parent has the result...
    import os; os._exit(0)              # ...then skip teardown entirely


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--scenes_per_gpu", type=int, default=20)
    ap.add_argument("--requests_per_gpu", type=int, default=200)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    a = ap.parse_args()

    gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
    total_scenes = len(gpus) * a.scenes_per_gpu
    total_req = len(gpus) * a.requests_per_gpu
    print(f"[mgpu] gpus={gpus} scenes/gpu={a.scenes_per_gpu} -> {total_scenes} scenes")
    print(f"[mgpu] requests/gpu={a.requests_per_gpu} -> {total_req} total", flush=True)

    ctx = mp.get_context("spawn")     # fork + EGL/CUDA in the parent do not mix
    q = ctx.Queue()
    barrier = ctx.Barrier(len(gpus))
    procs = [ctx.Process(target=worker,
                         args=(g, a.ply, a.scenes_per_gpu, a.requests_per_gpu,
                               a.width, a.height, q, barrier))
             for g in gpus]
    for p in procs:
        p.start()
    results = [q.get() for _ in gpus]
    t0 = min(time.perf_counter() - r["wall"] for r in results if "wall" in r)
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0

    print(f"\n[mgpu] {'gpu':>4} {'scenes':>7} {'load_s':>7} {'vram_MiB':>9} "
          f"{'per_scene':>10} {'reqs':>6} {'req/s':>7} {'p50_ms':>7} {'p95_ms':>7} {'err':>4}")
    ok = [r for r in results if "error" not in r]
    for r in sorted(results, key=lambda r: r["gpu"]):
        if "error" in r:
            print(f"[mgpu] {r['gpu']:>4}  FAILED after {r['loaded']} scenes: {r['error']}")
            continue
        print(f"[mgpu] {r['gpu']:>4} {r['loaded']:>7} {r['load_s']:>7.1f} {r['vram']:>9} "
              f"{r['per_scene']:>10.0f} {r['n']:>6} {r['n']/r['wall']:>7.1f} "
              f"{r['p50']*1e3:>7.0f} {r['p95']*1e3:>7.0f} {r['errors']:>4}")

    if ok:
        agg = sum(r["n"] for r in ok) / wall
        print(f"\n[mgpu] TOTAL scenes_resident={sum(r['loaded'] for r in ok)} "
              f"vram_all={sum(r['vram'] for r in ok)} MiB")
        print(f"[mgpu] TOTAL requests={sum(r['n'] for r in ok)} wall={wall:.1f}s "
              f"aggregate_throughput={agg:.1f} req/s")
        best = max(r["n"] / r["wall"] for r in ok)
        print(f"[mgpu] per-gpu best={best:.1f} req/s  "
              f"scaling_efficiency={agg/(best*len(ok))*100:.0f}% vs perfect linear")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
