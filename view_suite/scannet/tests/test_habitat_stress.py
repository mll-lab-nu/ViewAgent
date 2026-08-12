"""Large-scale stress test for the Habitat renderer, mirroring
test_scannet_http_stress.py (--num_scenes / --num_clients / --requests_per_client)
but hitting HabitatRenderer in-process instead of going over HTTP.

Point of the test: with Open3D every worker collapsed onto EGL device 0, so the only
way to scale was more single-GPU boxes. Habitat pins each Simulator to an explicit
gpu_device_id, so we want to know what ONE GPU can actually sustain:
  * how many scenes can stay resident (VRAM per scene)
  * what throughput looks like when many clients hammer them concurrently
Scenes stay resident, so scene-switch cost — the main pain with the old service —
should be ~0 once loaded.

Usage:
    python test_habitat_stress.py --ply <mesh.ply> --num_scenes 64 \
        --num_clients 128 --requests_per_client 10 --gpu 5 [--threads 16]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def gpu_used_mib(gpu: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True).stdout.strip()
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="scene mesh; loaded num_scenes times")
    ap.add_argument("--num_scenes", type=int, default=64)
    ap.add_argument("--num_clients", type=int, default=128)
    ap.add_argument("--requests_per_client", type=int, default=10)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16, help="concurrent render threads")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    a = ap.parse_args()

    from view_suite.scannet.render.habitat_render import HabitatRenderer

    base = gpu_used_mib(a.gpu)
    print(f"[stress] gpu={a.gpu} baseline={base} MiB")
    print(f"[stress] scenes={a.num_scenes} clients={a.num_clients} "
          f"req/client={a.requests_per_client} threads={a.threads} "
          f"total_requests={a.num_clients * a.requests_per_client}")

    # ---- load phase -------------------------------------------------------
    t0 = time.perf_counter()
    sims = []
    for i in range(a.num_scenes):
        sims.append(HabitatRenderer(a.ply, gpu_device_id=a.gpu,
                                    width=a.width, height=a.height))
        if (i + 1) % 8 == 0 or i == 0:
            u = gpu_used_mib(a.gpu)
            print(f"  loaded {i+1:3d}/{a.num_scenes}  gpu={u} MiB  "
                  f"per_scene~{(u-base)/(i+1):.0f} MiB", flush=True)
    load_s = time.perf_counter() - t0
    after_load = gpu_used_mib(a.gpu)
    print(f"[stress] load done in {load_s:.1f}s  "
          f"vram_total={after_load-base} MiB  per_scene={(after_load-base)/a.num_scenes:.0f} MiB")

    # ---- concurrent request phase ----------------------------------------
    # Each "client" walks the scene list, so requests spread across all resident
    # scenes; a Simulator is not thread-safe, so serialise per-scene with a lock and
    # let concurrency come from hitting DIFFERENT scenes at once.
    K = np.array([[525., 0, a.width / 2], [0, 525., a.height / 2], [0, 0, 1.]])
    locks = [threading.Lock() for _ in sims]
    lat: list[float] = []
    lat_lock = threading.Lock()
    errors = [0]

    def client(cid: int) -> None:
        rng = np.random.default_rng(cid)
        mine: list[float] = []
        for r in range(a.requests_per_client):
            idx = (cid + r) % len(sims)
            c2w = random_c2w(rng)
            s = time.perf_counter()
            try:
                with locks[idx]:
                    sims[idx].render_image_from_cam_param(K, c2w, a.width, a.height)
                mine.append(time.perf_counter() - s)
            except Exception:
                errors[0] += 1
        with lat_lock:
            lat.extend(mine)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        list(ex.map(client, range(a.num_clients)))
    wall = time.perf_counter() - t0

    n = len(lat)
    arr = np.array(lat) if n else np.array([0.0])
    print(f"[stress] RESULT total_requests={n} errors={errors[0]} "
          f"wall={wall:.1f}s  throughput={n/wall:.1f} req/s")
    print(f"[stress] latency ms: mean={arr.mean()*1e3:.0f} p50={np.percentile(arr,50)*1e3:.0f} "
          f"p95={np.percentile(arr,95)*1e3:.0f} max={arr.max()*1e3:.0f}")
    print(f"[stress] vram: {after_load-base} MiB for {a.num_scenes} scenes")

    for s in sims:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
