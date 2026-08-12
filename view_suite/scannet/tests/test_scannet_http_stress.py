#!/usr/bin/env python3
"""
ScanNet HTTP Render Service Stress Test (HRWRoutedAsyncUnifiedClient version)

This version uses your new HRWRoutedAsyncUnifiedClient (which internally uses decode_multipart
from multipart.py), so the test no longer duplicates multipart parsing logic.

Usage:
    python test_scannet_http_stress_async_client.py \
        --url=http://localhost:8765 \
        --num_scenes=64 \
        --num_clients=128 \
        --requests_per_client=10 \
        --scene_prefix=scene \
        --num_tasks_per_request=5 \
        --timeout=120 \
        --retries=3 \
        --backoff=0.5 \
        --max_connections=200
"""

import asyncio
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import fire
import httpx
import numpy as np

# Adjust this import to your actual package/module layout:
# Example 1: if async_client.py is next to this file:
#   from async_client import AsyncUnifiedClient
# Example 2: if it is inside a package:
#   from yourpkg.async_client import AsyncUnifiedClient
from view_suite.service_http.async_client_routed import HRWRoutedAsyncUnifiedClient  # <-- change if needed


@dataclass
class RequestStats:
    """Statistics for a single request."""
    client_id: int
    request_id: int
    scene_id: str
    num_tasks: int
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: Optional[str] = None
    num_images: int = 0


def generate_random_camera_task(width: int = 640, height: int = 480) -> Dict[str, Any]:
    """Generate a random camera pose task for testing."""
    # Fixed intrinsics on purpose. Habitat bakes FOV into the sensor, so randomising fx
    # makes every frame a full scene rebuild (~2 s) and measures reload cost, not render
    # cost. Real callers hold intrinsics fixed.
    fx = 462.07   # view_suite.envs.utils.scannet_utils.default_intrinsics()
    fy = 617.31
    cx = width / 2
    cy = height / 2
    intrinsics = [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ]

    # A pose INSIDE the room, not identity. At identity the camera sits at the world
    # origin looking at almost nothing: measured 1% non-background, a 1.9 KiB PNG and
    # 11 ms, versus 46 ms and 88 KiB for a real interior view. Benchmarking the empty
    # frame overstates throughput ~4x.
    yaw = random.uniform(0, 2 * math.pi)
    fwd = [math.cos(yaw), math.sin(yaw), 0.0]
    right = [fwd[1], -fwd[0], 0.0]
    down = [
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    ]
    eye = [random.uniform(2.0, 5.0), random.uniform(2.0, 5.0), 1.5]
    extrinsics = [
        [right[0], down[0], fwd[0], eye[0]],
        [right[1], down[1], fwd[1], eye[1]],
        [right[2], down[2], fwd[2], eye[2]],
        [0, 0, 0, 1],
    ]

    return {
        "mode": "cam_param",
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "size": [width, height],
    }


async def client_worker(
    client_id: int,
    num_requests: int,
    url: str,
    scene_ids: List[str],
    *,
    token: Optional[str],
    num_tasks_per_request: int,
    timeout: float,
    retries: int,
    backoff: float,
    max_connections: int,
) -> List[RequestStats]:
    """
    Worker function for a single logical client.

    Notes:
      - Each worker owns one HRWRoutedAsyncUnifiedClient to reuse its connection pool.
      - This avoids creating a new httpx client for every request.
    """
    client = HRWRoutedAsyncUnifiedClient(
        url,
        token=token,
        timeout=timeout,
        max_connections=max_connections,
    )

    stats: List[RequestStats] = []

    try:
        for req_id in range(num_requests):
            scene_id = random.choice(scene_ids)
            tasks = [generate_random_camera_task() for _ in range(num_tasks_per_request)]

            meta = {
                "scene_id": scene_id,
                "tasks": tasks,
            }

            start_time = time.perf_counter()
            try:
                response_meta, images = await client.render(
                    meta=meta,
                    images=None,          # Stress test sends metadata only
                    retries=retries,
                    backoff=backoff,
                )
                end_time = time.perf_counter()

                # A 200 is NOT success. handler.handle() answers internal errors and
                # missing scenes with HTTP 200 + meta={"error": ...} and zero images, and
                # the client does not raise on those. Counting them as successes is how a
                # renderer that produced NO images at all still scored 100% — twice. A
                # request only counts if it returned one image per task.
                ok = len(images) == len(tasks)
                stats.append(RequestStats(
                    client_id=client_id,
                    request_id=req_id,
                    scene_id=scene_id,
                    num_tasks=len(tasks),
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    success=ok,
                    num_images=len(images),
                    error=None if ok else f"expected {len(tasks)} images, got {len(images)}",
                ))
            except Exception as e:
                end_time = time.perf_counter()
                stats.append(RequestStats(
                    client_id=client_id,
                    request_id=req_id,
                    scene_id=scene_id,
                    num_tasks=len(tasks),
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    success=False,
                    error=str(e),
                ))
    finally:
        await client.aclose()

    return stats

import os
async def run_stress_test(
    *,
    url: str,
    num_scenes: int,
    num_clients: int,
    requests_per_client: int,
    scene_folder_path: str,
    token: Optional[str],
    num_tasks_per_request: int,
    timeout: float,
    retries: int,
    backoff: float,
    max_connections: int,
):
    """Run stress test with multiple concurrent client workers."""
    scene_folder_path+="/scans"
    scene_ids = [os.path.basename(o) for o in sorted(os.listdir(scene_folder_path)) if os.path.isdir(os.path.join(scene_folder_path, o))][:num_scenes]

    print(f"\n{'='*80}")
    print("ScanNet HTTP Render Service Stress Test (AsyncUnifiedClient)")
    print(f"{'='*80}")
    print(f"Service URL:            {url}")
    print(f"Number of scenes:       {num_scenes}")
    print(f"Number of clients:      {num_clients}")
    print(f"Requests per client:    {requests_per_client}")
    print(f"Tasks per request:      {num_tasks_per_request}")
    print(f"Total requests:         {num_clients * requests_per_client}")
    print(f"Total render tasks:     {num_clients * requests_per_client * num_tasks_per_request}")
    print(f"Client timeout:         {timeout}s")
    print(f"Client retries:         {retries} (total attempts = {retries + 1})")
    print(f"Client backoff:         {backoff}s (exp backoff + jitter)")
    print(f"Client max_connections: {max_connections}")
    print(f"{'='*80}\n")

    # Health check
    try:
        # Same self-signed-certificate switch the render client honours.
        verify = os.getenv("RENDER_TLS_NO_VERIFY", "0").strip().lower() not in (
            "1", "true", "yes",
        )
        async with httpx.AsyncClient(timeout=10.0, verify=verify) as hc:
            r = await hc.get(f"{url}/health")
            r.raise_for_status()
            print("Service health check: OK")
            print(f"Health response: {r.json()}\n")
    except Exception as e:
        print(f"ERROR: Service health check failed: {e}")
        print(f"Make sure the service is running at {url}")
        return

    print("Starting stress test...\n")
    overall_start = time.perf_counter()

    worker_tasks = [
        client_worker(
            client_id=i,
            num_requests=requests_per_client,
            url=url,
            scene_ids=scene_ids,
            token=token,
            num_tasks_per_request=num_tasks_per_request,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            max_connections=max_connections,
        )
        for i in range(num_clients)
    ]

    all_stats_lists = await asyncio.gather(*worker_tasks, return_exceptions=True)

    overall_end = time.perf_counter()
    overall_duration = overall_end - overall_start

    all_stats: List[RequestStats] = []
    for item in all_stats_lists:
        if isinstance(item, Exception):
            print(f"Client worker error: {item}")
        else:
            all_stats.extend(item)

    successful = [s for s in all_stats if s.success]
    failed = [s for s in all_stats if not s.success]

    print(f"\n{'='*80}")
    print("Stress Test Results")
    print(f"{'='*80}")
    print(f"Total time:             {overall_duration:.2f}s")
    print(f"Total requests:         {len(all_stats)}")
    print(f"Successful requests:    {len(successful)}")
    print(f"Failed requests:        {len(failed)}")
    print(f"Success rate:           {100 * len(successful) / max(1, len(all_stats)):.2f}%")
    print()

    if successful:
        durations = [s.duration for s in successful]
        images_returned = sum(s.num_images for s in successful)

        print("Request Latency Statistics (successful requests):")
        print(f"  Min:                  {min(durations):.3f}s")
        print(f"  Max:                  {max(durations):.3f}s")
        print(f"  Mean:                 {np.mean(durations):.3f}s")
        print(f"  Median:               {np.median(durations):.3f}s")
        print(f"  P95:                  {np.percentile(durations, 95):.3f}s")
        print(f"  P99:                  {np.percentile(durations, 99):.3f}s")
        print(f"  Std Dev:              {np.std(durations):.3f}s")
        print()
        print("Throughput:")
        print(f"  Requests/sec:         {len(successful) / overall_duration:.2f}")
        print(f"  Images/sec:           {images_returned / overall_duration:.2f}")
        print()

    if failed:
        print("Failed Request Errors:")
        error_counts: Dict[str, int] = {}
        for s in failed:
            err = s.error or "Unknown error"
            error_counts[err] = error_counts.get(err, 0) + 1

        for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {err}: {count} times")
        print()

    if successful:
        scene_stats: Dict[str, List[float]] = {}
        for s in successful:
            scene_stats.setdefault(s.scene_id, []).append(s.duration)

        print("Per-Scene Statistics (top 10 most requested):")
        sorted_scenes = sorted(scene_stats.items(), key=lambda x: -len(x[1]))[:10]
        for scene_id, ds in sorted_scenes:
            print(f"  {scene_id}:")
            print(f"    Requests:           {len(ds)}")
            print(f"    Mean latency:       {np.mean(ds):.3f}s")
            print(f"    Median latency:     {np.median(ds):.3f}s")

    print(f"{'='*80}\n")


def main(
    url: str = "http://localhost:8767",
    num_scenes: int = 128,
    num_clients: int = 128,
    requests_per_client: int = 10,
    scene_folder_path: str = "data/scannet",
    token: Optional[str] = None,
    num_tasks_per_request: int = 5,
    timeout: float = 120.0,
    retries: int = 3,
    backoff: float = 0.5,
    max_connections: int = 200,
):
    """
    Args:
        url: Service URL.
        num_scenes: Number of distinct scene IDs used in the test.
        num_clients: Number of concurrent client workers.
        requests_per_client: Number of requests sent per client worker.
        scene_folder_path: Path to the folder containing scene directories.
        token: Optional API token (passed as ?token=...).
        num_tasks_per_request: Number of tasks sent in each request meta.
        timeout: Per-request timeout (seconds) for the async client.
        retries: Number of retries after the first attempt (total attempts = retries + 1).
        backoff: Initial backoff delay (seconds) used for exponential backoff + jitter.
        max_connections: Connection pool size for each worker's AsyncUnifiedClient.
    """
    asyncio.run(run_stress_test(
        url=url,
        num_scenes=num_scenes,
        num_clients=num_clients,
        requests_per_client=requests_per_client,
        scene_folder_path=scene_folder_path,
        token=token,
        num_tasks_per_request=num_tasks_per_request,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        max_connections=max_connections,
    ))


if __name__ == "__main__":
    fire.Fire(main)
