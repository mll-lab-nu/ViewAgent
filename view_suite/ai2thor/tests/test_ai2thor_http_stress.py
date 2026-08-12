#!/usr/bin/env python3
"""
AI2-THOR HTTP Render Service Stress Test (AsyncUnifiedClient version)

This version uses AsyncUnifiedClient for testing the AI2-THOR HTTP render service.

Usage:
    python test_ai2thor_http_stress.py \
        --url=http://localhost:8765 \
        --num_scenes=30 \
        --num_clients=128 \
        --requests_per_client=10 \
        --scene_prefix=FloorPlan \
        --num_tasks_per_request=5 \
        --timeout=120 \
        --retries=3 \
        --backoff=0.5 \
        --max_connections=200
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import fire
import httpx
import numpy as np

from view_suite.service_http.async_client import AsyncUnifiedClient


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


def generate_random_pose_task(width: int = 640, height: int = 480, fov: float = 90.0) -> Dict[str, Any]:
    """Generate a random camera pose task for AI2-THOR."""
    # Random position in typical room range
    x = random.uniform(-5.0, 5.0)
    y = random.uniform(0.5, 2.5)  # Typical camera height
    z = random.uniform(-5.0, 5.0)

    # Random rotation (yaw only for simplicity)
    yaw = random.uniform(0, 360)
    pitch = random.uniform(-30, 30)
    roll = 0.0

    pose = {
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"x": pitch, "y": yaw, "z": roll},
    }

    return {
        "pose": pose,
        "width": width,
        "height": height,
        "fov": fov,
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
      - Each worker owns one AsyncUnifiedClient to reuse its connection pool.
      - This avoids creating a new httpx client for every request.
    """
    client = AsyncUnifiedClient(
        url,
        token=token,
        timeout=timeout,
        max_connections=max_connections,
    )

    stats: List[RequestStats] = []

    try:
        for req_id in range(num_requests):
            scene_id = random.choice(scene_ids)
            tasks = [generate_random_pose_task() for _ in range(num_tasks_per_request)]

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

                stats.append(RequestStats(
                    client_id=client_id,
                    request_id=req_id,
                    scene_id=scene_id,
                    num_tasks=len(tasks),
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    success=True,
                    num_images=len(images),
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


async def run_stress_test(
    *,
    url: str,
    num_scenes: int,
    num_clients: int,
    requests_per_client: int,
    scene_prefix: str,
    scene_start: int,
    token: Optional[str],
    num_tasks_per_request: int,
    timeout: float,
    retries: int,
    backoff: float,
    max_connections: int,
):
    """Run stress test with multiple concurrent client workers."""
    # Generate scene IDs (e.g., FloorPlan1, FloorPlan2, ...)
    scene_ids = [f"{scene_prefix}{i}" for i in range(scene_start, scene_start + num_scenes)]

    print(f"\n{'='*80}")
    print("AI2-THOR HTTP Render Service Stress Test (AsyncUnifiedClient)")
    print(f"{'='*80}")
    print(f"Service URL:            {url}")
    print(f"Scene prefix:           {scene_prefix}")
    print(f"Scene range:            {scene_start} to {scene_start + num_scenes - 1}")
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
        async with httpx.AsyncClient(timeout=10.0) as hc:
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
    url: str = "http://localhost:8765",
    num_scenes: int = 10,
    num_clients: int = 128,
    requests_per_client: int = 10,
    scene_prefix: str = "FloorPlan",
    scene_start: int = 1,
    token: Optional[str] = None,
    num_tasks_per_request: int = 5,
    timeout: float = 240.0,
    retries: int = 6,
    backoff: float = 2.0,
    max_connections: int = 200,
):
    """
    Run AI2-THOR HTTP service stress test.

    Args:
        url: Service URL.
        num_scenes: Number of distinct scene IDs used in the test.
        num_clients: Number of concurrent client workers.
        requests_per_client: Number of requests sent per client worker.
        scene_prefix: Scene ID prefix (default: "FloorPlan").
        scene_start: Starting scene number (default: 1, generates FloorPlan1, FloorPlan2, ...).
        token: Optional API token (passed as ?token=...).
        num_tasks_per_request: Number of tasks sent in each request meta.
        timeout: Per-request timeout (seconds) for the async client.
        retries: Number of retries after the first attempt (total attempts = retries + 1).
        backoff: Initial backoff delay (seconds) used for exponential backoff + jitter.
        max_connections: Connection pool size for each worker's AsyncUnifiedClient.

    Examples:
        # Test with kitchen scenes (FloorPlan1-30)
        python test_ai2thor_http_stress.py --scene_prefix=FloorPlan --scene_start=1 --num_scenes=30

        # Test with living room scenes (FloorPlan201-230)
        python test_ai2thor_http_stress.py --scene_prefix=FloorPlan --scene_start=201 --num_scenes=30

        # Test with bedroom scenes (FloorPlan301-330)
        python test_ai2thor_http_stress.py --scene_prefix=FloorPlan --scene_start=301 --num_scenes=30

        # Heavy load test
        python test_ai2thor_http_stress.py --num_clients=256 --requests_per_client=20
    """
    asyncio.run(run_stress_test(
        url=url,
        num_scenes=num_scenes,
        num_clients=num_clients,
        requests_per_client=requests_per_client,
        scene_prefix=scene_prefix,
        scene_start=scene_start,
        token=token,
        num_tasks_per_request=num_tasks_per_request,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        max_connections=max_connections,
    ))


if __name__ == "__main__":
    fire.Fire(main)
