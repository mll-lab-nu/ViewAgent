#!/usr/bin/env python3
"""
AI2-THOR Render HTTP Service

FastAPI-based service for rendering AI2-THOR 3D scenes with GPU acceleration.

Usage:
    python service.py --max_workers=24 --port=8765

Environment variables:
    UNIFIED_MAX_INFLIGHT: Maximum concurrent requests (default: 0 = unlimited)
    UNIFIED_API_KEY: Optional API key for authentication
    UNIFIED_ADMIT_TIMEOUT: Timeout for request admission in seconds (default: 2.0)
"""
import logging
import os
import subprocess
from typing import List, Optional, Sequence

import uvicorn

from view_suite.service_http.service import build_app
from view_suite.ai2thor.service_http.handler import AI2ThorRenderHandler

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _detect_gpus() -> List[int]:
    """Auto-detect visible GPU ids (borrowed pattern from vagen navigation.serve).

    Order:
      1) CUDA_VISIBLE_DEVICES env var (post-mask indices: 0..N-1)
      2) nvidia-smi output
      3) fallback to [0]
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [int(i) for i, d in enumerate(vis.split(",")) if d.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        )
        return [int(line.strip()) for line in out.strip().split("\n") if line.strip()]
    except Exception:
        return [0]

def run(
    max_workers: int = 24,
    gpu_ids: Sequence[int] | str | None = None,
    platform: str = "CloudRendering",
    agentMode: str = "default",
    width: int = 512,
    height: int = 512,
    fieldOfView: float = 90.0,
    host: str = "0.0.0.0",
    port: int = 8765,
    reload: bool = False,
    log_level: str = "info",
) -> None:
    """
    Start the AI2-THOR HTTP render service.

    Args:
        max_workers: Number of worker processes for rendering
        gpu_ids: GPU IDs to use (comma-separated string or list, e.g., "0,1,2,3")
        platform: AI2-THOR platform type (default: "CloudRendering")
        agentMode: AI2-THOR agent mode (default: "default")
        width: Default image width (default: 512)
        height: Default image height (default: 512)
        fieldOfView: Default field of view in degrees (default: 90.0)
        host: Host address to bind to (default: "0.0.0.0" for all interfaces)
        port: Port number to listen on (default: 8765)
        reload: Enable auto-reload for development (default: False)
        log_level: Logging level (default: "info")

    Environment variables (for concurrency control):
        UNIFIED_MAX_INFLIGHT: Max concurrent requests (0 = unlimited, default: 0)
        UNIFIED_API_KEY: Optional API key for authentication
    """
    # Read environment variables for display
    max_inflight_env = os.getenv("UNIFIED_MAX_INFLIGHT", "0")
    api_key_set = bool(os.getenv("UNIFIED_API_KEY"))

    # Auto-detect GPUs if not explicitly provided (matches vagen navigation's
    # behaviour and avoids the "all workers share CUDA_VISIBLE_DEVICES mask"
    # footgun where every Unity controller lands on the same card).
    if gpu_ids is None or (isinstance(gpu_ids, str) and not gpu_ids.strip()):
        detected = _detect_gpus()
        gpu_ids = detected
        print(f"[service] gpu_ids auto-detected -> {detected}")

    handler = AI2ThorRenderHandler(
        max_workers=max_workers,
        log_level=getattr(logging, log_level.upper(), logging.INFO),
        gpu_ids=gpu_ids,
        platform=platform,
        agentMode=agentMode,
        width=width,
        height=height,
        fieldOfView=fieldOfView,
    )

    app = build_app(handler)

    print("=" * 60)
    print(f"AI2-THOR HTTP Render Service starting on {host}:{port}")
    print("=" * 60)
    print(f"  max_workers:         {max_workers}")
    print(f"  gpu_ids:             {gpu_ids}")
    print(f"  platform:            {platform}")
    print(f"  agentMode:           {agentMode}")
    print(f"  size:                {width}x{height}")
    print(f"  fieldOfView:         {fieldOfView}")
    print(f"  max_inflight:        {max_inflight_env} (from env UNIFIED_MAX_INFLIGHT)")
    print(f"  api_key:             {'✓ set' if api_key_set else '✗ not set'}")
    print("=" * 60)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )

if __name__ == "__main__":
    import fire
    fire.Fire(run)
