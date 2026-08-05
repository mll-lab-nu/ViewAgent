from __future__ import annotations  # py3.9 (habitat env) parses 3.10 unions
#!/usr/bin/env python3
"""
ScanNet Render HTTP Service

FastAPI-based service for rendering ScanNet 3D scenes with GPU acceleration.

Usage:
    python service.py --scannet_root=data/scannet/scans --max_workers=24 --port=8765

Environment variables:
    UNIFIED_MAX_INFLIGHT: Maximum concurrent requests (default: 0 = unlimited)
    UNIFIED_API_KEY: Optional API key for authentication
    UNIFIED_ADMIT_TIMEOUT: Timeout for request admission in seconds (default: 2.0)
"""
import logging
import os
from typing import Sequence

import uvicorn

from view_suite.service_http.service import build_app
from view_suite.scannet.service_http.handler import ScanNetRenderHandler

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def run(
    scannet_root: str = "data/scannet/scans",
    gs_root: str = "data/scannet_3dgs_mcmc",
    backend: str = "open3d",
    max_workers: int = 24,
    gpu_ids: Sequence[int] | str | None = None,
    forced_render_size: Sequence[int] | int | str | None = None,
    host: str = "0.0.0.0",
    port: int = 8765,
    reload: bool = False,
    log_level: str = "info",
    render_timeout: float = 120.0,
) -> None:
    """
    Start the ScanNet HTTP render service.

    Args:
        scannet_root: Root directory of ScanNet dataset (must contain scene subdirs)
        max_workers: Number of worker processes for rendering. NOTE each worker
            caches exactly ONE scene (handler._ensure_scene_loaded), so with N
            GPUs this is also the cap on cached scenes: scenes_per_gpu =
            max_workers / len(gpu_ids). Keep it small (~10/GPU) when the GPU is
            shared with training — 32 workers on one GPU is what exhausted VRAM.
        gpu_ids: GPU IDs to use (comma-separated string or list, e.g., "0,1,2,3")
        forced_render_size: Force all renders to this size (e.g., "512,512", "512x512", or 512)
        host: Host address to bind to (default: "0.0.0.0" for all interfaces)
        port: Port number to listen on (default: 8765)
        reload: Enable auto-reload for development (default: False)
        log_level: Logging level (default: "info")
        render_timeout: Timeout for each render request in seconds (default: 30.0)

    Environment variables (for concurrency control):
        UNIFIED_MAX_INFLIGHT: Max concurrent requests (0 = unlimited, default: 0)
        UNIFIED_API_KEY: Optional API key for authentication
        UNIFIED_RENDER_TIMEOUT: Render timeout in seconds (overrides render_timeout arg)
    """
    # Read environment variables for display
    max_inflight_env = os.getenv("UNIFIED_MAX_INFLIGHT", "0")
    api_key_set = bool(os.getenv("UNIFIED_API_KEY"))

    # Allow render timeout to be overridden by environment variable
    timeout_env = os.getenv("UNIFIED_RENDER_TIMEOUT")
    if timeout_env:
        render_timeout = float(timeout_env)

    handler = ScanNetRenderHandler(
        max_workers=max_workers,
        scannet_root=scannet_root,
        gs_root=gs_root,
        backend=backend,
        log_level=getattr(logging, log_level.upper(), logging.INFO),
        gpu_ids=gpu_ids,
        forced_render_size=forced_render_size,
        render_timeout_s=render_timeout,
    )

    app = build_app(handler)

    print("=" * 60)
    print(f"ScanNet HTTP Render Service starting on {host}:{port}")
    print("=" * 60)
    print(f"  backend:             {backend}")
    print(f"  scannet_root:        {scannet_root}")
    print(f"  gs_root:             {gs_root}")
    print(f"  max_workers:         {max_workers}")
    print(f"  gpu_ids:             {gpu_ids}")
    print(f"  forced_render_size:  {forced_render_size}")
    print(f"  render_timeout:      {render_timeout}s")
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
