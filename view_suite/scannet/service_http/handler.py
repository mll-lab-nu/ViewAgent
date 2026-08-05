# handler.py - ScanNet Render Handler for HTTP Service
"""
ScanNet-specific handler implementation for HTTP service.
Handles 3D scene rendering using mesh renderers with GPU support.

Concurrency control is delegated to the service layer via UNIFIED_MAX_INFLIGHT env var.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from view_suite.service_http.handler import BaseHandler, HandlerResult
from view_suite.scannet.render.mesh_render import MeshRenderer
from view_suite.scannet.utils.path_utils import resolve_scene_ply, resolve_scene_gs_ply
from view_suite.service_http.multipart import numpy_to_png_bytes

SUPPORTED_BACKENDS = ("open3d", "gsplat", "habitat")

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)

# ---------- Worker-side globals (kept minimal) ----------
_ACTIVE_SCENE_ID: Optional[str] = None
_ACTIVE_RENDERER = None  # MeshRenderer or GaussianSplatRenderer
_RENDERER_LOCK = threading.Lock()
_FORCED_RENDER_SIZE: Optional[Tuple[int, int]] = None
# PHYSICAL gpu index this worker owns. Kept separate from CUDA_VISIBLE_DEVICES because
# habitat's gpu_device_id goes through EGL enumeration, which CUDA masking does not
# touch — after _bind_process_to_gpu(6) CUDA sees one device numbered 0, but habitat
# still needs the real "6" to pick the right EGL device.
_WORKER_GPU_ID: Optional[int] = None


def _bind_process_to_gpu(gpu_id: Optional[int]) -> None:
    """
    Bind this worker process to a single GPU.

    IMPORTANT: `CUDA_VISIBLE_DEVICES` alone is NOT enough for the open3d backend.
    Open3D renders through Filament -> EGL, and EGL enumerates devices independently
    of CUDA, so an Open3D context always lands on EGL device 0 no matter what CUDA
    sees. That is why multi-GPU open3d rendering never worked and only single-GPU
    boxes were usable. We therefore also mask at the driver/EGL level. The fully
    reliable fix for open3d is still container-level masking (NVIDIA_VISIBLE_DEVICES
    handed to the container runtime) or the habitat backend, which takes an explicit
    gpu_device_id and is verified to isolate.

    Args:
        gpu_id: GPU device ID to bind to, or None to skip binding
    """
    global _WORKER_GPU_ID
    if gpu_id is None:
        return
    _WORKER_GPU_ID = int(gpu_id)                          # habitat needs the PHYSICAL id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)      # CUDA (torch, gsplat)
    os.environ["NVIDIA_VISIBLE_DEVICES"] = str(gpu_id)    # container runtime masking
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)             # honoured by some EGL loaders
    os.environ.setdefault("MAGNUM_DEVICE", str(gpu_id))   # habitat/Magnum device pick
    try:
        import torch
        torch.cuda.set_device(0)
    except Exception:
        pass


def _normalize_gpu_ids(value: Sequence[int] | str | int | None) -> list[int]:
    """
    Parse GPU IDs from various input formats.

    Args:
        value: GPU IDs as comma-separated string, list of ints, or None

    Returns:
        List of valid GPU IDs (empty list means CPU/soft fallback)
    """
    if value is None:
        return []
    if isinstance(value, int):
        tokens = [str(value)]
    elif isinstance(value, str):
        tokens = [tok.strip() for tok in value.split(",") if tok.strip()]
    else:
        tokens = [str(v).strip() for v in value]
    out: list[int] = []
    for tok in tokens:
        try:
            out.append(int(tok))
        except Exception:
            LOGGER.warning("Ignoring invalid GPU id token: %s", tok)
    return out


def _normalize_forced_size(value: Sequence[int] | int | str | None) -> Optional[Tuple[int, int]]:
    """
    Parse forced render size from various input formats.

    Args:
        value: Size as 'W,H', 'WxH', int, [w,h], or None

    Returns:
        (width, height) tuple or None to use task-specific sizes

    Raises:
        ValueError: If format is invalid
    """
    if value is None:
        return None
    if isinstance(value, int):
        v = max(1, int(value))
        return (v, v)
    if isinstance(value, str):
        tokens = value.replace("x", ",").replace("X", ",").split(",")
        vals = [int(tok.strip()) for tok in tokens if tok.strip()]
    else:
        vals = [int(x) for x in value]
    if len(vals) != 2:
        raise ValueError("forced_render_size must be exactly two ints (width,height)")
    return (max(1, vals[0]), max(1, vals[1]))


def _ensure_scene_loaded(scene_id: str, scene_root: str, backend: str):
    """
    Keep one scene cached per process. Replace if scene_id changes.

    Args:
        scene_id: ScanNet scene identifier (e.g., "scene0011_00")
        scene_root: Root dir — mesh dataset root for backend="open3d",
                    3DGS dataset root for backend="gsplat"
        backend: "open3d" (MeshRenderer) or "gsplat" (GaussianSplatRenderer)

    Returns:
        Renderer instance for the requested scene.

    Raises:
        FileNotFoundError: If the expected PLY file is missing.
    """
    global _ACTIVE_SCENE_ID, _ACTIVE_RENDERER
    with _RENDERER_LOCK:
        if _ACTIVE_SCENE_ID == scene_id and _ACTIVE_RENDERER is not None:
            return _ACTIVE_RENDERER

        # Release previous renderer (best-effort cleanup)
        if _ACTIVE_RENDERER is not None:
            # habitat frees its Simulator (and the GPU memory) via close()
            if hasattr(_ACTIVE_RENDERER, "close"):
                with contextlib.suppress(Exception):
                    _ACTIVE_RENDERER.close()
            # gsplat renderer exposes .release() directly
            if hasattr(_ACTIVE_RENDERER, "release") and not hasattr(_ACTIVE_RENDERER, "mesh"):
                with contextlib.suppress(Exception):
                    _ACTIVE_RENDERER.release()
            # mesh renderer wraps the offscreen renderer
            off = getattr(_ACTIVE_RENDERER, "renderer", None)
            if off is not None and hasattr(off, "release"):
                with contextlib.suppress(Exception):
                    off.release()
        _ACTIVE_RENDERER = None
        _ACTIVE_SCENE_ID = None

        # Load new scene
        if backend == "gsplat":
            from view_suite.scannet.render.gsplat_render import GaussianSplatRenderer
            ply_path = resolve_scene_gs_ply(scene_root, scene_id)
            renderer = GaussianSplatRenderer(ply_path)
        elif backend == "open3d":
            ply_path = resolve_scene_ply(scene_root, scene_id)
            renderer = MeshRenderer(ply_path)
        elif backend == "habitat":
            from view_suite.scannet.render.habitat_render import HabitatRenderer
            ply_path = resolve_scene_ply(scene_root, scene_id)   # same mesh as open3d
            renderer = HabitatRenderer(ply_path, gpu_device_id=_WORKER_GPU_ID or 0)
        else:
            raise ValueError(f"Unsupported backend={backend!r}; expected one of {SUPPORTED_BACKENDS}")
        _ACTIVE_RENDERER = renderer
        _ACTIVE_SCENE_ID = scene_id
        LOGGER.info("[ScanNetRender] Loaded %s renderer scene_id=%s", backend, scene_id)
        return renderer


def _render_images_worker(
    scene_id: str,
    tasks: List[Dict[str, Any]],
    scene_root: str,
    backend: str,
    gpu_id: Optional[int] = None,
    forced_render_size: Optional[Tuple[int, int]] = None,
) -> List[bytes]:
    """
    Worker entry point: bind GPU, load scene, render tasks to PNG bytes.

    This function runs in a separate process. It configures GPU binding on first call,
    loads the requested scene (caching it for subsequent calls), and renders all tasks.

    Args:
        scene_id: ScanNet scene identifier
        tasks: List of render task dicts (each with mode, intrinsics, extrinsics, size)
        scene_root: Data root — mesh root for open3d, 3DGS root for gsplat
        backend: "open3d" or "gsplat"
        gpu_id: GPU device ID to bind to (None for CPU)
        forced_render_size: Override render size if specified

    Returns:
        List of rendered PNG bytes (one per task)

    Raises:
        Exception: Various rendering errors (propagated to caller)
    """
    # Configure per-process static state
    global _FORCED_RENDER_SIZE
    if _FORCED_RENDER_SIZE is None and forced_render_size is not None:
        _FORCED_RENDER_SIZE = forced_render_size

    # Bind GPU (idempotent if already set)
    _bind_process_to_gpu(gpu_id)

    renderer = _ensure_scene_loaded(scene_id, scene_root, backend)

    out: List[bytes] = []
    for idx, task in enumerate(tasks):
        # Determine render size
        if _FORCED_RENDER_SIZE is not None:
            w, h = _FORCED_RENDER_SIZE
        else:
            size = task.get("size") or [300, 300]
            w, h = int(size[0]), int(size[1])

        mode = (task.get("mode") or "").lower()

        try:
            if mode == "cam_param":
                K = np.array(task["intrinsics"], dtype=float)
                T = np.array(task["extrinsics"], dtype=float)
                img_array = renderer.render_image_from_cam_param(K, T, width=w, height=h)
                img_bytes = numpy_to_png_bytes(img_array.astype(np.uint8))
            else:
                LOGGER.warning(
                    "[ScanNetRender/Worker] Unknown mode '%s' for task #%d; transparent image",
                    mode, idx
                )
                img_bytes = numpy_to_png_bytes(np.zeros((h, w, 4), dtype=np.uint8))
        except Exception as exc:
            # Propagate to main process
            raise

        out.append(img_bytes)

    return out


# ---------- Worker Pool (fixed-size, minimal) ----------
@dataclass
class _WorkerSlot:
    """
    Represents a single worker process with GPU affinity.

    Attributes:
        executor: ProcessPoolExecutor managing this worker
        gpu_id: GPU device ID assigned to this worker (None for CPU)
        current_scene: Scene ID currently loaded in this worker (for sticky dispatch)
        last_used: Timestamp of last use (for LRU eviction)
    """
    executor: ProcessPoolExecutor
    gpu_id: Optional[int]
    current_scene: Optional[str] = None
    last_used: float = 0.0


class _WorkerPool:
    """
    Fixed-size pool with LRU dispatch and simple BrokenProcessPool recovery.

    This pool maintains a fixed number of worker processes, each potentially bound
    to a specific GPU. It implements:
    - Sticky scheduling: requests for the same scene go to the same worker
    - LRU eviction: when all workers are busy with different scenes
    - Crash recovery: automatic respawn of crashed worker processes
    """

    def __init__(
        self,
        max_workers: int = 4,
        scene_root: str = "",
        backend: str = "open3d",
        gpu_ids: Sequence[int] | None = None,
        forced_render_size: Optional[Tuple[int, int]] = None,
        *,
        crash_cooldown_s: float = 0.5,
    ):
        """
        Initialize worker pool.

        Args:
            max_workers: Number of worker processes to maintain
            scene_root: Data root for the selected backend
            backend: "open3d" or "gsplat"
            gpu_ids: List of GPU IDs to distribute across workers (round-robin)
            forced_render_size: Override all render sizes if specified
            crash_cooldown_s: Seconds to wait before respawning crashed worker
        """
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend={backend!r}; expected one of {SUPPORTED_BACKENDS}"
            )
        self.max_workers = max(1, int(max_workers))
        self.scene_root = scene_root
        self.backend = backend
        self.gpu_ids: List[Optional[int]] = [int(g) for g in (gpu_ids or [])] or [None]
        self.forced_render_size = forced_render_size
        self.crash_cooldown_s = float(crash_cooldown_s)

        self._ctx = mp.get_context("spawn")
        self.worker_slots: List[_WorkerSlot] = []
        for i in range(self.max_workers):
            gid = self.gpu_ids[i % len(self.gpu_ids)]
            self.worker_slots.append(
                _WorkerSlot(executor=self._create_executor(), gpu_id=gid)
            )

        self.scene_to_worker: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._metrics: Counter[str] = Counter()

    def _create_executor(self) -> ProcessPoolExecutor:
        """Create a single-worker process executor."""
        return ProcessPoolExecutor(max_workers=1, mp_context=self._ctx)

    def _assign_worker_locked(self, scene_id: str) -> int:
        """
        Assign a worker to handle the given scene (must hold self._lock).

        Strategy: sticky by scene, then prefer idle, then LRU eviction.

        Args:
            scene_id: Scene identifier

        Returns:
            Worker slot index
        """
        # Check if scene already mapped to a worker (sticky)
        mapped = self.scene_to_worker.get(scene_id)
        if mapped is not None:
            return mapped

        # Prefer idle worker
        for idx, slot in enumerate(self.worker_slots):
            if slot.current_scene is None:
                self.scene_to_worker[scene_id] = idx
                slot.current_scene = scene_id
                return idx

        # All workers busy: evict LRU
        idx = min(range(len(self.worker_slots)), key=lambda i: self.worker_slots[i].last_used)
        prev = self.worker_slots[idx].current_scene
        if prev:
            self.scene_to_worker.pop(prev, None)
        self.worker_slots[idx].current_scene = scene_id
        self.scene_to_worker[scene_id] = idx
        return idx

    async def render(self, scene_id: str, tasks: List[Dict[str, Any]]) -> List[bytes]:
        """
        Submit render tasks to worker pool.

        Args:
            scene_id: Scene identifier
            tasks: List of render task dictionaries

        Returns:
            List of rendered PNG bytes

        Raises:
            BrokenProcessPool: If worker crashes twice consecutively
            Various rendering errors propagated from worker
        """
        async with self._lock:
            idx = self._assign_worker_locked(scene_id)
            slot = self.worker_slots[idx]
            slot.last_used = time.monotonic()
            executor = slot.executor
            gpu_id = slot.gpu_id

        loop = asyncio.get_running_loop()
        try:
            self._metrics["submit"] += 1
            return await loop.run_in_executor(
                executor,
                _render_images_worker,
                scene_id,
                tasks,
                self.scene_root,
                self.backend,
                gpu_id,
                self.forced_render_size,
            )
        except BrokenProcessPool:
            # Tear down -> cool down -> respawn -> retry once
            self._metrics["broken_pool"] += 1
            LOGGER.error(
                "[ScanNetRender] BrokenProcessPool on slot=%d gpu=%s; respawning after %.2fs",
                idx, gpu_id, self.crash_cooldown_s
            )

            async with self._lock:
                slot = self.worker_slots[idx]
                if slot.current_scene:
                    self.scene_to_worker.pop(slot.current_scene, None)
                with contextlib.suppress(Exception):
                    slot.executor.shutdown(wait=True, cancel_futures=True)

            await asyncio.sleep(self.crash_cooldown_s)

            async with self._lock:
                slot = self.worker_slots[idx]
                with contextlib.suppress(Exception):
                    slot.executor.shutdown(wait=True, cancel_futures=True)
                slot.executor = self._create_executor()
                slot.current_scene = scene_id
                slot.last_used = time.monotonic()
                self.scene_to_worker[scene_id] = idx
                executor = slot.executor
                gpu_id = slot.gpu_id

            try:
                self._metrics["broken_pool_retry"] += 1
                return await loop.run_in_executor(
                    executor,
                    _render_images_worker,
                    scene_id,
                    tasks,
                    self.scene_root,
                    self.backend,
                    gpu_id,
                    self.forced_render_size,
                )
            except BrokenProcessPool:
                self._metrics["broken_pool_repeated"] += 1
                LOGGER.error(
                    "[ScanNetRender] repeated BrokenProcessPool on slot=%d gpu=%s; giving up",
                    idx, gpu_id
                )
                raise

    async def warm(self, scene_ids: Sequence[str]) -> None:
        """
        Warm up cache by loading specified scenes into workers.

        Args:
            scene_ids: List of scene identifiers to preload
        """
        for s in scene_ids:
            if not s:
                continue
            try:
                await self.render(s, [])
            except Exception as exc:
                LOGGER.warning("[ScanNetRender] warm failed scene=%s err=%s", s, exc)
                self._metrics["warm_fail"] += 1

    async def aclose(self) -> None:
        """Shutdown all workers gracefully."""
        for slot in self.worker_slots:
            slot.current_scene = None
            with contextlib.suppress(Exception):
                slot.executor.shutdown(wait=True, cancel_futures=True)

    def metrics(self) -> Dict[str, int]:
        """Return current metrics counters."""
        return dict(self._metrics)


# ---------- Public handler ----------
class ScanNetRenderHandler(BaseHandler):
    """
    ScanNet render handler for HTTP service framework.

    This handler manages a pool of worker processes for rendering ScanNet scenes.
    Each worker can be bound to a specific GPU and caches one scene at a time.

    Concurrency control: Handled at service level via UNIFIED_MAX_INFLIGHT env var.
    This handler focuses purely on rendering logic and resource management.

    Input meta format:
      {
        "scene_id": "scene0011_00",
        "tasks": [
          {
            "mode": "cam_param",
            "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "extrinsics": [[...4x4 matrix...]],
            "size": [width, height]
          },
          ...
        ]
      }

    Output:
      - meta: {"scene_id": "...", "count": N}
      - encoded_images: List[bytes] (PNG frames)
    """

    def __init__(
        self,
        max_workers: int = 4,
        scannet_root: str = "",
        gs_root: str = "",
        backend: str = "open3d",
        log_level: int = logging.INFO,
        gpu_ids: Sequence[int] | str | None = None,
        forced_render_size: Sequence[int] | int | str | None = None,
        *,
        crash_cooldown_s: float = 0.5,
        **kwargs: Any,
    ):
        """
        Initialize ScanNet render handler.

        Args:
            max_workers: Number of worker processes
            scannet_root: Root directory of mesh-PLY ScanNet dataset (open3d backend)
            gs_root: Root directory of 3DGS checkpoints (gsplat backend)
            backend: "open3d" (MeshRenderer) or "gsplat" (GaussianSplatRenderer)
            log_level: Logging level (default: INFO)
            gpu_ids: GPU IDs to use (comma-separated string or list)
            forced_render_size: Force all renders to this size (e.g., "512,512" or 512)
            crash_cooldown_s: Seconds to wait before respawning crashed worker
        """
        LOGGER.setLevel(log_level)
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend={backend!r}; expected one of {SUPPORTED_BACKENDS}"
            )
        normalized_gpu_ids = _normalize_gpu_ids(gpu_ids)
        normalized_size = _normalize_forced_size(forced_render_size)
        scene_root = gs_root if backend == "gsplat" else scannet_root

        self.backend = backend
        self.scene_root = scene_root
        self.pool = _WorkerPool(
            max_workers=max_workers,
            scene_root=scene_root,
            backend=backend,
            gpu_ids=normalized_gpu_ids,
            forced_render_size=normalized_size,
            crash_cooldown_s=crash_cooldown_s,
        )

        self._metrics: Counter[str] = Counter()

    async def handle(self, meta: Dict[str, Any], images: List[Image.Image]) -> HandlerResult:
        """
        Handle render request.

        Args:
            meta: Dict containing "scene_id" and "tasks"
            images: Not used for ScanNet rendering (input images ignored)

        Returns:
            HandlerResult with rendered images and metadata
        """
        scene_id: str = meta.get("scene_id") or ""
        tasks: List[Dict[str, Any]] = meta.get("tasks") or []

        if not scene_id:
            LOGGER.error("[ScanNetRender] Missing scene_id in request")
            return HandlerResult(
                meta={"error": "Missing scene_id"},
                images=[]
            )

        if not isinstance(tasks, list):
            LOGGER.error("[ScanNetRender] Invalid tasks format")
            return HandlerResult(
                meta={"error": "Invalid tasks format"},
                images=[]
            )

        LOGGER.info("[ScanNetRender] scene_id=%s tasks=%d", scene_id, len(tasks))

        try:
            rendered_images = await self.pool.render(scene_id, tasks)
            self._metrics["success"] += 1
            self._metrics["images_returned"] += len(rendered_images)

            return HandlerResult(
                meta={"scene_id": scene_id, "count": len(rendered_images)},
                encoded_images=rendered_images,
                image_format="PNG",
                image_mime="image/png",
            )
        except FileNotFoundError as exc:
            self._metrics["not_found"] += 1
            LOGGER.error("[ScanNetRender] Scene not found: %s", exc)
            return HandlerResult(
                meta={"error": f"Scene not found: {exc}"},
                images=[]
            )
        except Exception as exc:
            tb = traceback.format_exc()
            LOGGER.error("[ScanNetRender] Internal error: %s\n%s", exc, tb)
            self._metrics["internal_error"] += 1
            return HandlerResult(
                meta={"error": "Internal rendering error"},
                images=[]
            )

    async def aclose(self) -> None:
        """Cleanup: shutdown worker pool."""
        await self.pool.aclose()

    async def warm_cache(self, scene_ids: Sequence[str]) -> None:
        """
        Preload scenes into workers (optional warmup).

        Args:
            scene_ids: List of scene identifiers to preload
        """
        await self.pool.warm(scene_ids)

    def metrics_snapshot(self) -> Dict[str, int]:
        """
        Get current metrics snapshot.

        Returns:
            Dictionary with handler and pool metrics
        """
        merged = dict(self._metrics)
        try:
            merged.update({f"pool_{k}": v for k, v in self.pool.metrics().items()})
        except Exception:
            pass
        return merged
