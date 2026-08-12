# handler.py - AI2-THOR Render Handler for HTTP Service (Hybrid: multi-process + multi-thread + multi-slot)
"""
Hybrid concurrency model:
- Few processes (max_process) for isolation and CPU/GIL separation
- Inside each process: ThreadPoolExecutor (max_threads) driving multiple Controllers (max_slots)

Per-task overrides:
- Each task may override width/height/fov.
- If overrides differ from slot params, we RECONSTRUCT the controller (slow).
- RECONSTRUCT is expected to be rare; we log WARNING when it happens.

Scene switching:
- If scene differs and params are same, we RESET controller to target scene (fast).
- RESET and RECONSTRUCT logs/metrics are distinct.

Input/Output contract unchanged:
Input meta:
  {
    "scene_id": "FloorPlan1",
    "tasks": [
      {
        "pose": {"position": {...}, "rotation": {...}},
        "width": 512, "height": 512, "fov": 90.0   # optional
      },
      ...
    ]
  }

Output:
  HandlerResult(meta={"scene_id": "...", "count": N}, encoded_images=[PNG bytes...])
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from view_suite.service_http.handler import BaseHandler, HandlerResult

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)


# =============================================================================
# Shared utilities
# =============================================================================
def _map_platform(platform_str: str) -> Any:
    """Map platform string to AI2-THOR platform class."""
    if platform_str == "CloudRendering":
        from ai2thor.platform import CloudRendering
        return CloudRendering
    if platform_str == "Linux64":
        from ai2thor.platform import Linux64
        return Linux64
    from ai2thor.platform import CloudRendering
    return CloudRendering


def _normalize_gpu_ids(value: Sequence[int] | str | int | None) -> List[int]:
    """Parse gpu_ids from str / list / int / None -> List[int]. Empty list means "no pinning"."""
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        return [int(tok.strip()) for tok in s.split(",") if tok.strip()]
    return [int(v) for v in value]


def _add_third_party_camera(controller: Any, fov: float) -> None:
    """
    Add ThirdPartyCamera (assume id=0).

    IMPORTANT:
    - After controller.reset(...), third-party cameras are typically cleared.
      We MUST re-add it after reset.
    """
    controller.step(
        action="AddThirdPartyCamera",
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        rotation={"x": 0.0, "y": 0.0, "z": 0.0},
        fieldOfView=float(fov),
    )


def _reset_controller_to_scene(controller: Any, scene_id: str) -> None:
    """Reset controller to a new scene using common API patterns."""
    try:
        controller.reset(scene=scene_id)
        return
    except TypeError:
        pass
    except Exception:
        raise

    try:
        controller.reset(scene_id)
        return
    except TypeError:
        pass
    except Exception:
        raise

    # Best-effort fallback
    try:
        controller.step(action="Reset", sceneName=scene_id)
        return
    except Exception:
        pass

    raise RuntimeError("Unable to reset controller to new scene; unsupported ai2thor reset API")


def _create_controller(
    controller_config: Dict[str, Any],
    scene_id: str,
    width: int,
    height: int,
    fov: float,
    gpu_device: Optional[int] = None,
) -> Any:
    """Create a new AI2-THOR Controller (+ ThirdPartyCamera), optionally pinned to a GPU.

    If gpu_device is given, it is passed through to AI2-THOR's Controller so the
    underlying Unity process picks that physical device (CloudRendering honors
    this). This is our knob for real per-process GPU distribution, independent
    of the shell-level CUDA_VISIBLE_DEVICES mask.
    """
    from ai2thor.controller import Controller

    platform = _map_platform(controller_config.get("platform", "CloudRendering"))
    kwargs: Dict[str, Any] = dict(
        platform=platform,
        agentMode=controller_config.get("agentMode", "default"),
        scene=scene_id,
        width=int(width),
        height=int(height),
        fieldOfView=float(fov),
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )
    if gpu_device is not None:
        kwargs["gpu_device"] = int(gpu_device)
    controller = Controller(**kwargs)
    _add_third_party_camera(controller, fov)
    return controller


def _encode_rgb_to_png_bytes(frame_rgb: np.ndarray) -> bytes:
    """Encode RGB frame to PNG bytes."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    return buf.tobytes()


def _transparent_png(width: int, height: int) -> bytes:
    """Transparent placeholder PNG."""
    img = np.zeros((int(height), int(width), 4), dtype=np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def _task_params(task: Dict[str, Any], default_w: int, default_h: int, default_f: float) -> tuple[int, int, float]:
    """Extract per-task width/height/fov with fallback to defaults."""
    w = int(task.get("width", default_w))
    h = int(task.get("height", default_h))
    f = float(task.get("fov", default_f))
    return w, h, f


# =============================================================================
# Per-process thread-based pool (lives inside a worker process)
# =============================================================================
@dataclass
class _ControllerSlot:
    """One slot owns exactly one Controller instance."""
    controller: Any
    scene_id: str
    width: int
    height: int
    fov: float
    last_used: float
    lock: threading.Lock


class _ThreadControllerPool:
    """
    In-process pool of controller slots + threads.

    - max_slots: how many controllers we keep alive in this process
    - max_threads: how many threads execute blocking controller.step/reset concurrently
    - Defaults are per-process, but each task may override w/h/fov (rarely).
    """
    def __init__(
        self,
        *,
        max_slots: int,
        max_threads: int,
        controller_config: Dict[str, Any],
        default_width: int,
        default_height: int,
        default_fov: float,
        slot_gpu_ids: Optional[List[Optional[int]]] = None,
    ):
        self.max_slots = max(1, int(max_slots))
        self.max_threads = max(1, int(max_threads))
        self.controller_config = dict(controller_config)

        self.default_width = int(default_width)
        self.default_height = int(default_height)
        self.default_fov = float(default_fov)

        # Round-robin the provided GPU ids across slots; None -> no pinning.
        if slot_gpu_ids:
            self.slot_gpu_ids: List[Optional[int]] = [
                slot_gpu_ids[i % len(slot_gpu_ids)] for i in range(self.max_slots)
            ]
        else:
            self.slot_gpu_ids = [None] * self.max_slots

        self.slots: List[Optional[_ControllerSlot]] = [None] * self.max_slots
        self.scene_to_slot: Dict[str, int] = {}
        self._sched_lock = threading.Lock()

        # Fixed-size thread pool prevents "can't start new thread" runaway.
        self._executor = ThreadPoolExecutor(max_workers=self.max_threads)

        self._metrics: Counter[str] = Counter()

    def shutdown(self) -> None:
        """Stop controllers and shutdown threadpool."""
        with self._sched_lock:
            for i, slot in enumerate(self.slots):
                if slot is None:
                    continue
                with contextlib.suppress(Exception):
                    slot.controller.stop()
                self.slots[i] = None
            self.scene_to_slot.clear()

        with contextlib.suppress(Exception):
            self._executor.shutdown(wait=True, cancel_futures=True)

    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    def _pick_slot_locked(self, scene_id: str) -> int:
        """Sticky -> empty (not yet reserved) -> LRU.

        Concurrency note: the _sched_lock only covers this function; the actual
        slot Controller is created later in _ensure_slot_ready_for_request
        (outside this lock). If we only check `slot is None`, a burst of distinct
        first-time scenes all observe all-None and all claim slot 0. Guard by
        also excluding slot indices that already appear as the target in
        scene_to_slot.
        """
        mapped = self.scene_to_slot.get(scene_id)
        if mapped is not None:
            return mapped

        reserved_idx = set(self.scene_to_slot.values())
        for idx, slot in enumerate(self.slots):
            if slot is None and idx not in reserved_idx:
                self.scene_to_slot[scene_id] = idx
                return idx

        idx = min(range(len(self.slots)), key=lambda i: self.slots[i].last_used if self.slots[i] else 0.0)
        prev_slot = self.slots[idx]
        if prev_slot is not None:
            self.scene_to_slot.pop(prev_slot.scene_id, None)
        self.scene_to_slot[scene_id] = idx
        return idx

    def _create_slot(self, slot_idx: int, scene_id: str, width: int, height: int, fov: float) -> _ControllerSlot:
        """Create controller slot with specified params."""
        self._metrics["controller_create"] += 1
        gpu = self.slot_gpu_ids[slot_idx]
        LOGGER.info(
            "[AI2ThorRender/Hybrid][P] CREATE slot=%d scene=%s size=%dx%d fov=%.1f gpu=%s",
            slot_idx, scene_id, int(width), int(height), float(fov), gpu
        )
        controller = _create_controller(self.controller_config, scene_id, width, height, fov, gpu_device=gpu)
        return _ControllerSlot(
            controller=controller,
            scene_id=scene_id,
            width=int(width),
            height=int(height),
            fov=float(fov),
            last_used=time.monotonic(),
            lock=threading.Lock(),
        )

    def _reset_slot_scene(self, slot: _ControllerSlot, slot_idx: int, scene_id: str) -> None:
        """Reset controller scene (keeps params)."""
        self._metrics["controller_reset_scene"] += 1
        LOGGER.info(
            "[AI2ThorRender/Hybrid][P] RESET slot=%d old_scene=%s -> new_scene=%s (size=%dx%d fov=%.1f)",
            slot_idx, slot.scene_id, scene_id, slot.width, slot.height, slot.fov
        )
        _reset_controller_to_scene(slot.controller, scene_id)
        _add_third_party_camera(slot.controller, slot.fov)
        slot.scene_id = scene_id

    def _reconstruct_slot(self, slot: _ControllerSlot, slot_idx: int, scene_id: str, width: int, height: int, fov: float) -> None:
        """Recreate controller due to param change (expensive, should be rare)."""
        self._metrics["controller_reconstruct"] += 1
        gpu = self.slot_gpu_ids[slot_idx]
        LOGGER.warning(
            "[AI2ThorRender/Hybrid][P] RECONSTRUCT slot=%d scene=%s "
            "old=%dx%d fov=%.1f new=%dx%d fov=%.1f gpu=%s",
            slot_idx, scene_id,
            slot.width, slot.height, slot.fov,
            int(width), int(height), float(fov), gpu,
        )
        with contextlib.suppress(Exception):
            slot.controller.stop()
        slot.controller = _create_controller(self.controller_config, scene_id, width, height, fov, gpu_device=gpu)
        slot.scene_id = scene_id
        slot.width = int(width)
        slot.height = int(height)
        slot.fov = float(fov)

    def _ensure_slot_ready_for_request(self, slot_idx: int, scene_id: str) -> _ControllerSlot:
        """
        Ensure slot exists and has correct scene for the request-level scene_id.
        Uses DEFAULT params at request start (tasks may override later).
        """
        slot = self.slots[slot_idx]
        if slot is None:
            slot = self._create_slot(slot_idx, scene_id, self.default_width, self.default_height, self.default_fov)
            self.slots[slot_idx] = slot
            return slot

        # If scene differs, do a reset (fast path)
        if slot.scene_id != scene_id:
            with slot.lock:
                self._reset_slot_scene(slot, slot_idx, scene_id)

        return slot

    def _render_blocking(self, slot_idx: int, scene_id: str, tasks: List[Dict[str, Any]]) -> List[bytes]:
        """
        Blocking render function executed in this process' thread pool.

        Thread safety:
        - We lock the slot for the whole request, so one controller is used by one thread at a time.
        """
        slot = self._ensure_slot_ready_for_request(slot_idx, scene_id)

        with slot.lock:
            slot.last_used = time.monotonic()

            out: List[bytes] = []
            for i, task in enumerate(tasks):
                pose = task.get("pose")
                if not pose or "position" not in pose or "rotation" not in pose:
                    LOGGER.warning("[AI2ThorRender/Hybrid][P] Invalid pose for task #%d; transparent image", i)
                    out.append(_transparent_png(self.default_width, self.default_height))
                    continue

                # Per-task overrides (kept as requested)
                w, h, f = _task_params(task, self.default_width, self.default_height, self.default_fov)

                # If per-task params differ from current slot params, RECONSTRUCT (rare)
                if slot.width != w or slot.height != h or float(slot.fov) != float(f):
                    self._reconstruct_slot(slot, slot_idx, scene_id, w, h, f)

                # Step and encode
                slot.controller.step(
                    action="UpdateThirdPartyCamera",
                    thirdPartyCameraId=0,
                    position=pose["position"],
                    rotation=pose["rotation"],
                    fieldOfView=float(slot.fov),
                )

                ev = slot.controller.last_event
                if not ev.third_party_camera_frames:
                    raise RuntimeError("No third party camera frames available")

                frame = ev.third_party_camera_frames[0]
                out.append(_encode_rgb_to_png_bytes(frame))

            return out

    async def render(self, scene_id: str, tasks: List[Dict[str, Any]]) -> List[bytes]:
        """Async entry: schedule blocking render in thread pool."""
        with self._sched_lock:
            slot_idx = self._pick_slot_locked(scene_id)
            self._metrics["submit"] += 1

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._render_blocking, slot_idx, scene_id, tasks)


# =============================================================================
# Worker process entrypoint (one ThreadControllerPool per process)
# =============================================================================
_WORKER_POOL: Optional[_ThreadControllerPool] = None

def _worker_init(
    controller_config: Dict[str, Any],
    max_slots: int,
    max_threads: int,
    default_width: int,
    default_height: int,
    default_fov: float,
    slot_gpu_ids: Optional[List[Optional[int]]] = None,
) -> None:
    """Initializer for each worker process."""
    global _WORKER_POOL
    _WORKER_POOL = _ThreadControllerPool(
        max_slots=max_slots,
        max_threads=max_threads,
        controller_config=controller_config,
        default_width=default_width,
        default_height=default_height,
        default_fov=default_fov,
        slot_gpu_ids=slot_gpu_ids,
    )
    LOGGER.info(
        "[AI2ThorRender/Hybrid][P] Worker init max_slots=%d max_threads=%d default=%dx%d fov=%.1f gpu_ids=%s",
        int(max_slots), int(max_threads), int(default_width), int(default_height),
        float(default_fov), slot_gpu_ids,
    )

def _worker_render(scene_id: str, tasks: List[Dict[str, Any]]) -> List[bytes]:
    """Called by main process via ProcessPoolExecutor."""
    if _WORKER_POOL is None:
        raise RuntimeError("Worker pool is not initialized")
    # Run the pool's async render inside this process
    return asyncio.run(_WORKER_POOL.render(scene_id, tasks))


# =============================================================================
# Main-process scheduler across processes (sticky + LRU by scene)
# =============================================================================
@dataclass
class _ProcSlot:
    executor: ProcessPoolExecutor
    last_used: float = 0.0
    current_scene: Optional[str] = None


class _HybridPool:
    """Main-process scheduler that dispatches to one of N worker processes."""
    def __init__(
        self,
        *,
        max_process: int,
        max_slots: int,
        max_threads: int,
        controller_config: Dict[str, Any],
        default_width: int,
        default_height: int,
        default_fov: float,
        gpu_ids: Optional[List[int]] = None,
    ):
        self.max_process = max(1, int(max_process))

        self._ctx = mp.get_context("spawn")
        self.proc_slots: List[_ProcSlot] = []

        # Distribute one GPU per process (round-robin). All slots inside a
        # given process will share that process's GPU for memory efficiency.
        if gpu_ids:
            proc_gpu_ids: List[Optional[int]] = [
                int(gpu_ids[i % len(gpu_ids)]) for i in range(self.max_process)
            ]
        else:
            proc_gpu_ids = [None] * self.max_process

        for i in range(self.max_process):
            proc_gpu = proc_gpu_ids[i]
            slot_gpu_ids = [proc_gpu] * int(max_slots)
            ex = ProcessPoolExecutor(
                max_workers=1,
                mp_context=self._ctx,
                initializer=_worker_init,
                initargs=(
                    controller_config,
                    int(max_slots), int(max_threads),
                    int(default_width), int(default_height), float(default_fov),
                    slot_gpu_ids,
                ),
            )
            self.proc_slots.append(_ProcSlot(executor=ex))
        LOGGER.info(
            "[AI2ThorRender/Hybrid] spawned %d procs with gpu assignment: %s",
            self.max_process, proc_gpu_ids,
        )

        self.scene_to_proc: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._metrics: Counter[str] = Counter()

    async def aclose(self) -> None:
        for ps in self.proc_slots:
            with contextlib.suppress(Exception):
                ps.executor.shutdown(wait=True, cancel_futures=True)
        self.proc_slots.clear()
        self.scene_to_proc.clear()

    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    def _assign_proc_locked(self, scene_id: str) -> int:
        # Sticky -> first-empty -> LRU. We immediately claim a proc by setting
        # current_scene here (under the asyncio lock from render()), so a burst
        # of concurrent distinct scenes cannot all collide on proc 0.
        mapped = self.scene_to_proc.get(scene_id)
        if mapped is not None:
            return mapped

        for idx, ps in enumerate(self.proc_slots):
            if ps.current_scene is None:
                ps.current_scene = scene_id
                self.scene_to_proc[scene_id] = idx
                return idx

        idx = min(range(len(self.proc_slots)), key=lambda i: self.proc_slots[i].last_used)
        prev_scene = self.proc_slots[idx].current_scene
        if prev_scene:
            self.scene_to_proc.pop(prev_scene, None)
        self.proc_slots[idx].current_scene = scene_id
        self.scene_to_proc[scene_id] = idx
        return idx

    async def render(self, scene_id: str, tasks: List[Dict[str, Any]]) -> List[bytes]:
        async with self._lock:
            proc_idx = self._assign_proc_locked(scene_id)
            ps = self.proc_slots[proc_idx]
            ps.last_used = time.monotonic()
            executor = ps.executor
            self._metrics["submit"] += 1

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _worker_render, scene_id, tasks)


# =============================================================================
# Public handler
# =============================================================================
class AI2ThorRenderHandler(BaseHandler):
    """
    Hybrid handler with split knobs:
    - max_process: number of processes (default 1 => single-process mode)
    - max_threads: threads per process
    - max_slots: controllers per process

    Compatibility:
    - Accepts **kwargs so extra server-provided keys won't crash init.
    - Any missing config stays at default values.
    """

    def __init__(
        self,
        max_workers: int = 16,  # backward compatible default for max_threads if not set
        log_level: int = logging.INFO,
        gpu_ids: Sequence[int] | str | None = None,
        platform: str = "CloudRendering",
        agentMode: str = "default",
        width: int = 512,
        height: int = 512,
        fieldOfView: float = 90.0,
        *,
        max_process: int = 1,
        max_threads: Optional[int] = 24,
        max_slots: Optional[int] = None,
        **kwargs: Any,
    ):
        # kwargs is intentionally ignored to keep server integration robust.
        LOGGER.setLevel(log_level)

        self.default_width = int(width)
        self.default_height = int(height)
        self.default_fov = float(fieldOfView)

        controller_config = {"platform": platform, "agentMode": agentMode}

        mt = int(max_threads if max_threads is not None else max_workers)
        ms = int(max_slots if max_slots is not None else mt)
        mpn = int(max_process)

        norm_gpu_ids = _normalize_gpu_ids(gpu_ids)  # -> List[int] or [] if none

        self._metrics: Counter[str] = Counter()

        self._single_process_mode = (mpn == 1)

        if self._single_process_mode:
            # Single-process: each slot gets a GPU (round-robin). This lets one
            # Python process host multiple AI2-THOR Controllers on separate GPUs
            # because each Controller is a separate Unity subprocess.
            self.pool_single = _ThreadControllerPool(
                max_slots=ms,
                max_threads=mt,
                controller_config=controller_config,
                default_width=self.default_width,
                default_height=self.default_height,
                default_fov=self.default_fov,
                slot_gpu_ids=norm_gpu_ids or None,
            )
            self.pool_hybrid = None
        else:
            # Hybrid: each process gets a GPU (round-robin); all slots in that
            # process share the process's GPU.
            self.pool_single = None
            self.pool_hybrid = _HybridPool(
                max_process=mpn,
                max_slots=ms,
                max_threads=mt,
                controller_config=controller_config,
                default_width=self.default_width,
                default_height=self.default_height,
                default_fov=self.default_fov,
                gpu_ids=norm_gpu_ids or None,
            )

        LOGGER.info(
            "[AI2ThorRender] init mode=%s max_process=%d max_threads=%d max_slots=%d "
            "default=%dx%d fov=%.1f gpu_ids=%s",
            "single" if self._single_process_mode else "hybrid",
            mpn, mt, ms,
            self.default_width, self.default_height, self.default_fov,
            norm_gpu_ids,
        )

    async def handle(self, meta: Dict[str, Any], images: List[Image.Image]) -> HandlerResult:
        scene_id: str = meta.get("scene_id") or ""
        tasks: List[Dict[str, Any]] = meta.get("tasks") or []

        if not scene_id:
            LOGGER.error("[AI2ThorRender] Missing scene_id in request")
            return HandlerResult(meta={"error": "Missing scene_id"}, images=[])

        if not isinstance(tasks, list):
            LOGGER.error("[AI2ThorRender] Invalid tasks format")
            return HandlerResult(meta={"error": "Invalid tasks format"}, images=[])

        LOGGER.info("[AI2ThorRender] scene_id=%s tasks=%d", scene_id, len(tasks))

        try:
            if self._single_process_mode:
                rendered_images = await self.pool_single.render(scene_id, tasks)
            else:
                rendered_images = await self.pool_hybrid.render(scene_id, tasks)

            self._metrics["success"] += 1
            self._metrics["images_returned"] += len(rendered_images)

            return HandlerResult(
                meta={"scene_id": scene_id, "count": len(rendered_images)},
                encoded_images=rendered_images,
                image_format="PNG",
                image_mime="image/png",
            )
        except Exception as exc:
            tb = traceback.format_exc()
            LOGGER.error("[AI2ThorRender] Internal error: %s\n%s", exc, tb)
            self._metrics["internal_error"] += 1
            return HandlerResult(meta={"error": "Internal rendering error"}, images=[])

    async def aclose(self) -> None:
        if self._single_process_mode and self.pool_single is not None:
            self.pool_single.shutdown()
        if (not self._single_process_mode) and (self.pool_hybrid is not None):
            await self.pool_hybrid.aclose()

    async def warm_cache(self, scene_ids: Sequence[str]) -> None:
        # Best-effort warm: issue empty task list calls (will create controller + load scene)
        for s in scene_ids:
            if not s:
                continue
            try:
                await self.handle({"scene_id": s, "tasks": []}, [])
            except Exception:
                pass

    def metrics_snapshot(self) -> Dict[str, int]:
        merged = dict(self._metrics)
        try:
            if self._single_process_mode and self.pool_single is not None:
                merged.update({f"pool_{k}": v for k, v in self.pool_single.metrics().items()})
            if (not self._single_process_mode) and (self.pool_hybrid is not None):
                merged.update({f"pool_{k}": v for k, v in self.pool_hybrid.metrics().items()})
        except Exception:
            pass
        return merged
