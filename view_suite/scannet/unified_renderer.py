# unified_render.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from PIL import Image
import numpy as np

from view_suite.scannet.render.mesh_render import MeshRenderer
from view_suite.scannet.utils.path_utils import resolve_scene_ply
from view_suite.service_http.async_client_routed import HRWRoutedAsyncUnifiedClient

DEFAULT_CLIENT_OPEN_TIMEOUT = float(os.getenv("SCANNET_RENDER_CLIENT_OPEN_TIMEOUT", "60"))
DEFAULT_CLIENT_MAX_INFLIGHT = int(os.getenv("SCANNET_RENDER_CLIENT_MAX_INFLIGHT", "64"))

@dataclass
class RenderConfig:
    render_backend: str                 # "local" | "client" | "habitat"
    scannet_root: Optional[str] = None  # required for local; also used to resolve ply for client
    client_url: Optional[str] = None    # required for client
    client_origin: Optional[str] = None
    scene_id: Optional[str] = None      # current scene
    client_open_timeout: Optional[float] = None
    client_max_inflight: Optional[int] = None
    # habitat backend: renders in-process on an EXPLICIT gpu, unlike open3d/EGL
    # which always lands on device 0 regardless of CUDA_VISIBLE_DEVICES.
    gpu_device_id: int = 0
    habitat_light_intensity: float = 2.5   # matches Open3D mean exposure

def _to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x

def _ensure_K3x3(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K)
    if K.shape == (4, 4):
        return K[:3, :3]
    if K.shape == (3, 3):
        return K
    raise ValueError(f"Intrinsics must be 3x3 or 4x4; got {K.shape}")

class UnifiedRender:
    """
   unified renderer with lazy init and consistent return (PIL.Image).
    - local: MeshRenderer
    - client: HRWRoutedAsyncUnifiedClient (HTTP)
    """
    def __init__(self, render_backend: str, scannet_root: str | None,
                 client_url: str | None, client_origin: str | None, scene_id: str | None,
                 client_open_timeout: float | None = DEFAULT_CLIENT_OPEN_TIMEOUT,
                 client_max_inflight: int | None = DEFAULT_CLIENT_MAX_INFLIGHT):
        self.cfg = RenderConfig(
            render_backend,
            scannet_root,
            client_url,
            client_origin,
            scene_id,
            client_open_timeout,
            client_max_inflight,
        )
        self._mesh: Optional[MeshRenderer] = None
        self._client: Optional[HRWRoutedAsyncUnifiedClient] = None
        self._habitat = None
        self._ply: Optional[str] = None

    # ------------- lifecycle -------------
    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    def set_scene(self, scene_id: str) -> None:
        """Allow switching scenes on the fly."""
        if scene_id != self.cfg.scene_id:
            self.cfg.scene_id = scene_id
            self._ply = None
            self._mesh = None  # drop local cache to avoid mixing scenes

    # ------------- ensures -------------
    def _ensure_ply(self) -> str:
        if self._ply is None:
            if self.cfg.scannet_root is None and self.cfg.scene_id is not None:
                raise ValueError("scannet_root and scene_id are required")
            if self.cfg.scannet_root is None:
                raise ValueError("scannet_root is required")
            if self.cfg.scene_id is None:
                raise ValueError("scene_id is required")
            self._ply = resolve_scene_ply(self.cfg.scannet_root, self.cfg.scene_id)
        return self._ply

    def _ensure_local(self) -> MeshRenderer:
        if self._mesh is None:
            self._mesh = MeshRenderer(self._ensure_ply())
        return self._mesh

    def _ensure_habitat(self):
        """Habitat-Sim renderer pinned to cfg.gpu_device_id (real multi-GPU isolation)."""
        if self._habitat is None:
            from view_suite.scannet.render.habitat_render import HabitatRenderer
            self._habitat = HabitatRenderer(
                self._ensure_ply(),
                gpu_device_id=int(self.cfg.gpu_device_id or 0),
                light_intensity=float(self.cfg.habitat_light_intensity),
            )
        return self._habitat

    async def _ensure_client(self) -> HRWRoutedAsyncUnifiedClient:
        if self._client is None:
            assert self.cfg.client_url, "client_url required"
            self._client = HRWRoutedAsyncUnifiedClient(
                base_url=self.cfg.client_url,
            )
        return self._client

    @staticmethod
    def _to_pil(img: Image.Image | np.ndarray) -> Image.Image:
        return img if isinstance(img, Image.Image) else Image.fromarray(img)

    # ------------- public APIs (match your names) -------------
    async def render_image_from_cam_param(self, camera_intrinsics, camera_extrinsics, width=300, height=300) -> Image.Image:
        if self.cfg.render_backend == "local":
            img = self._ensure_local().render_image_from_cam_param(camera_intrinsics, camera_extrinsics, width, height)
            return self._to_pil(img)
        elif self.cfg.render_backend == "habitat":
            img = self._ensure_habitat().render_image_from_cam_param(camera_intrinsics, camera_extrinsics, width, height)
            return self._to_pil(img)
        elif self.cfg.render_backend == "client":
            K = _ensure_K3x3(np.asarray(camera_intrinsics, dtype=np.float32))
            E = np.asarray(camera_extrinsics, dtype=np.float32)

            tasks = [{"mode": "cam_param",
                    "intrinsics": _to_jsonable(K),
                    "extrinsics": _to_jsonable(E),
                    "size": [int(width), int(height)]}]
            client = await self._ensure_client()
            meta = {
                "scene_id": self.cfg.scene_id,
                "tasks": tasks,
            }
            response_meta, imgs = await client.render(meta)  # returns List[PIL.Image]
            return imgs[0]
        else:
            raise ValueError(f"unknown backend: {self.cfg.render_backend}")


    # (optional) support direct forwarding a list of tasks
    async def render_tasks(self, tasks: List[Dict[str, Any]]) -> List[Image.Image]:
        if self.cfg.render_backend == "local":
            out: List[Image.Image] = []
            r = self._ensure_local()
            for t in tasks:
                w, h = t.get("size", [300, 300])
                out.append(self._to_pil(r.render_image_from_cam_param(t["intrinsics"], t["extrinsics"], w, h)))
            return out
        else:
            client = await self._ensure_client()
            meta = {
                "scene_id": self.cfg.scene_id,
                "tasks": tasks,
            }
            _, imgs = await client.render(meta)
            return imgs
