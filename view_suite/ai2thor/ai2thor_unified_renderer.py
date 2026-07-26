"""AI2-THOR unified renderer — client-only analog of scannet's UnifiedRender.

Accepts the same (K, c2w, w, h) interface as scannet's renderer so
GymProxyTool and friends can be reused unchanged. Internally converts
OpenCV K+c2w -> AI2-THOR Unity pose + fov and POSTs to the AI2-THOR HTTP
render service (`view_suite.ai2thor.service_http`).

Only the "client" backend is implemented here; there's no need for a local
mode because AI2-THOR requires a live Controller per scene which is too heavy
to spin up per-env-instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from view_suite.ai2thor.pose_utils import build_render_task
from view_suite.service_http.async_client_routed import HRWRoutedAsyncUnifiedClient


DEFAULT_CLIENT_OPEN_TIMEOUT = float(os.getenv("AI2THOR_RENDER_CLIENT_OPEN_TIMEOUT", "60"))
DEFAULT_CLIENT_MAX_INFLIGHT = int(os.getenv("AI2THOR_RENDER_CLIENT_MAX_INFLIGHT", "64"))


@dataclass
class Ai2thorRenderConfig:
    client_url: Optional[str] = None
    client_origin: Optional[str] = None
    scene_id: Optional[str] = None
    client_open_timeout: Optional[float] = DEFAULT_CLIENT_OPEN_TIMEOUT
    client_max_inflight: Optional[int] = DEFAULT_CLIENT_MAX_INFLIGHT


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


class AI2ThorUnifiedRender:
    """Client-only renderer for AI2-THOR.

    Public interface mirrors scannet.UnifiedRender:
      - set_scene(scene_id)
      - render_image_from_cam_param(K, c2w, width, height) -> PIL.Image
      - render_tasks(tasks)  (raw passthrough; tasks already in AI2-THOR format)
      - close()
    """

    def __init__(
        self,
        client_url: Optional[str] = None,
        client_origin: Optional[str] = None,
        scene_id: Optional[str] = None,
        client_open_timeout: Optional[float] = DEFAULT_CLIENT_OPEN_TIMEOUT,
        client_max_inflight: Optional[int] = DEFAULT_CLIENT_MAX_INFLIGHT,
    ):
        self.cfg = Ai2thorRenderConfig(
            client_url=client_url,
            client_origin=client_origin,
            scene_id=scene_id,
            client_open_timeout=client_open_timeout,
            client_max_inflight=client_max_inflight,
        )
        self._client: Optional[HRWRoutedAsyncUnifiedClient] = None

    # ------------- lifecycle -------------
    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    def set_scene(self, scene_id: Optional[str]) -> None:
        if scene_id != self.cfg.scene_id:
            self.cfg.scene_id = scene_id

    # ------------- ensure -------------
    async def _ensure_client(self) -> HRWRoutedAsyncUnifiedClient:
        if self._client is None:
            if not self.cfg.client_url:
                raise ValueError(
                    "AI2ThorUnifiedRender requires client_url "
                    "(e.g. http://localhost:8765)"
                )
            self._client = HRWRoutedAsyncUnifiedClient(base_url=self.cfg.client_url)
        return self._client

    @staticmethod
    def _to_pil(img: Image.Image | np.ndarray) -> Image.Image:
        return img if isinstance(img, Image.Image) else Image.fromarray(img)

    # ------------- public APIs (match scannet interface) -------------
    async def render_image_from_cam_param(
        self,
        camera_intrinsics,
        camera_extrinsics,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        K = np.asarray(camera_intrinsics, dtype=np.float64)
        c2w = np.asarray(camera_extrinsics, dtype=np.float64)
        task = build_render_task(c2w, K, width=int(width), height=int(height))
        # Normalize numpy -> JSON-safe primitives for the HTTP layer.
        task = _to_jsonable(task)

        client = await self._ensure_client()
        meta = {
            "scene_id": self.cfg.scene_id,
            "tasks": [task],
        }
        _response_meta, imgs = await client.render(meta)
        return self._to_pil(imgs[0])

    async def render_tasks(self, tasks: List[Dict[str, Any]]) -> List[Image.Image]:
        """Forward raw tasks to the service.

        Tasks here are expected to already be in AI2-THOR format
        ({pose, width, height, fov}); use this for batch renders where the
        caller has already constructed tasks via build_render_task.
        """
        client = await self._ensure_client()
        meta = {
            "scene_id": self.cfg.scene_id,
            "tasks": [_to_jsonable(t) for t in tasks],
        }
        _, imgs = await client.render(meta)
        return [self._to_pil(i) for i in imgs]
