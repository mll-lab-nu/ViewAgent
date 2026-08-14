"""Client-side Habitat-GS renderer — the object the gym envs hold.

Same role as ``view_suite.ai2thor.ai2thor_unified_renderer.AI2ThorUnifiedRender``: a thin
async client that speaks the shared multipart render protocol and exposes the interface
every renderer in this repo exposes, so ``GymProxyTool`` and friends do not care which
world they are looking at.

Client-only on purpose. A resident Habitat-GS scene is a loaded gaussian cloud (the
corpus averages ~350 MB per ``.gs.ply``), far too heavy to instantiate per env replica;
the service keeps one scene per worker and routes by scene id.

Unlike the AI2-THOR client this sends OpenCV (K, c2w) straight through — the service's
``habitat_gs`` backend takes exactly that, so there is no Unity pose conversion in the
middle and nothing to get wrong.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from view_suite.service_http.async_client_routed import HRWRoutedAsyncUnifiedClient

DEFAULT_CLIENT_OPEN_TIMEOUT = float(os.getenv("HABITAT_GS_RENDER_CLIENT_OPEN_TIMEOUT", "60"))
DEFAULT_CLIENT_MAX_INFLIGHT = int(os.getenv("HABITAT_GS_RENDER_CLIENT_MAX_INFLIGHT", "64"))


@dataclass
class HabitatGSRenderConfig:
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


class HabitatGSUnifiedRender:
    """Interface mirrors scannet.UnifiedRender:
      - set_scene(scene_id)
      - render_image_from_cam_param(K, c2w, width, height) -> PIL.Image
      - render_tasks(tasks) -> [PIL.Image]
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
        self.cfg = HabitatGSRenderConfig(
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

    async def _ensure_client(self) -> HRWRoutedAsyncUnifiedClient:
        if self._client is None:
            if not self.cfg.client_url:
                raise ValueError(
                    "HabitatGSUnifiedRender requires client_url "
                    "(e.g. https://<host>:8812); see client_url_habitat_gs.txt"
                )
            self._client = HRWRoutedAsyncUnifiedClient(base_url=self.cfg.client_url)
        return self._client

    @staticmethod
    def _to_pil(img: Image.Image | np.ndarray) -> Image.Image:
        return img if isinstance(img, Image.Image) else Image.fromarray(img)

    # ------------- public APIs -------------
    @staticmethod
    def build_task(K, c2w, width: int, height: int) -> Dict[str, Any]:
        """The `cam_param` task shape the ScanNet-family service expects."""
        return {
            "mode": "cam_param",
            "intrinsics": _to_jsonable(np.asarray(K, dtype=np.float64)),
            "extrinsics": _to_jsonable(np.asarray(c2w, dtype=np.float64)),
            "size": [int(width), int(height)],
        }

    async def render_image_from_cam_param(
        self,
        camera_intrinsics,
        camera_extrinsics,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        task = self.build_task(camera_intrinsics, camera_extrinsics, width, height)
        client = await self._ensure_client()
        meta = {"scene_id": self.cfg.scene_id, "tasks": [task]}
        _response_meta, imgs = await client.render(meta)
        if not imgs:
            # The service answers internal errors with HTTP 200 and an empty image
            # list, so an empty response here is a real failure, not an empty batch.
            raise RuntimeError(
                f"render returned no images for scene_id={self.cfg.scene_id!r}"
            )
        return self._to_pil(imgs[0])

    async def render_tasks(self, tasks: List[Dict[str, Any]]) -> List[Image.Image]:
        client = await self._ensure_client()
        meta = {"scene_id": self.cfg.scene_id, "tasks": [_to_jsonable(t) for t in tasks]}
        _, imgs = await client.render(meta)
        if len(imgs) != len(tasks):
            raise RuntimeError(
                f"render returned {len(imgs)} images for {len(tasks)} tasks "
                f"(scene_id={self.cfg.scene_id!r})"
            )
        return [self._to_pil(i) for i in imgs]
