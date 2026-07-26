"""AI2-THOR render env (analog of scannet/gym_scannet_render_env.py).

Wraps AI2ThorUnifiedRender with ViewBaseEnv's required abstract methods so
downstream tool envs (GymAi2thorToolEnv) can call render_image_from_cam_param
with a consistent (K, c2w) interface regardless of the render backend.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from view_suite.envs.base.view_base_env import ViewBaseEnv
from view_suite.ai2thor.ai2thor_unified_renderer import (
    AI2ThorUnifiedRender,
    DEFAULT_CLIENT_MAX_INFLIGHT,
    DEFAULT_CLIENT_OPEN_TIMEOUT,
)


class GymAi2thorRenderEnv(ViewBaseEnv):
    """Render env using the AI2-THOR HTTP service as backend.

    Expected env_config keys:
      - client_url                : http url of the AI2-THOR render service
      - client_url_file_path      : (alt) file containing the url on first line
      - client_origin             : optional origin header (rare)
      - scene_id                  : current scene id (optional, can be set later)
      - client_open_timeout       : float
      - client_max_inflight       : int
      - render_size               : (w, h), default (512, 512)
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        client_url = env_config.get("client_url", None)
        client_url_file_path = env_config.get("client_url_file_path", None)
        if client_url is None and client_url_file_path is not None:
            with open(client_url_file_path, "r") as f:
                client_url = f.read().strip()
        client_origin = env_config.get("client_origin", None)
        scene_id = env_config.get("scene_id", None)
        client_open_timeout = env_config.get(
            "client_open_timeout", DEFAULT_CLIENT_OPEN_TIMEOUT
        )
        self.render_size = env_config.get("render_size", (512, 512))
        client_max_inflight = env_config.get(
            "client_max_inflight", DEFAULT_CLIENT_MAX_INFLIGHT
        )
        self.renderer = AI2ThorUnifiedRender(
            client_url=client_url,
            client_origin=client_origin,
            scene_id=scene_id,
            client_open_timeout=client_open_timeout,
            client_max_inflight=client_max_inflight,
        )

    async def render_image_from_cam_param(
        self, camera_intrinsics, camera_extrinsics, width=None, height=None
    ):
        if width is None:
            width = self.render_size[0]
        if height is None:
            height = self.render_size[1]
        return await self.renderer.render_image_from_cam_param(
            camera_intrinsics, camera_extrinsics, width, height
        )

    async def render_tasks(self, tasks: List[Dict[str, Any]]):
        return await self.renderer.render_tasks(tasks)

    @abstractmethod
    async def close(self) -> None:
        ...

    @abstractmethod
    async def system_prompt(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...

    @abstractmethod
    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        ...
