"""Habitat-GS render env (analog of ai2thor/gym_ai2thor_render_env.py).

Wraps HabitatGSUnifiedRender in ViewBaseEnv's abstract surface so the tool env above
it calls render_image_from_cam_param(K, c2w) without caring which world it is in.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from view_suite.envs.base.view_base_env import ViewBaseEnv
from view_suite.habitat_gs.habitat_gs_unified_renderer import (
    DEFAULT_CLIENT_MAX_INFLIGHT,
    DEFAULT_CLIENT_OPEN_TIMEOUT,
    HabitatGSUnifiedRender,
)


class GymHabitatGSRenderEnv(ViewBaseEnv):
    """Render env backed by the Habitat-GS HTTP render service.

    Expected env_config keys:
      - client_url            : url of the render service (';'-separate several)
      - client_url_file_path  : (alt) file whose first line holds that url
      - client_origin         : optional origin header (rare)
      - scene_id              : current scene id (may be set later)
      - client_open_timeout   : float
      - client_max_inflight   : int
      - render_size           : (w, h), default (512, 512)
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        client_url = env_config.get("client_url", None)
        client_url_file_path = env_config.get("client_url_file_path", None)
        if client_url is None and client_url_file_path is not None:
            with open(client_url_file_path, "r") as f:
                # One line, ';'-separated for several servers -- reading the whole file
                # keeps a trailing newline inside the last URL and makes it invalid.
                client_url = f.read().strip()
        self.render_size = env_config.get("render_size", (512, 512))
        self.renderer = HabitatGSUnifiedRender(
            client_url=client_url,
            client_origin=env_config.get("client_origin", None),
            scene_id=env_config.get("scene_id", None),
            client_open_timeout=env_config.get(
                "client_open_timeout", DEFAULT_CLIENT_OPEN_TIMEOUT),
            client_max_inflight=env_config.get(
                "client_max_inflight", DEFAULT_CLIENT_MAX_INFLIGHT),
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
