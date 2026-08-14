"""HabitatGSInteractiveViewPlanning: multi-turn IVP on Habitat-GS 3DGS scenes.

IVP ("active exploration"): the agent flies the camera around a live gaussian-splatting
scene (rendered on the fly by the Habitat-GS HTTP render service) to localize an unseen
target view, then submits a 6-DoF camera pose estimate. Multi-turn; the render service
URL is read from ``client_url_habitat_gs.txt``.

The real logic lives in the Habitat-GS ``GymProxyTool`` engine, which mirrors the
ScanNet IVP engine but drives a Habitat camera: ground-parallel turning, horizontal
translation, no roll.
"""
from __future__ import annotations

from view_suite.envs.habitat_gs_proxy_task.gym_proxy_tool import GymProxyTool


class HabitatGSInteractiveViewPlanning(GymProxyTool):
    """Identical behavior to the Habitat-GS GymProxyTool; distinct class for routing."""
    pass
