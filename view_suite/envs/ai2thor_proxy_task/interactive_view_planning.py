"""Ai2ThorInteractiveViewPlanning: multi-turn Interactive View Planning (IVP) on AI2-THOR.

IVP ("active exploration"): the agent flies the camera around a live AI2-THOR
scene (rendered on the fly by the AI2-THOR HTTP render service) to localize an
unseen target view, then submits a 6-DoF camera pose estimate. Multi-turn; the
render service URL is read from ``client_url_ai2thor.txt``.

The real logic lives in the AI2-THOR ``GymProxyTool`` engine
(``view_suite.envs.ai2thor_proxy_task.gym_proxy_tool``), which mirrors the
ScanNet IVP engine but talks to the AI2-THOR renderer. This is a thin alias so
the task can be registered under its own name.
"""
from __future__ import annotations

from view_suite.envs.ai2thor_proxy_task.gym_proxy_tool import GymProxyTool


class Ai2ThorInteractiveViewPlanning(GymProxyTool):
    """Identical behavior to the AI2-THOR GymProxyTool; distinct class for registry routing."""
    pass
