"""Ai2ThorPath2View: Path-to-View (P2V) MCQ env on AI2-THOR-generated data.

P2V ("forward dynamics"): given the initial view + an action sequence, pick the
resulting view among four image options. Single-turn, needs NO render service --
the images are pre-rendered in the JSONL by
``view_suite.envs.ai2thor_proxy_task.data_gen.generate_data``.

The base class ``Path2View`` (-> ``GymProxyNoTool``) is scene-agnostic: it only
reads the JSONL and its referenced images. This is a thin alias so the task can
be registered under its own name and routed to the AI2-THOR dataset path while
leaving the ScanNet configs untouched.
"""
from __future__ import annotations

from view_suite.envs.scannet_proxy_task.path_to_view import Path2View


class Ai2ThorPath2View(Path2View):
    """Identical behavior to Path2View; distinct class for registry routing."""
    pass
