"""HabitatGSView2Path: View-to-Path (V2P) MCQ env on Habitat-GS-generated data.

V2P ("inverse dynamics"): given the initial view, a top-down reference and the target
view, pick the action sequence (among four text options) that reaches the target.
Single-turn, needs NO render service -- the images are pre-rendered in the JSONL.

Thin subclass of ``View2Path`` (-> ``GymProxyNoTool``), registered under its own name
so the eval/train config `name:` selector can route to the Habitat-GS dataset.
"""
from __future__ import annotations

from view_suite.envs.scannet_proxy_task.view_to_path import View2Path


class HabitatGSView2Path(View2Path):
    """Identical behavior to View2Path; distinct class for registry routing."""
    pass
