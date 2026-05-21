# -*- coding: utf-8 -*-
"""
Interactive view planning task (a.k.a. active exploration / camera navigation).

The default behavior is identical to ``GymProxyTool``: the agent issues
``answer(tx, ty, tz, rx, ry, rz)`` to submit a final pose estimate, which
terminates the episode and grades success.

A single env-config knob ``no_submit`` flips the variant:

  - ``no_submit: false`` (default): standard submit-required mode.
  - ``no_submit: true``: no answer/submit action. The current camera pose
    is graded against the target pose every step; the episode terminates
    with success as soon as both position and rotation errors are within
    the per-item thresholds. The system prompt is rewritten to remove the
    submit requirement and any ``answer(...)`` tokens in the action list
    are silently stripped.
"""
from __future__ import annotations

import re
from functools import cached_property
from typing import Any, Dict, Tuple

import numpy as np

from view_suite.envs.scannet_proxy_task.gym_proxy_tool import GymProxyTool
from view_suite.envs.scannet_proxy_task.utils.gym_proxy_tool_utils import (
    geodesic_angle_deg,
    resolve_thresholds,
)
from view_suite.envs.utils.parse_utils import get_format_instruction


_ANSWER_TOKEN_RE = re.compile(r"^\s*answer\s*(\(|$)", re.IGNORECASE)
_ACTION_BLOCK_RE = re.compile(r"<action>([\s\S]*?)</action>", re.IGNORECASE)


class InteractiveViewPlanning(GymProxyTool):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.no_submit: bool = bool(env_config.get("no_submit", False))

    @property
    def _action_only_allowed(self) -> tuple[str, ...]:
        if not self.no_submit:
            return super()._action_only_allowed
        return tuple(a for a in super()._action_only_allowed if a != "answer")

    async def system_prompt(self) -> Dict[str, Any]:
        if not self.no_submit:
            return await super().system_prompt()

        fmt_instr = get_format_instruction(
            self.format, action_example="action_1|action_2|action_3..."
        )
        preface = (
            "You are solving an active-exploration navigation task.\n\n"
            "GOAL\n"
            "Go to the TARGET VIEW as close as possible. "
            "Move and rotate the camera so that your current view matches the target view.\n"
            "The episode succeeds AUTOMATICALLY once your camera pose is within the "
            "position and rotation thresholds of the target. "
            "There is NO submit/answer action — just navigate.\n\n"
            "TURN LIMIT\n"
            f"You have at most {int(self.max_turns)} turns.\n\n"
            "OUTPUT FORMAT (STRICT)\n"
            f"{fmt_instr}\n\n"
            "FORMAT RULES\n"
            "- Do NOT output any text outside the expected tags.\n"
            "- Use '|' to separate multiple actions.\n"
            "- Actions must be chosen from the supported action list.\n"
        )
        obs_str = (preface + "\n\n" + (self._tool_instruction or "")).strip()
        return {"obs_str": obs_str}

    @staticmethod
    def _strip_answer_actions(action_str: str) -> str:
        m = _ACTION_BLOCK_RE.search(action_str)
        if not m:
            return action_str
        inner = m.group(1)
        kept = [t for t in inner.split("|") if not _ANSWER_TOKEN_RE.match(t.strip())]
        new_inner = "|".join(t.strip() for t in kept)
        return action_str[: m.start(1)] + new_inner + action_str[m.end(1) :]

    def _compute_pose_error(self) -> Dict[str, float] | None:
        if not self._has_active_camera or self.target_view is None or self.current_item is None:
            return None
        cur = self.view_engine.get_se3(degrees=True)
        tgt = np.asarray(self.target_view["c2w_se3_deg"], dtype=np.float64)
        pos_err = float(np.linalg.norm(cur[:3] - tgt[:3]))
        ang_err = float(geodesic_angle_deg(cur[3:], tgt[3:]))
        pos_thr, ang_thr = resolve_thresholds(
            self.current_item, self.tol_trans_l2_m, self.tol_rot_l2_deg
        )
        return {
            "pos_err_m": pos_err,
            "ang_err_deg": ang_err,
            "pos_threshold_m": float(pos_thr),
            "ang_threshold_deg": float(ang_thr),
        }

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self.no_submit:
            return await super().step(action_str)

        action_str = self._strip_answer_actions(action_str)
        obs, reward, done, info = await super().step(action_str)

        metrics = self._compute_pose_error()
        if metrics is None:
            return obs, reward, done, info

        success = (
            metrics["pos_err_m"] <= metrics["pos_threshold_m"] + 1e-9
            and metrics["ang_err_deg"] <= metrics["ang_threshold_deg"] + 1e-9
        )

        merged = dict(info) if isinstance(info, dict) else {}
        merged.update(self._info_answer(success, metrics))
        if isinstance(info, dict) and info.get("error"):
            merged["error"] = info["error"]
        info = merged

        if done:
            return obs, reward, done, info

        if success:
            self.episode_done = True
            reward = float(reward) + self.answer_reward
            if self.is_format_correct:
                reward += self.format_reward
            msg = (
                f"[auto-success] pos_err={metrics['pos_err_m']:.3f}m "
                f"(<= {metrics['pos_threshold_m']:.3f}m), "
                f"ang_err={metrics['ang_err_deg']:.3f}° "
                f"(<= {metrics['ang_threshold_deg']:.3f}°)"
            )
            new_obs_str = (msg + "\n" + obs.get("obs_str", "")).strip()
            obs = {**obs, "obs_str": new_obs_str}
            return obs, reward, True, info

        return obs, reward, done, info


if __name__ == "__main__":
    # Interactive demo -- fly the camera around an IVP scene yourself.
    #
    # IVP REQUIRES a running ScanNet render service (see the README, "Start the
    # ScanNet Render Service"); the env talks to it over HTTP. Run (from the
    # repo root, or with VIEWSUITE_ROOT exported):
    #     python view_suite/envs/scannet_proxy_task/interactive_view_planning.py
    #
    # Defaults to action_only_mode + no_submit + snap-every-step: the camera
    # starts on the initial view (so you move right away, no select_view), there
    # is no answer() action -- the episode auto-succeeds once the camera pose is
    # within threshold of the target -- and rotations snap to step multiples.
    #
    # Each turn, type movement keys (any combination, e.g. "wwd"); the newly
    # rendered view is saved under --save_dir. The service URL comes from
    # --client_url, else client_url.txt, else http://0.0.0.0:8767.
    import asyncio
    import os
    import sys

    import fire

    # keyboard key -> env action  (matches GymScannetToolEnv._keymap)
    _KEYMAP = {
        "w": "move_forward", "s": "move_backward",
        "d": "move_right",   "a": "move_left",
        "y": "move_up",      "h": "move_down",
        "q": "turn_left",    "e": "turn_right",
        "r": "look_up",      "f": "look_down",
        "t": "rotate_ccw",   "g": "rotate_cw",
    }

    def _default_client_url() -> str:
        root = os.environ.get("VIEWSUITE_ROOT", ".")
        for p in (os.path.join(root, "client_url.txt"), "client_url.txt"):
            if os.path.isfile(p):
                with open(p) as fh:
                    for line in fh:
                        if line.strip():
                            return line.strip()
        return "http://0.0.0.0:8767"

    def _keys_to_actions(s: str) -> str:
        """Translate a string of keyboard keys into a |-joined action list."""
        actions = []
        for ch in s.lower():
            if ch in " ,|":
                continue
            mapped = _KEYMAP.get(ch)
            if mapped is None:
                print(f"  (ignoring unknown key '{ch}')")
            else:
                actions.append(mapped)
        return "|".join(actions)

    async def _play(
        jsonl_path: str = "",
        client_url: str = "",
        save_dir: str = "",
        seed: int = 0,
        scannet_root: str = "",
        format: str = "free_think",
    ) -> None:
        root = os.environ.get("VIEWSUITE_ROOT", ".")
        jsonl_path = jsonl_path or os.path.join(
            root, "data/viewsuite_15k/interactive_view_planning_test.jsonl"
        )
        scannet_root = scannet_root or os.path.join(root, "data/scannet/scans")
        client_url = client_url or _default_client_url()
        save_dir = os.path.abspath(save_dir or os.path.join(root, "tests/ivp_play"))
        os.makedirs(save_dir, exist_ok=True)

        print(f"[IVP demo] jsonl={jsonl_path}")
        print(f"[IVP demo] client_url={client_url}  (ScanNet render service)")
        print("[IVP demo] mode: action_only + no_submit + snap")
        print("[IVP demo] keyboard controls -- type any combination, e.g. 'wwd':")
        print("    w/s  move forward / backward    q/e  turn left / right")
        print("    a/d  move left / right          r/f  look up / down")
        print("    y/h  move up / down             t/g  rotate ccw / cw")
        print(f"[IVP demo] rendered views are saved under: {save_dir}/")
        print("[IVP demo] no_submit mode: just navigate -- the episode")
        print("           auto-succeeds when you reach the target. 'quit' to exit.\n")

        env = InteractiveViewPlanning({
            "jsonl_path": jsonl_path,
            "render_backend": "client",
            "client_url": client_url,
            "scannet_root": scannet_root,
            "format": format,
            "action_only_mode": True,    # camera starts on the initial view
            "no_submit": True,           # auto-success near target, no answer()
            "is_snap_every_step": True,  # snap rotations to step multiples
        })

        def _save(obs, tag: str) -> int:
            imgs = (obs.get("multi_modal_input") or {}).get("<image>", [])
            for i, img in enumerate(imgs):
                img.save(os.path.join(save_dir, f"{tag}_img{i}.png"))
            return len(imgs)

        try:
            obs, _ = await env.reset(seed=seed)
        except Exception as exc:
            print(f"[error] could not start the IVP env: {type(exc).__name__}: {exc}")
            print("  Is the ScanNet render service running? See README Step 3.")
            try:
                await env.close()
            except Exception:
                pass
            sys.exit(1)

        n = _save(obs, "reset")
        print("=" * 64)
        print(obs["obs_str"])
        print(f"[{n} image(s) saved: {save_dir}/reset_img*.png]\n")

        # In action_only_mode the camera already starts on the initial view,
        # so you can move right away (no select_view needed).
        turn = 0
        done = False
        while not done:
            try:
                s = input("Move [keys e.g. 'wwd', or 'quit']: ").strip()
            except EOFError:
                print()
                break
            if s.lower() in {"quit", "exit"}:
                break
            if not s:
                continue

            action = _keys_to_actions(s)
            if not action:
                print("  no valid keys -- try e.g. 'wwd'\n")
                continue

            try:
                obs, reward, done, info = await env.step(
                    f"<think>play</think><action>{action}</action>"
                )
            except Exception as exc:
                print(f"\n[error] render step failed: {type(exc).__name__}: {exc}")
                print("  The ScanNet render service looks unreachable -- see README Step 3.")
                break

            turn += 1
            n = _save(obs, f"turn{turn:02d}")
            print(f"  actions: {action}")
            print(f"  reward={reward:.3f}  done={done}  success={info.get('success')}")
            print(obs["obs_str"])
            print(f"[{n} image(s) saved: {save_dir}/turn{turn:02d}_img*.png]\n")

        if done:
            print("[IVP demo] episode finished.")

        await env.close()
        print("[IVP demo] bye.")

    fire.Fire(lambda **kw: asyncio.run(_play(**kw)))
