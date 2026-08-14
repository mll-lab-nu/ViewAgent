"""Tool-enabled camera env for Habitat-GS scenes.

Mirrors GymAi2thorToolEnv, with two deliberate differences:

  1. The camera is a ``HabitatGSViewManipulator``, so actions mean what they mean in
     Habitat: turning is ground-parallel, forward motion is horizontal regardless of
     pitch. The same manipulator drives data generation, so the ground truth in the
     JSONL and the transitions the agent experiences here are the same function.

  2. **No roll.** Habitat's default agent has no roll axis, so rotate_ccw / rotate_cw
     are absent from the vocabulary rather than present-and-ignored -- an action the
     prompt offers but the world does not implement is a silent scoring hazard.
"""
from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import Any, Dict, List, Tuple

from view_suite.envs.utils.parse_utils import FormatRegistry, ParsedAction, parse_actions
from view_suite.habitat_gs.gym_habitat_gs_render_env import GymHabitatGSRenderEnv
from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator


class GymHabitatGSToolEnv(GymHabitatGSRenderEnv):
    """Camera-control action vocabulary over a Habitat-GS scene."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.step_translation = float(env_config.get("step_translation", 0.5))
        self.step_rotation_deg = float(env_config.get("step_rotation_deg", 30.0))
        self.is_discrete = bool(env_config.get("is_discrete", True))
        self.pitch_limit_deg = float(env_config.get("pitch_limit_deg", 60.0))
        self.action_only_mode = bool(env_config.get("action_only_mode", False))
        self.view_engine = HabitatGSViewManipulator(
            step_translation=self.step_translation,
            step_rotation_deg=self.step_rotation_deg,
            pitch_limit_deg=self.pitch_limit_deg,
            discrete=self.is_discrete,
        )

    # -------------------------
    # Tool prompt / action vocab
    # -------------------------
    @cached_property
    def _keymap(self) -> Dict[str, str]:
        return {
            "move_forward": "w",
            "move_backward": "s",
            "move_right": "d",
            "move_left": "a",
            "move_up": "y",
            "move_down": "h",
            "turn_left": "q",
            "turn_right": "e",
            "look_up": "r",
            "look_down": "f",
        }

    @cached_property
    def _action_only_allowed(self) -> List[str]:
        return [
            "move_forward", "move_backward", "move_right", "move_left",
            "move_up", "move_down",
            "turn_left", "turn_right", "look_up", "look_down",
            "answer",
        ]

    @cached_property
    def _action_full(self) -> List[str]:
        return self._action_only_allowed[:-1] + [
            "query_pose", "select_view", "get_view", "answer",
        ]

    @cached_property
    def action_description(self) -> Dict[str, str]:
        # Rounded for display only: a per-scene step is an awkward float and
        # "0.6501199473505435 meters" in a prompt is noise. The camera uses the exact
        # value, which differs by <5 mm against a 0.5 m success threshold.
        t, r = round(self.step_translation, 2), round(self.step_rotation_deg, 2)
        return {
            "move_forward":  f"move forward on the ground plane by {t} meters.",
            "move_backward": f"move backward on the ground plane by {t} meters.",
            "move_left":     f"strafe left on the ground plane by {t} meters.",
            "move_right":    f"strafe right on the ground plane by {t} meters.",
            "move_up":       f"move the camera up by {t} meters.",
            "move_down":     f"move the camera down by {t} meters.",
            "turn_left":     f"turn left by {r} degrees, about the vertical axis.",
            "turn_right":    f"turn right by {r} degrees, about the vertical axis.",
            "look_up":       f"tilt the camera up by {r} degrees "
                             f"(clamped to +/-{self.pitch_limit_deg}).",
            "look_down":     f"tilt the camera down by {r} degrees "
                             f"(clamped to +/-{self.pitch_limit_deg}).",
            "query_pose":    "query_pose(view_name), return the 6-DoF pose of a named view "
                             "in DEGREES; does NOT change the camera.",
            "select_view":   "select_view(view_name), reset the camera to the named view "
                             "and render an image.",
            "get_view":      "get_view(tx, ty, tz, rx, ry, rz), directly set the camera "
                             "pose (c2w, Euler XYZ in DEGREES) and render an image.",
            # The spelled-out signature and the positional-only sentence are load-bearing:
            # with a vaguer description Qwen2.5-VL emitted answer(tx=..., ty=...) with
            # keyword arguments, the parser rejected every one of them, and IVP scored
            # 0/288 with 287 episodes running out of turns. The model was answering; the
            # prompt had not told it how.
            "answer":        "answer(tx, ty, tz, rx, ry, rz), where tx, ty, tz are "
                             "translation in meters and rx, ry, rz are rotation in "
                             "degrees. All arguments must be positional plain numbers. "
                             "This action is terminal and no further actions can be "
                             "taken.",
        }

    @cached_property
    def _tool_instruction(self) -> str:
        actions = self._action_only_allowed if self.action_only_mode else self._action_full
        lines = ["SUPPORTED ACTIONS", "-----------------",
                 "Arguments are inside parentheses.", ""]
        lines += [f"- {name} : {self.action_description[name]}" for name in actions]
        instruction = "\n".join(lines).strip()

        if not self.action_only_mode:
            instruction += (
                "\n\nACTION ORDER CONSTRAINTS\n"
                "------------------------\n"
                "- You MUST call exactly one of:\n"
                "    - select_view(view_name), or\n"
                "    - get_view(tx, ty, tz, rx, ry, rz)\n"
                "before performing ANY of the following actions:\n"
                "    move_*, turn_*, look_*.\n\n"
                "- Calling move / turn / look before a view is selected\n"
                "is INVALID and will result in failure.\n\n"
                "- query_pose(...) does NOT count as selecting a view.\n\n"
                "- The episode terminates immediately after calling answer(...).\n"
                "No further actions are allowed.\n"
            )
        else:
            instruction += (
                "\n- The episode terminates immediately after calling answer(...).\n"
                "No further actions are allowed.\n"
            )

        instruction += (
            "\nCAMERA MODEL\n"
            "------------\n"
            "- Turning is about the vertical axis only, so the horizon stays level.\n"
            "- Forward/backward/left/right move along the ground plane; they are NOT\n"
            "  affected by how far up or down the camera is tilted.\n"
            "- There is no roll action: the camera never rotates about its view axis.\n"
        )
        if self.is_discrete:
            instruction += (
                "\nDISCRETE MODE\n"
                "-------------\n"
                f"- translation step: {self.step_translation:.2f} meters\n"
                f"- rotation step: {self.step_rotation_deg:.0f} degrees\n"
            )
        return instruction

    @cached_property
    def _view_dict(self) -> Dict[str, Any]:
        return self._get_view_dict()

    # -------------------------
    # Action parse + execute
    # -------------------------
    def _parse_action_str(
        self, action_str: str, format: str = "free_think"
    ) -> Tuple[bool, List[ParsedAction]]:
        is_no_think = (format == "no_think")
        ft = FormatRegistry.parse(format, action_str)
        if not ft["ok"]:
            return False, []
        actions_ok, parsed_actions = parse_actions(ft["actions_blob"])
        if not actions_ok:
            return (True, []) if is_no_think else (False, [])
        return True, parsed_actions

    def _execute_action(self, action: "ParsedAction") -> Dict[str, Any]:
        if self.action_only_mode and action.name not in self._action_only_allowed:
            return {"success": False, "is_answer": False,
                    "result": f"action not allowed in action_only_mode: {action.name}",
                    "need_render": False}

        if action.name in self._keymap:
            try:
                self.view_engine.step(self._keymap[action.name])
                return {"success": True, "is_answer": False, "result": None,
                        "need_render": True}
            except Exception as e:
                return {"success": False, "is_answer": False, "result": str(e),
                        "need_render": False}

        match action.name:
            case "query_pose":
                view = self._view_dict.get(action.arg)
                if not view:
                    return {"success": False, "is_answer": False,
                            "result": f"view not found: {action.arg}", "need_render": False}
                return {"success": True, "is_answer": False,
                        "result": view.get("c2w_se3_deg"), "need_render": False}

            case "select_view":
                view = self._view_dict.get(action.arg)
                if not view:
                    return {"success": False, "is_answer": False,
                            "result": f"view not found: {action.arg}", "need_render": False}
                try:
                    self.view_engine.reset(view.get("c2w_extrinsic"))
                    return {"success": True, "is_answer": False, "result": None,
                            "need_render": True}
                except Exception as e:
                    return {"success": False, "is_answer": False, "result": str(e),
                            "need_render": False}

            case "get_view":
                try:
                    self.view_engine.set_se3(action.arg, degrees=True)
                    return {"success": True, "is_answer": False, "result": None,
                            "need_render": True}
                except Exception as e:
                    return {"success": False, "is_answer": False, "result": str(e),
                            "need_render": False}

            case "answer":
                return {"success": True, "is_answer": True, "result": action.arg,
                        "need_render": False}

            case _:
                return {"success": False, "is_answer": False,
                        "result": f"unknown action: {action.name}", "need_render": False}

    # -------------------------
    # Abstracts (filled by GymProxyTool)
    # -------------------------
    @abstractmethod
    def _get_view_dict(self) -> Dict[str, Any]:
        ...

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
