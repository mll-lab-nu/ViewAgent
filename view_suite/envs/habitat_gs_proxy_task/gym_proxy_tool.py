"""Habitat-GS proxy-tool env.

Adapted from view_suite.envs.scannet_proxy_task.gym_proxy_tool.GymProxyTool.

Differences vs the scannet version:
  - Inherits from GymHabitatGSToolEnv (HTTP Habitat-GS service backend).
  - Default intrinsics come from the configured (fov, size) rather than from the
    scannet calibration constant.
All task logic (prompting, grading, threshold resolution, tool dispatch,
multi-level success bookkeeping) is reused unchanged via the scannet utility
modules, which are scene-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from view_suite.habitat_gs.gym_habitat_gs_tool_env import GymHabitatGSToolEnv
from view_suite.habitat_gs.pose_utils import intrinsics_from_fov

from view_suite.envs.utils.jsonl_utils import (
    resolve_rel_image,
    count_lines,
    read_jsonl_line_by_index,
)
from view_suite.envs.utils.image_utils import safe_open_rgb

from view_suite.scannet.utils.pose_utils import c2w_extrinsic_to_se3

from view_suite.envs.utils.parse_utils import ParsedAction

from view_suite.envs.utils.scannet_utils import (
    parse_get_view_arg_deg,
    fmt_pose6_deg,
    ensure_K4x4,
)

# Scene-agnostic task utilities live under scannet_proxy_task — reused here.
from view_suite.envs.scannet_proxy_task.utils.gym_proxy_tool_utils import (
    geodesic_angle_deg,
    smooth_closeness_score,
    resolve_thresholds,
    parse_multi_level_success_rate,
    multi_level_success_flags,
    resolve_thresholds_per_action_len,
)
from view_suite.envs.scannet_proxy_task.utils.gym_proxy_tool_prompt import (
    build_system_prompt,
    build_reset_prompt,
)


def _default_intrinsics_gs(width: int = 512, height: int = 512, fov: float = 90.0) -> np.ndarray:
    """4x4 K for the GS render service (square pixels; Habitat cannot express fx != fy)."""
    K3 = intrinsics_from_fov(width, height, fov)
    return ensure_K4x4(K3)


class GymProxyTool(GymHabitatGSToolEnv):
    """Multi-turn tool-use proxy task on Habitat-GS 3DGS scenes (active exploration).

    Same obs/info contracts as scannet's GymProxyTool; see that module for
    the full reward / grading specification.
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.config = env_config

        # --- Dataset ---
        self.jsonl_path = Path(self.config["jsonl_path"])
        assert self.jsonl_path.is_file(), f"jsonl not found: {self.jsonl_path}"
        self.total_lines = int(self.config.get("total_lines") or count_lines(self.jsonl_path))
        assert self.total_lines > 0, "empty jsonl"
        self.dataset_root = self.config.get("dataset_root")

        # --- Runtime ---
        self.current_item: Optional[Dict[str, Any]] = None
        self.current_index: Optional[int] = None
        self.current_step: int = 0
        self.episode_done: bool = False

        self.images: List[Image.Image] = []  # (target, init, topdown)
        self._named_views: Dict[str, Dict[str, Any]] = {}
        self.target_view: Optional[Dict[str, Any]] = None

        self._has_active_camera: bool = False
        self.camera_intrinsics: Optional[np.ndarray] = None

        # --- Rewards ---
        self.format_reward = float(self.config.get("format_reward", 0.1))
        self.per_turn_format_reward = float(self.config.get("per_turn_format_reward", 0.0))
        self.answer_reward = float(self.config.get("answer_reward", 1.0))
        self.answer_close_reward = float(self.config.get("answer_close_reward", 0.0))
        self.give_answer_reward = float(self.config.get("give_answer_reward", 0.0))
        self.single_action_reward = float(self.config.get("single_action_reward", 0.0))

        # --- Prompting / Controls ---
        self.max_turns = int(self.config.get("max_turns", 10))
        self.image_size: Optional[Sequence[int]] = self.config.get("image_size", None)
        self.use_example_in_sys_prompt = bool(self.config.get("use_example_in_sys_prompt", True))
        self.example_count = int(self.config.get("example_count", 2))
        self.format: str = self.config.get("format", "free_think")
        self.action_only_mode = bool(self.config.get("action_only_mode", False))

        # --- Tolerances ---
        self.tol_per_action_len = self.config.get(
            "tol_per_action_len", "0.25,15;2:0.5,30;3-5:0.5,30;0.5,30"
        )
        if self.tol_per_action_len is None:
            self.tol_trans_l2_m: Optional[float] = float(self.config.get("tol_trans_l2_m", 1.0))
            self.tol_rot_l2_deg: Optional[float] = float(self.config.get("tol_rot_l2_deg", 30.0))
        else:
            self.tol_trans_l2_m = None
            self.tol_rot_l2_deg = None

        # --- Multi-level thresholds ---
        self.multi_level_success_rate: Optional[str] = self.config.get(
            "multi_level_success_rate", "1,60;1,90;2,60;2,90;3,60;3,90"
        )
        try:
            self._multi_levels = parse_multi_level_success_rate(self.multi_level_success_rate)
        except Exception as e:
            raise ValueError(
                f"Invalid multi_level_success_rate='{self.multi_level_success_rate}': {e}"
            )

        # --- Intrinsics ---
        self._override_K = self.config.get("camera_intrinsics_override", None)
        if self._override_K is not None:
            self._override_K = ensure_K4x4(np.array(self._override_K, dtype=np.float64))
        self.fix_intrinsics = bool(self.config.get("fix_intrinsics", True))
        # Default K: derived from render size + fov kept in env_config,
        # falling back to 512 / 90 which matches our data_gen output.
        default_width = int(self.config.get("render_width", 512))
        default_height = int(self.config.get("render_height", 512))
        default_fov = float(self.config.get("render_fov", 90.0))
        self._default_K = (
            _default_intrinsics_gs(default_width, default_height, default_fov)
            if self.fix_intrinsics else None
        )

    # ------------------ base hook ------------------
    def _get_view_dict(self) -> Dict[str, Any]:
        return self._named_views

    async def close(self) -> None:
        try:
            await self.renderer.close()
        except Exception:
            pass

    async def system_prompt(self) -> Dict[str, Any]:
        use_examples = (not self.action_only_mode) and bool(self.use_example_in_sys_prompt)
        obs_str = build_system_prompt(
            tool_instruction=self._tool_instruction,
            max_turns=self.max_turns,
            use_examples=use_examples,
            example_count=int(self.example_count if use_examples else 0),
            format_name=self.format,
        )
        return {"obs_str": obs_str}

    # ------------------ Slim info builders ------------------
    def _info_reset(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        pos_thr_m, ang_thr_deg = resolve_thresholds(item, self.tol_trans_l2_m, self.tol_rot_l2_deg)
        return {
            "success": False,
            "scene_id": item.get("scene_id"),
            "sample_id": item.get("sample_id"),
            "gt_action_seq": item.get("gt_action_seq"),
            "pos_threshold_m": float(pos_thr_m),
            "ang_threshold_deg": float(ang_thr_deg),
            "jsonl_idx": idx,
        }

    def _info_step(self, success: bool = False, error: Optional[str] = None) -> Dict[str, Any]:
        d = {"success": bool(success)}
        if error is not None:
            d["error"] = str(error)
        return d

    def _info_answer(self, success: bool, metrics: Dict[str, Any]) -> Dict[str, Any]:
        info = {
            "success": bool(success),
            "pos_err_m": float(metrics["pos_err_m"]),
            "ang_err_deg": float(metrics["ang_err_deg"]),
            "pos_threshold_m": float(metrics["pos_threshold_m"]),
            "ang_threshold_deg": float(metrics["ang_threshold_deg"]),
        }
        if getattr(self, "_multi_levels", None):
            flags = multi_level_success_flags(
                float(info["pos_err_m"]),
                float(info["ang_err_deg"]),
                self._multi_levels,
            )
            info.update(flags)
        return info

    # ------------------ Small helpers ------------------
    def _reset_runtime(self) -> None:
        self.episode_done = False
        self.current_step = 0
        self.current_item = None
        self.current_index = None
        self.images.clear()
        self._named_views.clear()
        self.target_view = None
        self._has_active_camera = False
        self.camera_intrinsics = None

    def _set_active_intrinsics(self, K: np.ndarray) -> None:
        self.camera_intrinsics = ensure_K4x4(K)

    def _obs(self, text: str, imgs: List[Image.Image]) -> Dict[str, Any]:
        return {"obs_str": text.strip(), "multi_modal_input": {"<image>": self._resize_image(imgs)}}

    def _validate_action(self, a: ParsedAction) -> Optional[str]:
        if self.action_only_mode and a.name not in self._action_only_allowed:
            return f"action must be one of {self._action_only_allowed}"
        if a.name not in self._action_full:
            return f"action must be one of {self._action_full}"
        if a.name in self._keymap and not self._has_active_camera:
            return "Call select_view(view_name) or get_view(tx,ty,tz,rx,ry,rz) before moving/rotating the cameraction."
        return None

    async def _render_current(self, width: int, height: int) -> Image.Image:
        assert self._has_active_camera, "camera not activated"
        E_c2w = self.view_engine.get_pose(mode="c2w")
        assert self.camera_intrinsics is not None, "active intrinsics not set"
        return await self.render_image_from_cam_param(self.camera_intrinsics, E_c2w, width=width, height=height)

    def _current_pose_block(self) -> Optional[str]:
        if not self._has_active_camera:
            return None
        se3_deg = self.view_engine.get_se3(degrees=True)
        return "\n".join([
            "Current camera 6-DoF (c2w, Euler XYZ, DEGREES):",
            fmt_pose6_deg(se3_deg),
        ])

    # ------------------ Obs string builders ------------------
    def _build_reset_obs_str(self, item: Dict[str, Any]) -> str:
        scene_line = f"You're in the scene {item.get('scene_id')}." if item.get("scene_id") else ""
        if self.action_only_mode:
            q_text = (
                "Please study the target view <image>, the initial view <image>, "
                "and the top-down view <image>.\n"
                "You start from the initial view. Move toward the target view using actions"
            )
            available_names = None
        else:
            q_text = (
                "Now please estimate the camera pose of the target view <image>.\n"
                "You can start by selecting the initial view <image> or the top-down view <image>."
            )
            available_names = list(self._named_views.keys()) or None

        if scene_line:
            q_text = scene_line + "\n" + q_text

        prompt = build_reset_prompt(q_text, "", available_names)
        parts: List[str] = [prompt]

        if "init_view" in self._named_views:
            parts += [
                "Initial view camera 6-DoF (c2w, Euler XYZ, DEGREES):",
                fmt_pose6_deg(self._named_views["init_view"]["c2w_se3_deg"]),
            ]
        if "top_down_view" in self._named_views:
            parts += [
                "Top-down view camera 6-DoF (c2w, Euler XYZ, DEGREES):",
                fmt_pose6_deg(self._named_views["top_down_view"]["c2w_se3_deg"]),
            ]

        pos_thr_m, ang_thr_deg = resolve_thresholds(item, self.tol_trans_l2_m, self.tol_rot_l2_deg)
        parts += [
            f"Success thresholds: position error <= {pos_thr_m}m, rotation error <= {ang_thr_deg}°",
            f"Step {self.current_step + 1}/{self.max_turns}",
        ]
        return "\n".join(parts)

    def _build_step_obs_str(self, ok: bool, error_msg: Optional[str] = None) -> str:
        lines: List[str] = []
        if ok:
            lines.append("format: ok")
            pose_block = self._current_pose_block()
            if pose_block:
                lines.append(pose_block)
        else:
            lines.append(f"format: error | {error_msg or ''}".rstrip())
        lines.append(f"Step {self.current_step + 1}/{self.max_turns}")
        return "\n".join(lines)

    def _build_answer_obs_str(self, answer_msg: str) -> str:
        lines: List[str] = [answer_msg]
        pose_block = self._current_pose_block()
        if pose_block:
            lines.append(pose_block)
        lines.append(f"Step {self.current_step + 1}/{self.max_turns}")
        return "\n".join(lines)

    @staticmethod
    def _inject_image_tag(obs_str: str, has_image: bool) -> str:
        if not has_image:
            return obs_str
        return obs_str.replace("\nStep ", "\n<image>\nStep ", 1)

    # ------------------ Episode lifecycle ------------------
    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert seed is not None, "reset(seed) requires an integer seed."
        self._reset_runtime()

        idx = seed % self.total_lines
        item = read_jsonl_line_by_index(self.jsonl_path, idx)
        self.current_item = item
        self.current_index = idx

        # Adaptive thresholds based on ground-truth action sequence length.
        # If tol_per_action_len is not configured, keep the fixed tol_trans/rot set in __init__.
        if self.tol_per_action_len:
            gt_action_seq = item.get("gt_action_seq") or item.get("meta", {}).get("gt_action_seq_letters") or []
            self.tol_trans_l2_m, self.tol_rot_l2_deg = resolve_thresholds_per_action_len(
                self.tol_per_action_len, len(gt_action_seq)
            )

        # The action step is a property of the SAMPLE, not of the env: this corpus
        # scales it per scene, from a 23 m room to a 538 m plaza, so a fixed value is
        # wrong for 92% of the test split. Take it from the row and reconfigure the
        # camera, or the agent is told one step size, moves another, and chases a target
        # that was built with a third -- silently, with every render looking fine.
        _m = item.get("meta") or {}
        _step_t = _m.get("step_translation_m")
        _step_r = _m.get("step_rotation_deg")
        if _step_t is not None:
            self.step_translation = float(_step_t)
            self.view_engine.step_t = float(_step_t)
        if _step_r is not None:
            self.step_rotation_deg = float(_step_r)
            self.view_engine.step_r = float(_step_r)
        _pitch = _m.get("pitch_limit_deg")
        if _pitch is not None:
            self.pitch_limit_deg = float(_pitch)
            self.view_engine.pitch_limit = float(_pitch)
        # The prompt quotes these, and it is cached on first access.
        for _attr in ("_tool_instruction", "action_description"):
            self.__dict__.pop(_attr, None)

        details = item["image_detail"]
        # Load target, init, topdown if present (tolerate partial schemas).
        for nm in ("target_view", "init_view", "top_down_view"):
            if nm in details and details[nm].get("path"):
                p = resolve_rel_image(self.jsonl_path, details[nm]["path"], self.dataset_root)
                self.images.append(safe_open_rgb(p))

        for name, pack in details.items():
            if not isinstance(pack, dict) or "c2w_extrinsics" not in pack:
                continue
            if pack["c2w_extrinsics"] is None:
                continue
            E_c2w = np.array(pack["c2w_extrinsics"], dtype=np.float64)
            K_raw = pack.get("c2w_intrinsics")
            if K_raw is None:
                K = self._default_K if self.fix_intrinsics else _default_intrinsics_gs()
            else:
                K = self._default_K if self.fix_intrinsics else ensure_K4x4(np.array(K_raw, dtype=np.float64))
            dp = {
                "c2w_extrinsic": E_c2w,
                "c2w_se3_deg": c2w_extrinsic_to_se3(E_c2w, degrees=True),
                "K": K,
            }
            if name == "target_view":
                self.target_view = dp
            else:
                self._named_views[str(name)] = dp

        if self.action_only_mode and "init_view" in self._named_views:
            init_view = self._named_views["init_view"]
            self.view_engine.reset(init_view["c2w_extrinsic"])
            self._set_active_intrinsics(init_view["K"])
            self._has_active_camera = True
        else:
            self.view_engine.reset(None)

        self.renderer.set_scene(item.get("scene_id"))

        obs_str = self._build_reset_obs_str(item)
        obs = self._obs(obs_str, self.images)
        info = self._info_reset(item, idx)
        self.is_format_correct = True
        return obs, info

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.current_step += 1
        if self.episode_done:
            obs_str = self._build_step_obs_str(ok=False, error_msg="episode_done")
            obs = self._obs(obs_str, [])
            return obs, 0.0, True, self._info_step(False, "episode_done")

        assert self.current_item is not None, "Call reset() before step()."
        item = self.current_item

        ok, actions = self._parse_action_str(action_str, format=self.format)
        if not ok:
            return self._format_error_step("free_think/actions parse error")

        total_reward = 0.0
        need_render = False
        last_render: Optional[Image.Image] = None

        for a in actions:
            if a.name == "get_view" and isinstance(a.arg, str):
                parsed = parse_get_view_arg_deg(a.arg)
                if parsed is None:
                    return self._format_error_step("get_view requires tx,ty,tz,rx,ry,rz (DEGREES)")
                a = ParsedAction(name="get_view", arg=parsed)

            err = self._validate_action(a)
            if err:
                return self._format_error_step(err)

            if a.name == "answer":
                success, metrics, msg = self._grade_answer_pose(a.arg, item)
                if not metrics:
                    return self._format_error_step(msg)
                closeness = float(metrics.get("pose_closeness", 0.0) or 0.0)
                answer_bonus = self.answer_reward if success else 0.0
                close_bonus = (self.answer_close_reward * closeness) if self.answer_close_reward > 0.0 else 0.0
                total_reward = total_reward + answer_bonus + close_bonus + self.give_answer_reward
                if self.is_format_correct:
                    total_reward += self.format_reward
                if len(actions) == 1 and self.single_action_reward:
                    total_reward += self.single_action_reward
                self.episode_done = True

                if need_render:
                    w, h = self.image_size if self.image_size is not None else (512, 512)
                    last_render = await self._render_current(width=w, height=h)

                obs_str = self._build_answer_obs_str(msg)
                imgs = [last_render] if last_render is not None else []
                obs_str = self._inject_image_tag(obs_str, has_image=bool(imgs))
                obs = self._obs(obs_str, imgs)
                info = self._info_answer(success, metrics)
                return obs, total_reward, True, info

            result = self._execute_action(a)
            if not result["success"]:
                return self._format_error_step(str(result["result"]))

            need_render = need_render or bool(result["need_render"])
            if a.name == "select_view" and a.arg in self._named_views:
                self._set_active_intrinsics(self._named_views[a.arg]["K"])
                self._has_active_camera = True
            elif a.name == "get_view":
                if self.camera_intrinsics is None:
                    K = self._override_K if self._override_K is not None else _default_intrinsics_gs()
                    self._set_active_intrinsics(K)
                self._has_active_camera = True

        if need_render:
            w, h = self.image_size if self.image_size is not None else (512, 512)
            last_render = await self._render_current(width=w, height=h)

        obs_str = self._build_step_obs_str(ok=True)
        imgs = [last_render] if last_render is not None else []
        obs_str = self._inject_image_tag(obs_str, has_image=bool(imgs))
        obs = self._obs(obs_str, imgs)
        total_reward += self.per_turn_format_reward
        if len(actions) == 1 and self.single_action_reward:
            total_reward += self.single_action_reward
        return obs, total_reward, False, self._info_step(False)

    def _format_error_step(self, msg: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.is_format_correct = False
        obs = self._obs(self._build_step_obs_str(ok=False, error_msg=msg), [])
        return obs, 0.0, False, self._info_step(False, msg)

    # ------------------ Answer grading ------------------
    def _grade_answer_pose(self, answer_arg: Any, item: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        if isinstance(answer_arg, str):
            parsed = parse_get_view_arg_deg(answer_arg)
            if parsed is None:
                return False, {}, "answer requires tx,ty,tz,rx,ry,rz (DEGREES)"
            pred = np.array(parsed, dtype=np.float64)
        elif isinstance(answer_arg, (list, tuple, np.ndarray)) and len(answer_arg) == 6:
            try:
                pred = np.array([float(x) for x in answer_arg], dtype=np.float64)
            except Exception:
                return False, {}, "answer values must be numeric (tx,ty,tz,rx,ry,rz)"
        else:
            return False, {}, "answer requires 6 numbers (tx,ty,tz,rx,ry,rz)"

        target = self.target_view["c2w_se3_deg"] if self.target_view is not None else None
        if target is None:
            return False, {}, "No target_view in the datapoint."
        target = np.array(target, dtype=np.float64)

        pos_thr_m, ang_thr_deg = resolve_thresholds(item, self.tol_trans_l2_m, self.tol_rot_l2_deg)

        pos_err_m = float(np.linalg.norm(pred[:3] - target[:3]))
        ang_err_deg = float(geodesic_angle_deg(pred[3:], target[3:]))
        success = (pos_err_m <= pos_thr_m + 1e-9) and (ang_err_deg <= ang_thr_deg + 1e-9)

        msg = (
            f"[answer] pred={fmt_pose6_deg(pred.tolist())} | "
            f"gt={fmt_pose6_deg(target.tolist())} | "
            f"pos_err={pos_err_m:.3f}m --> {pos_thr_m:.3f}m, "
            f"ang_err={ang_err_deg:.3f}° --> {ang_thr_deg:.3f}°"
        )

        pos_close = smooth_closeness_score(pos_err_m, pos_thr_m)
        ang_close = smooth_closeness_score(ang_err_deg, ang_thr_deg)
        pose_close = float(max(0.0, min(1.0, pos_close * ang_close)))
        msg += f", closeness={pose_close:.3f} -> success={success}"

        metrics = {
            "pos_err_m": pos_err_m,
            "ang_err_deg": ang_err_deg,
            "pos_threshold_m": float(pos_thr_m),
            "ang_threshold_deg": float(ang_thr_deg),
            "pose_closeness": pose_close,
        }
        return success, metrics, msg
