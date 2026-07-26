"""
AI2-THOR proxy-task data generator.

Produces a dataset with the SAME JSONL schema as ScanNet's viewsuite_5k, but
sourced from AI2-THOR iTHOR FloorPlans.

For each (scene, sample):
  * sample an initial (position, rotation) inside the scene's reachable region
  * sample a GT action sequence (length 2..6) of action letters
  * simulate the sequence via ViewManipulator to get a GT target pose
  * sample 3 distractor action sequences of similar length -> distractor poses
  * render 1 initial + 4 options + 1 top-down view (top-down shared per scene)
  * write meta.json + 5 PNGs + one jsonl row per task type

Output structure (task-centric naming, matches the ScanNet proxy-task datasets):
  out_root/
    path_to_view_test.jsonl               # P2V (forward dynamics)
    view_to_path_test.jsonl               # V2P (inverse dynamics)
    interactive_view_planning_test.jsonl  # IVP (active exploration)
    FloorPlan1/
      top_down_view.png
      sample_000/
        initial_view.png
        option_000.png option_001.png option_002.png option_003.png
        meta.json
      sample_001/
        ...

All non-essential knobs have defaults but are CLI-settable.
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from view_suite.ai2thor.scene_list import parse_subset
from view_suite.ai2thor.view_manipulator import ViewManipulator
from view_suite.ai2thor.pose_utils import (
    unity_pose_to_c2w,
    intrinsics_from_fov,
    top_down_c2w,
)


# ---------- Action vocabulary (matches ScanNet proxy task) ----------
ACTION_LETTERS = ["w", "s", "a", "d", "q", "e", "r", "f"]   # move_forward/back/left/right, yaw_left/right, look_up/down
ACTION_NAMES = {
    "w": "move_forward", "s": "move_backward",
    "a": "move_left",    "d": "move_right",
    "q": "turn_left",    "e": "turn_right",
    "r": "look_up",      "f": "look_down",
    "y": "move_up",      "h": "move_down",
    "t": "roll_ccw",     "g": "roll_cw",
}


@dataclass
class GenConfig:
    out_root: str = "/root/projects/viewsuite/data/ai2thor_test"
    scenes: str = "default"                 # see scene_list.parse_subset
    samples_per_scene: int = 16
    width: int = 512
    height: int = 512
    fov: float = 90.0
    step_translation: float = 0.5           # meters
    step_rotation_deg: float = 30.0         # degrees
    gt_seq_min: int = 2
    gt_seq_max: int = 6
    num_distractors: int = 3
    top_down_height_y: float = 4.0          # meters above scene centroid (legacy, unused now)
    seed: int = 0
    # ----- Quality filters (all configurable, defaults tuned for indoor iTHOR) -----
    inside_room_threshold_m: float = 0.4    # pose XZ must lie within this dist of any reachable point
    min_camera_y: float = 0.5               # reject options with y below this (floor)
    max_camera_y: float = 2.5               # reject options with y above this (ceiling)
    max_resample_tries: int = 30            # attempts per sample before giving up
    min_image_std: float = 0.0              # optional: min per-channel std for a valid render (0 = off)
    # Low-semantic-content filter: reject a render if a single near-identical
    # color covers more than this fraction of the image (e.g. a blank wall /
    # floor / ceiling view). 1.0 = off. Applied to the initial view and every
    # option (the target view is one of the options).
    max_dominant_pixel_frac: float = 0.8


# =============================================================================
# Sampling helpers
# =============================================================================
def _sample_initial_pose(
    controller: Controller,
    rng: random.Random,
) -> Dict:
    """Pick a random reachable position + yaw (pitch 0) as the initial pose dict."""
    ev = controller.step(action="GetReachablePositions")
    positions = ev.metadata["actionReturn"]
    if not positions:
        raise RuntimeError("No reachable positions for current scene")
    pos = positions[rng.randrange(len(positions))]
    yaw = float(rng.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))
    return {
        "position": {"x": float(pos["x"]), "y": float(pos["y"]) + 0.8, "z": float(pos["z"])},
        "rotation": {"x": 0.0, "y": yaw, "z": 0.0},
    }


def _sample_action_seq(rng: random.Random, length: int) -> List[str]:
    return [rng.choice(ACTION_LETTERS) for _ in range(length)]


def _seq_equal(a: List[str], b: List[str]) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def _apply_sequence(init_pose: Dict, seq: List[str], step_t: float, step_r: float) -> Dict:
    vm = ViewManipulator(
        init_pose=init_pose,
        step_translation=step_t,
        step_rotation_deg=step_r,
        is_discrete=True,
    )
    for a in seq:
        vm.step(a)
    return vm.get_pose_thor()


# =============================================================================
# Reachability + pose validation
# =============================================================================
def _reachable_xz(controller: Controller) -> np.ndarray:
    """Return (N, 2) float array of reachable XZ positions."""
    ev = controller.step(action="GetReachablePositions")
    pts = ev.metadata["actionReturn"] or []
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([[p["x"], p["z"]] for p in pts], dtype=np.float64)


def _pose_inside_room(
    pose: Dict,
    reachable_xz: np.ndarray,
    threshold_m: float,
) -> bool:
    """Check whether the pose's XZ is within threshold_m of any reachable point."""
    if reachable_xz.size == 0:
        return True  # no reachability data -> be permissive rather than drop everything
    p = np.array([pose["position"]["x"], pose["position"]["z"]], dtype=np.float64)
    d = float(np.linalg.norm(reachable_xz - p, axis=1).min())
    return d <= threshold_m


def _pose_y_in_range(pose: Dict, y_lo: float, y_hi: float) -> bool:
    y = float(pose["position"]["y"])
    return (y_lo <= y) and (y <= y_hi)


def _valid_option_pose(
    pose: Dict,
    reachable_xz: np.ndarray,
    cfg: "GenConfig",
) -> bool:
    return (
        _pose_inside_room(pose, reachable_xz, cfg.inside_room_threshold_m)
        and _pose_y_in_range(pose, cfg.min_camera_y, cfg.max_camera_y)
    )


def _image_std(img: np.ndarray) -> float:
    """Mean per-channel std — cheap proxy for 'meaningful content'."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        return float(a.reshape(-1, a.shape[-1]).std(axis=0).mean())
    return float(a.std())


def _dominant_pixel_fraction(img: np.ndarray, bits: int = 3) -> float:
    """Fraction of pixels sharing the single most-common (quantized) color.

    A cheap "low-semantic content" detector: a blank wall / floor / ceiling view
    is dominated by one near-uniform color. Each channel is quantized to
    ``2**(8-bits)`` levels (default: top 5 bits -> 32 levels) so near-identical
    pixels collapse to the same bin, then we return the largest bin's fraction.
    1.0 = completely uniform image; ~0 = highly varied.
    """
    a = np.asarray(img)
    if a.ndim == 3:
        q = (a[..., :3].astype(np.uint16) >> bits)
        key = (q[..., 0].astype(np.int32) << 12) | (q[..., 1].astype(np.int32) << 6) | q[..., 2].astype(np.int32)
    else:
        key = (a.astype(np.uint16) >> bits).astype(np.int32)
    key = key.reshape(-1)
    if key.size == 0:
        return 1.0
    counts = np.bincount(key)
    return float(counts.max()) / float(key.size)


# =============================================================================
# Scene centroid for top-down
# =============================================================================
def _scene_centroid_xz(controller: Controller) -> Tuple[float, float]:
    ev = controller.step(action="GetReachablePositions")
    pts = ev.metadata["actionReturn"] or [{"x": 0.0, "z": 0.0}]
    xs = [p["x"] for p in pts]
    zs = [p["z"] for p in pts]
    return (float(np.mean(xs)), float(np.mean(zs)))


def _render_topdown_mapview(controller: Controller) -> Tuple[np.ndarray, Dict]:
    """
    Render top-down via AI2-THOR's canonical map-view camera.

    Flow:
      1) GetMapViewCameraProperties -> position, rotation, orthographicSize, orthographic=True
      2) AddThirdPartyCamera (orthographic=True, using those props) -> cam index 1
      3) Grab third_party_camera_frames[1]
      4) Leave the camera attached; scene reset will clear it on next iteration

    Returns (rgb, top_down_pose_dict). The rgb is RGBA if AI2-THOR returned
    alpha — caller is expected to drop the alpha channel if needed.
    """
    ev = controller.step(action="GetMapViewCameraProperties")
    props = ev.metadata.get("actionReturn") or {}
    pos = dict(props.get("position", {"x": 0.0, "y": 2.5, "z": 0.0}))
    rot = dict(props.get("rotation", {"x": 90.0, "y": 0.0, "z": 0.0}))
    orthographic = bool(props.get("orthographic", True))
    ortho_size = float(props.get("orthographicSize", 3.0))
    fov = float(props.get("fieldOfView", 90.0))

    controller.step(
        action="AddThirdPartyCamera",
        position=pos,
        rotation=rot,
        fieldOfView=fov,
        orthographic=orthographic,
        orthographicSize=ortho_size,
    )
    frames = controller.last_event.third_party_camera_frames
    if not frames or len(frames) < 2:
        raise RuntimeError("Top-down third-party camera frame not available")
    rgb = np.asarray(frames[-1])
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[..., :3]

    cam_pose = {"position": pos, "rotation": rot}
    return rgb, cam_pose


# =============================================================================
# Rendering via Controller's third-party camera
# =============================================================================
def _ensure_third_party_camera(controller: Controller, fov: float):
    controller.step(
        action="AddThirdPartyCamera",
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        rotation={"x": 0.0, "y": 0.0, "z": 0.0},
        fieldOfView=float(fov),
    )


def _render_at(controller: Controller, pose: Dict, fov: float) -> np.ndarray:
    controller.step(
        action="UpdateThirdPartyCamera",
        thirdPartyCameraId=0,
        position=pose["position"],
        rotation=pose["rotation"],
        fieldOfView=float(fov),
    )
    ev = controller.last_event
    if not ev.third_party_camera_frames:
        raise RuntimeError("No third_party_camera_frames available")
    return np.asarray(ev.third_party_camera_frames[0])  # (H, W, 3) uint8 RGB


def _save_png(img: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


# =============================================================================
# Prompt builders  (match scannet jsonl text so prompts work with existing envs)
# =============================================================================
def _build_forward_prompt(step_t: float, step_r: float, action_seq_names: List[str]) -> str:
    return (
        f"Given the initial view <image> and a top-down reference <image>, "
        f"after you execute the following action sequence "
        f"(translation step = {step_t} m; rotation step = {step_r} degrees per step):\n"
        f"[{', '.join(action_seq_names)}]\n"
        f"which of the following images corresponds to the result?\n"
        f"A. <image>\nB. <image>\nC. <image>\nD. <image>\n"
    )


def _build_inverse_prompt(step_t: float, step_r: float,
                          option_action_seq_names: Dict[str, List[str]]) -> str:
    lines = [
        "Given the initial view <image> and a top-down reference <image>, "
        "which action sequence will reach the target view <image>?",
        f"(Action semantics: translation step = {step_t} m; rotation step = {step_r} degrees per step.)",
    ]
    for letter in ["A", "B", "C", "D"]:
        lines.append(f"{letter}. [{', '.join(option_action_seq_names[letter])}]")
    return "\n".join(lines) + "\n"


def _build_active_explore_prompt() -> str:
    # active_exploration prompt is typically inlined by the env class itself; we
    # still produce a sensible prompt string so the jsonl row is self-describing.
    return (
        "Given the initial view <image> and a top-down reference <image>, "
        "estimate the target view's 6-DoF pose relative to the world."
    )


# =============================================================================
# Per-sample generation
# =============================================================================
def _build_sample(
    controller: Controller,
    scene_id: str,
    sample_idx: int,
    scene_dir: str,
    out_root: str,
    cfg: GenConfig,
    rng: random.Random,
    K_matrix: np.ndarray,
    reachable_xz: np.ndarray,
) -> Optional[Dict]:
    """Produce one (scene, sample) unit and return its jsonl rows.

    Returns None if we cannot find a valid-inside-room GT or distractor set
    after cfg.max_resample_tries attempts (caller should resample a new initial
    pose or log+skip).
    """
    # 1) initial pose (sampled on a reachable point, so always inside)
    init_pose = _sample_initial_pose(controller, rng)
    init_c2w = unity_pose_to_c2w(init_pose)

    # 2) Sample GT action sequence such that the resulting pose stays in-room.
    gt_seq: Optional[List[str]] = None
    gt_pose: Optional[Dict] = None
    for _ in range(cfg.max_resample_tries):
        gt_len = rng.randint(cfg.gt_seq_min, cfg.gt_seq_max)
        seq = _sample_action_seq(rng, gt_len)
        pose = _apply_sequence(init_pose, seq, cfg.step_translation, cfg.step_rotation_deg)
        if _valid_option_pose(pose, reachable_xz, cfg):
            gt_seq, gt_pose = seq, pose
            break
    if gt_seq is None:
        return None  # couldn't find a valid GT — caller should resample the initial pose

    # 3) Sample distractor sequences: different from GT + each other, pose in-room.
    distractors: List[List[str]] = []
    distractor_poses: List[Dict] = []
    attempts = 0
    while len(distractors) < cfg.num_distractors and attempts < cfg.max_resample_tries * 4:
        attempts += 1
        dl = rng.randint(cfg.gt_seq_min, cfg.gt_seq_max)
        seq = _sample_action_seq(rng, dl)
        if _seq_equal(seq, gt_seq) or any(_seq_equal(seq, d) for d in distractors):
            continue
        pose = _apply_sequence(init_pose, seq, cfg.step_translation, cfg.step_rotation_deg)
        if not _valid_option_pose(pose, reachable_xz, cfg):
            continue
        distractors.append(seq)
        distractor_poses.append(pose)

    if len(distractors) < cfg.num_distractors:
        # Couldn't find enough valid distractors — reject this sample.
        return None

    # 5) assign options A..D with GT randomly placed
    option_letters = ["A", "B", "C", "D"]
    gt_idx_within_options = rng.randrange(4)
    option_action_seqs_letters: Dict[str, List[str]] = {}
    option_action_seqs_names: Dict[str, List[str]] = {}
    option_poses: Dict[str, Dict] = {}
    j = 0  # distractor index
    for i, letter in enumerate(option_letters):
        if i == gt_idx_within_options:
            option_action_seqs_letters[letter] = list(gt_seq)
            option_poses[letter] = gt_pose
        else:
            option_action_seqs_letters[letter] = list(distractors[j])
            option_poses[letter] = distractor_poses[j]
            j += 1
        option_action_seqs_names[letter] = [ACTION_NAMES.get(a, a) for a in option_action_seqs_letters[letter]]

    gt_answer_letter = option_letters[gt_idx_within_options]

    # 6) render images
    initial_rgb = _render_at(controller, init_pose, cfg.fov)
    option_rgbs = [_render_at(controller, option_poses[l], cfg.fov) for l in option_letters]

    # 6b) optional image-content filter: reject if any render is too uniform
    #     (blank wall / outside-skybox slip-through)
    if cfg.min_image_std > 0:
        all_stds = [_image_std(initial_rgb)] + [_image_std(r) for r in option_rgbs]
        if min(all_stds) < cfg.min_image_std:
            return None

    # 6c) low-semantic-content filter: reject if the initial view or any option
    #     is dominated by a single near-identical color (blank wall / floor /
    #     ceiling), which carries little spatial information. The target view is
    #     one of the options, so this also guarantees a meaningful target.
    if cfg.max_dominant_pixel_frac < 1.0:
        all_imgs = [initial_rgb] + option_rgbs
        if max(_dominant_pixel_fraction(im) for im in all_imgs) > cfg.max_dominant_pixel_frac:
            return None

    sample_dir_rel = f"{scene_id}/sample_{sample_idx:03d}"
    sample_dir_abs = os.path.join(out_root, sample_dir_rel)
    _save_png(initial_rgb, os.path.join(sample_dir_abs, "initial_view.png"))
    option_paths_rel = []
    for i, rgb in enumerate(option_rgbs):
        rel = f"sample_{sample_idx:03d}/option_{i:03d}.png"
        _save_png(rgb, os.path.join(scene_dir, rel))
        option_paths_rel.append(f"{scene_id}/{rel}")

    # 7) meta.json (mirror scannet meta schema where possible)
    initial_view_meta = {
        "pose_c2w": init_c2w.tolist(),
        "intrinsics": K_matrix.tolist(),
        "image": "initial_view.png",
        "unity_pose": init_pose,
    }
    options_meta = []
    for i, letter in enumerate(option_letters):
        p = option_poses[letter]
        options_meta.append({
            "is_gt": (letter == gt_answer_letter),
            "action_seq": option_action_seqs_letters[letter],
            "action_seq_names": option_action_seqs_names[letter],
            "pose_c2w": unity_pose_to_c2w(p).tolist(),
            "intrinsics": K_matrix.tolist(),
            "image": f"option_{i:03d}.png",
            "unity_pose": p,
        })

    meta = {
        "scene_id": scene_id,
        "sample_id": f"{scene_id}_sample_{sample_idx}",
        "initial": initial_view_meta,
        "options": options_meta,
        "gt_index": gt_idx_within_options,
        "gt_answer_letter": gt_answer_letter,
        "gt_action_seq_letters": list(gt_seq),
        "gt_action_seq_names": [ACTION_NAMES.get(a, a) for a in gt_seq],
        "step_translation_m": cfg.step_translation,
        "step_rotation_deg": cfg.step_rotation_deg,
        "source": "ai2thor",
    }
    with open(os.path.join(sample_dir_abs, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 8) jsonl rows for each of the 3 tasks
    init_path_rel = f"{scene_id}/sample_{sample_idx:03d}/initial_view.png"
    top_down_path_rel = f"{scene_id}/sample_{sample_idx:03d}/../top_down_view.png"
    target_path_rel = option_paths_rel[gt_idx_within_options]

    # view_N fields follow the order they appear in image_path
    def _detail_view(path_rel: str, pose: Dict) -> Dict:
        return {
            "path": path_rel,
            "c2w_extrinsics": unity_pose_to_c2w(pose).tolist(),
            "c2w_intrinsics": K_matrix.tolist(),
        }

    # -- forward_dynamics: init + top_down + 4 options  (GT image is at A/B/C/D)
    forward_row = {
        "scene_id": scene_id,
        "sample_id": meta["sample_id"],
        "prompt": _build_forward_prompt(
            cfg.step_translation, cfg.step_rotation_deg, meta["gt_action_seq_names"]
        ),
        "image_path": [init_path_rel, top_down_path_rel] + [
            f"{scene_id}/sample_{sample_idx:03d}/option_{i:03d}.png" for i in range(4)
        ],
        "image_detail": {
            "init_view":     _detail_view(init_path_rel, init_pose),
            "top_down_view": {
                "path": top_down_path_rel,
                "c2w_extrinsics": None,   # filled in by caller (scene-level)
                "c2w_intrinsics": K_matrix.tolist(),
            },
            **{f"view_{i}": _detail_view(
                f"{scene_id}/sample_{sample_idx:03d}/option_{i:03d}.png",
                option_poses[option_letters[i]],
            ) for i in range(4)},
        },
        "gt_answer": gt_answer_letter,
        "meta": {
            "step_translation_m": cfg.step_translation,
            "step_rotation_deg": cfg.step_rotation_deg,
            "gt_label": gt_answer_letter,
            "gt_action_seq_letters": meta["gt_action_seq_letters"],
            "gt_action_seq_names":   meta["gt_action_seq_names"],
            "option_action_seq_letters": option_action_seqs_letters,
            "option_action_seq_names":   option_action_seqs_names,
        },
    }

    # -- inverse_dynamics: init + top_down + target-only (pick target = GT option)
    inverse_row = {
        "scene_id": scene_id,
        "sample_id": meta["sample_id"],
        "prompt": _build_inverse_prompt(
            cfg.step_translation, cfg.step_rotation_deg, option_action_seqs_names
        ),
        "image_path": [init_path_rel, top_down_path_rel, target_path_rel],
        "image_detail": {
            "init_view":     _detail_view(init_path_rel, init_pose),
            "top_down_view": {
                "path": top_down_path_rel,
                "c2w_extrinsics": None,
                "c2w_intrinsics": K_matrix.tolist(),
            },
            "target_view":   _detail_view(target_path_rel, gt_pose),
        },
        "gt_answer": gt_answer_letter,
        "meta": {
            "used_only_gt": True,
            "step_translation_m": cfg.step_translation,
            "step_rotation_deg": cfg.step_rotation_deg,
            "gt_label": gt_answer_letter,
            "gt_action_seq_letters": meta["gt_action_seq_letters"],
            "gt_action_seq_names":   meta["gt_action_seq_names"],
            "option_action_seq_letters": option_action_seqs_letters,
            "option_action_seq_names":   option_action_seqs_names,
        },
    }

    # -- active_explore: prompt asks agent to reach target; target = GT option
    active_explore_row = {
        "scene_id": scene_id,
        "sample_id": meta["sample_id"],
        "prompt": _build_active_explore_prompt(),
        "image_path": [init_path_rel, top_down_path_rel, target_path_rel],
        "image_detail": {
            "init_view":     _detail_view(init_path_rel, init_pose),
            "top_down_view": {
                "path": top_down_path_rel,
                "c2w_extrinsics": None,
                "c2w_intrinsics": K_matrix.tolist(),
            },
            "target_view":   _detail_view(target_path_rel, gt_pose),
        },
        "gt_answer": {
            "pose_c2w": unity_pose_to_c2w(gt_pose).tolist(),
            "unity_pose": gt_pose,
        },
        "meta": {
            "step_translation_m": cfg.step_translation,
            "step_rotation_deg": cfg.step_rotation_deg,
            "gt_action_seq_letters": meta["gt_action_seq_letters"],
            "gt_action_seq_names":   meta["gt_action_seq_names"],
        },
    }

    return {
        "forward":          forward_row,
        "inverse":          inverse_row,
        "active_explore":   active_explore_row,
        "top_down_c2w":     None,   # caller fills in (scene-level)
    }


# =============================================================================
# Main driver
# =============================================================================
def run(
    out_root: str = GenConfig.out_root,
    scenes: str = GenConfig.scenes,
    samples_per_scene: int = GenConfig.samples_per_scene,
    width: int = GenConfig.width,
    height: int = GenConfig.height,
    fov: float = GenConfig.fov,
    step_translation: float = GenConfig.step_translation,
    step_rotation_deg: float = GenConfig.step_rotation_deg,
    seed: int = GenConfig.seed,
    gt_seq_min: int = GenConfig.gt_seq_min,
    gt_seq_max: int = GenConfig.gt_seq_max,
    top_down_height_y: float = GenConfig.top_down_height_y,
    # ---- Quality filter knobs ----
    inside_room_threshold_m: float = GenConfig.inside_room_threshold_m,
    min_camera_y: float = GenConfig.min_camera_y,
    max_camera_y: float = GenConfig.max_camera_y,
    max_resample_tries: int = GenConfig.max_resample_tries,
    min_image_std: float = GenConfig.min_image_std,
    max_dominant_pixel_frac: float = GenConfig.max_dominant_pixel_frac,
    gpu_id: int = 0,
):
    """Generate the AI2-THOR proxy-task dataset."""
    cfg = GenConfig(
        out_root=out_root, scenes=scenes, samples_per_scene=samples_per_scene,
        width=width, height=height, fov=fov,
        step_translation=step_translation, step_rotation_deg=step_rotation_deg,
        seed=seed, gt_seq_min=gt_seq_min, gt_seq_max=gt_seq_max,
        top_down_height_y=top_down_height_y,
        inside_room_threshold_m=inside_room_threshold_m,
        min_camera_y=min_camera_y,
        max_camera_y=max_camera_y,
        max_resample_tries=max_resample_tries,
        min_image_std=min_image_std,
        max_dominant_pixel_frac=max_dominant_pixel_frac,
    )
    os.makedirs(out_root, exist_ok=True)
    if gpu_id is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))

    scene_ids = parse_subset(cfg.scenes)
    print(f"[gen] scenes={len(scene_ids)} samples_per_scene={cfg.samples_per_scene}")

    K = intrinsics_from_fov(cfg.width, cfg.height, cfg.fov)

    # open output jsonls (overwrite). Filenames use the task-centric naming
    # (p2v / v2p / ivp) to match the ScanNet proxy-task datasets:
    #   forward_dynamics  -> path_to_view            (P2V)
    #   inverse_dynamics  -> view_to_path            (V2P)
    #   active_explore    -> interactive_view_planning (IVP)
    jsonl_paths = {
        "forward":        os.path.join(out_root, "path_to_view.jsonl"),
        "inverse":        os.path.join(out_root, "view_to_path.jsonl"),
        "active_explore": os.path.join(out_root, "interactive_view_planning.jsonl"),
    }
    jsonl_files = {k: open(v, "w") for k, v in jsonl_paths.items()}

    def _make_controller(scene):
        c = Controller(
            platform=CloudRendering,
            agentMode="default",
            scene=scene,
            width=cfg.width, height=cfg.height, fieldOfView=cfg.fov,
            renderDepthImage=False, renderInstanceSegmentation=False,
        )
        _ensure_third_party_camera(c, cfg.fov)
        return c

    def _safe_stop(c):
        try:
            c.stop()
        except Exception:
            pass  # Unity teardown often BrokenPipes; harmless

    def _gen_one_scene(controller, scene_id):
        """Generate one scene's rows into an in-memory buffer. Raises on Unity
        crash (caller restarts the controller and retries). Images are written
        to disk during sampling; the buffered rows are returned so the caller
        only commits them (and the top-down png) once the scene fully succeeds."""
        controller.reset(scene=scene_id)
        _ensure_third_party_camera(controller, cfg.fov)
        # Warmup frame: AI2THOR sometimes returns a stale/blank frame on the
        # first capture after reset+AddThirdPartyCamera. Render+discard once.
        controller.step(
            action="UpdateThirdPartyCamera", thirdPartyCameraId=0,
            position={"x": 0.0, "y": 1.0, "z": 0.0},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            fieldOfView=float(cfg.fov),
        )
        _ = controller.last_event.third_party_camera_frames
        scene_rng = random.Random((cfg.seed * 1_000_003) ^ hash(scene_id))
        reachable_xz = _reachable_xz(controller)
        td_rgb, td_pose = _render_topdown_mapview(controller)
        if td_rgb.shape[0] != cfg.height or td_rgb.shape[1] != cfg.width:
            td_rgb = np.asarray(
                Image.fromarray(td_rgb).resize((cfg.width, cfg.height), Image.BILINEAR)
            )
        scene_dir = os.path.join(out_root, scene_id)
        td_c2w = unity_pose_to_c2w(td_pose).tolist()

        buf = {"forward": [], "inverse": [], "active_explore": []}
        k = attempted = skipped = 0
        while k < cfg.samples_per_scene:
            attempted += 1
            if attempted > cfg.samples_per_scene * 5:
                break
            rows = None
            for _ in range(5):
                rows = _build_sample(
                    controller=controller, scene_id=scene_id, sample_idx=k,
                    scene_dir=scene_dir, out_root=out_root, cfg=cfg,
                    rng=scene_rng, K_matrix=K, reachable_xz=reachable_xz,
                )
                if rows is not None:
                    break
            if rows is None:
                skipped += 1
                continue
            for key in ("forward", "inverse", "active_explore"):
                rows[key]["image_detail"]["top_down_view"]["c2w_extrinsics"] = td_c2w
                buf[key].append(rows[key])
            k += 1
        # scene fully generated -> commit top-down png (rows reference it)
        _save_png(td_rgb, os.path.join(scene_dir, "top_down_view.png"))
        return buf, k, skipped

    controller = _make_controller(scene_ids[0])
    t_all = time.time()
    n_rows = 0
    try:
        for si, scene_id in enumerate(scene_ids, 1):
            ok = False
            for attempt in range(3):
                t_s = time.time()
                try:
                    buf, k, skipped = _gen_one_scene(controller, scene_id)
                    for key in ("forward", "inverse", "active_explore"):
                        for row in buf[key]:
                            jsonl_files[key].write(json.dumps(row) + "\n")
                            n_rows += 1
                        jsonl_files[key].flush()
                    print(f"  [{si}/{len(scene_ids)}] {scene_id}: kept {k}/{cfg.samples_per_scene} "
                          f"skipped {skipped}  in {time.time()-t_s:.1f}s  ({n_rows} rows total)")
                    ok = True
                    break
                except Exception as e:
                    print(f"  [warn] {scene_id} attempt {attempt+1}/3 crashed: "
                          f"{type(e).__name__}: {e}; restarting controller", flush=True)
                    _safe_stop(controller)
                    time.sleep(3)
                    controller = _make_controller(scene_id)
            if not ok:
                print(f"  [skip] {scene_id}: failed after 3 attempts", flush=True)
    finally:
        _safe_stop(controller)
        for f in jsonl_files.values():
            f.close()

    elapsed = time.time() - t_all
    print(f"[done] wrote {n_rows} rows ({n_rows//3} samples) in {elapsed:.1f}s -> {out_root}")


if __name__ == "__main__":
    import fire
    fire.Fire(run)
