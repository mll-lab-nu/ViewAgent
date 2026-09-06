"""Generate P2V / V2P / IVP samples from Habitat-GS 3DGS scenes.

Row schema matches the ScanNet and AI2-THOR generators exactly -- the P2V and V2P envs
are scene-agnostic JSONL readers and are reused unchanged -- but the sampling underneath
differs in three ways that this corpus forces:

**Poses are tied to the navmesh.** A 3DGS reconstruction is only faithful near the views
it was trained on; a camera in the middle of a wall renders smeared fog rather than
failing. Positions are therefore snapped to walkable space and raised to eye height, and
pitch is limited. This is necessary but not sufficient -- see the quality gate below.

**The translation step is per-scene.** The corpus spans two families whose scale differs
by an order of magnitude: `interior_*` rooms have a ~23 m navmesh diagonal, the `sceneNN`
scenes (many outdoor) have a ~90 m median and up to 538 m. A fixed 0.5 m step, which is
right for a room, moves 3 m across a 538 m plaza over a whole ground-truth path -- the
target view is then indistinguishable from the initial one and every sample is thrown
away by the degeneracy check. Scaling the step with the navmesh keeps the *apparent*
view change roughly constant, which is also the physically sensible thing: outdoor
content is far away, so it takes more translation to produce the same parallax. The step
is written into the prompt, as it already was, so it stays disclosed to the model.

**Every rendered view passes a quality gate.** Cheap image statistics first, then an
optional VLM pass (`filter_low_semantic.py`). The bar is *basic recognisability*: a view
must depict something a reader could identify and tell apart from other views. That one
criterion covers all three failure modes seen in this corpus -- off-manifold smear, a
blank wall or floor, and a frame that is simply too dark.

Actions come from ``HabitatGSViewManipulator``, the same object the IVP env drives, so
ground truth here and transitions there are the same function. (The AI2-THOR generator
and its env do not share one, and silently disagree once pitch is non-zero.)
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from view_suite.habitat_gs.habitat_gs_render import HabitatGSRenderer
from view_suite.habitat_gs.pose_utils import (
    intrinsics_from_fov, pose_distance, top_down_c2w,
)
from view_suite.habitat_gs.scene_list import (
    default_root, scene_navmesh, scene_ply, split_of,
)
from view_suite.habitat_gs.view_manipulator import (
    ACTION_LETTERS, ACTION_NAMES, HabitatGSViewManipulator,
)

OPTION_LETTERS = ["A", "B", "C", "D"]


@dataclass
class GenConfig:
    out_root: str = "data/viewagent15k_habitat_gs"
    root: str = field(default_factory=default_root)
    samples_per_scene: int = 24
    width: int = 512
    height: int = 512
    fov: float = 90.0
    # --- camera placement ---
    eye_height_m: float = 1.5           # above the walkable surface
    pitch_limit_deg: float = 60.0
    step_rotation_deg: float = 30.0
    # --- per-scene translation step ---
    # step = clamp(navmesh_diagonal / step_scene_divisor, lo, hi). The divisor is set so
    # a median room (23 m diagonal) lands on 0.5 m, the value the other two envs use.
    step_scene_divisor: float = 46.0
    step_translation_min_m: float = 0.25
    step_translation_max_m: float = 8.0
    # --- ground-truth path ---
    gt_seq_min: int = 2
    gt_seq_max: int = 6
    num_distractors: int = 3
    # --- rejection ---
    max_resample_tries: int = 30
    min_image_std: float = 8.0          # flat frame => nothing to recognise
    min_image_mean: float = 12.0        # near-black
    max_dominant_pixel_frac: float = 0.8
    degen_pos_thr_frac: float = 1.0     # x step_translation
    degen_ang_thr: float = 30.0
    seed: int = 0


# =============================================================================
# Image quality -- the cheap half of the gate
# =============================================================================
def _dominant_pixel_fraction(img: np.ndarray, bits: int = 3) -> float:
    """Fraction of pixels sharing a coarse colour bucket.

    A blank wall, a floor, or an off-manifold smear all collapse into very few
    buckets; a recognisable view does not.
    """
    q = (np.asarray(img)[:, :, :3] >> (8 - bits)).astype(np.uint32)
    keys = (q[:, :, 0] << (2 * bits)) | (q[:, :, 1] << bits) | q[:, :, 2]
    counts = np.bincount(keys.ravel())
    return float(counts.max()) / float(keys.size)


def image_is_usable(img: np.ndarray, cfg: GenConfig) -> Tuple[bool, str]:
    a = np.asarray(img)
    if float(a.std()) < cfg.min_image_std:
        return False, "flat"
    if float(a.mean()) < cfg.min_image_mean:
        return False, "dark"
    if _dominant_pixel_fraction(a) > cfg.max_dominant_pixel_frac:
        return False, "dominant_colour"
    return True, ""


# =============================================================================
# Scene scale -> action step
# =============================================================================
def scene_step_translation(renderer: HabitatGSRenderer, cfg: GenConfig) -> float:
    pf = renderer.pathfinder
    if pf is None or not pf.is_loaded:
        return 0.5
    lo, hi = pf.get_bounds()
    ext = np.asarray(hi) - np.asarray(lo)
    diag = float(np.hypot(ext[0], ext[2]))
    step = diag / float(cfg.step_scene_divisor)
    # Rounded: the value is quoted in the prompt and stored in every row, and a
    # 15-digit float there is noise.
    return round(float(np.clip(step, cfg.step_translation_min_m,
                               cfg.step_translation_max_m)), 2)


# =============================================================================
# Sampling
# =============================================================================
def _sample_pose(renderer: HabitatGSRenderer, rng: random.Random, cfg: GenConfig,
                 step_t: float) -> Optional[HabitatGSViewManipulator]:
    p = renderer.sample_navigable_point()
    if p is None:
        return None
    return HabitatGSViewManipulator(
        position=(float(p[0]), float(p[1]) + cfg.eye_height_m, float(p[2])),
        yaw_deg=float(rng.choice(range(0, 360, int(cfg.step_rotation_deg)))),
        pitch_deg=0.0,
        step_translation=step_t,
        step_rotation_deg=cfg.step_rotation_deg,
        pitch_limit_deg=cfg.pitch_limit_deg,
        discrete=True,
    )


def _sample_action_seq(rng: random.Random, length: int) -> List[str]:
    return [rng.choice(ACTION_LETTERS) for _ in range(length)]


def _seq_equal(a: Sequence[str], b: Sequence[str]) -> bool:
    return list(a) == list(b)


def _apply(vm_state: HabitatGSViewManipulator, seq: Sequence[str], step_t: float,
           cfg: GenConfig) -> HabitatGSViewManipulator:
    vm = HabitatGSViewManipulator(
        position=tuple(vm_state.pos), yaw_deg=vm_state.yaw, pitch_deg=vm_state.pitch,
        step_translation=step_t, step_rotation_deg=cfg.step_rotation_deg,
        pitch_limit_deg=cfg.pitch_limit_deg, discrete=True,
    )
    for a in seq:
        vm.step(a)
    return vm


def _pose_ok(renderer: HabitatGSRenderer, vm: HabitatGSViewManipulator,
             cfg: GenConfig, tol_m: float) -> bool:
    """Reject a camera that has wandered off walkable space.

    Checked in the horizontal plane only: the camera is deliberately above the floor,
    so is_navigable() at the camera position is always false.
    """
    ground = np.array([vm.pos[0], vm.pos[1] - cfg.eye_height_m, vm.pos[2]])
    snapped = renderer.snap_to_navmesh(ground)
    if snapped is None:
        return False
    return float(np.hypot(snapped[0] - ground[0], snapped[2] - ground[2])) <= tol_m


# =============================================================================
# Per-sample generation
# =============================================================================
def build_sample(renderer: HabitatGSRenderer, scene_id: str, sample_idx: int,
                 rng: random.Random, cfg: GenConfig, step_t: float,
                 K: np.ndarray, out_dir: str, top_down_rel: str,
                 top_down_c2w_mat: np.ndarray) -> Optional[Dict]:
    off_navmesh_tol = max(1.0, 2.0 * step_t)

    for _ in range(cfg.max_resample_tries):
        init = _sample_pose(renderer, rng, cfg, step_t)
        if init is None:
            return None
        init_img = renderer.render_image_from_cam_param(K, init.get_c2w(),
                                                        cfg.width, cfg.height)
        ok, _why = image_is_usable(init_img, cfg)
        if not ok:
            continue

        gt_seq = _sample_action_seq(rng, rng.randint(cfg.gt_seq_min, cfg.gt_seq_max))
        gt_vm = _apply(init, gt_seq, step_t, cfg)
        if not _pose_ok(renderer, gt_vm, cfg, off_navmesh_tol):
            continue
        # Trivially-solved sample: the target is already where the agent starts.
        d_pos, d_ang = pose_distance(init.get_c2w(), gt_vm.get_c2w())
        if d_pos <= cfg.degen_pos_thr_frac * step_t and d_ang <= cfg.degen_ang_thr:
            continue
        gt_img = renderer.render_image_from_cam_param(K, gt_vm.get_c2w(),
                                                      cfg.width, cfg.height)
        ok, _why = image_is_usable(gt_img, cfg)
        if not ok:
            continue

        # Distractors: different action sequences, each landing somewhere usable.
        seqs, vms, imgs = [gt_seq], [gt_vm], [gt_img]
        tries = 0
        while len(seqs) < 1 + cfg.num_distractors and tries < cfg.max_resample_tries:
            tries += 1
            cand = _sample_action_seq(rng, rng.randint(cfg.gt_seq_min, cfg.gt_seq_max))
            if any(_seq_equal(cand, s) for s in seqs):
                continue
            cand_vm = _apply(init, cand, step_t, cfg)
            if not _pose_ok(renderer, cand_vm, cfg, off_navmesh_tol):
                continue
            # Must be visibly distinct from the target, or the MCQ has two right answers.
            dp, da = pose_distance(gt_vm.get_c2w(), cand_vm.get_c2w())
            if dp <= cfg.degen_pos_thr_frac * step_t and da <= cfg.degen_ang_thr:
                continue
            cand_img = renderer.render_image_from_cam_param(K, cand_vm.get_c2w(),
                                                            cfg.width, cfg.height)
            ok, _why = image_is_usable(cand_img, cfg)
            if not ok:
                continue
            seqs.append(cand)
            vms.append(cand_vm)
            imgs.append(cand_img)

        if len(seqs) < 1 + cfg.num_distractors:
            continue

        order = list(range(len(seqs)))
        rng.shuffle(order)
        gt_slot = order.index(0)
        gt_letter = OPTION_LETTERS[gt_slot]

        sample_dir = os.path.join(out_dir, scene_id, f"sample_{sample_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        init_rel = f"{scene_id}/sample_{sample_idx:03d}/init.png"
        Image.fromarray(init_img).save(os.path.join(out_dir, init_rel))
        opt_rels = []
        for slot, src in enumerate(order):
            rel = f"{scene_id}/sample_{sample_idx:03d}/option_{slot:03d}.png"
            Image.fromarray(imgs[src]).save(os.path.join(out_dir, rel))
            opt_rels.append(rel)
        target_rel = opt_rels[gt_slot]

        Kl = K.tolist()

        def detail(rel: str, vm: HabitatGSViewManipulator) -> Dict:
            return {"path": rel, "c2w_extrinsics": vm.get_c2w().tolist(),
                    "c2w_intrinsics": Kl}

        seq_names = {OPTION_LETTERS[slot]: [ACTION_NAMES.get(a, a) for a in seqs[src]]
                     for slot, src in enumerate(order)}
        seq_letters = {OPTION_LETTERS[slot]: list(seqs[src])
                       for slot, src in enumerate(order)}
        common_meta = {
            "step_translation_m": step_t,
            "step_rotation_deg": cfg.step_rotation_deg,
            # Recorded so the env can adopt it: the ground truth was built with this
            # limit, and an env that lets the camera pitch further is a different task.
            "pitch_limit_deg": cfg.pitch_limit_deg,
            "gt_label": gt_letter,
            "gt_action_seq_letters": list(gt_seq),
            "gt_action_seq_names": [ACTION_NAMES.get(a, a) for a in gt_seq],
            "option_action_seq_letters": seq_letters,
            "option_action_seq_names": seq_names,
        }
        top_down_detail = {"path": top_down_rel,
                           "c2w_extrinsics": top_down_c2w_mat.tolist(),
                           "c2w_intrinsics": Kl}
        sample_id = f"{scene_id}_sample_{sample_idx}"

        forward_row = {
            "scene_id": scene_id, "sample_id": sample_id,
            "prompt": _forward_prompt(step_t, cfg.step_rotation_deg,
                                      common_meta["gt_action_seq_names"]),
            "image_path": [init_rel, top_down_rel] + opt_rels,
            "image_detail": {
                "init_view": detail(init_rel, init),
                "top_down_view": top_down_detail,
                **{f"view_{slot}": detail(opt_rels[slot], vms[order[slot]])
                   for slot in range(len(order))},
            },
            "gt_answer": gt_letter,
            "meta": dict(common_meta),
        }
        inverse_row = {
            "scene_id": scene_id, "sample_id": sample_id,
            "prompt": _inverse_prompt(step_t, cfg.step_rotation_deg, seq_names),
            "image_path": [init_rel, top_down_rel, target_rel],
            "image_detail": {
                "init_view": detail(init_rel, init),
                "top_down_view": top_down_detail,
                "target_view": detail(target_rel, gt_vm),
            },
            "gt_answer": gt_letter,
            "meta": {"used_only_gt": True, **common_meta},
        }
        active_row = {
            "scene_id": scene_id, "sample_id": sample_id,
            "prompt": _active_prompt(),
            "image_path": [init_rel, top_down_rel, target_rel],
            "image_detail": {
                "init_view": detail(init_rel, init),
                "top_down_view": top_down_detail,
                "target_view": detail(target_rel, gt_vm),
            },
            "gt_answer": {"pose_c2w": gt_vm.get_c2w().tolist(),
                          "gs_pose": gt_vm.get_state()},
            "meta": {k: common_meta[k] for k in
                     ("step_translation_m", "step_rotation_deg",
                      "gt_action_seq_letters", "gt_action_seq_names")},
        }
        return {"forward": forward_row, "inverse": inverse_row,
                "active_explore": active_row}

    return None


def _forward_prompt(step_t: float, step_r: float, names: List[str]) -> str:
    return (
        f"Given the initial view <image> and a top-down reference <image>, "
        f"after you execute the following action sequence "
        f"(translation step = {step_t} m; rotation step = {step_r} degrees per step):\n"
        f"[{', '.join(names)}]\n"
        f"which of the following images corresponds to the result?\n"
        f"A. <image>\nB. <image>\nC. <image>\nD. <image>\n"
    )


def _inverse_prompt(step_t: float, step_r: float, names: Dict[str, List[str]]) -> str:
    lines = [
        "Given the initial view <image> and a top-down reference <image>, "
        "which action sequence will reach the target view <image>?",
        f"(Action semantics: translation step = {step_t} m; "
        f"rotation step = {step_r} degrees per step.)",
    ]
    for letter in OPTION_LETTERS:
        lines.append(f"{letter}. [{', '.join(names[letter])}]")
    return "\n".join(lines) + "\n"


def _active_prompt() -> str:
    return (
        "Given the initial view <image> and a top-down reference <image>, "
        "estimate the target view's 6-DoF pose relative to the world."
    )


# =============================================================================
# Driver
# =============================================================================
def run_scene(scene_id: str, cfg: GenConfig, gpu: int = 0) -> Dict[str, List[Dict]]:
    split = split_of(cfg.root, scene_id)
    if split is None:
        raise FileNotFoundError(f"scene {scene_id!r} not under {cfg.root}")
    # Images sit directly under out_root, one directory per scene: that is what
    # resolve_rel_image() assumes when a config gives no dataset_root, and it is the
    # layout the AI2-THOR corpus uses. An extra images/ level makes every lookup miss,
    # and safe_open_rgb turns a miss into None rather than an error.
    out_dir = cfg.out_root
    os.makedirs(out_dir, exist_ok=True)

    K = intrinsics_from_fov(cfg.width, cfg.height, cfg.fov)
    renderer = HabitatGSRenderer(
        scene_ply(cfg.root, scene_id, split), gpu_device_id=gpu,
        width=cfg.width, height=cfg.height,
        navmesh_path=scene_navmesh(cfg.root, scene_id, split),
    )
    rows: Dict[str, List[Dict]] = {"forward": [], "inverse": [], "active_explore": []}
    try:
        step_t = scene_step_translation(renderer, cfg)
        pf = renderer.pathfinder
        lo, hi = pf.get_bounds()
        centre = (np.asarray(lo) + np.asarray(hi)) / 2.0
        # High enough to see the whole footprint at the configured FOV.
        extent = float(max(np.asarray(hi)[0] - np.asarray(lo)[0],
                           np.asarray(hi)[2] - np.asarray(lo)[2]))
        td_c2w = top_down_c2w(centre, height_y=0.75 * extent)
        td_rel = f"{scene_id}/top_down.png"
        os.makedirs(os.path.join(out_dir, scene_id), exist_ok=True)
        td_img = renderer.render_image_from_cam_param(K, td_c2w, cfg.width, cfg.height)
        Image.fromarray(td_img).save(os.path.join(out_dir, td_rel))

        rng = random.Random(f"{cfg.seed}:{scene_id}")
        for i in range(cfg.samples_per_scene):
            s = build_sample(renderer, scene_id, i, rng, cfg, step_t, K,
                             out_dir, td_rel, td_c2w)
            if s is None:
                continue
            for k in rows:
                rows[k].append(s[k])
    finally:
        renderer.close()
    return rows


def write_jsonl(rows: Dict[str, List[Dict]], out_root: str, suffix: str = "") -> None:
    names = {"forward": "path_to_view", "inverse": "view_to_path",
             "active_explore": "interactive_view_planning"}
    os.makedirs(out_root, exist_ok=True)
    for key, base in names.items():
        path = os.path.join(out_root, f"{base}{suffix}.jsonl")
        with open(path, "a") as f:
            for r in rows[key]:
                f.write(json.dumps(r) + "\n")
