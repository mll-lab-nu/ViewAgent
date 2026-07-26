"""
Pose/intrinsics conversion between ScanNet-style OpenCV convention
(used by our metadata + VLM-facing views) and AI2-THOR / Unity format
(used by the ai2thor.controller render service).

ScanNet / OpenCV convention:
  - World frame: arbitrary
  - Camera frame: +X right, +Y down, +Z forward (into scene)
  - Extrinsics: 4x4 c2w
  - Intrinsics: 3x3 K  (fx 0 cx; 0 fy cy; 0 0 1)

AI2-THOR / Unity convention:
  - World frame: Y up, right-handed (as used in our ViewManipulator)
  - Camera / ThirdPartyCamera rotation is given in Euler degrees
    {x: pitch-ish, y: yaw, z: roll}, where x positive means look DOWN
    (see view_manipulator.py get_pose_thor()).
  - AI2-THOR pinhole is determined by width/height + fieldOfView (horizontal FOV).

This module supports round-trip conversions:
    unity_pose <-> (c2w 4x4, K 3x3)

so that data_gen and the runtime env can store ScanNet-style metadata while
still feeding the AI2-THOR renderer its native pose dicts.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def _rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


# Rotation matrices in Unity convention (Y-up, left-handed-ish — we follow
# ViewManipulator's construction which already works end-to-end with AI2THOR).
# Internal canonical (yaw, pitch, roll) with +yaw=turn-right, +pitch=look-up.
# Unity render uses unity_x = -pitch, unity_y = yaw, unity_z = -roll.


def _R_yaw_pitch_roll(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from intuitive yaw/pitch/roll (degrees).

    Camera local axes: +X right, +Y up, +Z forward (view_manipulator convention).
    Returns R such that world_vec = R @ cam_vec_local (same as view_manipulator._basis).
    """
    y, p, r = _deg2rad(yaw_deg), _deg2rad(pitch_deg), _deg2rad(roll_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    Ry = np.array([[ cy, 0.0,  sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0,  cy]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cp, -sp],
                   [0.0,  sp,  cp]], dtype=np.float64)
    Rz = np.array([[ cr, -sr, 0.0],
                   [ sr,  cr, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    return Ry @ Rx @ Rz


# OpenCV camera basis: +X right, +Y DOWN, +Z forward.
# ViewManipulator local camera basis: +X right, +Y UP, +Z forward.
# So the mapping from (our intuitive) camera frame -> OpenCV camera frame is a
# sign flip on Y and Z (flip Y AND flip Z so that Z stays forward while Y
# becomes down). Matrix form:
_M_INT_TO_CV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)  # 3x3


def unity_pose_to_c2w(pose: Dict) -> np.ndarray:
    """Convert AI2-THOR pose dict to 4x4 c2w in OpenCV convention.

    pose: {"position": {"x","y","z"}, "rotation": {"x","y","z"}}  (degrees)
    """
    pos = pose["position"]
    rot = pose["rotation"]
    # inverse of ViewManipulator.get_pose_thor():
    pitch = -float(rot["x"])
    yaw   =  float(rot["y"])
    roll  = -float(rot["z"])
    R_intuitive = _R_yaw_pitch_roll(yaw, pitch, roll)  # camera-frame (Y-up) -> world
    # Camera axis relabeling: OpenCV cam -> intuitive cam is diag(1,-1,-1).
    # c2w in OpenCV = R_intuitive @ _M_INT_TO_CV
    R_c2w = R_intuitive @ _M_INT_TO_CV

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_c2w
    T[:3, 3] = [float(pos["x"]), float(pos["y"]), float(pos["z"])]
    return T


def c2w_to_unity_pose(c2w: np.ndarray) -> Dict:
    """Convert 4x4 c2w (OpenCV convention) back to AI2-THOR pose dict.

    Recovers (yaw, pitch, roll) from the intuitive rotation matrix and flips to
    Unity Euler (unity_x = -pitch, unity_z = -roll).
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    assert c2w.shape == (4, 4), c2w.shape
    R_intuitive = c2w[:3, :3] @ _M_INT_TO_CV.T  # since _M_INT_TO_CV is its own inverse

    # Decompose R_intuitive = Ry(yaw) * Rx(pitch) * Rz(roll) following the
    # same construction as _R_yaw_pitch_roll.
    # Column layout:
    #   R[:, 0] = right (world)
    #   R[:, 1] = up    (world)
    #   R[:, 2] = fwd   (world)
    # Using the ZYX-ish decomposition for this specific basis:
    #   pitch = asin( R[1, 2] )   (sin pitch = up.y? no — derive explicitly)
    # Derivation from Ry*Rx*Rz:
    #   R[0,1] = sy*sp*cr - cy*sr + 0? ...
    # Easier: compute yaw from forward-X and forward-Z after zeroing pitch.
    R = R_intuitive
    # Forward in world (3rd column of R_intuitive) under _R_yaw_pitch_roll is
    # fwd = [sin(yaw)*cos(pitch), -sin(pitch), cos(yaw)*cos(pitch)]
    # so pitch = -asin(fwd[1])  (NOT +asin(fwd[1]) — the negative is required
    # because Rx(pitch) maps local +Z -> [0, -sin(pitch), cos(pitch)] ).
    fwd = R[:, 2]
    pitch = -math.degrees(math.asin(max(-1.0, min(1.0, float(fwd[1])))))
    yaw = math.degrees(math.atan2(float(fwd[0]), float(fwd[2])))
    # roll: from up vector once pitch/yaw are known
    # up_cam after undoing yaw/pitch should be (sin(-roll), cos(roll), 0)
    R_ypi = _R_yaw_pitch_roll(yaw, pitch, 0.0)
    # up residual = R_ypi.T @ R_intuitive @ [0,1,0]
    up_res = R_ypi.T @ (R_intuitive @ np.array([0.0, 1.0, 0.0]))
    roll = math.degrees(math.atan2(-float(up_res[0]), float(up_res[1])))

    return {
        "position": {"x": float(c2w[0, 3]), "y": float(c2w[1, 3]), "z": float(c2w[2, 3])},
        "rotation": {"x": float(-pitch), "y": float(yaw), "z": float(-roll)},
    }


def intrinsics_from_fov(width: int, height: int, fov_x_deg: float) -> np.ndarray:
    """
    OpenCV 3x3 K for a centered pinhole matching AI2THOR's fieldOfView
    (which is the HORIZONTAL FOV in degrees).

    fx = (width / 2) / tan(fov_x / 2)
    fy = fx                                  (square pixels)
    cx = width / 2
    cy = height / 2
    """
    w = float(width)
    h = float(height)
    fx = (w / 2.0) / math.tan(_deg2rad(fov_x_deg) / 2.0)
    fy = fx
    K = np.array([[fx, 0.0, w / 2.0],
                  [0.0, fy, h / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def fov_from_K(K: np.ndarray, width: int) -> float:
    """Recover horizontal FOV (degrees) from K (OpenCV), given target width."""
    fx = float(np.asarray(K)[0, 0])
    return _rad2deg(2.0 * math.atan((float(width) / 2.0) / fx))


def build_render_task(
    c2w: np.ndarray,
    K: np.ndarray | None = None,
    *,
    width: int = 512,
    height: int = 512,
    fov: float | None = None,
) -> Dict:
    """Assemble a single task dict for AI2ThorRenderHandler.

    If K is given (and fov is None), derive fov from K+width. Otherwise use the
    supplied fov (defaults to 90 if neither is provided).
    """
    if fov is None:
        if K is None:
            fov = 90.0
        else:
            fov = fov_from_K(K, width)
    pose = c2w_to_unity_pose(np.asarray(c2w, dtype=np.float64))
    return {"pose": pose, "width": int(width), "height": int(height), "fov": float(fov)}


def top_down_c2w(position_xyz: Tuple[float, float, float], *, height_y: float = 5.0) -> np.ndarray:
    """Build a top-down c2w: camera at (x, height_y, z) looking straight down (-Y).

    OpenCV convention: forward = +Z_cam must equal -Y_world (pointing down), so
    pitch = -90 (look down 90 deg) with yaw/roll = 0 in our intuitive frame.
    """
    x, _y, z = position_xyz
    pose = {
        "position": {"x": float(x), "y": float(height_y), "z": float(z)},
        "rotation": {"x": 90.0, "y": 0.0, "z": 0.0},  # unity_x=90 -> pitch=-90
    }
    return unity_pose_to_c2w(pose)
