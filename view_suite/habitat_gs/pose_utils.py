"""Pose and intrinsics helpers for Habitat-GS.

Much smaller than the AI2-THOR equivalent, which needs a Unity<->OpenCV conversion in
both directions. Habitat-GS is fed OpenCV (K, c2w) directly, so nothing here converts
between frames -- these are only the conveniences the task code needs.

World convention: Y-up, forward is -Z (Habitat's own).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def intrinsics_from_fov(width: int, height: int, fov_x_deg: float = 90.0) -> np.ndarray:
    """Pinhole K from a horizontal FOV, square pixels.

    Square by construction: Habitat derives the vertical FOV from the framebuffer
    aspect and cannot express fx != fy, so anything else would be silently resampled
    by the renderer.
    """
    fx = 0.5 * float(width) / math.tan(math.radians(float(fov_x_deg)) / 2.0)
    return np.array([
        [fx, 0.0, float(width) / 2.0],
        [0.0, fx, float(height) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def fov_from_K(K: np.ndarray, width: int) -> float:
    K = np.asarray(K, dtype=np.float64)
    return float(np.degrees(2.0 * math.atan(0.5 * float(width) / K[0, 0])))


def c2w_from_yaw_pitch(position, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """OpenCV c2w for a Habitat-style camera: yaw about world +Y, then pitch."""
    from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator
    return HabitatGSViewManipulator(
        position=position, yaw_deg=yaw_deg, pitch_deg=pitch_deg, discrete=False
    ).get_c2w()


def top_down_c2w(center_xyz, height_y: float = 8.0) -> np.ndarray:
    """Camera above `center_xyz` looking straight down, used for the V2P reference map.

    Straight down is the gimbal-locked orientation, so the remaining degree of freedom
    (which way is "up" in the image) is fixed here rather than left to whatever a
    decomposition happens to return: image-up is world -Z, so the map is oriented the
    same way in every scene and the reference is comparable across samples.
    """
    c = np.asarray(center_xyz, dtype=np.float64).reshape(3)
    # OpenCV camera axes expressed in world: +X right, +Y down, +Z forward(view dir).
    x_axis = np.array([1.0, 0.0, 0.0])    # image right  -> world +X
    z_axis = np.array([0.0, -1.0, 0.0])   # view dir     -> world -Y (straight down)
    y_axis = np.cross(z_axis, x_axis)     # image down   -> world -Z
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = x_axis, y_axis, z_axis
    c2w[:3, 3] = [c[0], c[1] + float(height_y), c[2]]
    return c2w


def pose_distance(c2w_a, c2w_b) -> Tuple[float, float]:
    """(metres, degrees) between two camera poses."""
    a = np.asarray(c2w_a, dtype=np.float64).reshape(4, 4)
    b = np.asarray(c2w_b, dtype=np.float64).reshape(4, 4)
    d_pos = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    R = a[:3, :3].T @ b[:3, :3]
    cos = (np.trace(R) - 1.0) / 2.0
    d_ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return d_pos, d_ang
