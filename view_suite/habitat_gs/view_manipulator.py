"""Camera control for Habitat-GS, using Habitat's own action semantics.

Deliberately NOT a copy of the ScanNet or AI2-THOR manipulator. The three envs do not
agree on what an action means, and this one follows the simulator it runs on:

| | ScanNet | AI2-THOR | here (Habitat-GS) |
|---|---|---|---|
| turn  | camera-local +Y (`R_c2w @ R_y`) — tilts once pitched | world +Y | world +Y |
| pitch | camera-local X | camera-local X | camera-local X |
| roll  | in the action set | in the action set | **absent** |
| move  | full camera basis (pitch-coupled) | full camera basis | **horizontal** |

Habitat gets the last two rows from structure rather than from special-casing: an agent
is a *body* node with a *sensor* child (``habitat_sim/agent/controls/default_controls.py``).
``turn_left`` is ``rotate_y_local`` on the body, ``look_up`` is ``rotate_x_local`` on the
sensor, and ``move_forward`` is ``translate_local`` along the body's -Z. Because pitch
lives on the sensor, the body never tilts: its local +Y stays world-up, so turning is
ground-parallel and forward motion is horizontal, for free.

We reproduce that split with two angles instead of two scene nodes — the sensor sits at
the body origin, so the camera position *is* the body position:

    R_camera_gl = R_y(yaw) @ R_x(pitch)

Default action set matches Habitat's (``agent.py::_default_action_space`` plus the
strafe/look actions the proxy tasks need): no roll, and up/down are separate actions
rather than a consequence of pitching and moving forward.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# OpenCV camera (+X right, +Y down, +Z forward) <-> Habitat/OpenGL camera
# (+X right, +Y up, -Z forward). Self-inverse.
_CV_TO_GL_3 = np.diag([1.0, -1.0, -1.0])

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)

# Letters are the shared proxy-task vocabulary; the names are Habitat's own control
# names, which they already matched one-for-one.
ACTION_NAMES: Dict[str, str] = {
    "w": "move_forward", "s": "move_backward",
    "a": "move_left",    "d": "move_right",
    "q": "turn_left",    "e": "turn_right",
    "r": "look_up",      "f": "look_down",
    "y": "move_up",      "h": "move_down",
}
# What data generation samples from. Mirrors the other two envs (which also exclude
# up/down), and excludes roll because Habitat's default agent has no roll action.
ACTION_LETTERS = ["w", "s", "a", "d", "q", "e", "r", "f"]


def _deg2rad(d: float) -> float:
    return float(d) * np.pi / 180.0


def _snap(val: float, step: float) -> float:
    return step * round(val / step)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _R_y(deg: float) -> np.ndarray:
    c, s = np.cos(_deg2rad(deg)), np.sin(_deg2rad(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _R_x(deg: float) -> np.ndarray:
    c, s = np.cos(_deg2rad(deg)), np.sin(_deg2rad(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


class HabitatGSViewManipulator:
    """Body-yaw + sensor-pitch camera, matching Habitat's default agent."""

    def __init__(
        self,
        position=(0.0, 1.5, 0.0),
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        step_translation: float = 0.25,   # Habitat's default move amount
        step_rotation_deg: float = 30.0,  # the proxy tasks' step, not Habitat's 10
        pitch_limit_deg: float = 60.0,
        discrete: bool = True,
    ):
        self.pos = np.asarray(position, dtype=np.float64).copy()
        self.yaw = float(yaw_deg)
        self.pitch = float(pitch_deg)
        self.step_t = float(step_translation)
        self.step_r = float(step_rotation_deg)
        self.pitch_limit = float(pitch_limit_deg)
        self.is_discrete = bool(discrete)
        if self.is_discrete:
            self._snap_angles()

    # ── state ────────────────────────────────────────────────────────────────
    def _snap_angles(self) -> None:
        if not self.is_discrete:
            return
        self.yaw = _snap(self.yaw, self.step_r)
        self.pitch = _snap(self.pitch, self.step_r)

    def _body_basis(self):
        """Body frame: yaw only, so these stay in the horizontal plane."""
        Rb = _R_y(self.yaw)
        right = Rb @ np.array([1.0, 0.0, 0.0])
        fwd = Rb @ np.array([0.0, 0.0, -1.0])   # Habitat forward is -Z
        return right, fwd

    def rotation_gl(self) -> np.ndarray:
        """Camera rotation in Habitat/OpenGL convention."""
        return _R_y(self.yaw) @ _R_x(self.pitch)

    def get_c2w(self) -> np.ndarray:
        """4x4 OpenCV-style camera-to-world, the form every renderer here takes."""
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = self.rotation_gl() @ _CV_TO_GL_3
        c2w[:3, 3] = self.pos
        return c2w

    def set_c2w(self, c2w) -> None:
        """Inverse of get_c2w. Any roll in the input is dropped -- this agent cannot
        represent it, and silently keeping it would desynchronise pose from action."""
        c2w = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
        R_gl = c2w[:3, :3] @ _CV_TO_GL_3        # self-inverse
        fwd = R_gl @ np.array([0.0, 0.0, -1.0])
        self.pos = c2w[:3, 3].copy()
        # fwd = R_y(yaw) @ R_x(pitch) @ (0,0,-1) = (-sin y cos p, sin p, -cos y cos p),
        # so both components need negating. Getting this wrong costs a sign on yaw and
        # mirrors every pose that arrives through select_view / get_view -- which still
        # renders a perfectly plausible image.
        self.yaw = float(np.degrees(np.arctan2(-fwd[0], -fwd[2])))
        self.pitch = float(np.degrees(np.arcsin(_clamp(fwd[1], -1.0, 1.0))))
        self._snap_angles()

    # ── SE(3) interface, matching scannet.ViewManipulator ────────────────────
    # Same convention on purpose: [cx, cy, cz, rx, ry, rz] with intrinsic 'xyz' Euler
    # angles of the *c2w* rotation, degrees. The IVP answer format and the success
    # metric are written against this, so diverging here would silently change what
    # counts as a correct pose estimate.
    def get_se3(self, degrees: bool = True) -> np.ndarray:
        from scipy.spatial.transform import Rotation as _R
        c2w = self.get_c2w()
        eul = _R.from_matrix(c2w[:3, :3]).as_euler("xyz", degrees=degrees)
        return np.concatenate([c2w[:3, 3], eul])

    def set_se3(self, pose6, degrees: bool = True) -> None:
        """Any roll in the request is dropped -- this agent has no roll axis, and
        keeping it would put the pose out of sync with what the actions can reach."""
        from scipy.spatial.transform import Rotation as _R
        pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
        if pose6.shape[0] != 6:
            raise ValueError(f"pose6 must have shape (6,), got {pose6.shape}")
        c2w = np.eye(4)
        c2w[:3, :3] = _R.from_euler("xyz", pose6[3:], degrees=degrees).as_matrix()
        c2w[:3, 3] = pose6[:3]
        self.set_c2w(c2w)

    def reset(self, c2w=None) -> None:
        if c2w is None:
            self.pos[:] = 0.0
            self.yaw = self.pitch = 0.0
        else:
            self.set_c2w(c2w)

    def get_pose(self, mode: str = "c2w") -> np.ndarray:
        """4x4 camera pose, matching scannet.ViewManipulator's signature.

        Named and shaped to that interface on purpose: the shared IVP engine calls
        get_pose(mode="c2w") on whatever manipulator it is given. An identically-named
        method returning something else -- as a dict of angles did here -- type-errors
        on the first render, after reset has already succeeded.
        """
        c2w = self.get_c2w()
        if mode == "c2w":
            return c2w
        if mode == "w2c":
            return np.linalg.inv(c2w)
        raise ValueError(f"mode must be 'c2w' or 'w2c', got {mode!r}")

    def get_state(self) -> Dict:
        """The human-readable state: position plus the two angles this camera has."""
        return {
            "position": {"x": float(self.pos[0]), "y": float(self.pos[1]),
                         "z": float(self.pos[2])},
            "yaw": float(self.yaw),
            "pitch": float(self.pitch),
        }

    def set_state(self, pose: Dict) -> None:
        p = pose["position"]
        self.pos[:] = [float(p["x"]), float(p["y"]), float(p["z"])]
        self.yaw = float(pose.get("yaw", 0.0))
        self.pitch = float(pose.get("pitch", 0.0))
        self._snap_angles()

    # ── body actions (horizontal by construction) ────────────────────────────
    def move_forward(self, d: float) -> None:
        _, fwd = self._body_basis()
        self.pos += fwd * float(d)

    def move_right(self, d: float) -> None:
        right, _ = self._body_basis()
        self.pos += right * float(d)

    def turn_right(self) -> None:
        self.yaw -= self.step_r      # Habitat turn_right is a negative Y rotation
        self._snap_angles()

    def turn_left(self) -> None:
        self.yaw += self.step_r
        self._snap_angles()

    # ── sensor actions ───────────────────────────────────────────────────────
    def look_up(self) -> None:
        self.pitch = _clamp(self.pitch + self.step_r, -self.pitch_limit, self.pitch_limit)
        self._snap_angles()

    def look_down(self) -> None:
        self.pitch = _clamp(self.pitch - self.step_r, -self.pitch_limit, self.pitch_limit)
        self._snap_angles()

    def move_up(self, d: float) -> None:
        # Habitat's MoveUp is sensor-local +Y, i.e. it tilts with pitch. Kept faithful
        # even though the sampled action set does not use it.
        self.pos += (self.rotation_gl() @ np.array([0.0, 1.0, 0.0])) * float(d)

    # ── unified step ─────────────────────────────────────────────────────────
    def step(self, action: str) -> Dict:
        a = (action or "").strip().lower()
        if a == "w":
            self.move_forward(+self.step_t)
        elif a == "s":
            self.move_forward(-self.step_t)
        elif a == "a":
            self.move_right(-self.step_t)
        elif a == "d":
            self.move_right(+self.step_t)
        elif a == "q":
            self.turn_left()
        elif a == "e":
            self.turn_right()
        elif a == "r":
            self.look_up()
        elif a == "f":
            self.look_down()
        elif a == "y":
            self.move_up(+self.step_t)
        elif a == "h":
            self.move_up(-self.step_t)
        return self.get_state()

    def apply_sequence(self, actions) -> Dict:
        for a in actions:
            self.step(a)
        return self.get_state()
