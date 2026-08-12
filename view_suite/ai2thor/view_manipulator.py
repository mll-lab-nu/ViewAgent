import os
from datetime import datetime

import numpy as np
import cv2

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _snap(val: float, step: float) -> float:
    return step * round(val / step)


def _deg2rad(d):
    return d * np.pi / 180.0


def _rot_yaw_pitch_roll_matrix(yaw_deg, pitch_deg, roll_deg):
    """
    Intuitive rotation basis (Unity world: Y-up):
      - yaw   : around WORLD +Y (yaw increases -> turn right)
      - pitch : around CAMERA local +X (pitch increases -> look up)
      - roll  : around CAMERA local +Z

    Local camera axes used for movement:
      +X right, +Y up, +Z forward
    """
    y = _deg2rad(yaw_deg)
    p = _deg2rad(pitch_deg)
    r = _deg2rad(roll_deg)

    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)

    # yaw about world Y
    R_y = np.array([
        [ cy, 0.0,  sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0,  cy],
    ], dtype=np.float64)

    # pitch about local X
    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  cp, -sp],
        [0.0,  sp,  cp],
    ], dtype=np.float64)

    # roll about local Z
    R_z = np.array([
        [ cr, -sr, 0.0],
        [ sr,  cr, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return R_y @ R_x @ R_z


class ViewManipulator:
    """
    Controller-free, intuitive ViewManipulator for AI2-THOR ThirdPartyCamera pose.

    Internal canonical state (human-readable):
      - position: world xyz
      - yaw/pitch/roll: degrees
        yaw   : + -> turn right
        pitch : + -> look up
        roll  : + -> CCW (our internal convention)

    Output pose format (AI2-THOR acceptable):
      {"position": {"x":..,"y":..,"z":..},
       "rotation": {"x":..,"y":..,"z":..}}
    """

    def __init__(
        self,
        init_pose: dict | None,
        step_translation: float = 0.25,
        step_rotation_deg: float = 30.0,
        pitch_limit_deg: float = 89.0,
        roll_enabled: bool = True,
        is_discrete: bool = False,
    ):
        self.step_t = float(step_translation)
        self.step_r = float(step_rotation_deg)

        self.pitch_limit = float(pitch_limit_deg)
        self.roll_enabled = bool(roll_enabled)
        self.is_discrete = bool(is_discrete)

        # canonical pose (intuitive)
        self.pos = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        if init_pose is not None:
            self.set_pose_thor(init_pose)
        if self.is_discrete:
            self._snap_angles()

    # ---------------- basis ----------------
    def _basis(self):
        Rm = _rot_yaw_pitch_roll_matrix(self.yaw, self.pitch, self.roll)
        right = Rm @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = Rm @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        fwd = Rm @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return right, up, fwd

    # ---------------- snapping ----------------
    def _snap_angles(self):
        if not self.is_discrete:
            return
        self.yaw = _snap(self.yaw, self.step_r)
        self.pitch = _snap(self.pitch, self.step_r)
        self.roll = _snap(self.roll, self.step_r)

    # ---------------- I/O pose ----------------
    def get_pose_thor(self) -> dict:
        """
        Map our intuitive angles to Unity rotation dict.

        Empirically, THOR/Unity camera X positive tends to look DOWN.
        We want pitch positive = look UP, so unity_x = -pitch.

        yaw: unity_y = yaw  (yaw positive -> turn right)
        roll: unity_z = -roll to keep "roll + = CCW" more intuitive for image content
        """
        unity_x = -self.pitch
        unity_y = self.yaw
        unity_z = -self.roll

        return {
            "position": {"x": float(self.pos[0]), "y": float(self.pos[1]), "z": float(self.pos[2])},
            "rotation": {"x": float(unity_x), "y": float(unity_y), "z": float(unity_z)},
        }

    def set_pose_thor(self, pose: dict):
        """
        Initialize from THOR pose dict (position/rotation degrees).

        Inverse mapping of get_pose_thor():
          pitch = -unity_x
          yaw   = unity_y
          roll  = -unity_z
        """
        p = pose["position"]
        r = pose["rotation"]

        self.pos[:] = [float(p["x"]), float(p["y"]), float(p["z"])]
        self.pitch = -float(r["x"])
        self.yaw = float(r["y"])
        self.roll = -float(r["z"])

        if self.is_discrete:
            self._snap_angles()

    # ---------------- movement ----------------
    def move_forward(self, d):
        _, _, fwd = self._basis()
        self.pos += fwd * float(d)

    def move_right(self, d):
        right, _, _ = self._basis()
        self.pos += right * float(d)

    def move_up(self, d):
        _, up, _ = self._basis()
        self.pos += up * float(d)

    # ---------------- rotation ----------------
    def yaw_left(self):
        self.yaw -= self.step_r
        self._snap_angles()

    def yaw_right(self):
        self.yaw += self.step_r
        self._snap_angles()

    def look_up(self):
        self.pitch = _clamp(self.pitch + self.step_r, -self.pitch_limit, self.pitch_limit)
        self._snap_angles()

    def look_down(self):
        self.pitch = _clamp(self.pitch - self.step_r, -self.pitch_limit, self.pitch_limit)
        self._snap_angles()

    def roll_ccw(self):
        if self.roll_enabled:
            self.roll += self.step_r
            self._snap_angles()

    def roll_cw(self):
        if self.roll_enabled:
            self.roll -= self.step_r
            self._snap_angles()

    # ---------------- unified step ----------------
    def step(self, action: str) -> dict:
        a = action.strip().lower()
        if not a:
            return self.get_pose_thor()

        # translation
        if a == "w":
            self.move_forward(+self.step_t)
        elif a == "s":
            self.move_forward(-self.step_t)
        elif a == "a":
            self.move_right(-self.step_t)
        elif a == "d":
            self.move_right(+self.step_t)
        elif a == "y":
            self.move_up(+self.step_t)
        elif a == "h":
            self.move_up(-self.step_t)

        # rotation
        elif a == "q":
            self.yaw_left()
        elif a == "e":
            self.yaw_right()
        elif a == "r":
            self.look_up()
        elif a == "f":
            self.look_down()
        elif a == "t":
            self.roll_ccw()
        elif a == "g":
            self.roll_cw()
        else:
            raise ValueError(f"Unsupported action: {action}")

        return self.get_pose_thor()


# --------------------------
# Test with AI2-THOR
# --------------------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _save_rgb(rgb, path):
    # rgb: HxWx3 uint8 RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def main():
    out_dir = "thor_vm_test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    _ensure_dir(out_dir)

    # 1) init controller (you control FOV etc here)
    controller = Controller(
        platform=CloudRendering,
        agentMode="default",
        scene="FloorPlan319",
        width=512,
        height=512,
        fieldOfView=90,
    )
    controller.reset(scene="FloorPlan319")

    # 2) init third party camera pose (this is the init_pose you pass to ViewManipulator)
    init_pose = {
        "position": {"x": -1.25, "y": 1.0, "z": -1.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    fov = 90.0
    cam_id = 0

    controller.step(
        action="AddThirdPartyCamera",
        position=init_pose["position"],
        rotation=init_pose["rotation"],
        fieldOfView=fov,
    )

    # 3) create manipulator from init_pose
    vm = ViewManipulator(
        init_pose=init_pose,
        step_translation=0.25,
        step_rotation_deg=15.0,
        is_discrete=True,
        roll_enabled=True,
    )

    # 4) push initial pose once (optional but explicit)
    controller.step(
        action="UpdateThirdPartyCamera",
        thirdPartyCameraId=cam_id,
        position=init_pose["position"],
        rotation=init_pose["rotation"],
        fieldOfView=fov,
    )

    # save initial frame
    ev = controller.last_event
    frame0 = ev.third_party_camera_frames[0]
    _save_rgb(frame0, os.path.join(out_dir, "step_0000_init.png"))
    print("Saved:", os.path.join(out_dir, "step_0000_init.png"))

    print("\nControls:")
    print("  w/s: forward/back")
    print("  a/d: left/right")
    print("  y/h: up/down")
    print("  q/e: yaw left/right")
    print("  r/f: look up/down")
    print("  t/g: roll CCW/CW")
    print("  p  : print current THOR pose")
    print("  x  : exit\n")

    i = 1
    while True:
        cmd = input("action> ").strip().lower()
        if not cmd:
            continue
        if cmd == "x":
            break
        if cmd == "p":
            print(vm.get_pose_thor())
            continue

        cmd = cmd[0]
        try:
            pose = vm.step(cmd)
        except Exception as e:
            print("Error:", e)
            continue

        # push to THOR
        controller.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=cam_id,
            position=pose["position"],
            rotation=pose["rotation"],
            fieldOfView=fov,
        )

        ev = controller.last_event
        frame = ev.third_party_camera_frames[0]
        img_path = os.path.join(out_dir, f"step_{i:04d}_{cmd}.png")
        _save_rgb(frame, img_path)
        print("Saved:", img_path)
        print("Pose:", pose)

        i += 1

    controller.stop()
    print("Done. Output folder:", out_dir)


if __name__ == "__main__":
    main()
