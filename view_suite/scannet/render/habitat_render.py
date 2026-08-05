"""Habitat-Sim mesh renderer — drop-in replacement for MeshRenderer.

Why this exists: Open3D renders through Filament -> EGL, and EGL enumerates devices
independently of CUDA. `CUDA_VISIBLE_DEVICES` therefore does NOT move an Open3D
context, so every render worker lands on EGL device 0 and multi-GPU rendering is
impossible (only single-GPU 4090/5090 boxes ever worked). Habitat-Sim exposes an
explicit `gpu_device_id` (it does eglQueryDevicesEXT + eglGetPlatformDisplayEXT
internally), which is verified to isolate: requesting gpu_device_id=6 grows GPU 6's
memory and no other GPU's.

Interface matches BaseRenderer so it can be swapped in behind UnifiedRenderer:
    render_image_from_cam_param(camera_intrinsics, camera_extrinsics, width, height)
with OpenCV-style c2w extrinsics, same as MeshRenderer.

Requires the separate `habitat` conda env (habitat-sim 0.3.3 headless):
    conda create -y -n habitat python=3.9
    conda install -y -n habitat habitat-sim headless -c conda-forge -c aihabitat
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from PIL import Image

# OpenCV camera (+X right, +Y down, +Z forward)  ->  Habitat/OpenGL camera
# (+X right, +Y up, -Z forward). Flipping Y and Z converts between them.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

# ScanNet worlds are Z-up (verified: scene0329_00 bounds x 0..9.6, y 0..11, z 0..2.8
# — z is clearly the ~2.8m ceiling height). Habitat assumes Y-up, so poses expressed
# in the ScanNet frame must be rotated -90deg about X before being handed to Habitat.
_ZUP_TO_YUP = np.array([[1., 0., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])


def _hfov_deg_from_K(K: np.ndarray, width: int) -> float:
    """Horizontal FOV implied by fx. Habitat takes an FOV, not a full K."""
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (4, 4):
        K = K[:3, :3]
    fx = float(K[0, 0])
    if fx <= 0:
        return 90.0
    return float(np.degrees(2.0 * math.atan(0.5 * width / fx)))


class HabitatRenderer:
    """Renders a ScanNet mesh through Habitat-Sim on an explicitly chosen GPU."""

    def __init__(self, file_path: str, gpu_device_id: int = 0,
                 width: int = 512, height: int = 512, hfov_deg: float = 90.0):
        import habitat_sim  # imported lazily: only the `habitat` env has it

        self._hs = habitat_sim
        self.file_path = file_path
        self.gpu_device_id = int(gpu_device_id)
        self._w, self._h, self._hfov = int(width), int(height), float(hfov_deg)
        self._sim = None
        self._build()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _build(self) -> None:
        hs = self._hs
        cfg = hs.SimulatorConfiguration()
        cfg.scene_id = self.file_path
        cfg.gpu_device_id = self.gpu_device_id      # the whole point — real isolation
        cfg.enable_physics = False
        # ScanNet meshes carry vertex colours and ship no lighting rig; NO_LIGHT_KEY
        # gives flat vertex-colour shading (without it every render comes back black).
        cfg.scene_light_setup = self._hs.gfx.NO_LIGHT_KEY

        spec = hs.CameraSensorSpec()
        spec.uuid = "rgb"
        spec.sensor_type = hs.SensorType.COLOR
        spec.resolution = [self._h, self._w]
        spec.hfov = self._hfov
        spec.position = [0.0, 0.0, 0.0]             # sensor at the agent origin
        agent_cfg = hs.agent.AgentConfiguration(sensor_specifications=[spec])
        self._sim = hs.Simulator(hs.Configuration(cfg, [agent_cfg]))

    def _rebuild_if_needed(self, width: int, height: int, hfov: float) -> None:
        """Habitat fixes sensor resolution/FOV at construction, so a different
        request size means rebuilding the simulator."""
        if (width, height) == (self._w, self._h) and abs(hfov - self._hfov) < 1e-6:
            return
        self.close()
        self._w, self._h, self._hfov = width, height, hfov
        self._build()

    def close(self) -> None:
        if self._sim is not None:
            try:
                self._sim.close()
            finally:
                self._sim = None

    # ── rendering ────────────────────────────────────────────────────────────
    def render_image_from_cam_param(
        self,
        camera_intrinsics,
        camera_extrinsics,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """camera_extrinsics is a 4x4 OpenCV-style c2w matrix (same as MeshRenderer)."""
        import quaternion  # provided by habitat-sim

        c2w = np.asarray(camera_extrinsics, dtype=np.float64).reshape(4, 4)
        hfov = _hfov_deg_from_K(camera_intrinsics, width)
        self._rebuild_if_needed(int(width), int(height), hfov)

        # ScanNet Z-up world -> Habitat Y-up world, then OpenCV cam -> OpenGL cam
        c2w_gl = _ZUP_TO_YUP @ c2w @ _CV_TO_GL
        R, t = c2w_gl[:3, :3], c2w_gl[:3, 3]

        state = self._hs.AgentState()
        state.position = t.astype(np.float32)
        state.rotation = quaternion.from_rotation_matrix(R)
        # sensor sits at the agent origin; give it the same pose
        for s in state.sensor_states.values():
            s.position = state.position
            s.rotation = state.rotation
        self._sim.get_agent(0).set_state(state, infer_sensor_states=False)

        rgb = np.asarray(self._sim.get_sensor_observations()["rgb"])
        if rgb.ndim == 3 and rgb.shape[2] == 4:      # RGBA -> RGB
            rgb = rgb[:, :, :3]
        return Image.fromarray(rgb.astype(np.uint8))
