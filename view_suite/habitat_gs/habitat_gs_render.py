"""Habitat-GS renderer — 3D Gaussian Splatting scenes behind the ScanNet renderer API.

Habitat-GS (https://github.com/zju3dv/habitat-gs) is Habitat-Sim with a CUDA gaussian
rasterizer bolted onto the render path. From this repo's point of view it is the same
simulator as the ScanNet habitat backend, so this class is deliberately a close sibling
of ``view_suite.scannet.render.habitat_render.HabitatRenderer`` and keeps its interface:

    render_image_from_cam_param(camera_intrinsics, camera_extrinsics, width, height)

with an OpenCV-style 4x4 c2w. Three things differ, and each one is load-bearing:

1. **The stage is a ``.gs.ply``.** Habitat-GS picks the gaussian asset path purely by
   filename suffix (``src/esp/assets/Asset.cpp``: ``.gs.ply`` / ``.3dgs.ply`` /
   ``.4dgs.ply`` / ``.4dgs.npz``). A correctly-formatted gaussian cloud under any other
   name loads as an ordinary point cloud and renders as confetti, with no error.

2. **No lighting rig.** Gaussians carry baked view-dependent radiance; there is nothing
   for a light to illuminate. The ScanNet renderer registers a six-light rig because
   ScanNet meshes are unlit vertex-colour geometry — doing that here is at best a no-op
   and at worst double-counts exposure, so we pin NO_LIGHT_KEY.

3. **The world is Y-up.** ScanNet meshes are Z-up and Habitat loads them unrotated, which
   is why that renderer applies no world transform and its "up" is +Z. These scenes ship
   Habitat-format ``.navmesh`` files, so they are already in Habitat's own Y-up frame:
   "up" is +Y. Everything downstream (pose sampling, move_up/move_down) must agree.

Requires the separate ``habitat-gs`` conda env — habitat-gs cannot coexist with the
habitat-sim 0.3.3 build that serves the ScanNet backend:
    conda create -y -n habitat-gs -c conda-forge python=3.12 cmake=3.31
    HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF HABITAT_BUILD_GUI_VIEWERS=OFF \
        pip install . --no-build-isolation
"""
from __future__ import annotations

import contextlib
import math
import os
from typing import Optional, Tuple

import numpy as np

# OpenCV camera (+X right, +Y down, +Z forward) -> Habitat/OpenGL camera
# (+X right, +Y up, -Z forward).
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

# Habitat's own up axis. Unlike ScanNet (Z-up meshes loaded unrotated), the GS scenes
# are authored in Habitat's frame — their .navmesh files would not load otherwise.
WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _geometry_from_K(K: np.ndarray, width: int, height: int) -> Tuple[float, int]:
    """Map a full pinhole K onto what habitat can express.

    Habitat takes a single horizontal FOV and derives the vertical one from the
    framebuffer aspect, so it can only represent SQUARE pixels. For a non-square K we
    render at a height that makes the implied vertical FOV match fy, then resample:
    we want tan(vfov/2) = H/(2*fy) but habitat gives tan(vfov/2) = H_render/(2*fx),
    so H_render = H * fx / fy.

    Returns (hfov_deg, render_height).
    """
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (4, 4):
        K = K[:3, :3]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if fx <= 0:
        return 90.0, int(height)
    hfov = float(np.degrees(2.0 * math.atan(0.5 * width / fx)))
    if fy <= 0:
        return hfov, int(height)
    return hfov, max(1, int(round(height * fx / fy)))


class HabitatGSRenderer:
    """Renders a 3DGS scene through Habitat-GS on an explicitly chosen GPU."""

    def __init__(self, file_path: str, gpu_device_id: int = 0,
                 width: int = 512, height: int = 512, hfov_deg: float = 90.0,
                 navmesh_path: Optional[str] = None,
                 background=(0.0, 0.0, 0.0, 1.0)):
        import habitat_sim  # lazy: only the `habitat-gs` env has this build

        if not str(file_path).endswith((".gs.ply", ".3dgs.ply")):
            # Fail loudly. The alternative is a silent fallback to the point-cloud
            # loader, which returns images -- just not of this scene.
            raise ValueError(
                f"{file_path!r} is not a gaussian stage asset. Habitat-GS selects the "
                f"gaussian renderer by suffix; the file must end in .gs.ply or .3dgs.ply"
            )
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        self._hs = habitat_sim
        self.file_path = file_path
        self.navmesh_path = navmesh_path
        self.gpu_device_id = int(gpu_device_id)
        self._w, self._h, self._hfov = int(width), int(height), float(hfov_deg)
        self._bg = tuple(background)
        self._sim = None
        self._build()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _build(self) -> None:
        hs = self._hs
        cfg = hs.SimulatorConfiguration()
        cfg.scene_id = self.file_path
        cfg.gpu_device_id = self.gpu_device_id   # real per-GPU isolation, as for ScanNet
        cfg.enable_physics = False
        # Gaussians are pre-lit. See module docstring.
        cfg.scene_light_setup = hs.gfx.NO_LIGHT_KEY

        spec = hs.CameraSensorSpec()
        spec.uuid = "rgb"
        spec.sensor_type = hs.SensorType.COLOR
        spec.resolution = [self._h, self._w]
        spec.hfov = self._hfov
        spec.position = [0.0, 0.0, 0.0]          # sensor at the agent origin
        spec.clear_color = list(self._bg)
        agent_cfg = hs.agent.AgentConfiguration(sensor_specifications=[spec])
        self._sim = hs.Simulator(hs.Configuration(cfg, [agent_cfg]))

        if self.navmesh_path and os.path.exists(self.navmesh_path):
            # Only needed by pose sampling (data_gen keeps cameras near walkable
            # space, because a 3DGS reconstruction degrades away from its training
            # views). Rendering itself never consults the navmesh, so a missing one
            # is not fatal here.
            with contextlib.suppress(Exception):
                self._sim.pathfinder.load_nav_mesh(self.navmesh_path)

    def _rebuild_if_needed(self, width: int, height: int, hfov: float) -> None:
        """Adapt to a new request geometry, rebuilding only when unavoidable.

        Resolution and FOV are baked in at construction. habitat-sim has no working
        in-place FOV path (setting spec.hfov + set_projection_params is accepted and
        changes nothing -- verified on the ScanNet backend, where believing it worked
        silently served every frame at the wrong FOV), so either changing costs a full
        rebuild. Callers with fixed intrinsics pay this once.
        """
        if (width, height) != (self._w, self._h):
            prev = (self._w, self._h, self._hfov)
            self.close()
            self._w, self._h, self._hfov = width, height, hfov
            try:
                self._build()
            except Exception:
                # Never leave a half-dead renderer in the handler's cache: _sim would
                # be None while the dims claim otherwise, so every later render dies
                # on a plain exception the pool does not treat as a broken worker, and
                # the scene stays dead for the life of the service.
                self._w, self._h, self._hfov = prev
                raise
            return
        if abs(hfov - self._hfov) < 1e-6:
            return
        prev = self._hfov
        self.close()
        self._hfov = hfov
        try:
            self._build()
        except Exception:
            self._hfov = prev
            raise

    def close(self) -> None:
        if self._sim is not None:
            try:
                # Tearing down a simulator that is not the current GL context aborts
                # the process (SIGABRT, "no current context"). Make it current first.
                with contextlib.suppress(Exception):
                    self._sim.renderer.acquire_gl_context()
                self._sim.close()
            except Exception:
                pass
            finally:
                self._sim = None

    # ── navmesh (used by data generation, not by rendering) ──────────────────
    @property
    def pathfinder(self):
        return None if self._sim is None else self._sim.pathfinder

    def is_navigable(self, point_xyz) -> bool:
        pf = self.pathfinder
        if pf is None or not pf.is_loaded:
            return False
        return bool(pf.is_navigable(np.asarray(point_xyz, dtype=np.float32)))

    def snap_to_navmesh(self, point_xyz) -> Optional[np.ndarray]:
        """Nearest walkable point, or None if there is no navmesh / no such point."""
        pf = self.pathfinder
        if pf is None or not pf.is_loaded:
            return None
        snapped = np.asarray(pf.snap_point(np.asarray(point_xyz, dtype=np.float32)),
                             dtype=np.float64)
        return None if not np.all(np.isfinite(snapped)) else snapped

    def sample_navigable_point(self) -> Optional[np.ndarray]:
        pf = self.pathfinder
        if pf is None or not pf.is_loaded:
            return None
        return np.asarray(pf.get_random_navigable_point(), dtype=np.float64)

    # ── rendering ────────────────────────────────────────────────────────────
    def render_image_from_cam_param(
        self,
        camera_intrinsics,
        camera_extrinsics,
        width: int = 512,
        height: int = 512,
    ) -> np.ndarray:
        """camera_extrinsics is a 4x4 OpenCV-style c2w, as for every other backend."""
        import quaternion  # provided by habitat-sim

        c2w = np.asarray(camera_extrinsics, dtype=np.float64).reshape(4, 4)
        hfov, render_h = _geometry_from_K(camera_intrinsics, int(width), int(height))
        self._rebuild_if_needed(int(width), render_h, hfov)

        # OpenCV cam -> OpenGL cam. No world rotation: the scene is already in
        # Habitat's Y-up frame.
        c2w_gl = c2w @ _CV_TO_GL
        R, t = c2w_gl[:3, :3], c2w_gl[:3, 3]

        # Habitat keeps ONE current GL context per process and each new Simulator
        # steals it, so with N resident scenes only the last-built one can draw.
        # This is a no-op when already current, so single-scene workers pay nothing.
        with contextlib.suppress(Exception):
            self._sim.renderer.acquire_gl_context()

        state = self._hs.AgentState()
        state.position = t.astype(np.float32)
        state.rotation = quaternion.from_rotation_matrix(R)
        for s in state.sensor_states.values():
            s.position = state.position
            s.rotation = state.rotation
        self._sim.get_agent(0).set_state(state, infer_sensor_states=False)

        rgb = np.asarray(self._sim.get_sensor_observations()["rgb"])
        if rgb.ndim == 3 and rgb.shape[2] == 4:      # RGBA -> RGB
            rgb = rgb[:, :, :3]
        # ndarray, not PIL: the service calls .astype(np.uint8) on this. Returning a
        # PIL Image raised inside the worker and handle()'s catch-all turned that into
        # HTTP 200 with zero images -- renders silently produced nothing.
        rgb = rgb.astype(np.uint8)
        if rgb.shape[0] != int(height):
            from PIL import Image
            rgb = np.asarray(
                Image.fromarray(rgb).resize((int(width), int(height)), Image.BILINEAR))
        return rgb
