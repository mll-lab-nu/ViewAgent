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

import contextlib
import math
from typing import Optional, Sequence

import numpy as np

# OpenCV camera (+X right, +Y down, +Z forward)  ->  Habitat/OpenGL camera
# (+X right, +Y up, -Z forward). Flipping Y and Z converts between them.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

# NOTE: ScanNet worlds are Z-up (scene0329_00 bounds x 0..9.6, y 0..11, z 0..2.8 —
# z is the ~2.8m ceiling height) while Habitat is nominally Y-up, BUT Habitat loads
# the mesh without re-orienting it, so no world rotation must be applied here.
# Measured: with a Z-up->Y-up rotation every frame is black; without it, 33% coverage.


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
                 width: int = 512, height: int = 512, hfov_deg: float = 90.0,
                 lighting: bool = True, light_intensity: float = 2.5,
                 background=(1.0, 1.0, 1.0, 1.0)):
        import habitat_sim  # imported lazily: only the `habitat` env has it

        self._hs = habitat_sim
        self.file_path = file_path
        self.gpu_device_id = int(gpu_device_id)
        self._w, self._h, self._hfov = int(width), int(height), float(hfov_deg)
        self._bg = tuple(background)
        self._lighting = bool(lighting)
        self._light_intensity = float(light_intensity)
        self._sim = None
        self._build()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _build(self) -> None:
        hs = self._hs
        cfg = hs.SimulatorConfiguration()
        cfg.scene_id = self.file_path
        cfg.gpu_device_id = self.gpu_device_id      # the whole point — real isolation
        cfg.enable_physics = False
        # ScanNet meshes carry vertex colours and ship no lighting rig. NO_LIGHT_KEY
        # gives flat vertex-colour shading — correct but dark/flat. With lighting=True we
        # register a rig of directional lights to approximate Open3D's defaultLit PBR
        # (sun light + indirect intensity 20000) so the two renderers look comparable.
        if self._lighting:
            # override_scene_light_defaults is REQUIRED: without it the ScanNet .ply is
            # loaded on a flat vertex-colour path that ignores lights entirely (proved:
            # identical output at light intensity 3/8/20). With it, the mesh is rendered
            # lit so the rig below actually does something.
            cfg.scene_light_setup = self._hs.gfx.DEFAULT_LIGHTING_KEY
            cfg.override_scene_light_defaults = True
        else:
            cfg.scene_light_setup = self._hs.gfx.NO_LIGHT_KEY

        spec = hs.CameraSensorSpec()
        spec.uuid = "rgb"
        spec.sensor_type = hs.SensorType.COLOR
        spec.resolution = [self._h, self._w]
        spec.hfov = self._hfov
        spec.position = [0.0, 0.0, 0.0]             # sensor at the agent origin
        # Match MeshRenderer, whose background defaults to WHITE (1,1,1,1).
        # Habitat clears to black by default, which alone made renders look ~5x
        # darker than Open3D in mean-pixel terms even where geometry matched.
        spec.clear_color = list(self._bg)
        agent_cfg = hs.agent.AgentConfiguration(sensor_specifications=[spec])
        self._sim = hs.Simulator(hs.Configuration(cfg, [agent_cfg]))
        if self._lighting:
            self._apply_lights()

    def _apply_lights(self) -> None:
        """Directional rig approximating Open3D's sun + indirect ambient.

        w=0 in the vector means a DIRECTIONAL light. One bright key from above plus
        weaker fills from the four sides so interiors are not lit from one face only
        (a single sun leaves most of a room black).
        """
        hs = self._hs
        I = self._light_intensity
        def L(vec, k):
            return hs.gfx.LightInfo(vector=vec, color=[k * I, k * I, k * I],
                                    model=hs.gfx.LightPositionModel.Global)
        lights = [
            L([0.0, 0.0, -1.0, 0.0], 1.0),   # key, from above (scene is Z-up)
            L([1.0, 0.0, -0.3, 0.0], 0.4),
            L([-1.0, 0.0, -0.3, 0.0], 0.4),
            L([0.0, 1.0, -0.3, 0.0], 0.4),
            L([0.0, -1.0, -0.3, 0.0], 0.4),
            L([0.0, 0.0, 1.0, 0.0], 0.25),   # bounce from the floor
        ]
        self._sim.set_light_setup(lights, hs.gfx.DEFAULT_LIGHTING_KEY)

    def _rebuild_if_needed(self, width: int, height: int, hfov: float) -> None:
        """Adapt to a new request geometry, rebuilding only when unavoidable.

        Resolution is baked into the sensor's framebuffer, so a size change still costs
        a full rebuild (~1.4s). FOV does not: retargeting the projection in place keeps
        the scene loaded. That matters because callers routinely vary focal length while
        holding the size fixed — rebuilding on every fx would turn each render into a
        scene reload (measured: 43 renders/s instead of ~200 on 8 GPUs).
        """
        if (width, height) != (self._w, self._h):
            prev = (self._w, self._h, self._hfov)
            self.close()
            self._w, self._h, self._hfov = width, height, hfov
            try:
                self._build()
            except Exception:
                # Do not leave a half-dead renderer behind: _sim is None but the dims
                # say otherwise, so the handler's cache keeps handing this object out
                # and every later render dies on `self._sim.get_agent(0)`. That is a
                # plain exception, not a BrokenProcessPool, so the pool never recycles
                # the slot and the scene stays dead for the life of the service.
                # Restore the old geometry and re-raise so the caller can retry.
                self._w, self._h, self._hfov = prev
                raise
            return
        if abs(hfov - self._hfov) < 1e-6:
            return
        try:
            spec = self._sim.get_agent(0).agent_config.sensor_specifications[0]
            spec.hfov = hfov
            self._sim.get_agent(0)._sensors["rgb"].set_projection_params(spec)
            self._hfov = hfov
        except Exception:                      # no in-place path -> fall back to rebuild
            self.close()
            self._hfov = hfov
            self._build()

    def close(self) -> None:
        if self._sim is not None:
            try:
                # Same context caveat as rendering: tearing down a simulator that is
                # not the current one aborts the process (SIGABRT on "no current
                # context"). Make it current first.
                with contextlib.suppress(Exception):
                    self._sim.renderer.acquire_gl_context()
                self._sim.close()
            except Exception:
                pass
            finally:
                self._sim = None

    # ── rendering ────────────────────────────────────────────────────────────
    def render_image_from_cam_param(
        self,
        camera_intrinsics,
        camera_extrinsics,
        width: int = 512,
        height: int = 512,
    ) -> np.ndarray:
        """camera_extrinsics is a 4x4 OpenCV-style c2w matrix (same as MeshRenderer)."""
        import quaternion  # provided by habitat-sim

        c2w = np.asarray(camera_extrinsics, dtype=np.float64).reshape(4, 4)
        hfov = _hfov_deg_from_K(camera_intrinsics, width)
        self._rebuild_if_needed(int(width), int(height), hfov)

        # OpenCV cam -> OpenGL cam only. Do NOT rotate the world: Habitat loads the
        # ScanNet mesh in its native (Z-up) frame without converting it, so poses are
        # already in the right frame. Verified: applying a Z-up->Y-up rotation renders
        # pure black, passing the pose straight through renders the scene.
        c2w_gl = c2w @ _CV_TO_GL
        R, t = c2w_gl[:3, :3], c2w_gl[:3, 3]

        # Habitat keeps ONE current GL context per process, and constructing a new
        # Simulator steals it from every existing one — so with N resident scenes only
        # the last-built one can draw ("GL::Context::current(): no current context").
        # acquire_gl_context() makes THIS simulator current again; it is a no-op when
        # it already is, so the single-scene path pays nothing.
        with contextlib.suppress(Exception):
            self._sim.renderer.acquire_gl_context()

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
        # Return an ndarray, NOT a PIL Image: BaseRenderer's contract is what
        # MeshRenderer does (`return np.asarray(pil)`), and the service calls
        # `.astype(np.uint8)` on the result. Returning a PIL Image raised AttributeError
        # inside the worker, which handle()'s catch-all turned into HTTP 200 with zero
        # images — every habitat render silently produced nothing while the stress
        # harness counted it as a success.
        return rgb.astype(np.uint8)
