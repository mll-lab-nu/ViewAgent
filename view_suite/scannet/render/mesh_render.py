from __future__ import annotations  # py3.9 (habitat env) parses 3.10 unions
import os
os.environ['__EGL_VENDOR_LIBRARY_DIRS'] = '/usr/share/glvnd/egl_vendor.d'
os.environ['__GLX_VENDOR_LIBRARY_DIRS'] = '/usr/share/glvnd/glx_vendor.d'
import numpy as np
import traceback
try:
    import open3d as o3d
except ImportError as e:
    traceback.print_exc()
    print(f"WARNING: Open3D not available: {e}")
    o3d = None
except OSError as e:
    traceback.print_exc()
    print(f"WARNING: Open3D cannot be loaded due to system library issue: {e}")
    o3d = None
from .base_render import BaseRenderer
from view_suite.scannet.utils.pose_utils import extrinsic_c2w_to_w2c

class MeshRenderer(BaseRenderer):
    """Renderer for meshes using Open3D offscreen rendering.
    Adds SSAA (super-sampling) and subpixel jitter accumulation (TAA-like).
    Keeps the same field-of-view across arbitrary output sizes by
    scaling intrinsics and letterboxing when aspect ratios differ.
    """

    def __init__(self, file_path, brightness=1.2, roughness=0.6, metallic=0.1):
        super().__init__(file_path)

        # Load mesh
        self.mesh = o3d.io.read_triangle_mesh(file_path)
        if not self.mesh.has_vertices():
            raise ValueError("The mesh is empty or invalid.")

        # Ensure normals for proper shading
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()

        # Vertex-color brightness correction (fallback to gray if no color)
        if self.mesh.has_vertex_colors():
            colors = np.asarray(self.mesh.vertex_colors)
            self.mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors * brightness, 0, 1))
        else:
            self.mesh.paint_uniform_color([0.7, 0.7, 0.7])

        self.roughness = roughness
        self.metallic = metallic

        # Offscreen renderer state
        self.renderer = None
        self.width = -1
        self.height = -1

        print("Mesh loaded and preprocessed successfully.")

    # --------------------- internal helpers ---------------------

    @staticmethod
    def _to_K3x3(K: np.ndarray) -> np.ndarray:
        """Accept 3x3 or 4x4, return 3x3 intrinsics."""
        K = np.asarray(K, dtype=np.float64)
        if K.shape == (4, 4):
            return K[:3, :3]
        if K.shape == (3, 3):
            return K
        raise ValueError(f"camera_intrinsics must be 3x3 or 4x4, got {K.shape}")

    @staticmethod
    def _infer_base_size_from_K(K3: np.ndarray) -> tuple[int, int]:
        """Heuristic: original image size ≈ (2*cx, 2*cy). Works well for ScanNet-like data."""
        cx, cy = float(K3[0, 2]), float(K3[1, 2])
        w0 = max(1, int(round(cx * 2.0)))
        h0 = max(1, int(round(cy * 2.0)))
        return w0, h0

    @staticmethod
    def _scale_K_with_letterbox(K3: np.ndarray, target_w: int, target_h: int) -> tuple[float, float, float, float, float]:
        """
        Scale K to target frame while preserving FOV.
        If aspect ratio differs, letterbox and shift principal point accordingly.
        Returns (fx, fy, cx, cy, scale) for the target (width=target_w, height=target_h).
        """
        fx0, fy0 = float(K3[0, 0]), float(K3[1, 1])
        cx0, cy0 = float(K3[0, 2]), float(K3[1, 2])
        base_w, base_h = MeshRenderer._infer_base_size_from_K(K3)

        # Uniform scale to fit within target (no cropping)
        sx = target_w / base_w
        sy = target_h / base_h
        s = min(sx, sy)

        new_w = base_w * s
        new_h = base_h * s
        pad_x = 0.5 * (target_w - new_w)  # left/right letterbox
        pad_y = 0.5 * (target_h - new_h)  # top/bottom letterbox

        fx = fx0 * s
        fy = fy0 * s
        cx = cx0 * s + pad_x
        cy = cy0 * s + pad_y
        return fx, fy, cx, cy, s

    def _rebuild_renderer(self, width: int, height: int, background=(1.0, 1.0, 1.0, 1.0)):
        """Ensure the offscreen renderer matches the requested size and keep geometry intact."""
        needs_recreate = False
        if self.renderer is None:
            needs_recreate = True
        elif self.width != width or self.height != height:
            try:
                self.renderer.release()
            except Exception:
                pass
            self.renderer = None
            needs_recreate = True

        if needs_recreate:
            self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            self.width, self.height = width, height

            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLit"  # PBR shading
            mat.base_roughness = self.roughness
            mat.base_metallic = self.metallic
            self.renderer.scene.add_geometry("mesh", self.mesh, mat)

            try:
                self.renderer.scene.set_indirect_light_intensity(20000)
                self.renderer.scene.enable_sun_light(True)
                self.renderer.scene.set_sun_light(
                    direction=[-0.577, -0.577, -0.577], intensity=65000, color=[1.0, 1.0, 1.0]
                )
            except Exception:
                pass  # Some Open3D builds may not support all lighting controls.

        if self.renderer is not None:
            self.renderer.scene.set_background(list(background))

    @staticmethod
    def _jitter_sequence(pattern: str, n: int, seed: int | None = None) -> list[tuple[float, float]]:
        """
        Generate a subpixel jitter sequence for (cx, cy).
        Values are in pixel units (e.g., +/- 0.25 px).
        """
        import random
        if n <= 1:
            return [(0.0, 0.0)]
        pattern = (pattern or "fixed").lower()

        if pattern == "fixed":
            base = [(0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)]
            out = []
            i = 0
            while len(out) < n:
                out.append(base[i % len(base)])
                i += 1
            return out[:n]

        if pattern == "random":
            rng = random.Random(seed)
            return [(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)) for _ in range(n)]

        # Halton-like simple low-discrepancy on [−0.5, 0.5]
        def halton(k: int, b: int) -> float:
            f, r = 1.0, 0.0
            while k > 0:
                f /= b
                r += f * (k % b)
                k //= b
            return r
        seq = []
        for i in range(1, n + 1):
            u = halton(i, 2) - 0.5
            v = halton(i, 3) - 0.5
            seq.append((u, v))
        return seq

    @staticmethod
    def _downsample(img_np: np.ndarray, target_w: int, target_h: int, method: str = "lanczos") -> np.ndarray:
        """Downsample with a high-quality filter using PIL."""
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img_np)
        method = method.lower()
        if method == "nearest":
            resample = PILImage.NEAREST
        elif method == "bilinear":
            resample = PILImage.BILINEAR
        elif method == "bicubic":
            resample = PILImage.BICUBIC
        else:
            resample = PILImage.LANCZOS
        pil = pil.resize((target_w, target_h), resample=resample)
        return np.asarray(pil)

    # ------------------------- public API -------------------------

    def render_image_from_cam_param(
        self,
        camera_intrinsics,
        camera_extrinsics,
        width: int = 300,
        height: int = 300,
        *,
        ssaa: int = 1,
        taa_samples: int = 1,
        jitter: str = "fixed",
        jitter_seed: int | None = None,
        background=(1.0, 1.0, 1.0, 1.0),
        downsample: str = "lanczos",
    ):
        """
        Render from camera intrinsics + extrinsics (OpenCV-style).
        - SSAA: render at (width*ssaa, height*ssaa) then downsample.
        - TAA-like accumulation: average multiple jittered renders (subpixel shifts on cx, cy).
        - K is scaled to the target frame and letterboxed to preserve FOV/composition.

        All new parameters are optional; defaults reproduce the legacy behavior.
        camera_extrinsics is a 4*4 c2w matrix
        """
        camera_extrinsics = np.asarray(camera_extrinsics, dtype=np.float64)
        camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
        camera_extrinsics = extrinsic_c2w_to_w2c(camera_extrinsics)
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return None

        ssaa = max(1, int(ssaa))
        taa_samples = max(1, int(taa_samples))
        big_w, big_h = width * ssaa, height * ssaa

        # Build/rebuild renderer at super-sampled resolution
        self._rebuild_renderer(big_w, big_h, background=background)

        # Base K scaled to the big canvas (without jitter)
        K3_in = self._to_K3x3(np.asarray(camera_intrinsics, dtype=np.float64))
        fx0, fy0, cx0, cy0, scale = self._scale_K_with_letterbox(K3_in, big_w, big_h)

        # Generate jitter sequence in pixel units (on the super-sampled canvas)
        jitters = self._jitter_sequence(jitter, taa_samples, seed=jitter_seed)

        acc = None
        for jx, jy in jitters:
            fx, fy = fx0, fy0
            cx, cy = cx0 + jx, cy0 + jy

            intr = o3d.camera.PinholeCameraIntrinsic(big_w, big_h, fx, fy, cx, cy)
            self.renderer.setup_camera(intr, np.asarray(camera_extrinsics, dtype=np.float64))
            big = np.asarray(self.renderer.render_to_image()).astype(np.float32)

            acc = big if acc is None else (acc + big)

        # Average accumulations
        big_avg = (acc / float(taa_samples)).clip(0, 255).astype(np.uint8)

        # Downsample if SSAA > 1
        if ssaa > 1:
            out = self._downsample(big_avg, width, height, method=downsample)
        else:
            out = big_avg
        return out

    
