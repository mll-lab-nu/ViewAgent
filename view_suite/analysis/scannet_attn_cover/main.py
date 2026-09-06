"""
Attention × Coverage analysis for Qwen2.5-VL trajectories.

For each trajectory, extracts the attention of the last token before the
first assistant response to the three initial images (target, init, top-down),
maps that attention to ScanNet mesh vertices, and computes the mean attention
on vertices visible in all three views (overlap) vs the rest (non-overlap).

Usage:
    python -m view_suite.analysis.scannet_attn_cover.main run \
        --rollout_dir /path/to/rollout_dir \
        --model_path /path/to/qwen25vl \
        --scannet_dir /path/to/scannet

    python -m view_suite.analysis.scannet_attn_cover.main run \
        --rollout_dir /root/projects/viewsuite/rollouts/iter5_rl_960/tag_ae_loose_no_example \
        --model_path /root/projects/viewsuite/iter5_rl_960 \
        --scannet_dir /root/projects/viewsuite/data/viewagent_scannet
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import fire

from view_suite.proxy_analysis.scannet_attn.model_manager import ModelManager
from view_suite.proxy_analysis.scannet_attn.attention_extractor import (
    _rebuild_conversation,
    _tokenize_conversation,
    _forward_with_attention,
)
from view_suite.proxy_analysis.scannet_attn.token_image_mapper import (
    find_image_token_spans,
    find_response_token_spans,
)
from view_suite.proxy_analysis.scannet_point_by_turn.trajectory_parser import TrajectoryParser
from view_suite.proxy_analysis.scannet_point_by_turn.visibility import VisibilityComputer
from view_suite.proxy_analysis.scannet_attn_cover.vertex_attention import (
    compute_vertex_attention,
    compute_overlap_stats,
)
from view_suite.proxy_analysis.scannet_attn_cover.plotter import plot_overlap_bar
from view_suite.scannet.utils.path_utils import resolve_scene_ply
from view_suite.envs.utils.scannet_utils import default_intrinsics


# Image order in first user message: target, init, top-down
_IMG_NAMES = ["target", "init", "topdown"]


def _parse_layer_indices(layer_indices) -> List[int]:
    if isinstance(layer_indices, int):
        return [layer_indices]
    if isinstance(layer_indices, (list, tuple)):
        return [int(x) for x in layer_indices]
    if isinstance(layer_indices, str):
        parsed = json.loads(layer_indices)
        if isinstance(parsed, int):
            return [parsed]
        return [int(x) for x in parsed]
    return [int(layer_indices)]


def _process_one_trajectory(
    traj_dir: str,
    manager: ModelManager,
    traj_info,
    vc: VisibilityComputer,
) -> Dict[int, Dict[str, float]]:
    """
    Process one trajectory: extract attention → map to vertices → overlap stats.

    Returns:
        dict mapping layer_idx → overlap stats dict.
    """
    traj_path = Path(traj_dir)

    # 1. Reconstruct conversation & tokenize
    messages, image_files = _rebuild_conversation(traj_path)
    inputs, input_ids_flat = _tokenize_conversation(messages, image_files, manager)

    # 2. Forward pass
    attn_by_layer = _forward_with_attention(inputs, manager)

    # 3. Find token spans
    image_spans = find_image_token_spans(
        input_ids_flat, inputs["image_grid_thw"], manager.spatial_merge_size,
    )
    response_spans = find_response_token_spans(
        input_ids_flat, manager.processor.tokenizer,
    )

    if not response_spans:
        print(f"  WARNING: No response spans found, skipping.")
        return {}

    first_resp = response_spans[0]
    # Last token position before the first assistant response
    query_pos = first_resp.token_start - 1

    # The first 3 images are target, init, top-down
    if len(image_spans) < 3:
        print(f"  WARNING: Expected >=3 images, got {len(image_spans)}, skipping.")
        return {}

    # c2w mapping: image order in conversation → c2w
    c2ws = [
        traj_info.target_view_c2w,   # turn_01_01 = target
        traj_info.init_view_c2w,     # turn_01_02 = init
        traj_info.top_down_view_c2w, # turn_01_03 = top-down
    ]

    results_by_layer: Dict[int, Dict[str, float]] = {}

    for layer_idx, attn_tensor in attn_by_layer.items():
        # Average across heads: (1, H, L, L) → (L, L)
        attn = attn_tensor[0].float().mean(dim=0).cpu().numpy()

        # Attention of last-token-before-response to all positions
        query_attn = attn[query_pos, :]  # (L,)

        # For each of the 3 images, map attention to vertices
        vertex_attns: List[Dict[int, float]] = []
        for img_i in range(3):
            span = image_spans[img_i]
            img_attn = query_attn[span.token_start:span.token_end]
            va = compute_vertex_attention(
                img_attn, span.grid_h, span.grid_w, vc, c2ws[img_i],
            )
            vertex_attns.append(va)
            print(f"    Layer {layer_idx}, {_IMG_NAMES[img_i]}: "
                  f"{len(va)} visible vertices")

        # Debug: pairwise overlaps (only print for first layer)
        if layer_idx == list(attn_by_layer.keys())[0]:
            sets = [set(d.keys()) for d in vertex_attns]
            for a, b in [(0, 1), (0, 2), (1, 2)]:
                n = len(sets[a] & sets[b])
                print(f"    Pairwise {_IMG_NAMES[a]}∩{_IMG_NAMES[b]}: {n}")

        stats = compute_overlap_stats(vertex_attns)
        results_by_layer[layer_idx] = stats
        print(f"    Layer {layer_idx}: overlap={stats['n_overlap']}, "
              f"non_overlap={stats['n_non_overlap']}, "
              f"overlap_mean={stats['overlap_mean']:.6f}, "
              f"non_overlap_mean={stats['non_overlap_mean']:.6f}")

    return results_by_layer


class AttnCoverAnalyzer:
    """CLI for attention × coverage analysis."""

    def run(
        self,
        rollout_dir: str,
        model_path: str,
        scannet_dir: str,
        layer_indices: Union[int, str] = '[0,7,14,21,27]',
        output_dir: Optional[str] = None,
        device: str = "cuda:0",
        max_trajs: int = 0,
        jsonl_path: Optional[str] = None,
    ) -> None:
        """
        Full pipeline: extract attention, map to vertices, compute overlap stats, plot.

        Args:
            rollout_dir:  Directory with trajectory sub-folders.
            model_path:   Path to Qwen2.5-VL model checkpoint.
            scannet_dir:  Root of ScanNet data (contains scans/).
            layer_indices: Layers to extract attention from.
            output_dir:   Output directory (default: <rollout_dir>/attn_cover_analysis/).
            device:       CUDA device.
            max_trajs:    Max trajectories (0 = all).
            jsonl_path:   JSONL path override.
        """
        t0 = time.time()
        indices = _parse_layer_indices(layer_indices)
        scannet_scans = str(Path(scannet_dir) / "scans")
        output_path = Path(output_dir) if output_dir else Path(rollout_dir) / "attn_cover_analysis"
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Load model ---
        print("Loading model ...")
        manager = ModelManager(model_path, layer_indices=indices, device=device)
        manager.load()

        # --- Parse trajectories for c2w poses ---
        print("Parsing trajectories ...")
        parser = TrajectoryParser(jsonl_path)
        trajectories = parser.parse_all(rollout_dir)

        if max_trajs > 0:
            trajectories = trajectories[:max_trajs]

        # Index by traj_id
        traj_info_map = {t.traj_id: t for t in trajectories}

        # Group by scene for VisibilityComputer reuse
        scene_trajs: Dict[str, List] = defaultdict(list)
        for t in trajectories:
            scene_trajs[t.scene_id].append(t)

        K4 = default_intrinsics()
        K3 = K4[:3, :3].copy()

        # --- Process ---
        # Collect per-trajectory per-layer stats
        all_results: List[Dict[str, Any]] = []
        total = len(trajectories)
        processed = 0
        vc = None

        for scene_id, scene_traj_list in scene_trajs.items():
            # Clean up previous VisibilityComputer to avoid Open3D segfault
            if vc is not None:
                del vc
                gc.collect()

            print(f"\n{'='*60}")
            print(f"Scene: {scene_id} ({len(scene_traj_list)} trajectories)")
            mesh_path = resolve_scene_ply(scannet_scans, scene_id)
            vc = VisibilityComputer(mesh_path, K3, width=512, height=512)
            print(f"  Mesh loaded: {vc.total_vertices:,} vertices")

            for traj_info in scene_traj_list:
                traj_dir = str(Path(rollout_dir) / traj_info.traj_id)
                processed += 1
                print(f"\n[{processed}/{total}] {traj_info.traj_id}")

                try:
                    layer_stats = _process_one_trajectory(
                        traj_dir, manager, traj_info, vc,
                    )
                    all_results.append({
                        "traj_id": traj_info.traj_id,
                        "scene_id": scene_id,
                        "layer_stats": {
                            str(k): v for k, v in layer_stats.items()
                        },
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

        # Final cleanup
        if vc is not None:
            del vc
            gc.collect()

        # --- Aggregate per layer ---
        print(f"\n{'='*60}")
        print("Aggregating ...")
        summary = _aggregate(all_results, indices)

        # --- Save ---
        payload = {
            "config": {
                "rollout_dir": rollout_dir,
                "model_path": model_path,
                "scannet_dir": scannet_dir,
                "layer_indices": indices,
            },
            "trajectories": all_results,
            "summary": summary,
        }
        results_path = output_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {results_path}")

        # --- Plot ---
        plot_overlap_bar(summary, output_path)

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s. Output → {output_path}")

    def plot_only(
        self,
        result_json: str,
        output_dir: str = "./attn_cover_replot",
    ) -> None:
        """Re-generate plots from saved results.json."""
        with open(result_json) as f:
            payload = json.load(f)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_overlap_bar(payload["summary"], output_path)
        print(f"Plots → {output_path}")


def _aggregate(
    all_results: List[Dict[str, Any]],
    layer_indices: List[int],
) -> Dict[str, Any]:
    """
    Aggregate per-trajectory per-layer stats into overall means.

    Returns:
        summary dict with per-layer means and overall means.
    """
    per_layer: Dict[str, Dict[str, List[float]]] = {}

    for lidx in layer_indices:
        key = str(lidx)
        overlap_vals = []
        non_overlap_vals = []

        for r in all_results:
            ls = r["layer_stats"].get(key)
            if ls is None:
                continue
            overlap_vals.append(ls["overlap_mean"])
            non_overlap_vals.append(ls["non_overlap_mean"])

        per_layer[key] = {
            "overlap_mean": float(np.mean(overlap_vals)) if overlap_vals else 0.0,
            "overlap_std": float(np.std(overlap_vals)) if overlap_vals else 0.0,
            "non_overlap_mean": float(np.mean(non_overlap_vals)) if non_overlap_vals else 0.0,
            "non_overlap_std": float(np.std(non_overlap_vals)) if non_overlap_vals else 0.0,
            "n_trajectories": len(overlap_vals),
        }

    return {"per_layer": per_layer}


if __name__ == "__main__":
    fire.Fire(AttnCoverAnalyzer)
