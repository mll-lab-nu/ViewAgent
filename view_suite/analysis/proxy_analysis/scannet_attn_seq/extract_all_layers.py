"""
All-layer attention-to-image extraction (uniform eager attention, query-tiled).

Every transformer layer uses the SAME attention implementation, so there is no
eager/SDPA mixing and no cross-layer contamination artifact. All 28 layers are
read from a single, fully-consistent forward pass.

To fit a 24 GB GPU on ~6 k-token / 12-image trajectories, attention is computed
with a custom interface that tiles over the query dimension: the math is exactly
eager (fp32 softmax of Q·Kᵀ·scale + additive mask), but only one query tile's
(H, T, L) score block is alive at a time. Inside each tile we directly reduce to
the per-query-row image-attention fraction, so the full (H, L, L) matrix is never
materialized.

Output schema matches what ``scripts/plot_attention_5layer.py`` consumes:
  results.json -> {"records": [ {"layers": {str(idx): {turn_stats, global_mean_fraction, ...}}} ], ...}

Usage:
    python -m view_suite.analysis.proxy_analysis.scannet_attn_seq.extract_all_layers run \
        --rollout_dir /path/to/.../tag_active_exploration \
        --model_path  /path/to/model \
        --output_dir  /path/to/out_dir \
        --device cuda:0
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import fire

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv

from view_suite.analysis.scannet_attn.model_manager import ModelManager
from view_suite.analysis.scannet_attn.attention_extractor import (
    _rebuild_conversation,
    _tokenize_conversation,
)
from view_suite.analysis.scannet_attn.token_image_mapper import (
    find_image_token_spans,
    find_response_token_spans,
)


# ── Per-forward context + per-layer results (set/cleared each trajectory) ──
_CTX: dict = {}
_RESULT: dict = {}
_TILE = 1024


def _reduce_layer(frac: np.ndarray, response_spans) -> dict:
    """Reduce a (L,) per-position image-attention fraction to a layer record."""
    turn_stats = []
    for resp in response_spans:
        rf = frac[resp.token_start:resp.token_end]
        turn_stats.append({
            "turn_idx": resp.turn_idx,
            "mean_img_fraction": float(rf.mean()),
            "std_img_fraction": float(rf.std()),
            "n_tokens": int(resp.token_end - resp.token_start),
        })
    return {
        "turn_stats": turn_stats,
        "global_mean_fraction": float(frac.mean()),
        "response_mean_fraction": float(
            np.mean([t["mean_img_fraction"] for t in turn_stats])
        ) if turn_stats else 0.0,
    }


def _tiled_frac_attention(module, query, key, value, attention_mask,
                          scaling, dropout=0.0, **kwargs):
    """Eager attention computed in query tiles; side-channels image fraction.

    Returns (attn_output [B, L, H, D], None) so the model runs normally.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)      # (B, H, Lk, D)
    value_states = repeat_kv(value, module.num_key_value_groups)
    B, H, L, D = query.shape
    Lk = key_states.shape[2]

    out = torch.empty(B, H, L, D, dtype=query.dtype, device=query.device)
    maskf = _CTX["img_mask_f"]                                    # (Lk,) fp32, image cols = 1
    img_rows = torch.empty(L, dtype=torch.float32, device=query.device)
    tot_rows = torch.empty(L, dtype=torch.float32, device=query.device)

    kT = key_states.transpose(2, 3)                              # (B, H, D, Lk)
    for st in range(0, L, _TILE):
        en = min(st + _TILE, L)
        q = query[:, :, st:en]                                   # (B, H, t, D)
        scores = torch.matmul(q, kT) * scaling                   # (B, H, t, Lk)
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, st:en, :Lk]
        else:
            qpos = torch.arange(st, en, device=query.device).unsqueeze(1)
            kpos = torch.arange(Lk, device=query.device).unsqueeze(0)
            scores = scores.masked_fill((kpos > qpos)[None, None], float("-inf"))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)   # (B, H, t, Lk) fp32
        out[:, :, st:en] = torch.matmul(probs.to(query.dtype), value_states)
        p = probs[0]                                             # (H, t, Lk) fp32
        img_rows[st:en] = torch.matmul(p, maskf).sum(dim=0)      # (t,)
        tot_rows[st:en] = p.sum(dim=2).sum(dim=0)                # (t,)
        del scores, probs, p, q

    frac = (img_rows / tot_rows.clamp_min(1e-9)).cpu().numpy()
    _RESULT[module.layer_idx] = _reduce_layer(frac, _CTX["response_spans"])

    attn_output = out.transpose(1, 2).contiguous()              # (B, L, H, D)
    return attn_output, None


ALL_ATTENTION_FUNCTIONS.register("frac_eager", _tiled_frac_attention)


class AllLayerAttnExtractor:

    def run(
        self,
        rollout_dir: str,
        model_path: str,
        output_dir: str,
        n_layers: int = 28,
        tile: int = 1024,
        max_trajs: int = 0,
        device: str = "cuda:0",
    ):
        global _TILE
        _TILE = int(tile)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        layer_indices = list(range(n_layers))
        print(f"Loading model (uniform query-tiled eager, all {n_layers} layers) "
              f"from {model_path} ...")
        manager = ModelManager(model_path, layer_indices=layer_indices, device=device)
        manager.load()
        manager.model.config.use_cache = False

        # Force EVERY attention layer to the custom uniform implementation.
        layers = manager.model.model.language_model.layers
        for i in range(len(layers)):
            layers[i].self_attn.config._attn_implementation = "frac_eager"

        rd = Path(rollout_dir)
        traj_dirs = sorted([
            d for d in rd.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (d / "messages.json").exists()
        ])
        if max_trajs > 0:
            traj_dirs = traj_dirs[:max_trajs]
        print(f"Found {len(traj_dirs)} trajectories | tile={_TILE}")

        all_records = []
        t0 = time.time()
        for i, td in enumerate(traj_dirs):
            try:
                messages, image_files = _rebuild_conversation(td)
                inputs, input_ids_flat = _tokenize_conversation(messages, image_files, manager)

                image_spans = find_image_token_spans(
                    input_ids_flat, inputs["image_grid_thw"], manager.spatial_merge_size,
                )
                response_spans = find_response_token_spans(
                    input_ids_flat, manager.processor.tokenizer,
                )
                if not response_spans or len(image_spans) < 3:
                    print(f"  [{i+1}] SKIP {td.name}: no response or <3 images")
                    continue

                seq_len = int(input_ids_flat.shape[0])
                img_mask_f = torch.zeros(seq_len, dtype=manager.dtype, device=manager.device)
                for s in image_spans:
                    img_mask_f[s.token_start:s.token_end] = 1
                n_img_tokens = int(img_mask_f.sum().item())

                _CTX.clear()
                _CTX["img_mask_f"] = img_mask_f.float()
                _CTX["response_spans"] = response_spans
                _RESULT.clear()

                with torch.no_grad():
                    manager.model(**inputs, use_cache=False)

                if len(_RESULT) != n_layers:
                    print(f"  [{i+1}] WARN {td.name}: captured {len(_RESULT)}/{n_layers} layers")

                record = {
                    "traj": td.name,
                    "seq_len": seq_len,
                    "n_img_tokens": n_img_tokens,
                    "n_images": len(image_spans),
                    "n_response_turns": len(response_spans),
                    "layers": {str(k): _RESULT[k] for k in sorted(_RESULT.keys())},
                }
                all_records.append(record)

                del inputs, img_mask_f
                torch.cuda.empty_cache()

                if (i + 1) % 25 == 0 or i == 0:
                    el = time.time() - t0
                    s = record["layers"].get("14", {})
                    print(f"  [{i+1}/{len(traj_dirs)}] {el:.0f}s | seq={seq_len} "
                          f"turns={len(response_spans)} imgs={len(image_spans)} "
                          f"resp_frac(L14)={s.get('response_mean_fraction', 0):.4f}")

            except torch.cuda.OutOfMemoryError as e:
                print(f"  [{i+1}] OOM {td.name}: {e}")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [{i+1}] ERROR {td.name}: {e}")
                import traceback
                traceback.print_exc()

        if not all_records:
            print("No records collected!")
            return

        payload = {
            "config": {
                "rollout_dir": rollout_dir,
                "model_path": model_path,
                "layer_indices": layer_indices,
                "attn_implementation": "frac_eager (uniform, query-tiled)",
                "tile": _TILE,
            },
            "records": all_records,
        }
        out_path = out_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(payload, f)
        el = time.time() - t0
        print(f"\n{len(all_records)} trajectories | {n_layers} layers | {el:.0f}s")
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    fire.Fire(AllLayerAttnExtractor)
