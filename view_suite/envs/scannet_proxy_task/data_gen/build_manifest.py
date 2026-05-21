"""
Build a unified manifest from multiple viewsuite jsonl files.

Scans each input jsonl, extracts unique (scene_id, sample_id) pairs, and writes
a manifest jsonl with one row per unique sample. Each row lists:
  - scene_id, sample_id, sample_dir (relative path under viewsuite_15k/)
  - images: standard 5 files to render (initial_view + option_000..003)
  - sources: which input jsonls referenced this sample

Also prints unique scene_id list (useful for the download script).

Usage:
    python -m view_suite.envs.scannet_proxy_task.data_gen.build_manifest \
        --jsonls viewsuite_15k/active_explore_test.jsonl \
                 viewsuite_15k/inverse_dynamics_test.jsonl \
                 viewsuite_15k/forward_dynamics_test.jsonl \
        --out  ViewSuite/view_suite/envs/scannet_proxy_task/data_gen/viewsuite_15k_gs_test_manifest.jsonl

Defaults (no args) resolve to the three *_test.jsonl files under
data/viewsuite_15k/ and write the manifest next to this file.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Iterable, Sequence


# ---- Defaults ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DATA = "/root/projects/viewsuite/data/viewsuite_15k"
DEFAULT_JSONLS = [
    os.path.join(_REPO_DATA, "active_explore_test.jsonl"),
    os.path.join(_REPO_DATA, "inverse_dynamics_test.jsonl"),
    os.path.join(_REPO_DATA, "forward_dynamics_test.jsonl"),
]
DEFAULT_OUT = os.path.join(_THIS_DIR, "viewsuite_15k_gs_test_manifest.jsonl")

# ScanNet viewsuite samples have a fixed set of images per sample_dir.
STANDARD_SAMPLE_IMAGES = (
    "initial_view.png",
    "option_000.png",
    "option_001.png",
    "option_002.png",
    "option_003.png",
)


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build(jsonls: Sequence[str], out_path: str) -> dict:
    # sample_key -> {"scene_id", "sample_id", "sample_dir", "sources": set()}
    samples: dict[tuple[str, str], dict] = {}

    for jp in jsonls:
        if not os.path.exists(jp):
            raise FileNotFoundError(f"jsonl not found: {jp}")
        src_tag = os.path.splitext(os.path.basename(jp))[0]
        n_lines = 0
        for row in _iter_jsonl(jp):
            scene_id = row["scene_id"]
            sample_id = row["sample_id"]
            # Some sample_id values include the scene prefix; derive sample_dir
            # deterministically from image_path (first entry starts with "scene*/sample_*/").
            first_img = row["image_path"][0]
            sample_dir = os.path.dirname(first_img)
            key = (scene_id, sample_dir)
            entry = samples.setdefault(
                key,
                {
                    "scene_id": scene_id,
                    "sample_id": sample_id,
                    "sample_dir": sample_dir,
                    "sources": set(),
                },
            )
            entry["sources"].add(src_tag)
            n_lines += 1
        print(f"[scan] {jp}  rows={n_lines}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scene_counts: dict[str, int] = defaultdict(int)
    with open(out_path, "w") as f:
        for (scene_id, sample_dir) in sorted(samples.keys()):
            e = samples[(scene_id, sample_dir)]
            row = {
                "scene_id": e["scene_id"],
                "sample_id": e["sample_id"],
                "sample_dir": e["sample_dir"],
                "images": list(STANDARD_SAMPLE_IMAGES),
                "sources": sorted(e["sources"]),
            }
            f.write(json.dumps(row) + "\n")
            scene_counts[scene_id] += 1

    summary = {
        "num_samples": len(samples),
        "num_scenes": len(scene_counts),
        "out": out_path,
        "top_scenes_by_sample_count": sorted(
            scene_counts.items(), key=lambda kv: -kv[1]
        )[:5],
    }
    print(
        f"[done] samples={summary['num_samples']}  scenes={summary['num_scenes']}  "
        f"→ {out_path}"
    )
    return summary


def _read_scene_ids(manifest_path: str) -> list[str]:
    """Convenience: unique scene_ids from a manifest, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    with open(manifest_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["scene_id"]
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def main(
    jsonls: Sequence[str] | None = None,
    out: str | None = None,
) -> None:
    jsonls = list(jsonls) if jsonls else DEFAULT_JSONLS
    out = out or DEFAULT_OUT
    build(jsonls, out)


if __name__ == "__main__":
    import fire
    fire.Fire(main)