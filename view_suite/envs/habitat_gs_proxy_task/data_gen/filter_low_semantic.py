"""Drop Habitat-GS samples whose views are not basically recognisable.

The cheap statistics in ``generate_data`` catch a flat or near-black frame, and they
are not enough. The failure this corpus actually produces is a *textured* view with no
content: after a look_down in an outdoor scene the camera sees nothing but cobblestone,
which has healthy variance, ordinary brightness and no dominant colour, and sails
through every scalar test while being impossible to tell apart from the next
cobblestone view. Off-manifold gaussian smear does the same thing -- it is busy, not
blank. Only a judge that can say "I cannot tell where this is" separates them.

The bar is **basic recognisability**: a view must show something a reader could
identify and use to tell this viewpoint from another one.

Two backends. `--backend=openrouter` reuses the AI2-THOR client and needs OPENROUTER_API.
`--backend=cli` shells out to whatever vision-capable CLI VIEW_JUDGE_CMD names, which is
the way in when no API key is available. Either way the rubric is ours: the AI2-THOR one
is written for indoor furniture, and half of this corpus is outdoors.

    python -m view_suite.envs.habitat_gs_proxy_task.data_gen.filter_low_semantic \
        --data_root=$VIEWSUITE_ROOT/data/viewagent15k_habitat_gs --workers=24 \
        --review_dir=$VIEWSUITE_ROOT/gs_filter_review
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import fire

# The OpenRouter client is imported lazily inside run(): it pulls in `requests`, which
# the habitat-gs env does not have and does not need for the default backend.

_DEFAULT_BACKEND_GS = "cli"
# A vision-capable CLI that takes `-g <image> <prompt>` and prints a verdict. Set
# VIEW_JUDGE_CMD to whichever one your site provides; there is no default worth
# hardcoding, and a site-specific tool name does not belong in a public repo.
_JUDGE_CMD = os.environ.get("VIEW_JUDGE_CMD", "")
TASKS = ("path_to_view", "view_to_path", "interactive_view_planning")

GS_FILTER_PROMPT = """You are curating first-person camera views, indoor AND outdoor, \
for a visual spatial-reasoning benchmark. An agent must move a camera to reproduce a \
specific TARGET view, so every view has to be recognisable: someone must be able to \
tell from the image roughly where the camera is and which way it points, and to tell \
this view apart from a different one nearby.

These images are renderings of 3D Gaussian-Splatting reconstructions, so some of them \
are degraded in a particular way: away from the viewpoints the reconstruction was \
trained on, the image dissolves into smeared streaks and floating blobs. That looks \
busy and detailed while showing nothing identifiable. Judge it on recognisability, \
not on how much texture it has.

Judge THIS single image:

- Reply "FILTER" if you cannot tell what you are looking at or where you are. That \
includes: a near-featureless plain surface (blank wall, bare ceiling); a frame filled \
almost entirely by ground -- floor, pavement, cobblestones, grass, road -- with no \
landmark in it, even when that ground is richly textured; smeared or melted geometry \
and floating artefacts from the reconstruction; or an image too dark, washed out or \
blurred to read.
- Reply "KEEP" if the image contains at least one clear, distinctive landmark that \
makes the viewpoint identifiable -- indoors a piece of furniture, appliance, doorway, \
window or fixture; outdoors a building facade, tree, sign, railing, vehicle, street \
furniture or a distinctive skyline.

Two examples. A frame that is 90% cobblestone pavement with a sliver of sky and no \
building -> FILTER. A frame showing a shopfront across a square, even if the lower \
half is pavement -> KEEP.

Answer with exactly one word: KEEP or FILTER."""


def judge_image_cli(path: str, timeout: float = 180.0, max_retries: int = 3,
                    judge_model: Optional[str] = None) -> str:
    """Judge one image by shelling out to VIEW_JUDGE_CMD. KEEP / FILTER / ERROR.

    Exists because an API key is not always available, while a site often provides a
    vision-capable CLI that authenticates on its own -- which also means no secret to
    configure and none to leak. The command must accept `-g <image> <prompt>` and print
    the verdict as its last line; `-m <model>` is passed when judge_model is set.

    One process per image is not elegant -- each pays the CLI's startup, ~9 s a call in
    our case -- but the pool below hides the latency.

    Pass `judge_model` to pin the judge. Left unset you get whatever the CLI currently
    defaults to, which is convenient and NOT reproducible: that default can change
    without notice and a later run would screen the corpus with a different judge,
    silently. Pin it for anything whose provenance matters.
    """
    if not _JUDGE_CMD:
        raise RuntimeError(
            "backend='cli' needs VIEW_JUDGE_CMD set to a vision-capable CLI, "
            "or use --backend=openrouter with OPENROUTER_API")
    cmd = _JUDGE_CMD.split() + ["-d", "-p"]
    if judge_model:
        cmd += ["-m", judge_model]
    cmd += ["-g", os.path.abspath(path), GS_FILTER_PROMPT]
    for attempt in range(max_retries):
        try:
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 text=True, timeout=timeout).stdout
        except subprocess.TimeoutExpired:
            continue
        # The CLI streams; the verdict is the last non-empty line.
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if lines:
            verdict = lines[-1].upper()
            if "FILTER" in verdict:
                return "FILTER"
            if "KEEP" in verdict:
                return "KEEP"
        time.sleep(2.0 * (2 ** attempt))
    return "ERROR"


def _views_to_judge(data_root: str, suffix: str = "") -> Dict[str, List[str]]:
    """{sample_id: [abs image path, ...]} covering every view the sample shows.

    Reads the P2V file as well as the IVP one, and that is the point: an IVP row lists
    only init and target, so judging it alone would leave the three distractors
    unscreened -- and an unreadable distractor makes the multiple choice unanswerable
    just as surely as an unreadable target does.
    """
    out: Dict[str, Set[str]] = {}
    for base in ("path_to_view", "interactive_view_planning"):
        path = os.path.join(data_root, f"{base}{suffix}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                paths = {os.path.normpath(os.path.join(data_root, p))
                         for p in d.get("image_path", [])
                         # The top-down reference is a map, not a view to localise
                         # from; it is rendered far off-manifold by construction and
                         # would be filtered every time.
                         if not p.endswith("top_down.png")}
                out.setdefault(d["sample_id"], set()).update(paths)
    return {k: sorted(v) for k, v in out.items()}


def run(
    data_root: str,
    tasks: Tuple[str, ...] = TASKS,
    backend: str = _DEFAULT_BACKEND_GS,
    model: Optional[str] = None,
    workers: int = 16,
    review_dir: Optional[str] = None,
    suffix: str = "",
    dry_run: bool = False,
    judge_model: Optional[str] = None,
) -> None:
    if backend == "cli":
        def judge(p):
            return judge_image_cli(p, judge_model=judge_model)
        model = judge_model or f"{_JUDGE_CMD} default (UNPINNED -- not reproducible)"
    else:
        from view_suite.envs.ai2thor_proxy_task.data_gen import filter_low_semantic as _base
        model = model or _base._DEFAULT_MODEL[backend]
        auth = _base._openrouter_key() if backend == "openrouter" else _base._user_cert()
        def judge(p):  # noqa: E306
            return _base.judge_image(p, model, auth, backend=backend,
                                     prompt=GS_FILTER_PROMPT)

    per_sample = _views_to_judge(data_root, suffix)
    uniq = sorted({p for ps in per_sample.values() for p in ps})
    print(f"{len(per_sample)} samples, {len(uniq)} distinct views to judge "
          f"({backend}/{model}, {workers} workers)")

    # Resume: judging this corpus is hours, and losing it to one interruption is not
    # acceptable. Verdicts are appended as they land and reloaded on restart.
    cache_path = os.path.join(data_root, f"_verdicts{suffix}.jsonl")
    verdicts: Dict[str, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if d["verdict"] != "ERROR":
                        verdicts[d["path"]] = d["verdict"]
        print(f"resuming: {len(verdicts)} verdicts already on disk")

    todo = [p for p in uniq if p not in verdicts]
    t0 = time.time()
    with open(cache_path, "a") as cache, ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (path, verdict) in enumerate(zip(todo, ex.map(judge, todo)), 1):
            verdicts[path] = verdict
            cache.write(json.dumps({"path": path, "verdict": verdict}) + "\n")
            if i % 200 == 0:
                rate = i / max(1e-6, time.time() - t0)
                cache.flush()
                print(f"  {i}/{len(todo)} judged, {rate:.1f}/s, "
                      f"~{(len(todo) - i) / max(1e-6, rate) / 60:.0f} min left", flush=True)

    n_err = sum(1 for v in verdicts.values() if v == "ERROR")
    n_filter = sum(1 for v in verdicts.values() if v == "FILTER")
    print(f"views: KEEP={len(verdicts) - n_filter - n_err} FILTER={n_filter} ERROR={n_err}")

    # Who judged this corpus, written next to it. Without this the dataset carries no
    # record of what screened it, and "we filtered it with a VLM" is not a provenance.
    with open(os.path.join(data_root, f"_filter_provenance{suffix}.json"), "w") as f:
        json.dump({"backend": backend, "model": model, "workers": workers,
                   "views_judged": len(verdicts), "views_filtered": n_filter,
                   "views_error": n_err, "prompt": GS_FILTER_PROMPT}, f, indent=2)
    if n_err:
        # A view with no verdict is not evidence of quality. Keeping it silently is how
        # a rate-limited run turns into a dataset nobody screened.
        print(f"[warn] {n_err} views got no verdict; their samples are kept. "
              f"Re-run to judge them.")

    drop: Set[str] = {sid for sid, ps in per_sample.items()
                      if any(verdicts.get(p) == "FILTER" for p in ps)}
    print(f"dropping {len(drop)}/{len(per_sample)} samples "
          f"({100.0 * len(drop) / max(1, len(per_sample)):.1f}%)")

    if review_dir:
        os.makedirs(review_dir, exist_ok=True)
        for path, v in verdicts.items():
            if v == "FILTER":
                shutil.copy(path, os.path.join(
                    review_dir, "__".join(path.split(os.sep)[-3:])))
        print(f"filtered views copied to {review_dir} for eyeballing")

    if dry_run:
        print("dry_run: no jsonl rewritten")
        return

    for task in tasks:
        path = os.path.join(data_root, f"{task}{suffix}.jsonl")
        if not os.path.exists(path):
            continue
        kept = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and json.loads(line)["sample_id"] not in drop:
                    kept.append(line)
        # Keep the pre-filter file: regenerating this corpus is hours of GPU time.
        shutil.move(path, path + ".prefilter")
        with open(path, "w") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
        print(f"  {task}: {len(kept)} rows kept")


if __name__ == "__main__":
    fire.Fire(run)
