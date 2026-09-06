"""VLM-based low-semantic-content filter for the AI2-THOR proxy-task dataset.

Some AI2-THOR views (a blank wall / bare floor / ceiling, or a corner with no
recognizable object) carry almost no spatial information, so an agent cannot
localize against them. We drop any *sample* whose **initial view** or **target
view** is judged low-content by a multimodal LLM (Gemini 2.5 Flash, called
through Meta's AI Gateway over mTLS -- no API key, just the x509 user cert).

Pipeline (post-processing over an already-generated dataset):
  1. read interactive_view_planning.jsonl (authoritative per-sample list; it
     carries both init_view and target_view paths in image_detail)
  2. for each sample, ask the VLM to judge its init + target view (KEEP/FILTER)
  3. keep a sample only if BOTH views are USEFUL
  4. rewrite all task jsonls (path_to_view / view_to_path /
     interactive_view_planning) keeping only surviving sample_ids; originals are
     backed up to <stem>.raw.jsonl

Usage:
  conda activate viewagent_thor
  python -m view_suite.envs.ai2thor_proxy_task.data_gen.filter_low_semantic \
      --data_root=$VIEWSUITE_ROOT/data/viewagent15k_ai2thor --workers=16
  # then re-split with view_suite.envs.utils.split_jsonl_by_scene
"""
from __future__ import annotations

import base64
import json
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import requests

# --- Backends ---------------------------------------------------------------
# "openrouter": external OpenRouter API (needs an OPENROUTER_API key, and an
#     egress proxy on networks that require one). Real quota -> the recommended path.
# "ai_gateway": an internal mTLS gateway to Vertex/Gemini (no key, but the shared
#     `playground` gateway is heavily rate-limited).
_DEFAULT_BACKEND = "openrouter"
_DEFAULT_MODEL = {"openrouter": "qwen/qwen3.7-plus", "ai_gateway": "gemini-2.5-flash"}

# OpenRouter (external). Set EGRESS_PROXY where outbound traffic needs a proxy;
# unset means connect directly.
_OR_URL = "https://openrouter.ai/api/v1/chat/completions"
_EGRESS_PROXY = os.environ.get("EGRESS_PROXY") or os.environ.get("HTTPS_PROXY")
_OR_PROXIES = ({"http": _EGRESS_PROXY, "https": _EGRESS_PROXY}
               if _EGRESS_PROXY else None)

# AI Gateway (Vertex/Gemini) config.
# Base URL of an internal/self-hosted Gemini-compatible gateway. No default: the
# endpoint is site-specific and does not belong in a public repo.
_GATEWAY_BASE = os.environ.get("AI_GATEWAY_BASE_URL", "")
_GCP_PROJECT = os.environ.get("AI_GATEWAY_PROJECT", "")


def _gateway_url(model: str) -> str:
    return (
        f"{_GATEWAY_BASE}"
        f"/v1/projects/{_GCP_PROJECT}/locations/global"
        f"/publishers/google/models/{model}:generateContent"
    )


def _user_cert() -> str:
    user = subprocess.check_output(["whoami"]).decode().strip()
    return f"/var/facebook/credentials/{user}/x509/{user}.pem"


def _openrouter_key() -> str:
    """Read the OpenRouter key from env or the repo .env (OPENROUTER_API)."""
    for k in ("OPENROUTER_API", "OPENROUTER_API_KEY", "OPENROUTER_KEY"):
        if os.environ.get(k):
            return os.environ[k]
    for envp in (
        os.path.join(os.environ.get("VIEWSUITE_ROOT", ""), "..", ".env"),
        "$ENV_FILE",
    ):
        if envp and os.path.exists(envp):
            for line in open(envp):
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in ("OPENROUTER_API", "OPENROUTER_API_KEY", "OPENROUTER_KEY"):
                    return v.strip().strip('"').strip("'")
    raise RuntimeError("OpenRouter key not found (set OPENROUTER_API in env or .env)")


# --- The filter prompt ------------------------------------------------------
# One image in, one word out. We judge the init view and target view separately;
# a sample survives only if BOTH are USEFUL.
FILTER_PROMPT = """You are curating first-person indoor camera views for a visual \
spatial-reasoning benchmark. In the task, an agent must move a camera to \
reproduce a specific TARGET view, so every view must contain distinctive, \
recognizable content that lets someone identify *where* the camera is and \
*which way* it points.

Judge THIS single image:

- Reply "FILTER" if the image is low-content: dominated by a plain, near-\
featureless surface such as a blank wall, bare floor, or ceiling; a bare corner \
with no identifiable object; or so dark/washed-out/blurry that you cannot tell \
what you are looking at. Such views give an agent nothing to localize against.
- Reply "KEEP" if the image contains at least one clear, distinctive object, \
piece of furniture, appliance, or structural landmark (e.g. a table, chair, \
sink, bed, cabinet, window, doorway with surroundings) that makes the viewpoint \
identifiable.

Example: an image that is almost entirely a beige wall with only a faint edge \
and no identifiable object -> FILTER.

Answer with exactly one word: KEEP or FILTER."""


def _mime(path: str) -> str:
    return "image/png" if path.lower().endswith(".png") else "image/jpeg"


def _request_verdict(backend: str, model: str, b64: str, mime: str, auth, timeout: float,
                     prompt: str = None):
    """One HTTP call; returns (status_code, verdict_text_or_None). Raises on network error.

    `prompt` defaults to FILTER_PROMPT. It is a parameter so another corpus can supply
    its own rubric without duplicating this client -- Habitat-GS needs one that also
    covers outdoor scenes and gaussian-splatting smear.
    """
    prompt = prompt if prompt is not None else FILTER_PROMPT
    if backend == "openrouter":
        body = {
            "model": model, "temperature": 0.0, "max_tokens": 8,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
        }
        r = requests.post(
            _OR_URL, headers={"Authorization": f"Bearer {auth}", "Content-Type": "application/json"},
            json=body, proxies=_OR_PROXIES, timeout=timeout,
        )
        if r.status_code == 200:
            return 200, r.json()["choices"][0]["message"]["content"]
        return r.status_code, None
    else:  # ai_gateway (Vertex/Gemini over mTLS)
        body = {
            "contents": [{"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": b64}},
            ]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8,
                                 "thinkingConfig": {"thinkingBudget": 0}},
        }
        r = requests.post(
            _gateway_url(model), cert=auth,
            headers={"content-type": "application/json",
                     "x-calling-product": os.environ.get("AI_GATEWAY_PRODUCT", "viewsuite-data-filter")},
            json=body, timeout=timeout,
        )
        if r.status_code == 200:
            return 200, r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return r.status_code, None


def judge_image(path: str, model: str, auth, backend: str = _DEFAULT_BACKEND,
                timeout: float = 60.0, max_retries: int = 8,
                prompt: str = None) -> str:
    """Return 'KEEP' / 'FILTER' for one image, or 'ERROR' if all retries fail.

    Retries on 429 / transient 5xx with exponential backoff + jitter. 'ERROR'
    (never a silent KEEP) lets the caller tell "judged KEEP" from "no verdict".
    """
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = _mime(path)
    for attempt in range(max_retries):
        try:
            code, txt = _request_verdict(backend, model, b64, mime, auth, timeout, prompt)
            if code == 200:
                if not txt:
                    return "ERROR"
                return "FILTER" if "FILTER" in txt.strip().upper() else "KEEP"
            if code in (429, 500, 502, 503, 504):
                time.sleep(min(2.0 * (2 ** attempt), 30.0) + random.uniform(0, 1.5))
                continue
            print(f"[warn] {path}: HTTP {code}")
            return "ERROR"
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                print(f"[warn] judge exhausted retries for {path}")
                return "ERROR"
            time.sleep(min(2.0 * (2 ** attempt), 30.0) + random.uniform(0, 1.5))
        except Exception as e:
            print(f"[warn] judge failed for {path}: {type(e).__name__}: {e}")
            return "ERROR"
    return "ERROR"


def _sample_views(data_root: str, ivp_jsonl: str) -> List[Tuple[str, str, str]]:
    """Return [(sample_id, init_abs_path, target_abs_path), ...] from the IVP jsonl."""
    out = []
    with open(ivp_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            det = d["image_detail"]
            init_rel = det["init_view"]["path"]
            tgt_rel = det["target_view"]["path"]
            out.append((
                d["sample_id"],
                os.path.normpath(os.path.join(data_root, init_rel)),
                os.path.normpath(os.path.join(data_root, tgt_rel)),
            ))
    return out


def run(
    data_root: str,
    tasks: Tuple[str, ...] = ("path_to_view", "view_to_path", "interactive_view_planning"),
    backend: str = _DEFAULT_BACKEND,
    model: Optional[str] = None,
    workers: int = 12,
    dry_run: bool = False,
    review_dir: Optional[str] = None,
    review_keep_sample: int = 40,
    cache_path: Optional[str] = None,
):
    """Filter low-semantic samples out of the AI2-THOR proxy-task jsonls.

    backend: "openrouter" (default; needs OPENROUTER_API, plus EGRESS_PROXY on
        networks that require one) or "ai_gateway" (internal mTLS -> Gemini;
        the shared gateway is rate-limited).
    model: VLM id; default per backend (openrouter -> qwen/qwen3.7-plus).
    workers: OpenRouter tolerates ~12; drop to ~2 for the ai_gateway playground.
    review_dir: if set, copy judged images into <review_dir>/{keep,filter}/ for
        a human to eyeball (all FILTER + a sample of KEEP).
    cache_path: JSON verdict cache (default <data_root>/.filter_verdicts.json);
        judged images are skipped on re-run.
    """
    model = model or _DEFAULT_MODEL.get(backend, _DEFAULT_MODEL["openrouter"])
    auth = _openrouter_key() if backend == "openrouter" else _user_cert()
    ivp_jsonl = os.path.join(data_root, "interactive_view_planning.jsonl")
    samples = _sample_views(data_root, ivp_jsonl)
    print(f"[filter] {len(samples)} samples; judging init+target via "
          f"{backend}:{model} (workers={workers})")

    # Unique image list (init + target of every sample).
    uniq: Dict[str, Optional[str]] = {}
    for _, ip, tp in samples:
        uniq.setdefault(ip, None)
        uniq.setdefault(tp, None)

    # Verdict cache: skip already-judged images on re-run.
    cache_path = cache_path or os.path.join(data_root, ".filter_verdicts.json")
    cache: Dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}
    todo = [p for p in uniq if cache.get(p) not in ("KEEP", "FILTER")]
    print(f"[filter] {len(uniq)} unique imgs; {len(uniq) - len(todo)} cached, {len(todo)} to judge")
    if todo:
        import threading
        lock = threading.Lock()
        done = [0]
        def _judge(p):
            v = judge_image(p, model, auth, backend=backend)
            with lock:
                cache[p] = v
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(todo):
                    json.dump(cache, open(cache_path, "w"), indent=0)
                    print(f"    judged {done[0]}/{len(todo)}", flush=True)
            return v
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_judge, todo))
        json.dump(cache, open(cache_path, "w"), indent=0)
    uniq = {p: cache.get(p, "ERROR") for p in uniq}

    # Optional: dump images into keep/ and filter/ review folders for human eyeball.
    if review_dir:
        for sub in ("keep", "filter"):
            d = os.path.join(review_dir, sub)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        keeps = [p for p, v in uniq.items() if v == "KEEP"]
        filts = [p for p, v in uniq.items() if v == "FILTER"]
        rng = random.Random(0)
        rng.shuffle(keeps)
        sel_keep = keeps[:review_keep_sample] if review_keep_sample > 0 else keeps
        def _flat(p):  # scene_sample_viewname.png
            parts = p.replace(data_root, "").strip("/").split("/")
            return "_".join(parts)
        for p in sel_keep:
            shutil.copy(p, os.path.join(review_dir, "keep", _flat(p)))
        for p in filts:
            shutil.copy(p, os.path.join(review_dir, "filter", _flat(p)))
        print(f"[filter] review dump -> {review_dir}/  (keep sample={len(sel_keep)}, filter={len(filts)})")

    # Drop a sample only on an EXPLICIT FILTER verdict; ERROR (retries exhausted)
    # is treated as keep so we never lose good data to a flaky API.
    kept_ids, dropped = set(), []
    for sid, ip, tp in samples:
        vi, vt = uniq[ip], uniq[tp]
        if vi == "FILTER" or vt == "FILTER":
            why = [w for w, v in (("init", vi), ("target", vt)) if v == "FILTER"]
            dropped.append((sid, "+".join(why)))
        else:
            kept_ids.add(sid)

    n_img_filt = sum(1 for v in uniq.values() if v == "FILTER")
    n_img_err = sum(1 for v in uniq.values() if v == "ERROR")
    print(f"[filter] images: {n_img_filt}/{len(uniq)} low-content, "
          f"{n_img_err} un-judged (API errors, kept)")
    print(f"[filter] samples: keep {len(kept_ids)}, drop {len(dropped)} / {len(samples)}")
    for sid, why in dropped[:20]:
        print(f"    drop {sid}  ({why})")
    if len(dropped) > 20:
        print(f"    ... (+{len(dropped) - 20} more)")

    if dry_run:
        print("[filter] dry-run: no files written")
        return {"kept": len(kept_ids), "dropped": len(dropped)}

    for stem in tasks:
        src = os.path.join(data_root, f"{stem}.jsonl")
        if not os.path.exists(src):
            print(f"[filter] skip missing {src}")
            continue
        rows = [json.loads(l) for l in open(src) if l.strip()]
        keep = [r for r in rows if r.get("sample_id") in kept_ids]
        raw = os.path.join(data_root, f"{stem}.raw.jsonl")
        if not os.path.exists(raw):
            os.rename(src, raw)  # back up original once
        with open(src, "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        print(f"[filter] {stem}: {len(rows)} -> {len(keep)} rows  (backup: {os.path.basename(raw)})")

    return {"kept": len(kept_ids), "dropped": len(dropped)}


if __name__ == "__main__":
    import fire
    fire.Fire(run)
