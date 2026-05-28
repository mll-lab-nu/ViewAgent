#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fire-based CLI to download *.tar.gz files from a (private) Hugging Face dataset repo,
extract them into an output directory, and delete the archives afterward.
"""

import os

# These must be set BEFORE importing huggingface_hub: both are frozen into
# module constants at import time, so setting them afterwards has no effect.
#   HF_HUB_ENABLE_HF_TRANSFER: fast multi-stream download (else slow single HTTP).
#   HF_HUB_DOWNLOAD_TIMEOUT:   read timeout (s) so a stalled connection RAISES
#                              instead of hanging forever -> lets retry kick in.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import sys
import time
import tarfile
from pathlib import Path
from typing import Iterable, Sequence

import fire
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


def _ensure_out_dir(path: Path) -> None:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _is_within_directory(directory: Path, target: Path) -> bool:
    """Prevent path traversal by ensuring 'target' stays under 'directory'."""
    directory = directory.resolve()
    try:
        target = target.resolve()
    except FileNotFoundError:
        # If target doesn't exist yet, validate its parent
        target = target.parent.resolve()
    return str(target).startswith(str(directory))


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    """Safely extract all members, rejecting path-traversal entries."""
    for member in tar.getmembers():
        dest = path / member.name
        if not _is_within_directory(path, dest):
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    tar.extractall(path)


def _download_one(
    repo_id: str,
    filename: str,
    out_dir: Path,
    repo_type: str,
    revision: str,
    token: str | None,
    max_retries: int = 8,
) -> Path:
    """
    Download a single file into out_dir, retrying on transient failures.

    Each retry resumes from the partial .incomplete file, so progress is not
    lost. The first attempt uses hf_transfer (fast); once a partial exists,
    huggingface_hub falls back to regular HTTP + resume on later attempts.
    """
    _ensure_out_dir(out_dir)
    delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            return Path(hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                revision=revision,
                token=token,
                local_dir=str(out_dir),
            ))
        except HfHubHTTPError as e:
            # 4xx (404/401/...) are real errors; retrying is pointless.
            code = getattr(e.response, "status_code", 0)
            if code and code < 500 and code != 429:
                raise
            err = e
        except (ConnectionError, TimeoutError, OSError) as e:
            err = e

        if attempt == max_retries:
            raise RuntimeError(
                f"{filename}: failed after {max_retries} attempts"
            ) from err
        print(
            f"[retry {attempt}/{max_retries}] {filename}: "
            f"{type(err).__name__}: {err} -- resuming in {delay}s",
            file=sys.stderr,
        )
        time.sleep(delay)
        delay = min(delay * 2, 120)  # 5,10,20,...,capped at 120s


def _normalize_files(files: Sequence[str] | str) -> list[str]:
    """
    Accept either:
      - a sequence of filenames, or
      - a single comma-separated string: "a.tar.gz,b.tar.gz"
    """
    if isinstance(files, str):
        # Split by comma and strip whitespace
        return [x.strip() for x in files.split(",") if x.strip()]
    return list(files)


def cli(
    files: str="viewsuite_15k.tar.gz",
    repo: str="MLL-Lab/viewsuite",
    out: str="data/",
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | None = None,
):
    """
    Download -> Extract -> Cleanup for multiple tar.gz files.

    Args:
      repo: HF repo_id, e.g. "user_or_org/my-dataset"
      out: Output directory to extract into
      *files: One or more archive names in the repo (e.g., "a.tar.gz", "b.tar.gz")
      repo_type: "dataset" | "model" | "space" (default: dataset)
      revision: Git branch/tag/sha (default: main)
      token: HF access token (defaults to $HF_TOKEN if not provided)

    Usage patterns:
      python script.py REPO OUT file1.tar.gz file2.tar.gz ...
      python script.py REPO OUT "file1.tar.gz,file2.tar.gz"   # comma-separated single arg
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")

    out_dir = Path(out).expanduser().resolve()
    names = _normalize_files(files if files else ())

    if not names:
        print("[ERROR] No files provided. Pass at least one *.tar.gz.", file=sys.stderr)
        sys.exit(2)

    for fname in names:
        if not fname.endswith(".tar.gz"):
            print(f"[WARN] '{fname}' does not end with .tar.gz; proceeding anyway...", file=sys.stderr)

        print(f"[1/3] Downloading: {fname}")
        archive_path = _download_one(repo, fname, out_dir, repo_type, revision, token)
        if not archive_path.exists():
            raise FileNotFoundError(f"Downloaded file not found: {archive_path}")

        print(f"[2/3] Extracting: {archive_path.name} -> {out_dir}")
        # Select extraction mode
        mode = "r:gz" if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".gz" else "r:*"
        with tarfile.open(archive_path, mode) as tar:
            _safe_extract(tar, out_dir)

        print(f"[3/3] Deleting archive: {archive_path.name}")
        try:
            archive_path.unlink()
        except Exception as e:
            print(f"[WARN] Failed to delete {archive_path}: {e}", file=sys.stderr)

    print(f"Done. Extracted contents are under: {out_dir}")


if __name__ == "__main__":
    # Fire maps CLI -> function arguments. See examples in the module docstring.
    fire.Fire(cli)
