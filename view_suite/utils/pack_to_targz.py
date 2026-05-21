#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast packer: prefers system `tar` + parallel gzip (`pigz`), falls back to Python tarfile.

Defaults:
  folder  = data/scannet        # relative to current working directory
  output  = scannet.tar.gz
  include = ["*.ply"]           # set to None/"none" to keep all files
  no_follow_symlinks = True
  level = 6                     # compression level (1..9)
  threads = 0                   # pigz threads; 0 = auto

Examples:
  # Use defaults (only *.ply, preserve top-level folder name)
  python pack_to_targz_fast.py

  # Keep all files (ignore patterns)
  python pack_to_targz_fast.py --include=None

  # Multiple patterns (comma or whitespace separated)
  python pack_to_targz_fast.py --include="*.ply,*.sens,*.txt"

  # Change folder/output; both can be relative paths
  python pack_to_targz_fast.py --folder=data/scannet --output=out/scannet.tar.gz

  # Follow symlinks (archive link targets instead of symlink entries)
  python pack_to_targz_fast.py --no_follow_symlinks=False

  # Faster but lower compression
  python pack_to_targz_fast.py --level=3

  # Limit pigz threads (default uses all cores)
  python pack_to_targz_fast.py --threads=8
"""

import os
import sys
import tarfile
import shutil
import subprocess
from pathlib import Path
from fnmatch import fnmatch
from typing import List, Optional, Union


def _normalize_include(include: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize the include patterns into a list of strings."""
    if include is None:
        return []
    if isinstance(include, str):
        s = include.strip()
        if s == "" or s.lower() == "none":
            return []
        parts = [p.strip() for chunk in s.split(",") for p in chunk.split()]
        return [p for p in parts if p]
    return [p for p in include if isinstance(p, str) and p.strip()]


def _python_tar(
    folder: Path,
    output: Path,
    patterns: List[str],
    no_follow_symlinks: bool,
    level: int,
):
    """
    Python fallback: single-threaded gzip via tarfile.
    Preserves the top-level folder name by using arcname=folder.name.
    """
    deref = not no_follow_symlinks
    try:
        tf = tarfile.open(
            output, mode="w:gz",
            compresslevel=max(1, min(level, 9)),
            dereference=deref
        )
    except TypeError:
        # Older Python: `dereference` not supported in tarfile.open()
        tf = tarfile.open(output, mode="w:gz")

    def _filter(ti: tarfile.TarInfo):
        # Always keep directories to preserve structure
        if ti.isdir():
            return ti
        # If no patterns, keep all files
        if not patterns:
            return ti
        base = ti.name.rsplit("/", 1)[-1]
        for pat in patterns:
            if fnmatch(base, pat) or fnmatch(ti.name, pat):
                return ti
        # Drop non-matching files
        return None

    with tf:
        tf.add(folder, arcname=folder.name, recursive=True, filter=_filter)


def _system_tar_all(
    folder: Path, output: Path, no_follow_symlinks: bool, level: int, threads: int
):
    """
    System path: tar the entire folder.
    Key rule: position-sensitive options must precede the file arguments.
    We use `-C <parent>` BEFORE passing the top-level folder name.
    """
    parent = str(folder.parent)
    top = folder.name
    tar_bin = shutil.which("tar")
    if not tar_bin:
        raise FileNotFoundError("`tar` not found in PATH")

    pigz = shutil.which("pigz")
    if pigz:
        pigz_arg = f"{pigz} -{max(1, min(level, 9))}"
        if threads and threads > 0:
            pigz_arg += f" -p {threads}"
        # Correct order: [-C parent] [--dereference?] [-I pigz ...] [-cf output] [top]
        cmd = [tar_bin, "-C", parent, "-I", pigz_arg, "-cf", str(output), top]
        if not no_follow_symlinks:
            cmd.insert(1, "--dereference")
        return subprocess.run(cmd, check=True)
    else:
        # Use -z and pass GZIP level via environment
        env = os.environ.copy()
        env["GZIP"] = f"-{max(1, min(level, 9))}"
        cmd = [tar_bin, "-C", parent, "-czf", str(output), top]
        if not no_follow_symlinks:
            cmd.insert(1, "--dereference")
        return subprocess.run(cmd, check=True, env=env)


def _system_tar_filtered(
    folder: Path, output: Path, patterns: List[str], no_follow_symlinks: bool, level: int, threads: int
):
    """
    System path: feed a filtered file list to tar via `find ... -print0 | tar --null --files-from=-`.
    Key rules:
      - Run `find` in the parent directory so paths start with the top folder name.
      - Place `-C <parent>` BEFORE `--files-from=-` (tar's position-sensitive option).
      - When using `-z`, set GZIP level via environment and pass env to tar.
    """
    parent = str(folder.parent)
    top = folder.name
    tar_bin = shutil.which("tar")
    if not tar_bin:
        raise FileNotFoundError("`tar` not found in PATH")

    find_bin = shutil.which("find")
    if not find_bin:
        raise FileNotFoundError("`find` not found in PATH")

    # Build `find` args in the parent dir to output "top/..." relative paths
    find_args = [find_bin, top, "-type", "f"]
    if patterns:
        find_args += ["("]
        for i, pat in enumerate(patterns):
            if i > 0:
                find_args += ["-o"]
            find_args += ["-name", pat]
        find_args += [")"]
    find_args += ["-print0"]

    pigz = shutil.which("pigz")

    # Correct order for tar: [-C parent] [--dereference?] [compression opts] [-cf output] [--null --files-from=-]
    tar_args = [tar_bin, "-C", parent]
    if not no_follow_symlinks:
        tar_args.append("--dereference")

    env = os.environ.copy()
    if pigz:
        pigz_arg = f"{pigz} -{max(1, min(level, 9))}"
        if threads and threads > 0:
            pigz_arg += f" -p {threads}"
        tar_args += ["-I", pigz_arg]
    else:
        tar_args += ["-z"]
        env["GZIP"] = f"-{max(1, min(level, 9))}"

    tar_args += ["-cf", str(output), "--null", "--files-from=-"]

    # Pipeline: (cd parent; find ...) | tar ...
    p_find = subprocess.Popen(find_args, cwd=parent, stdout=subprocess.PIPE)
    try:
        p_tar = subprocess.Popen(tar_args, stdin=p_find.stdout, env=env)
        p_find.stdout.close()
        rc_tar = p_tar.wait()
        rc_find = p_find.wait()
        if rc_find != 0 or rc_tar != 0:
            raise subprocess.CalledProcessError(rc_tar or rc_find, tar_args)
    finally:
        try:
            p_find.kill()
        except Exception:
            pass


def pack(
    folder: str = "data/scannet",
    output: str = "scannet.tar.gz",
    include: Optional[Union[str, List[str]]] = "*.ply",
    no_follow_symlinks: bool = True,
    level: int = 6,
    threads: int = 0,
    method: str = "auto",  # "auto" | "system" | "python"
):
    """
    Pack a folder into .tar.gz, preserving the top-level folder name.
    Works with relative paths (relative to current working directory).
    """
    folder_p = Path(folder).resolve()
    if not folder_p.exists() or not folder_p.is_dir():
        raise SystemExit(f"[Error] Invalid folder: {folder_p}")

    output_p = Path(output)
    if not str(output_p).endswith(".tar.gz"):
        output_p = output_p.with_suffix(".gz") if output_p.suffix == ".tar" else Path(str(output_p) + ".tar.gz")
    output_p = output_p.resolve()
    output_p.parent.mkdir(parents=True, exist_ok=True)

    patterns = _normalize_include(include)
    print(f"[Info] Method={method}  Compressor={'pigz' if shutil.which('pigz') else 'gzip/Python'}  Level={level}  Threads={threads}")
    print(f"[Info] Include={patterns if patterns else '(none) -> keep all'}  no_follow_symlinks={no_follow_symlinks}")
    print(f"[Info] Folder={folder_p}  Output={output_p}")

    if method in ("auto", "system"):
        try:
            if patterns:
                _system_tar_filtered(folder_p, output_p, patterns, no_follow_symlinks, level, threads)
            else:
                _system_tar_all(folder_p, output_p, no_follow_symlinks, level, threads)
            print(f"[OK] Created (system): {output_p}")
            return str(output_p)
        except Exception as e:
            if method == "system":
                raise
            print(f"[Warn] System path failed, falling back to Python: {e}", file=sys.stderr)

    # Python fallback
    _python_tar(folder_p, output_p, patterns, no_follow_symlinks, level)
    print(f"[OK] Created (python): {output_p}")
    return str(output_p)


def main(
    folder: str = "data/viewsuite_15k",
    output: str = "viewsuite_15k.tar.gz",
    include: Optional[Union[str, List[str]]] = None,
    no_follow_symlinks: bool = True,
    level: int = 1,
    threads: int = 16,
    method: str = "auto",
):
    """Entry point exposed via Fire."""
    return pack(folder, output, include, no_follow_symlinks, level, threads, method)


if __name__ == "__main__":
    try:
        import fire
    except Exception as e:
        raise SystemExit("Please install Google Fire first: pip install fire\n" + str(e))
    fire.Fire(main)
