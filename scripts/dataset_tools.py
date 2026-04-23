from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests


CHUNK_SIZE = 1024 * 1024


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_requests(url: str, dest: Path, overwrite: bool = False) -> Path:
    if dest.exists() and not overwrite:
        print(f"[skip] file exists: {dest}")
        return dest

    ensure_dir(dest.parent)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total > 0:
                    pct = done * 100.0 / total
                    print(f"\r[download] {dest.name}: {pct:6.2f}% ({done}/{total})", end="")
    if total > 0:
        print()
    return dest


def download_with_gdown(url: str, dest: Path, overwrite: bool = False) -> Path:
    if dest.exists() and not overwrite:
        print(f"[skip] file exists: {dest}")
        return dest

    ensure_dir(dest.parent)
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise SystemExit("gdown is required for Google Drive downloads. Install it with `pip install gdown`.") from exc

    output = str(dest)
    result = gdown.download(url=url, output=output, quiet=False, fuzzy=True)
    if result is None:
        raise RuntimeError(f"Failed to download from Google Drive: {url}")
    return Path(result)


def extract_archive(archive_path: Path, dest_dir: Path, overwrite: bool = False) -> Path:
    ensure_dir(dest_dir)
    sentinel = dest_dir / ".extracted"
    if sentinel.exists() and not overwrite:
        print(f"[skip] already extracted: {dest_dir}")
        return dest_dir

    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    sentinel.write_text("ok", encoding="utf-8")
    return dest_dir


def move_contents(src_dir: Path, dest_dir: Path) -> Path:
    ensure_dir(dest_dir.parent)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.move(str(src_dir), str(dest_dir))
    return dest_dir


def find_preferred_subdir(root: Path, candidates: list[str]) -> Optional[Path]:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path

    for item in root.iterdir():
        if item.is_dir() and item.name.lower() in {c.lower() for c in candidates}:
            return item
    return None


def normalize_dataset_dir(extract_root: Path, dataset_name: str, candidates: list[str]) -> Path:
    normalized = extract_root / dataset_name
    if normalized.exists():
        print(f"[ok] normalized dataset dir: {normalized}")
        return normalized

    found = find_preferred_subdir(extract_root, candidates)
    if found is not None:
        move_contents(found, normalized)
        print(f"[ok] normalized dataset dir: {normalized}")
        return normalized

    children = [p for p in extract_root.iterdir() if p.is_dir()]
    if len(children) == 1:
        move_contents(children[0], normalized)
        print(f"[ok] normalized dataset dir: {normalized}")
        return normalized

    raise FileNotFoundError(
        f"Could not infer extracted dataset folder under {extract_root}. "
        f"Looked for {candidates}."
    )
