from __future__ import annotations

import argparse
import os
from pathlib import Path

from dataset_tools import ensure_dir, extract_archive, normalize_dataset_dir, sha256sum


DEFAULT_URL = "https://drive.google.com/file/d/1JhhA36qmH8lKCgiX9lU6v8D7B1Y3Xa7r/view?usp=drive_link"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the MVTec dataset.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Google Drive share link for the MVTec archive.")
    parser.add_argument("--data-root", default="data", help="Root directory that will contain raw archives and extracted datasets.")
    parser.add_argument("--archive-name", default="", help="Optional archive name override. Leave empty to keep the file name inferred from Drive.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download and re-extract even if files already exist.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    raw_dir = ensure_dir(data_root / "raw")
    extract_root = ensure_dir(data_root / "extracted" / "mvtec")
    archive_path = download_mvtec_archive(args.url, raw_dir, args.archive_name, overwrite=args.overwrite)
    print(f"[info] archive: {archive_path}")
    print(f"[info] sha256: {sha256sum(archive_path)}")

    extract_archive(archive_path, extract_root, overwrite=args.overwrite)
    final_dir = normalize_dataset_dir(
        extract_root,
        dataset_name="mvtec",
        candidates=["mvtec", "mvtec_anomaly_detection", "MVTec-AD", "MVTec_AD"],
    )
    print(f"[done] MVTec ready at: {final_dir}")


def download_mvtec_archive(url: str, raw_dir: Path, archive_name: str, overwrite: bool = False) -> Path:
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise SystemExit("gdown is required for Google Drive downloads. Install it with `pip install gdown`.") from exc

    if archive_name:
        dest = raw_dir / archive_name
        if dest.exists() and not overwrite:
            print(f"[skip] file exists: {dest}")
            return dest
        result = gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)
        if result is None:
            raise RuntimeError(f"Failed to download from Google Drive: {url}")
        return Path(result)

    old_cwd = Path.cwd()
    try:
        os.chdir(raw_dir)
        result = gdown.download(url=url, output=None, quiet=False, fuzzy=True)
        if result is None:
            raise RuntimeError(f"Failed to download from Google Drive: {url}")
        archive_path = raw_dir / result
    finally:
        os.chdir(old_cwd)

    if archive_path.exists() and overwrite:
        print(f"[info] overwrite requested; keeping latest downloaded archive at {archive_path}")
    return archive_path


if __name__ == "__main__":
    main()
