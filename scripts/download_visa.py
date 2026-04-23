from __future__ import annotations

import argparse
from pathlib import Path

from dataset_tools import download_with_requests, ensure_dir, extract_archive, normalize_dataset_dir, sha256sum


DEFAULT_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the VisA dataset.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Direct URL for the VisA archive.")
    parser.add_argument("--data-root", default="data", help="Root directory that will contain raw archives and extracted datasets.")
    parser.add_argument("--archive-name", default="VisA_20220922.tar", help="File name for the downloaded archive.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download and re-extract even if files already exist.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    raw_dir = ensure_dir(data_root / "raw")
    extract_root = ensure_dir(data_root / "extracted" / "visa")
    archive_path = raw_dir / args.archive_name

    archive_path = download_with_requests(args.url, archive_path, overwrite=args.overwrite)
    print(f"[info] archive: {archive_path}")
    print(f"[info] sha256: {sha256sum(archive_path)}")

    extract_archive(archive_path, extract_root, overwrite=args.overwrite)
    final_dir = normalize_dataset_dir(
        extract_root,
        dataset_name="visa",
        candidates=["visa", "VisA", "VisA_20220922", "VisA_pytorch", "VisA_20220922_pytorch"],
    )
    print(f"[done] VisA ready at: {final_dir}")


if __name__ == "__main__":
    main()
