#!/usr/bin/env python3
"""Safely download the pinned official DSV4 checkpoint with resume support."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
EXPECTED_REPO_BYTES = 166_898_661_074
DEFAULT_HEADROOM_BYTES = 20 * 1024**3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--headroom-gib", type=float, default=20.0)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is required; install the optional 'hub' dependency"
        ) from error

    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination).free
    headroom_bytes = round(args.headroom_gib * 1024**3)
    existing_bytes = sum(
        path.stat().st_size for path in destination.rglob("*") if path.is_file()
    )
    remaining_bytes = max(0, EXPECTED_REPO_BYTES - existing_bytes)
    required_bytes = remaining_bytes + headroom_bytes
    if free_bytes < required_bytes:
        raise SystemExit(
            f"refusing download: {free_bytes:,} bytes free, "
            f"{required_bytes:,} bytes required including headroom"
        )

    path = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=destination,
        max_workers=args.max_workers,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

