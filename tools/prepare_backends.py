#!/usr/bin/env python3
"""Prepare pinned optional backend sources and apply the shipped patches.

This command fetches source code only. It does not download model weights,
build executables, or start inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = ROOT / "vendor" / "backends.lock.json"
MARKER = ".moeengine-backend.json"


def run(*arguments: str, cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def prepare(backend: dict[str, str], cache_root: Path | None) -> None:
    backend_id = backend["id"]
    destination = ROOT / "vendor" / backend_id
    patch = ROOT / backend["patch"]
    patch_digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    if patch_digest != backend["patch_sha256"]:
        raise ValueError(f"patch checksum mismatch: {patch}")
    expected = {
        "repository": backend["repository"],
        "revision": backend["revision"],
        "patch_sha256": patch_digest,
    }
    if destination.exists():
        marker = destination / MARKER
        if marker.is_file() and json.loads(marker.read_text()) == expected:
            run("git", "apply", "--reverse", "--check", str(patch), cwd=destination)
            print(f"{backend_id}: already prepared; patch verified")
            return
        raise FileExistsError(
            f"refusing to replace existing source tree: {destination}"
        )

    source = backend["repository"]
    if cache_root is not None:
        cached_source = cache_root / backend_id
        if not (cached_source / ".git").exists():
            raise FileNotFoundError(f"Git source cache not found: {cached_source}")
        source = str(cached_source.resolve())

    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=ROOT / "vendor") as temporary:
        checkout = Path(temporary) / "source"
        checkout.mkdir()
        run("git", "init", "--quiet", str(checkout), cwd=ROOT)
        run("git", "remote", "add", "origin", backend["repository"], cwd=checkout)
        run("git", "fetch", "--no-tags", "--depth", "1", source,
            backend["revision"], cwd=checkout)
        revision = subprocess.check_output(
            ["git", "rev-parse", "FETCH_HEAD"], cwd=checkout, text=True
        ).strip()
        if revision != backend["revision"]:
            raise ValueError(f"unexpected upstream revision: {revision}")
        run("git", "checkout", "--quiet", "--detach", revision, cwd=checkout)
        run("git", "apply", "--check", str(patch), cwd=checkout)
        run("git", "apply", str(patch), cwd=checkout)
        run("git", "apply", "--reverse", "--check", str(patch), cwd=checkout)
        # Prepared trees are generated source, not nested Git repositories.
        shutil.rmtree(checkout / ".git")
        (checkout / MARKER).write_text(json.dumps(expected, indent=2) + "\n")
        checkout.rename(destination)
    print(f"{backend_id}: prepared {backend['revision']}")


def main() -> int:
    manifest = json.loads(LOCKFILE.read_text())
    backends = {item["id"]: item for item in manifest["fetched_backends"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backends", nargs="*", choices=sorted(backends))
    parser.add_argument(
        "--cache-root", type=Path,
        help="optional directory containing local Git checkouts, for offline preparation",
    )
    args = parser.parse_args()
    try:
        for backend_id in args.backends or backends:
            prepare(backends[backend_id], args.cache_root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"backend preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
