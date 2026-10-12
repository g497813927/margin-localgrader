#!/usr/bin/env python3
"""Explicit one-time Clef-Flash download. Never imports or reads grading data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "ggml-org/Clef-Flash-GGUF"
REVISION = "6b4fe4c59098479652d94f0fe11a40facbebd1ec"
FILENAME = "Clef-Flash-Q4_K_M.gguf"
SIZE = 6_486_448_288
SHA256 = "3626116623ca8d89641952471fac9ce12af84fb3517520029a2ac6f5146868a2"
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}?download=true"


def verify_model(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != SIZE:
        raise ValueError(f"The complete model must contain exactly {SIZE:,} bytes. Run scripts/setup_clef.py to finish setup.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError("Clef model checksum mismatch. The file will not be used. Move the corrupt file aside and run setup again.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/clef-flash")
    parser.add_argument("--offline", action="store_true", help="Verify existing complete local files only; never access the network.")
    args = parser.parse_args()
    directory = args.model_dir.expanduser().resolve()
    model = directory / FILENAME
    partial = directory / (FILENAME + ".part")
    if not model.is_file():
        if not partial.is_file() or partial.stat().st_size != SIZE:
            if args.offline:
                parser.exit(1, "Clef-Flash is not fully downloaded. Offline verification cannot download missing weights.\n")
            curl = shutil.which("curl")
            if not curl:
                parser.exit(1, "Install curl first, then rerun this explicit setup command.\n")
            directory.mkdir(parents=True, exist_ok=True)
            present = partial.stat().st_size if partial.exists() else 0
            if present > SIZE:
                parser.exit(1, "The partial file is larger than the published model. Move it aside before retrying.\n")
            if shutil.disk_usage(directory).free < SIZE - present + 100 * 1024 * 1024:
                parser.exit(1, "There is not enough disk space for the remaining Clef download.\n")
            print(f"Downloading {REPOSITORY}, pinned revision {REVISION}.", flush=True)
            print(f"Total: {SIZE / 1e9:.2f} GB. Resuming from {present / 1e6:.1f} MB. Ctrl-C preserves the partial download.", flush=True)
            try:
                completed = subprocess.run([
                    curl, "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
                    "--connect-timeout", "30", "--max-time", "21600", "--retry", "2", "--retry-max-time", "21600",
                    "--speed-time", "120", "--speed-limit", "1024", "--max-filesize", str(SIZE),
                    "--continue-at", "-", "--output", str(partial), URL,
                ], check=False)
            except KeyboardInterrupt:
                parser.exit(130, "\nDownload stopped. Run this setup command again to resume the retained .part file.\n")
            if completed.returncode:
                parser.exit(1, "Download did not finish. The .part file was retained; run setup again to resume.\n")
        print("Verifying the complete model SHA-256…", flush=True)
        try:
            verify_model(partial)
        except ValueError as exc:
            parser.exit(1, str(exc) + "\n")
        partial.replace(model)
    else:
        print("Verifying existing Clef-Flash model…", flush=True)
        try:
            verify_model(model)
        except ValueError as exc:
            parser.exit(1, str(exc) + "\n")
    manifest = {
        "repository": REPOSITORY, "revision": REVISION, "source_model": "Cloudflare/clef-flash",
        "filename": FILENAME, "bytes": SIZE, "sha256": SHA256, "quantization": "Q4_K_M",
        "license": "Apache-2.0", "model_card": f"https://huggingface.co/{REPOSITORY}",
        "vision_projector": "not downloaded; text-only grading",
    }
    (directory / "local-model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {model}\nStart scripts/run_clef.py separately. Runtime never downloads model files.")


if __name__ == "__main__":
    main()
