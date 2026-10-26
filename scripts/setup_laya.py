#!/usr/bin/env python3
"""One-time online setup, separate from the strictly offline grading process."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = "convaiinnovations/laya-multilingual"
REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"
FILES = ["README.md", "config.json", "rl_agent_config.json", "encoder/config.json", "model.safetensors", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
REQUIREMENTS = ROOT / "requirements-laya.txt"
EXPECTED_SHA256 = {
    "model.safetensors": "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
    "tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-deps", action="store_true", help="Install the pinned Laya runtime into this Python environment first.")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/laya-multilingual")
    parser.add_argument("--offline", action="store_true", help="Only copy already-cached model files; do not contact Hugging Face.")
    args = parser.parse_args()
    if args.install_deps:
        if args.offline:
            parser.error("--offline cannot be combined with --install-deps.")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)], check=True)
    # Setup is deliberately a different process from the app. It never imports
    # the grader or reads submissions, rubrics, or stored grades.
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["HF_HOME"] = str(ROOT / "models/.hf-cache")
    os.environ["HF_HUB_OFFLINE"] = "1" if args.offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if args.offline else "0"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.exit(1, "Missing model setup dependencies. Run this command again with --install-deps.\n")
    print(f"Preparing {REPO} at pinned revision {REVISION}.", flush=True)
    print("This setup downloads approximately 680 MB of model files and does not read student documents.", flush=True)
    directory = args.model_dir.expanduser().resolve()
    snapshot_download(repo_id=REPO, revision=REVISION, allow_patterns=FILES,
                      local_dir=directory, local_files_only=args.offline, token=False)
    missing = [name for name in FILES if not (directory / name).is_file()]
    if missing:
        parser.exit(1, "Incomplete checkpoint: " + ", ".join(missing) + "\n")
    checksums = {}
    for name in FILES:
        digest = hashlib.sha256()
        with (directory / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        checksums[name] = digest.hexdigest()
        if name in EXPECTED_SHA256 and checksums[name] != EXPECTED_SHA256[name]:
            parser.exit(1, f"Checksum mismatch for {name}; the checkpoint is not ready for use.\n")
    versions = {}
    for name in ["laya", "torch", "transformers", "huggingface_hub", "safetensors", "numpy"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    manifest = {"repository": REPO, "revision": REVISION, "sha256": checksums, "runtime_versions": versions}
    (directory / "local-model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {directory}\nStart the app in a separate process; inference uses only these local files.")


if __name__ == "__main__":
    main()
