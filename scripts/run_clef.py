#!/usr/bin/env python3
"""Run the verified local Clef-Flash decision model on loopback only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

from setup_clef import FILENAME, ROOT, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/clef-flash")
    parser.add_argument("--llama-server", type=Path, help="Path to an installed llama-server binary (tested with v0.6.0).")
    parser.add_argument("--port", type=int, default=1235)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a local port between 1024 and 65535.")
    binary = str(args.llama_server.expanduser()) if args.llama_server else shutil.which("llama-server")
    if not binary:
        binary = next((str(path) for path in (Path("/opt/homebrew/bin/llama-server"), Path("/usr/local/bin/llama-server")) if path.is_file()), None)
    if not binary:
        parser.exit(1, "Install llama.cpp v0.6.0 or later before starting Clef (for example: brew install llama.cpp). This launcher never installs software.\n")
    model = args.model_dir.expanduser().resolve() / FILENAME
    print("Verifying local Clef-Flash weights…", flush=True)
    try:
        verify_model(model)
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")
    # Prevent inherited download, model, remote-RPC, host, or log settings from
    # altering the explicit local-only invocation.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_")}
    environment.update({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"})
    try:
        version_result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15, env=environment, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Could not inspect llama-server ({type(exc).__name__}). Check the binary path.\n")
    version_text = version_result.stdout + version_result.stderr
    version = re.search(r"version:\s*(\d+)\.(\d+)\.(\d+)", version_text)
    if not version or tuple(map(int, version.groups())) < (0, 6, 0):
        parser.exit(1, "Use llama.cpp v0.6.0 or later with Clef and /v1/systemone support.\n")
    command = [
        binary, "--model", str(model), "--alias", "clef-flash", "--host", "127.0.0.1", "--port", str(args.port),
        "--offline", "--log-disable", "--no-webui", "--no-slots", "--cors-origins", "http://127.0.0.1",
        "--ctx-size", "8192", "--batch-size", "8192", "--ubatch-size", "8192", "--parallel", "1",
        "--no-context-shift", "--cache-ram", "0", "--no-cache-idle-slots",
    ]
    print(f"Starting local Clef-Flash at http://127.0.0.1:{args.port}/v1/systemone", flush=True)
    print("Model files are offline. Server logs and prompt caching are disabled. Ctrl-C stops this server.", flush=True)
    # Nothing from the model process, including a possible input-bearing error,
    # is written to a terminal or logfile. The app receives API errors directly.
    try:
        result = subprocess.run(command, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except KeyboardInterrupt:
        parser.exit(0, "\nLocal Clef server stopped.\n")
    except OSError as exc:
        parser.exit(1, f"Could not start llama-server ({type(exc).__name__}).\n")
    if result.returncode:
        parser.exit(1, f"Clef server stopped with exit code {result.returncode}. Check available memory, whether the port is occupied, and the installed llama.cpp version.\n")


if __name__ == "__main__":
    main()
