#!/usr/bin/env python3
"""Run a separate, quiet Ollama daemon for local DeepSeek explanations.

Reads the already-installed model cache. Copies its selected immutable model
files with macOS clonefile (APFS copy-on-write); never downloads model files.
The existing Ollama application, its model cache, and HOME stay unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "deepseek-r1:70b"
MANIFEST = Path("manifests/registry.ollama.ai/library/deepseek-r1/70b")
MANIFEST_SHA256 = "d37b54d01a76512a5adbaa9da4ee1c00833bf2dfefb0d4cc54351ba529e0f4da"
TESTED_VERSION = "0.40.3"
PORT = 11435
MAX_METADATA_BYTES = 65536


class LaunchError(Exception):
    """Static, user-readable launcher errors; never include native output."""


def offline_environment(model_cache: Path, inherited=None):
    inherited = os.environ if inherited is None else inherited
    # A small allowlist prevents inherited request logging, native tracing,
    # proxies, remote runners, or alternate model settings from being enabled.
    environment = {key: inherited[key] for key in ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL")
                   if key in inherited}
    environment.update({
        "OLLAMA_HOST": f"127.0.0.1:{PORT}",
        "OLLAMA_MODELS": str(model_cache),
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_DEBUG": "false",
        "OLLAMA_DEBUG_LOG_REQUESTS": "false",
        "OLLAMA_NOHISTORY": "1",
        "OLLAMA_NOPRUNE": "1",
        "OLLAMA_CONTEXT_LENGTH": "16384",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_QUEUE": "1",
        "OLLAMA_KEEP_ALIVE": "0",
        "OLLAMA_ORIGINS": f"http://127.0.0.1:{PORT}",
        "DO_NOT_TRACK": "1",
    })
    return environment


def clone_model_file(source: Path, destination: Path):
    """Independent APFS file contents, not hardlinks into the user's cache."""
    if sys.platform != "darwin":
        raise LaunchError("This launcher requires macOS copy-on-write file cloning; no large file copy was attempted.")
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    clonefile = library.clonefile
    clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    clonefile.restype = ctypes.c_int
    if clonefile(os.fsencode(source), os.fsencode(destination), 0) != 0:
        raise LaunchError("Could not clone the cached model. Use a project and source cache on a compatible APFS volume; no download or full-size copy was attempted.")
    destination.chmod(0o600)


def prepare_model_cache(source: Path, destination: Path):
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if (source == destination or source in destination.parents or destination in source.parents):
        raise LaunchError("Use a separate project cache, outside the existing Ollama model cache.")
    try:
        manifest_bytes = (source / MANIFEST).read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
            raise LaunchError("The cached deepseek-r1:70b manifest does not match the locally verified release. No model was started or downloaded.")
        manifest = json.loads(manifest_bytes)
        descriptors = [manifest["config"], *manifest["layers"]]
        files = []
        for descriptor in descriptors:
            digest, size = descriptor["digest"], descriptor["size"]
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest) or type(size) is not int or size <= 0:
                raise LaunchError("The cached model manifest is invalid.")
            relative = Path("blobs") / digest.replace(":", "-")
            item = source / relative
            if not item.is_file() or item.stat().st_size != size:
                raise LaunchError("The existing DeepSeek cache is incomplete. This launcher never downloads missing files.")
            files.append((relative, size))
        # All source preflight checks finish before any cache is created.
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        for relative, size in files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.exists() or target.is_symlink():
                if target.is_symlink() or not target.is_file() or target.stat().st_size != size:
                    raise LaunchError("The project model cache contains an unexpected file. Use a new project cache directory.")
                continue
            temporary = target.with_name(target.name + ".clone-" + uuid.uuid4().hex)
            try:
                clone_model_file(source / relative, temporary)
                if temporary.stat().st_size != size:
                    raise LaunchError("The project model clone is incomplete.")
                temporary.rename(target)
            finally:
                temporary.unlink(missing_ok=True)
        target = destination / MANIFEST
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.read_bytes() != manifest_bytes:
                raise LaunchError("The project model manifest differs from the verified release. Use a new project cache directory.")
        else:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(manifest_bytes)
        return destination
    except LaunchError:
        raise
    except Exception:
        raise LaunchError("Could not prepare the private model cache. Check the local source cache and project permissions.") from None


def check_existing_key():
    # `ollama serve` generates keys in HOME when none exist. Do not trigger that
    # write or read the key itself; this project uses an existing installation.
    if not (Path.home() / ".ollama/id_ed25519").is_file():
        raise LaunchError("Ollama's existing installation key is absent. Initialize Ollama separately before using this launcher; no files were written in your home directory.")


def check_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            # Match the daemon's restart behavior after closed connections.
            # This permits TIME_WAIT reuse, never sharing an active listener.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", PORT))
        except OSError:
            raise LaunchError("Local explanation port 11435 is already occupied. Stop its existing process before starting this dedicated daemon.") from None


def find_binary(explicit: Path | None):
    candidates = ([explicit.expanduser()] if explicit else
                  [Path(found) for found in [shutil.which("ollama")] if found] +
                  [Path("/Applications/Ollama.app/Contents/Resources/ollama")])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise LaunchError("An installed Ollama executable is required. This launcher does not install software.")


def check_version(binary: Path, environment: dict):
    try:
        result = subprocess.run([str(binary), "--version"], env=environment, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=15, check=False)
        versions = re.findall(r"(?:client version is|ollama version is)\s+(\d+\.\d+\.\d+)",
                              result.stdout + result.stderr)
        if not versions or any(version != TESTED_VERSION for version in versions):
            raise LaunchError("Use the verified Ollama 0.40.3 runtime for this launcher. Other versions require a fresh privacy and context-limit review.")
    except LaunchError:
        raise
    except Exception:
        raise LaunchError("Could not check the installed Ollama version; no daemon was started.") from None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise LaunchError("The dedicated local Ollama endpoint returned a redirect.")


def wait_ready(process, timeout=45):
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LaunchError("The dedicated Ollama process stopped during startup. Check the executable, port, and project cache.")
        try:
            with client.open(f"http://127.0.0.1:{PORT}/api/version", timeout=1) as response:
                raw = response.read(MAX_METADATA_BYTES + 1)
                if response.status != 200 or len(raw) > MAX_METADATA_BYTES:
                    raise LaunchError("The dedicated Ollama endpoint returned invalid metadata.")
                metadata = json.loads(raw)
                if metadata.get("version") != TESTED_VERSION:
                    raise LaunchError("The dedicated daemon does not match the verified Ollama version.")
                if process.poll() is not None:
                    raise LaunchError("The dedicated Ollama process stopped during startup.")
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.2)
    raise LaunchError("The dedicated Ollama process did not become ready. No explanation request was sent.")


def stop_owned(process):
    if process is None:
        return
    # This process group was created by this launcher; never find/kill another
    # user's Ollama app or a process inferred only from its listening port.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", type=Path, help="Existing Ollama 0.40.3 executable.")
    parser.add_argument("--source-models", type=Path, default=Path.home() / ".ollama/models",
                        help="Existing cache read only as the source; never downloaded into or pruned.")
    parser.add_argument("--model-cache", type=Path, default=ROOT / "models/ollama-explanations",
                        help="Separate project cache populated with APFS clones of the selected model.")
    args = parser.parse_args(argv)
    process = None
    try:
        check_existing_key()
        check_port()
        binary = find_binary(args.ollama)
        environment = offline_environment(args.model_cache.expanduser().resolve())
        check_version(binary, environment)
        print("Preparing the local DeepSeek cache with APFS clones; no model downloads.", flush=True)
        cache = prepare_model_cache(args.source_models, args.model_cache)
        environment = offline_environment(cache)
        process = subprocess.Popen([str(binary), "serve"], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        wait_ready(process)
        print("Local DeepSeek explanations are ready at http://127.0.0.1:11435. Cloud access and request/runtime logs are disabled. Ctrl-C stops this daemon.", flush=True)
        process.wait()
        raise LaunchError("The dedicated Ollama daemon stopped. No explanation was fabricated.")
    except KeyboardInterrupt:
        print("\nStopping the dedicated local explanation daemon.", flush=True)
        return 0
    except LaunchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("Could not start the dedicated local explanation daemon. Check installation, cache, and permissions.", file=sys.stderr)
        return 1
    finally:
        stop_owned(process)


if __name__ == "__main__":
    raise SystemExit(main())
