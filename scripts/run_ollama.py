#!/usr/bin/env python3
"""Run a separate, quiet Ollama daemon with a project-local model store.

Starts with an empty store on a fresh clone; never downloads model files.
Import separately downloaded, verified GGUFs with import_explanation_candidate.py.
Ollama may initialize its standard local identity on first use; no account or
pre-existing model cache is required. The personal model store is not used.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
TESTED_VERSION = "0.40.3"
REVIEWED_VERSIONS = ("0.40.3", "0.40.5")
PORT = 11435
IS_WINDOWS = os.name == "nt"
MAX_METADATA_BYTES = 65536


class LaunchError(Exception):
    """Static, user-readable launcher errors; never include native output."""


def offline_environment(model_cache: Path, inherited=None):
    inherited = os.environ if inherited is None else inherited
    # A small allowlist prevents inherited request logging, native tracing,
    # proxies, remote runners, or alternate model settings from being enabled.
    environment = {key: inherited[key] for key in ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL",
            "SYSTEMROOT", "SystemRoot", "WINDIR", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP")
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


def copy_model_file(source: Path, destination: Path):
    """Make an independent copy on any supported filesystem, never a hardlink."""
    # Called only after checksum verification, during explicit model import.
    # Refuse replacement so an interrupted/parallel import cannot lose data.
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=8 * 1024 * 1024)
    except Exception:
        raise LaunchError("Could not copy the verified model into the project store. Check free disk space and project permissions.") from None


def prepare_model_cache(destination: Path):
    """Create an empty private store without inspecting any personal settings."""
    destination = destination.expanduser().absolute()
    paths = (destination, *destination.parents)
    if ".." in destination.parts or ROOT not in destination.parents or any(
            path.is_symlink() for path in paths if path == ROOT or ROOT in path.parents):
        raise LaunchError("Use a model cache inside this project, without symbolic links.")
    try:
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        blobs = destination / "blobs"
        if blobs.is_symlink():
            raise LaunchError("The project blob store must not be a symbolic link.")
        blobs.mkdir(exist_ok=True, mode=0o700)
    except LaunchError:
        raise
    except OSError:
        raise LaunchError("Could not prepare the project model store. Check project permissions.") from None
    return destination


def check_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            # Match the daemon's restart behavior after closed connections.
            # This permits TIME_WAIT reuse, never sharing an active listener.
            listener.setsockopt(socket.SOL_SOCKET,
                socket.SO_EXCLUSIVEADDRUSE if IS_WINDOWS else socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", PORT))
        except OSError:
            raise LaunchError("Local explanation port 11435 is already occupied. Stop its existing process before starting this dedicated daemon.") from None


def find_binary(explicit: Path | None):
    candidates = ([explicit.expanduser()] if explicit else
                  [Path(found) for found in [shutil.which("ollama")] if found] +
                  [Path("/Applications/Ollama.app/Contents/Resources/ollama")])
    if IS_WINDOWS and not explicit:
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(Path(local) / "Programs/Ollama/ollama.exe")
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
        if not versions or any(version not in REVIEWED_VERSIONS for version in versions):
            raise LaunchError("Use a reviewed Ollama runtime (0.40.3 or 0.40.5) for this launcher. Other versions require a fresh privacy and context-limit review.")
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
                if metadata.get("version") not in REVIEWED_VERSIONS:
                    raise LaunchError("The dedicated daemon does not match the verified Ollama version.")
                with client.open(f"http://127.0.0.1:{PORT}/api/status", timeout=1) as status:
                    raw_status = status.read(MAX_METADATA_BYTES + 1)
                    if status.status != 200 or len(raw_status) > MAX_METADATA_BYTES:
                        raise LaunchError("The dedicated Ollama endpoint returned invalid metadata.")
                    cloud = json.loads(raw_status).get("cloud", {})
                    if cloud.get("disabled") is not True or cloud.get("source") not in ("env", "both"):
                        raise LaunchError("The dedicated daemon did not confirm OLLAMA_NO_CLOUD=1.")
                if process.poll() is not None:
                    raise LaunchError("The dedicated Ollama process stopped during startup.")
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.2)
    raise LaunchError("The dedicated Ollama process did not become ready. No explanation request was sent.")


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    if IS_WINDOWS:
        # /T targets only descendants of this still-owned live process. Never
        # kill by port or executable name; those may belong to another app.
        system_root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR")
        if not system_root:
            raise LaunchError("Windows SystemRoot is required to stop the owned local daemon.")
        result = subprocess.run([str(Path(system_root) / "System32/taskkill.exe"),
                                 "/PID", str(process.pid), "/T", "/F"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=15, check=False)
        if result.returncode != 0 and process.poll() is None:
            raise LaunchError("Could not confirm cleanup of the owned local daemon.")
        process.wait(timeout=5)
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
    parser.add_argument("--ollama", type=Path, help="Existing reviewed Ollama 0.40.3 or 0.40.5 executable.")
    parser.add_argument("--model-cache", type=Path, default=ROOT / "models/ollama-explanations",
                        help="Project-local model store; starts empty until a model is imported.")
    args = parser.parse_args(argv)
    process = None
    try:
        check_port()
        binary = find_binary(args.ollama)
        cache = prepare_model_cache(args.model_cache)
        environment = offline_environment(cache)
        check_version(binary, environment)
        process_options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WINDOWS
                           else {"start_new_session": True})
        process = subprocess.Popen([str(binary), "serve"], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, **process_options)
        wait_ready(process)
        print("Local explanation daemon is ready at http://127.0.0.1:11435. Cloud access and request/runtime logs are disabled. Import a verified model separately if the store is empty. Ctrl-C stops this daemon.", flush=True)
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
        try:
            stop_owned(process)
        except Exception:
            print("Could not confirm cleanup of the owned local daemon. Check that process before restarting.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
