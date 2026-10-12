#!/usr/bin/env python3
"""Serve a pinned local thinking model for synthetic comparisons.

This launcher never downloads weights, enables tools, or changes other servers.
Only its own child process group is stopped when the launcher exits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SIZE = 16384
ALLOWED_PORTS = (11436, 11437)
TESTED_COMMIT = "d81235049"
MAX_METADATA_BYTES = 262144
CANDIDATES = {
    "pro": {
        "alias": "v4-pro-distill-9b-q8",
        "repo": "Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B-MTP-GGUF",
        "revision": "99b43f206bfe6e4e4c311d8fe5bf9b3a9d06d08e",
        "path": "models/v4-pro-distill-9b-q8/DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q8_0.gguf",
        "size": 9786061920,
        "sha256": "33a1ae7ed00f8a84ce11e98e47b9b32ef6c9feb628ecf14abd9315d948470b08",
    },
    "flash": {
        "alias": "v4-flash-distill-9b-q8",
        "repo": "Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF",
        "revision": "e842f7426b567ec1e335d5cc348f4c86cdcf1453",
        "path": "models/v4-flash-distill-9b-q8/Qwen3.5-9B-DeepSeek-V4-Flash-Q8_0.gguf",
        "size": 9527497984,
        "sha256": "be8cc2a28d5e39a0e94da66c7f925df92be48885cebfc8ef031b05177b691402",
    },
    "qwen38": {
        "alias": "qwen38-27b-q4",
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
        "path": "models/qwen38-27b-q4/Qwen3.8-27B-UD-Q4_K_M.gguf",
        "size": 16464440224,
        "sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
    },
    "qwen38_4b": {
        "alias": "qwen38-distill-4b-q4",
        "repo": "empero-ai/Qwen3.8-4B-Distill-GGUF",
        "revision": "391fc7d103e3942a408def3e4f51c2f85d464417",
        "path": "models/qwen38-distill-4b-q4/Qwen3.8-4B-Q4_K_M.gguf",
        "size": 2783446304,
        "sha256": "dec96e8cf2e11b613bb46513dec485377f9ca5a351e71712ee0e244f287c6790",
    },
    "qwen36": {
        "alias": "qwen36-35b-a3b-ud-q4-k-xl",
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "revision": "a483e9e6cbd595906af30beda3187c2663a1118c",
        "path": "models/qwen36-35b-a3b-ud-q4-k-xl/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "size": 22360456160,
        "sha256": "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450",
    },
}
REQUIRED_FLAGS = (
    "--offline", "--log-disable", "--no-webui", "--no-slots", "--no-agent",
    "--no-webui-mcp-proxy", "--no-mmproj", "--no-context-shift",
    "--no-cache-prompt", "--no-cache-idle-slots", "--cache-ram",
    "--jinja", "--reasoning-format", "--reasoning", "--no-reasoning-preserve",
    "--no-cors-credentials", "--gpu-layers",
)


class LaunchError(Exception):
    """Static diagnostic safe to show without exposing native output."""


def offline_environment(inherited=None):
    inherited = os.environ if inherited is None else inherited
    # Preserve HOME, not an override. Do not inherit any native model, proxy,
    # tool, download, tracing, dynamic-library injection, or log configuration.
    environment = {key: inherited[key] for key in ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL")
                   if key in inherited}
    environment.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "OTEL_SDK_DISABLED": "true",
    })
    return environment


def candidate_spec(name):
    if name not in CANDIDATES:
        raise LaunchError("Choose a pinned candidate: " + ", ".join(CANDIDATES) + ".")
    return CANDIDATES[name]


def validate_port(port):
    if type(port) is not int or port not in ALLOWED_PORTS:
        raise LaunchError("Candidate servers use only loopback ports 11436 or 11437.")


def verify_model(candidate):
    spec = candidate_spec(candidate)
    model = ROOT / spec["path"]
    try:
        # Pinned project locations only; reject links instead of following a
        # model path into another project or mutable external cache.
        if any(path.is_symlink() for path in (model, model.parent, ROOT / "models")):
            raise LaunchError("The candidate model must be a regular file in its pinned project directory.")
        with model.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != spec["size"]:
                raise LaunchError("The candidate weights are missing, incomplete, or the wrong release. No download was attempted.")
            digest = hashlib.sha256()
            while block := source.read(8 * 1024 * 1024):
                digest.update(block)
            after = os.fstat(source.fileno())
        current = model.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (
                after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev):
            raise LaunchError("The candidate weights changed during verification. No server was started.")
        if (current.st_ino, current.st_dev, current.st_size, current.st_mtime_ns) != (
                after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns):
            raise LaunchError("The candidate weights changed during verification. No server was started.")
        if digest.hexdigest() != spec["sha256"]:
            raise LaunchError("The candidate checksum does not match the pinned release. No server was started.")
    except LaunchError:
        raise
    except OSError:
        raise LaunchError("Could not read the pinned candidate weights. Complete the separate download before launching.") from None
    return model


def find_binary(explicit=None):
    paths = ([explicit.expanduser()] if explicit else
             [Path(value) for value in (shutil.which("llama-server"),
                                       "/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server") if value])
    for path in paths:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise LaunchError("An installed llama-server executable is required. This launcher never installs software.")


def check_runtime(binary, environment):
    try:
        version = subprocess.run([str(binary), "--version"], env=environment,
                                 stdin=subprocess.DEVNULL, capture_output=True,
                                 text=True, timeout=15, check=True)
        text = version.stdout + version.stderr
        if not re.search(r"version:\s*0\.6\.0\s*\(build\s+11429,\s*commit\s+" + TESTED_COMMIT + r"\)", text):
            raise LaunchError("Use the reviewed llama.cpp 0.6.0 build 11429, commit d81235049. Other builds need a fresh compatibility review.")
        help_result = subprocess.run([str(binary), "--help"], env=environment,
                                     stdin=subprocess.DEVNULL, capture_output=True,
                                     text=True, timeout=15, check=True)
        help_text = help_result.stdout + help_result.stderr
        if any(flag not in help_text for flag in REQUIRED_FLAGS):
            raise LaunchError("The installed runtime lacks a required privacy or thinking flag. No server was started.")
    except LaunchError:
        raise
    except Exception:
        raise LaunchError("Could not inspect llama-server. Check its installation; native diagnostic output was suppressed.") from None


def check_port(port):
    validate_port(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            raise LaunchError("The selected candidate port is occupied. Existing services were left running.") from None


def build_command(binary, candidate, model, port):
    validate_port(port)
    spec = candidate_spec(candidate)
    if model != ROOT / spec["path"]:
        raise LaunchError("Only the pinned candidate model path is supported.")
    return [
        str(binary), "--model", str(model), "--alias", spec["alias"],
        "--host", "127.0.0.1", "--port", str(port), "--offline",
        "--gpu-layers", "all", "--parallel", "1", "--ctx-size", str(CONTEXT_SIZE),
        "--batch-size", "512", "--ubatch-size", "512", "--n-predict", "4096",
        "--jinja", "--reasoning", "on", "--reasoning-format", "deepseek",
        "--no-reasoning-preserve", "--no-context-shift", "--no-mmproj",
        "--no-agent", "--no-webui-mcp-proxy", "--no-webui", "--no-slots",
        "--cache-ram", "0", "--no-cache-prompt", "--no-cache-idle-slots",
        "--cors-origins", f"http://127.0.0.1:{port}", "--no-cors-credentials",
        "--log-disable", "--timeout", "300",
    ]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise LaunchError("The local candidate endpoint returned a redirect.")


def metadata(client, port, path):
    validate_port(port)
    if path not in ("/health", "/props", "/v1/models"):
        raise LaunchError("Only local startup metadata may be requested by this launcher.")
    with client.open(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
        raw = response.read(MAX_METADATA_BYTES + 1)
        if response.status != 200 or len(raw) > MAX_METADATA_BYTES:
            raise LaunchError("The candidate returned invalid startup metadata.")
        try:
            parsed = json.loads(raw)
        except (ValueError, UnicodeError):
            raise LaunchError("The candidate returned invalid startup metadata.") from None
        if not isinstance(parsed, dict):
            raise LaunchError("The candidate returned invalid startup metadata.")
        return parsed


def validate_metadata(props, models, candidate):
    alias = candidate_spec(candidate)["alias"]
    try:
        valid = (
            props["model_alias"] == alias and props["total_slots"] == 1
            and props["default_generation_settings"]["n_ctx"] == CONTEXT_SIZE
            and props["ui"] is False and props["cors_proxy_enabled"] is False
            and all(props["modalities"][name] is False for name in ("vision", "video", "audio"))
            and TESTED_COMMIT in props["build_info"]
            and len(models["data"]) == 1 and models["data"][0]["id"] == alias
            and models["data"][0]["meta"]["n_ctx"] == CONTEXT_SIZE
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise LaunchError("The running candidate does not match the required model, context, or private runtime configuration.")


def wait_ready(process, candidate, port, timeout=180):
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LaunchError("The candidate stopped during startup. Check available memory, pinned weights, and the reviewed runtime; native output is suppressed.")
        try:
            health = metadata(client, port, "/health")
            if health.get("status") == "ok":
                validate_metadata(metadata(client, port, "/props"),
                                  metadata(client, port, "/v1/models"), candidate)
                if process.poll() is not None:
                    raise LaunchError("The candidate stopped during startup.")
                return
        except urllib.error.HTTPError as exc:
            if exc.code != 503:
                raise LaunchError("The local candidate rejected its startup metadata check.") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.2)
    raise LaunchError("The candidate did not become ready within three minutes. No comparison request was sent.")


def stop_owned(process):
    if process is None:
        return
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


def request_shutdown(signum, frame):
    # A background launcher's SIGTERM/SIGHUP must take the same cleanup path
    # as Ctrl-C, instead of leaving its separate native process group alive.
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--port", type=int, choices=ALLOWED_PORTS, default=11436)
    parser.add_argument("--llama-server", type=Path, help="Existing reviewed llama-server executable.")
    args = parser.parse_args(argv)
    process = None
    shutdown_signals = (signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {item: signal.getsignal(item) for item in shutdown_signals}
    for item in shutdown_signals:
        signal.signal(item, request_shutdown)
    try:
        check_port(args.port)
        binary = find_binary(args.llama_server)
        environment = offline_environment()
        check_runtime(binary, environment)
        print("Verifying the pinned local candidate checksum; no downloads.", flush=True)
        model = verify_model(args.candidate)
        command = build_command(binary, args.candidate, model, args.port)
        process = subprocess.Popen(command, env=environment, cwd=ROOT,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        wait_ready(process, args.candidate, args.port)
        print(f"Local {candidate_spec(args.candidate)['alias']} ready at http://127.0.0.1:{args.port}. "
              "Thinking is enabled; native logs, prompt caching, tools, and context shifting are disabled. "
              "Ctrl-C stops this candidate only.", flush=True)
        process.wait()
        raise LaunchError("The candidate server stopped. No comparison result was fabricated.")
    except KeyboardInterrupt:
        print("\nStopping this local candidate only.", flush=True)
        return 0
    except LaunchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("Could not start the local candidate. Check the runtime, weights, port, and memory; native output was suppressed.", file=sys.stderr)
        return 1
    finally:
        # Ignore repeated termination requests while the owned group is being
        # reaped, then restore the caller's handlers (also useful in tests).
        for item in shutdown_signals:
            signal.signal(item, signal.SIG_IGN)
        try:
            stop_owned(process)
        except (OSError, subprocess.SubprocessError):
            print("Could not confirm the candidate process stopped. Check the launcher-owned process; no other service was targeted.", file=sys.stderr)
        finally:
            for item, previous in previous_handlers.items():
                signal.signal(item, previous)


if __name__ == "__main__":
    raise SystemExit(main())
