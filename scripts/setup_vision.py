#!/usr/bin/env python3
"""Explicit setup-only download of a vision model into Margin's private Ollama.

Never imports the grading application, reads submissions, or performs inference.
Only this command may pull weights; the grading UI never downloads models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import httpx

from run_ollama import REVIEWED_VERSIONS, TESTED_VERSION, prepare_model_cache

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "models/ollama-explanations"
ENDPOINT = "http://127.0.0.1:11435"
DEFAULT_MODEL = "openbmb/minicpm-v4.7:1b"
MAX_METADATA = 1024 * 1024
MAX_PROGRESS = 16 * 1024 * 1024
MAX_LINE = 65536
DOWNLOAD_SECONDS = 12 * 60 * 60


class SetupError(Exception):
    """Fixed diagnostics without native response bodies or personal paths."""


def model_name(value):
    if (not isinstance(value, str) or len(value) > 160 or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?:[A-Za-z0-9][A-Za-z0-9_.-]*", value)
            or value.lower().endswith(":latest") or "cloud" in value.lower()
            or ("/" in value and ("." in value.split("/", 1)[0] or value.split("/", 1)[0].lower() == "localhost"))):
        raise argparse.ArgumentTypeError("Use an explicit local model tag such as openbmb/minicpm-v4.7:1b; URLs, cloud models, and latest are not allowed.")
    return value


def metadata(client, method, route, payload=None):
    try:
        with client.stream(method, ENDPOINT + route, json=payload) as response:
            if response.status_code != 200 or response.headers.get("content-encoding", "identity") != "identity":
                raise SetupError("The private Ollama endpoint rejected the setup check. Check the daemon and installed model; no fallback was used.")
            raw = bytearray()
            for block in response.iter_bytes(chunk_size=65536):
                raw.extend(block)
                if len(raw) > MAX_METADATA:
                    raise SetupError("The private daemon returned oversized model metadata.")
        value = json.loads(raw)
        if not isinstance(value, dict) or "error" in value:
            raise ValueError
        return value
    except (httpx.HTTPError, ValueError, UnicodeError):
        raise SetupError("The private Ollama setup check failed. Native response details were suppressed.") from None


def verify_daemon(client):
    version = metadata(client, "GET", "/api/version").get("version")
    if version not in REVIEWED_VERSIONS:
        raise SetupError("Use a reviewed Ollama runtime (0.40.3 or 0.40.5); no model was downloaded.")
    cloud = metadata(client, "GET", "/api/status").get("cloud", {})
    if not isinstance(cloud, dict) or cloud.get("disabled") is not True or cloud.get("source") not in {"env", "both"}:
        raise SetupError("The daemon must confirm OLLAMA_NO_CLOUD=1 before setup.")
    return version


def verify_cache(client):
    """A content-addressed random probe proves this daemon sees this project store."""
    cache = prepare_model_cache(CACHE)
    nonce = os.urandom(32)
    digest = hashlib.sha256(nonce).hexdigest()
    probe = cache / "blobs" / ("sha256-" + digest)
    try:
        descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(nonce)
        response = client.head(ENDPOINT + "/api/blobs/sha256:" + digest)
        if response.status_code != 200:
            raise SetupError("The daemon does not use this project's private model store. Start scripts/run_ollama.py; no model was downloaded.")
    except (OSError, httpx.HTTPError):
        raise SetupError("Could not verify the project's private model store; no model was downloaded.") from None
    finally:
        probe.unlink(missing_ok=True)


def verify_model(client, model, runtime_version=TESTED_VERSION):
    value = metadata(client, "POST", "/api/show", {"model": model, "verbose": False})
    if any(value.get(key) for key in ("remote_model", "remote_host")):
        raise SetupError("Remote models are not accepted; choose a fully installed local vision model.")
    required = value.get("requires")
    if required:
        if not isinstance(required, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", required):
            raise SetupError("The model declares an unrecognized runtime requirement; compatibility could not be verified.")
        if tuple(map(int, required.split("."))) > tuple(map(int, runtime_version.split("."))):
            raise SetupError("This model requires a newer runtime than the reviewed Ollama version. Choose a compatible installed model; no runtime was changed.")
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, list) or "vision" not in capabilities:
        raise SetupError("This installed model does not advertise vision support. No image was sent.")


def pull_model(client, model):
    started = time.monotonic()
    consumed = 0
    buffer = bytearray()
    success = False
    last_percent = -1
    try:
        with client.stream("POST", ENDPOINT + "/api/pull", json={"model": model, "stream": True}) as response:
            if response.status_code != 200 or response.headers.get("content-encoding", "identity") != "identity":
                raise SetupError("The model download was rejected. Check the explicit model tag and internet connection.")
            for block in response.iter_bytes(chunk_size=8192):
                consumed += len(block)
                buffer.extend(block)
                if consumed > MAX_PROGRESS or time.monotonic() - started > DOWNLOAD_SECONDS:
                    raise SetupError("The download exceeded its setup limit. Run this command again to resume.")
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    if len(line) > MAX_LINE:
                        raise SetupError("The download returned oversized progress metadata.")
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or "error" in value:
                        raise SetupError("The model download failed. Check the tag, free space, and compatible runtime; run again to resume.")
                    if value.get("status") == "success":
                        success = True
                    total, completed = value.get("total"), value.get("completed")
                    if type(total) is int and type(completed) is int and 0 <= completed <= total and total > 0:
                        percent = completed * 100 // total
                        if percent // 10 != last_percent // 10:
                            print(f"Downloading model layer: {percent}%", flush=True)
                            last_percent = percent
                if len(buffer) > MAX_LINE:
                    raise SetupError("The download returned oversized progress metadata.")
        if buffer.strip() or not success:
            raise SetupError("The download did not confirm completion. Run this command again to resume.")
    except (httpx.HTTPError, ValueError, UnicodeError):
        raise SetupError("The model download failed or timed out. Run this command again to resume; native details were suppressed.") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=model_name, default=DEFAULT_MODEL,
                        help="Explicit vision tag; alternative: qwen3.5:2b-q4_K_M.")
    parser.add_argument("--offline", action="store_true", help="Check local installation only; never pull weights.")
    args = parser.parse_args(argv)
    try:
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(60, connect=5),
                          headers={"Accept-Encoding": "identity"}, limits=httpx.Limits(max_connections=1)) as client:
            runtime_version = verify_daemon(client)
            verify_cache(client)
            if not args.offline:
                print("Downloading public model weights into the project store. No student content is read or sent.", flush=True)
                pull_model(client, args.model)
            verify_model(client, args.model, runtime_version)
        print(f"Local vision model available: {args.model}. Metadata checks passed; actual image recognition is checked when used locally.", flush=True)
        return 0
    except KeyboardInterrupt:
        print("Setup interrupted. Run the same command again to resume.", file=sys.stderr)
        return 130
    except SetupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("Vision setup failed. Start the private Ollama daemon and check local installation and permissions.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
