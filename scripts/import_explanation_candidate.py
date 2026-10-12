#!/usr/bin/env python3
"""Import a pinned public GGUF into the already-running private Ollama daemon.

Only project files and 127.0.0.1:11435 are used. No pull, generation, CLI,
daemon restart, environment credentials, or personal model cache is involved.
Repeating a successful import verifies its identity without recreating it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import sys
import uuid

import httpx

from run_explanation_candidate import CANDIDATES, ROOT, verify_model
from run_ollama import PORT, TESTED_VERSION, copy_model_file

ENDPOINT = f"http://127.0.0.1:{PORT}"
CACHE = ROOT / "models/ollama-explanations"
CONTEXT = 16384
MAX_METADATA = 1024 * 1024
MODEL_NAMES = {
    "pro": "v4-pro-distill-9b:q8_0",
    "flash": "v4-flash-distill-9b:q8_0",
    "qwen38": "qwen38-27b:q4_k_m",
    "qwen38_4b": "qwen38-distill-4b:q4_k_m",
    "qwen36": "qwen36-35b-a3b:ud_q4_k_xl",
}
# The descriptive Unsloth tag is not a GGUF file-type name. Identity is bound
# to the full pinned SHA-256; Ollama reports the GGUF's underlying base type.
QUANTIZATION_LEVELS = {MODEL_NAMES["qwen36"]: "q4_k_m"}


class ImportError(RuntimeError):
    """Fixed diagnostics only; no native output or response bodies."""


def no_links(path):
    if ROOT not in path.parents or any(p.is_symlink() for p in (path, *path.parents) if p == ROOT or ROOT in p.parents):
        raise ImportError("Import paths must stay in the project without symbolic links.")


def checked_bytes(path, maximum=MAX_METADATA):
    no_links(path)
    with path.open("rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ImportError("Import metadata must use regular project files.")
        data = source.read(maximum + 1)
    if len(data) > maximum:
        raise ImportError("Import metadata exceeds its size limit.")
    return data


def checked_json(path):
    try:
        value = json.loads(checked_bytes(path))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, UnicodeError):
        raise ImportError("The project model metadata is invalid; it was left untouched.") from None


def verify_blob(path, spec):
    no_links(path)
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != spec["size"]:
            raise ImportError("An existing project blob differs from the pinned file; it was left untouched.")
        digest = hashlib.sha256()
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(source.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if digest.hexdigest() != spec["sha256"] or identity(before) != identity(after) or identity(after) != identity(path.stat()):
        raise ImportError("The project blob checksum or identity differs; it was left untouched.")


def metadata(client, method, route, payload=None):
    try:
        with client.stream(method, ENDPOINT + route, json=payload) as response:
            if response.status_code != 200:
                raise ImportError("The dedicated Ollama endpoint rejected the metadata request. No fallback was attempted.")
            raw = bytearray()
            for block in response.iter_bytes(chunk_size=65536):
                raw.extend(block)
                if len(raw) > MAX_METADATA:
                    raise ImportError("The dedicated endpoint returned oversized metadata.")
        value = json.loads(raw)
        if not isinstance(value, dict) or "error" in value:
            raise ValueError
        return value
    except (httpx.HTTPError, ValueError, UnicodeError):
        raise ImportError("The dedicated Ollama metadata request failed. Native response details were suppressed.") from None


def tags(client):
    models = metadata(client, "GET", "/api/tags").get("models")
    if not isinstance(models, list) or any(not isinstance(m, dict) or not isinstance(m.get("name"), str) for m in models):
        raise ImportError("The dedicated endpoint returned invalid model names.")
    result = {m["name"]: m for m in models}
    if len(result) != len(models):
        raise ImportError("The dedicated endpoint returned duplicate model names.")
    return result


def verify_daemon_cache(client):
    """Prove this loopback daemon sees the project store before model mutation.

    A fresh random content-addressed probe exists only in this store. The HEAD
    request cannot import or download anything. Remove the probe on all paths;
    no baseline model or personal installation state is needed.
    """
    nonce = os.urandom(32)
    digest = hashlib.sha256(nonce).hexdigest()
    probe = CACHE / "blobs" / ("sha256-" + digest)
    no_links(probe)
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(nonce)
        try:
            response = client.head(ENDPOINT + "/api/blobs/sha256:" + digest)
        except httpx.HTTPError:
            raise ImportError("Could not verify the dedicated daemon's project store. No model was imported.") from None
        if response.status_code != 200:
            raise ImportError("The daemon uses a different model cache. No model was imported or modified.")
    finally:
        probe.unlink(missing_ok=True)


def manifest_paths(name):
    model, tag = name.split(":")
    return (CACHE / "manifests-v2/ollama.com/library" / model / tag,
            CACHE / "manifests/registry.ollama.ai/library" / model / tag)


def manifest_exists(name):
    return any(p.exists() or p.is_symlink() for p in manifest_paths(name))


def model_manifest(name, spec, api_digest):
    paths = [p for p in manifest_paths(name) if p.exists() or p.is_symlink()]
    if len(paths) != 1:
        raise ImportError("The model name has missing or ambiguous project manifests; nothing was overwritten.")
    path = paths[0]
    no_links(path.parent)
    # Ollama 0.40.3 uses a v2 manifest symlink into its content-addressed store.
    if path.is_symlink():
        target = path.resolve(strict=True)
        if target.parent != CACHE / "blobs" or not re.fullmatch(r"sha256-[a-f0-9]{64}", target.name):
            raise ImportError("The model manifest points outside the project blob store.")
        path = target
    raw = checked_bytes(path)
    if (not isinstance(api_digest, str) or not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", api_digest)
            or hashlib.sha256(raw).hexdigest() != api_digest.removeprefix("sha256:")):
        raise ImportError("The dedicated endpoint and project manifest identities differ.")
    try:
        manifest = json.loads(raw)
        layers = manifest["layers"]
        weights = [layer for layer in layers if layer["mediaType"] == "application/vnd.ollama.image.model"]
        params = [layer for layer in layers if layer["mediaType"] == "application/vnd.ollama.image.params"]
        if (len(layers) != 2 or len(weights) != 1 or len(params) != 1
                or weights[0]["digest"] != "sha256:" + spec["sha256"] or weights[0]["size"] != spec["size"]):
            raise ValueError
        for descriptor in [manifest["config"], params[0]]:
            digest = descriptor["digest"]
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
                raise ValueError
            content = checked_bytes(CACHE / "blobs" / digest.replace(":", "-"))
            if len(content) != descriptor["size"] or hashlib.sha256(content).hexdigest() != digest[7:]:
                raise ValueError
            decoded = json.loads(content)
            if descriptor is params[0] and decoded != {"num_ctx": CONTEXT}:
                raise ValueError
            if descriptor is manifest["config"] and (decoded.get("model_format") != "gguf" or decoded.get("remote_host") or decoded.get("remote_model")):
                raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ImportError("An existing model manifest differs from this pinned import; it was left untouched.") from None


def validate_import(client, name, spec):
    model = tags(client).get(name)
    if model is None:
        raise ImportError("The imported model is missing from the dedicated daemon.")
    model_manifest(name, spec, model.get("digest"))
    verify_blob(CACHE / "blobs" / ("sha256-" + spec["sha256"]), spec)
    shown = metadata(client, "POST", "/api/show", {"model": name})
    details = shown.get("details", {})
    parameters = shown.get("parameters", "")
    if (shown.get("remote_host") or shown.get("remote_model") or shown.get("system")
            or details.get("format") != "gguf" or "completion" not in shown.get("capabilities", [])
            or not shown.get("template") or not isinstance(parameters, str)
            or not re.fullmatch(r"\s*num_ctx\s+16384\s*", parameters)
            or details.get("quantization_level", "").lower() != QUANTIZATION_LEVELS.get(name, name.split(":")[1])):
        raise ImportError("The imported model metadata differs from the required local configuration.")
    return {"model": name, "endpoint": ENDPOINT, "version": TESTED_VERSION,
            "parameters": {"num_ctx": CONTEXT}, "details": details,
            "capabilities": shown["capabilities"], "sha256": spec["sha256"]}


@contextmanager
def import_lock():
    path = CACHE / ".candidate-import.lock"
    no_links(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ImportError("The import lock must be a regular project file.")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ImportError("Another candidate import is in progress.") from None
        yield
    finally:
        os.close(fd)


def import_candidate(candidate, *, client=None):
    if candidate not in MODEL_NAMES or candidate not in CANDIDATES:
        raise ImportError("Choose a pinned explanation candidate: " + ", ".join(MODEL_NAMES) + ".")
    spec, name = CANDIDATES[candidate], MODEL_NAMES[candidate]
    source = verify_model(candidate)
    no_links(CACHE / "blobs")
    if not (CACHE / "blobs").is_dir():
        raise ImportError("Start the dedicated project Ollama launcher before importing.")
    own_client = client is None
    if own_client:
        client = httpx.Client(trust_env=False, follow_redirects=False,
                              timeout=httpx.Timeout(120, connect=5))
    try:
        if metadata(client, "GET", "/api/version").get("version") != TESTED_VERSION:
            raise ImportError("The dedicated endpoint must run the verified Ollama 0.40.3 version.")
        cloud = metadata(client, "GET", "/api/status").get("cloud", {})
        if cloud.get("disabled") is not True or cloud.get("source") not in ("env", "both"):
            raise ImportError("The dedicated daemon must explicitly use OLLAMA_NO_CLOUD=1.")
        verify_daemon_cache(client)
        with import_lock():
            if name in tags(client):
                return validate_import(client, name, spec)
            if manifest_exists(name):
                raise ImportError("An unrecognized local model manifest exists; it was left untouched.")
            blob = CACHE / "blobs" / ("sha256-" + spec["sha256"])
            if blob.exists() or blob.is_symlink():
                verify_blob(blob, spec)
            else:
                temporary = blob.with_name(blob.name + ".copy-" + uuid.uuid4().hex)
                try:
                    copy_model_file(source, temporary)
                    verify_blob(temporary, spec)
                    os.link(temporary, blob)  # Atomic, never replaces an existing blob.
                finally:
                    temporary.unlink(missing_ok=True)
            response = client.head(ENDPOINT + "/api/blobs/sha256:" + spec["sha256"])
            if response.status_code != 200:
                raise ImportError("The dedicated daemon cannot see the verified project blob. No model was created.")
            if name in tags(client) or manifest_exists(name):
                raise ImportError("The model name appeared during import; it was left untouched.")
            created = metadata(client, "POST", "/api/create", {
                "model": name, "files": {source.name: "sha256:" + spec["sha256"]},
                "parameters": {"num_ctx": CONTEXT}, "stream": False,
            })
            if created.get("status") != "success":
                raise ImportError("Ollama did not confirm the import. No generation was attempted.")
            return validate_import(client, name, spec)
    finally:
        if own_client:
            client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(MODEL_NAMES), required=True)
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        result = import_candidate(args.candidate)
        print(json.dumps(result, indent=2, allow_nan=False))
        print("Verified local import. No generation, download, or default-model change.")
        return 0
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
    except Exception:
        print("The local candidate import could not be verified. Existing models were not replaced; native details were suppressed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
