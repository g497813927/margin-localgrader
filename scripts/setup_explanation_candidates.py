#!/usr/bin/env python3
"""Download pinned public candidate weights with verified, resumable ranges.

This setup-only command imports no grading application or student documents.
Saved progress contains public model metadata only; signed CDN URLs are never logged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
from urllib.parse import urljoin, urlsplit

import httpx

from run_explanation_candidate import CANDIDATES, ROOT

CHUNK_BYTES = 8 * 1024 * 1024
MAX_MODEL_BYTES = 64 * 1024**3
MAX_STATE_BYTES = 4 * 1024 * 1024
DISK_MARGIN = 64 * 1024 * 1024
PROGRESS_INTERVAL = 5.0


class DownloadError(RuntimeError):
    """Messages are fixed diagnostics, never server bodies or signed URLs."""


def workers_value(value):
    try:
        workers = int(value)
        if not 1 <= workers <= 16:
            raise ValueError
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Workers must be between 1 and 16.") from None
    return workers


def safe_url(value):
    try:
        parsed = urlsplit(value)
        hostname = (parsed.hostname or "").lower()
        host_ok = hostname in ("huggingface.co", "hf.co") or hostname.endswith((".huggingface.co", ".hf.co"))
        if (parsed.scheme != "https" or not host_ok or parsed.port not in (None, 443)
                or parsed.username is not None or parsed.password is not None or parsed.fragment):
            raise ValueError
    except (ValueError, TypeError):
        raise DownloadError("Only HTTPS Hugging Face and its HF CDN hosts are allowed.") from None
    return value


def no_links(path, root):
    if root.is_symlink() or any(part.is_symlink() for part in (path, *path.parents) if part == root or root in part.parents):
        raise DownloadError("Model download paths must not contain symbolic links.")


def spec_paths(candidate, spec, root):
    relative = Path(spec["path"])
    if (relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "models"
            or not relative.name.endswith(".gguf") or type(spec["size"]) is not int
            or not 0 < spec["size"] <= MAX_MODEL_BYTES
            or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", spec["repo"])
            or not re.fullmatch(r"[a-f0-9]{40}", spec["revision"])
            or not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"])):
        raise DownloadError("The pinned candidate metadata is invalid.")
    final = root / relative
    no_links(final, root)
    state_dir = final.parent / ("." + final.name + ".parallel-download")
    no_links(state_dir, root)
    url = safe_url(f'https://huggingface.co/{spec["repo"]}/resolve/{spec["revision"]}/{relative.name}')
    identity = {"schema_version": 1, "candidate": candidate, "repo": spec["repo"],
                "revision": spec["revision"], "filename": relative.name, "size": spec["size"],
                "sha256": spec["sha256"], "chunk_bytes": CHUNK_BYTES}
    return final, state_dir, url, identity


def sha_region(fd, offset, size):
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        data = os.pread(fd, min(1024 * 1024, remaining), offset)
        if not data:
            raise DownloadError("The saved model data is incomplete.")
        digest.update(data)
        remaining -= len(data)
        offset += len(data)
    return digest.hexdigest()


def open_regular(path, flags, mode=0o600):
    try:
        fd = os.open(path, flags | os.O_NOFOLLOW, mode)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise DownloadError("Download state must use regular files.")
        return fd
    except OSError:
        raise DownloadError("Could not open the local model or download state safely.") from None


def verify_file(path, spec):
    fd = open_regular(path, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        if before.st_size != spec["size"] or sha_region(fd, 0, spec["size"]) != spec["sha256"]:
            raise DownloadError("Existing model size or checksum differs from the pinned release. It was left untouched.")
        after = os.fstat(fd)
        current = path.stat()
        key = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        if key(before) != key(after) or key(after) != key(current):
            raise DownloadError("The model changed during verification. It was not accepted.")
    finally:
        os.close(fd)


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".download-state-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_state(path):
    fd = open_regular(path, os.O_RDONLY)
    try:
        if os.fstat(fd).st_size > MAX_STATE_BYTES:
            raise DownloadError("Download state exceeds its size limit.")
        with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as stream:
            value = json.load(stream)
    except (ValueError, UnicodeError):
        raise DownloadError("Download state is invalid; existing files were left untouched.") from None
    finally:
        os.close(fd)
    if not isinstance(value, dict):
        raise DownloadError("Download state is invalid; existing files were left untouched.")
    return value


@contextmanager
def download_lock(path):
    fd = open_regular(path, os.O_RDWR | os.O_CREAT)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DownloadError("Another parallel downloader is already working on this model.") from None
        yield
    finally:
        os.close(fd)


def fetch_range(client, url, start, end, total):
    """Follow only checked HTTPS HF redirects and accept the exact requested range."""
    requested = end - start + 1
    for attempt in range(3):
        current = safe_url(url)
        began = time.monotonic()
        try:
            for redirect in range(9):
                with client.stream("GET", current, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location or redirect == 8:
                            raise DownloadError("The model server returned an invalid redirect chain.")
                        current = safe_url(urljoin(current, location))
                        continue
                    if response.status_code != 206:
                        raise DownloadError("The model server did not honor the exact byte-range request.")
                    if response.headers.get("content-range") != f"bytes {start}-{end}/{total}":
                        raise DownloadError("The model server returned the wrong byte range or release size.")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise DownloadError("Compressed range responses are not accepted.")
                    length = response.headers.get("content-length")
                    if length is not None and length != str(requested):
                        raise DownloadError("The model server returned the wrong range length.")
                    data = bytearray()
                    for block in response.iter_raw(chunk_size=65536):
                        if time.monotonic() - began > 180:
                            raise DownloadError("A model chunk exceeded its time limit. Run again to resume.")
                        if len(data) + len(block) > requested:
                            raise DownloadError("The model server returned an oversized chunk.")
                        data.extend(block)
                    if len(data) != requested:
                        raise DownloadError("The model server returned an incomplete chunk.")
                    return bytes(data)
            raise DownloadError("The model server returned an invalid redirect chain.")
        except httpx.HTTPError:
            if attempt == 2:
                raise DownloadError("A model download request failed or timed out. Run again to resume.") from None
            time.sleep(attempt + 1)
    raise DownloadError("The public model download did not complete.")


def chunk_bounds(index, total):
    start = index * CHUNK_BYTES
    return start, min(total, start + CHUNK_BYTES)


def download_candidate(candidate, *, root=ROOT, workers=8, offline=False, progress_path=None, client=None):
    workers = workers_value(workers)
    spec = CANDIDATES[candidate]
    root = Path(root).absolute()
    final, state_dir, url, identity = spec_paths(candidate, spec, root)
    if final.exists() or final.is_symlink():
        verify_file(final, spec)
        print(f"Verified {candidate}: the pinned model is already complete.", flush=True)
        return final
    if offline:
        raise DownloadError("The complete pinned model is not present. Offline mode never downloads or promotes partial files.")
    final.parent.mkdir(parents=True, exist_ok=True)
    no_links(state_dir, root)
    state_dir.mkdir(exist_ok=True)
    part, state_path = state_dir / "weights.part", state_dir / "state.json"
    for path in (part, state_path, state_dir / "writer.lock"):
        no_links(path, root)
    if progress_path and Path(progress_path).is_symlink():
        raise DownloadError("The progress path must not be a symbolic link.")
    progress_path = Path(progress_path).resolve() if progress_path else None
    if progress_path:
        if progress_path.suffix != ".json" or progress_path.is_symlink() or not progress_path.parent.is_dir() or progress_path in (state_path.resolve(), final.resolve(), part.resolve()) or state_dir.resolve() in progress_path.parents:
            raise DownloadError("Use a separate JSON progress path in an existing directory.")
        if progress_path.exists():
            previous = read_state(progress_path)
            if previous.get("kind") != "margin-public-model-download" or previous.get("candidate") != candidate:
                raise DownloadError("The progress path contains unrelated data and was left untouched.")
    own_client = client is None
    if own_client:
        client = httpx.Client(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(60, connect=20),
                              limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers))
    try:
        with download_lock(state_dir / "writer.lock"):
            if final.exists() or final.is_symlink():
                verify_file(final, spec)
                return final
            if state_path.exists():
                state = read_state(state_path)
                if {key: state.get(key) for key in identity} != identity or not isinstance(state.get("completed"), dict):
                    raise DownloadError("Saved chunks belong to different model metadata. Existing files were left untouched.")
                if not part.exists():
                    raise DownloadError("Saved chunk state has no partial file. Existing state was left untouched.")
                fd = open_regular(part, os.O_RDWR)
                if os.fstat(fd).st_size != spec["size"]:
                    os.close(fd)
                    raise DownloadError("The saved partial file has the wrong size and was left untouched.")
            else:
                if part.exists():
                    raise DownloadError("An unrecognized partial file exists and was left untouched.")
                if shutil.disk_usage(final.parent).free < spec["size"] + DISK_MARGIN:
                    raise DownloadError("Not enough free disk space for the pinned model and verification margin.")
                fd = open_regular(part, os.O_RDWR | os.O_CREAT | os.O_EXCL)
                os.ftruncate(fd, spec["size"])
                os.fsync(fd)
                state = {**identity, "completed": {}}
                atomic_json(state_path, state)
            stop = threading.Event()
            state_lock = threading.Lock()
            try:
                count = (spec["size"] + CHUNK_BYTES - 1) // CHUNK_BYTES
                valid = {}
                for index_text, digest in state["completed"].items():
                    if not re.fullmatch(r"0|[1-9][0-9]*", index_text) or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                        raise DownloadError("Saved chunk checksums are invalid; state was left untouched.")
                    index = int(index_text)
                    if index >= count:
                        raise DownloadError("Saved chunk indices are invalid; state was left untouched.")
                    start, end = chunk_bounds(index, spec["size"])
                    if sha_region(fd, start, end - start) == digest:
                        valid[index_text] = digest
                state["completed"] = valid
                completed_bytes = sum(chunk_bounds(int(index), spec["size"])[1] - chunk_bounds(int(index), spec["size"])[0] for index in valid)
                remaining = spec["size"] - completed_bytes
                if shutil.disk_usage(final.parent).free < remaining + DISK_MARGIN:
                    raise DownloadError("Not enough free disk space to finish the remaining chunks.")
                atomic_json(state_path, state)
                started, last_progress = time.monotonic(), 0.0
                def progress(status, force=False):
                    nonlocal last_progress
                    now = time.monotonic()
                    if not force and now - last_progress < PROGRESS_INTERVAL:
                        return
                    with state_lock:
                        done = sum(chunk_bounds(int(index), spec["size"])[1] - chunk_bounds(int(index), spec["size"])[0] for index in state["completed"])
                    value = {"kind": "margin-public-model-download", "candidate": candidate, "status": status,
                             "verified_bytes": done, "total_bytes": spec["size"], "workers": workers,
                             "elapsed_seconds": round(now - started, 1)}
                    if progress_path:
                        atomic_json(progress_path, value)
                    print(f'{candidate}: {status}; verified {done}/{spec["size"]} bytes ({100 * done / spec["size"]:.1f}%).', flush=True)
                    last_progress = now
                def one_chunk(index):
                    if stop.is_set():
                        return
                    start, end = chunk_bounds(index, spec["size"])
                    data = fetch_range(client, url, start, end - 1, spec["size"])
                    if stop.is_set():
                        return
                    digest = hashlib.sha256(data).hexdigest()
                    written = 0
                    while written < len(data):
                        size = os.pwrite(fd, data[written:], start + written)
                        if size <= 0:
                            raise DownloadError("Could not save a model chunk.")
                        written += size
                    with state_lock:
                        os.fsync(fd)
                        state["completed"][str(index)] = digest
                        atomic_json(state_path, state)
                progress("downloading", True)
                try:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        pending = {pool.submit(one_chunk, index) for index in range(count) if str(index) not in valid}
                        try:
                            while pending:
                                finished, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                                for future in finished:
                                    future.result()
                                progress("downloading")
                        except BaseException:
                            stop.set()
                            for future in pending:
                                future.cancel()
                            raise
                except BaseException:
                    stop.set()
                    raise
                progress("verifying", True)
                os.fsync(fd)
                if sha_region(fd, 0, spec["size"]) != spec["sha256"]:
                    # Chunk digests prove local consistency, not release authenticity.
                    # Retain the bytes but invalidate their completion records so a
                    # subsequent explicit run can recover from incorrect server data.
                    state["completed"] = {}
                    atomic_json(state_path, state)
                    raise DownloadError("The completed file does not match the pinned SHA-256. No model was promoted; partial bytes were retained. Run again to fetch fresh chunks.")
                if os.fstat(fd).st_size != spec["size"]:
                    raise DownloadError("The completed partial file changed size. No model was promoted.")
                # Hard-link promotion is atomic and cannot overwrite a file another tool created.
                try:
                    os.link(part, final)
                except FileExistsError:
                    raise DownloadError("Another tool created the final model path. Both files were left untouched.") from None
                part.unlink()
                progress("verified", True)
                return final
            finally:
                stop.set()
                os.close(fd)
    finally:
        if own_client:
            client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--workers", type=workers_value, default=8)
    parser.add_argument("--offline", action="store_true", help="Verify a complete local model only; no network or writes.")
    parser.add_argument("--progress-json", type=Path, help="Optional explicit public-model progress file; never contains signed URLs.")
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        download_candidate(args.candidate, workers=args.workers, offline=args.offline, progress_path=args.progress_json)
    except KeyboardInterrupt:
        print("Download interrupted. Verified chunks are retained for the next run.", file=sys.stderr)
        return 130
    except DownloadError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("The public model download could not finish. Existing model files were not replaced; run again to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
