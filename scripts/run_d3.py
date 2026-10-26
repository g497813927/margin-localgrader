#!/usr/bin/env python3
"""Serve pinned d3-mini locally, text only, without downloads or request logs.

The first load verifies the complete official checkpoint. Use --device mps for
an explicit Apple GPU trial or --device cpu for the PyTorch reference path.
There is no automatic device fallback. No training or model downloads occur.
"""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import ctypes
import hashlib
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_ID = "vllm-sr/d3-mini"
REVISION = "a475367de0d8133a3c1a2a0dc12303d89cfa2ab4"
MODEL_NAME = "d3-mini"
MANIFEST_SHA256 = "05b33ccfce265f3794d60a35b58ea831841f63cd9d4bb296aca6811a8522efc7"
VENDOR_HASHES = {
    "d3_runtime.py": "6c049aa7d3da8da3738602a2812572879d0a47b03c28c3e33f36d7300257de89",
    "d3_format.py": "187a0f359f6f89a844e723377769556f78133b118851e702e4b0fe5123ccb0f1",
    "d3_fast.py": "68a1fb9a2c3bde9ac2763bf504decae592ca12722698fbde1b39a31262955535",
    "d3_kernels.py": "8f7534ca1e1846ebf7c4c73058002f3be4dbf0d5d426f0815150ae0cbe7677bd",
}
MAX_BODY_BYTES = 1_000_000
MAX_STATE_CHARS = 200_000
MAX_QUESTIONS = 32


class InvalidRequest(ValueError):
    pass


class ContextExceeded(ValueError):
    pass


class InvalidOutput(RuntimeError):
    pass


def configure_offline():
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "OTEL_SDK_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
        "D3_FAST": "0", "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    sys.dont_write_bytecode = True
    logging.disable(logging.CRITICAL)
    from grader.privacy import install_network_guard
    install_network_guard()


def flush_native_output():
    """Flush the same C runtime used by Python before restoring output handles."""
    # Windows Python uses the Universal CRT; CDLL(None) is the executable there,
    # and does not export fflush. On POSIX the process symbol table contains libc.
    library = "ucrtbase.dll" if sys.platform == "win32" else None
    runtime = ctypes.CDLL(library)
    runtime.fflush.argtypes = [ctypes.c_void_p]
    runtime.fflush.restype = ctypes.c_int
    if runtime.fflush(None) != 0:
        raise RuntimeError("Could not flush the local runtime output safely.")


@contextmanager
def silence_process():
    """Silence native/Python output for the whole single-worker server lifetime."""
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    saved = []
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            for fd in (1, 2):
                saved.append((fd, os.dup(fd)))
                os.dup2(sink.fileno(), fd)
            with redirect_stdout(sink), redirect_stderr(sink):
                try:
                    yield
                finally:
                    sink.flush()
                    flush_native_output()
    finally:
        for fd, original in reversed(saved):
            try:
                os.dup2(original, fd)
            finally:
                os.close(original)


def verify_local_sources(model_dir):
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Local checkpoint is missing.")
    manifest = root / "MODEL_MANIFEST.json"
    if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Checkpoint manifest is not the pinned d3-mini release.")
    for name, digest in VENDOR_HASHES.items():
        path = ROOT / "vendor/d3" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Pinned local runtime source verification failed.")
    return root


def load_runtime(model_dir, device, max_context):
    """Import only reviewed local code; use the official prediction implementation."""
    configure_offline()
    root = verify_local_sources(model_dir)
    if device not in {"mps", "cpu"} or not 256 <= max_context <= 8192:
        raise ValueError("Unsupported device or context budget.")
    from grader.dependency_pins import pinned_version
    try:
        expected_version = pinned_version(ROOT / "requirements-candidates.txt", "transformers")
    except (OSError, ValueError) as exc:
        raise RuntimeError("The pinned model requirements are missing or invalid.") from exc
    if version("transformers") != expected_version:
        raise RuntimeError(f"The pinned runtime requires transformers {expected_version}.")
    import torch
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; choose CPU explicitly.")
    from vendor.d3.d3_runtime import D3, kernel_report
    kernels = kernel_report()
    if any(value.startswith(("fla", "causal_conv1d"))
           for key, value in kernels.items() if key != "fla"):
        raise RuntimeError("CPU/MPS requires the PyTorch reference kernels, without CUDA-only kernels.")
    model = D3.from_pretrained(
        str(root), local_files_only=True, verify="full", device=device,
        batch_size=1, max_length=max_context, model_name=MODEL_NAME,
        permutation_average=False,
    )
    # The reference forward pass already specifies use_cache=False and rejects
    # KV caches for noncausal decisions. Keep configuration aligned as well.
    model.backbone.config.use_cache = False
    model.backbone.config.text_config.use_cache = False
    return model


def validate_request(body):
    if not isinstance(body, dict) or set(body) != {"model", "state", "questions"}:
        raise InvalidRequest()
    if body["model"] not in (MODEL_NAME, REPO_ID):
        raise InvalidRequest()
    state, questions = body["state"], body["questions"]
    if not isinstance(state, str) or not state.strip() or len(state) > MAX_STATE_CHARS:
        raise InvalidRequest()
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise InvalidRequest()
    for key, question in questions.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key):
            raise InvalidRequest()
        if not isinstance(question, dict) or set(question) - {"type", "instructions", "criteria"}:
            raise InvalidRequest()
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise InvalidRequest()
        kind, criteria = question.get("type"), question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise InvalidRequest()
            if any(not isinstance(k, str) or not k or not isinstance(v, (str, type(None)))
                   for k, v in criteria.items()):
                raise InvalidRequest()
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise InvalidRequest()
            if any(not isinstance(level, str) or not level for level in criteria):
                raise InvalidRequest()
        elif kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - {"false", "true"}
                or any(not isinstance(v, str) for v in criteria.values())
            ):
                raise InvalidRequest()
        else:
            raise InvalidRequest()
    return state, questions


def finite_probability(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def safe_result(response, questions):
    """Preserve native probabilities while excluding errors and arbitrary echoes."""
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise InvalidOutput()
    answers = response["answers"]
    if set(answers) != set(questions):
        raise InvalidOutput()
    clean = {}
    for key, definition in questions.items():
        answer, kind = answers[key], definition["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind or "error" in answer:
            raise InvalidOutput()
        if kind == "noul":
            if not finite_probability(answer.get("noul")):
                raise InvalidOutput()
            clean[key] = {"type": kind, "noul": answer["noul"]}
            continue
        expected = (set(definition["criteria"]) if kind == "choice"
                    else {str(i) for i in range(len(definition["criteria"]))})
        probs = answer.get("probabilities")
        if (not isinstance(probs, dict) or set(probs) != expected
                or any(not finite_probability(p) for p in probs.values())
                or not 0.98 <= sum(probs.values()) <= 1.02):
            raise InvalidOutput()
        clean[key] = {"type": kind, "probabilities": dict(probs)}
        if "confidence" in answer:
            if not finite_probability(answer["confidence"]):
                raise InvalidOutput()
            clean[key]["confidence"] = answer["confidence"]
        if kind == "choice":
            if answer.get("choice") not in expected:
                raise InvalidOutput()
            clean[key]["choice"] = answer["choice"]
        else:
            score = answer.get("score")
            if (not isinstance(score, (int, float)) or isinstance(score, bool)
                    or not math.isfinite(score) or not 0 <= score <= len(expected) - 1):
                raise InvalidOutput()
            clean[key]["score"] = score  # Expected index, never rubric points.
            clean[key]["legend"] = {str(i): text for i, text in enumerate(definition["criteria"])}
    usage = response.get("usage")
    if (not isinstance(usage, dict) or type(usage.get("input_tokens")) is not int
            or usage["input_tokens"] < 0 or usage.get("output_tokens") != 0
            or usage.get("truncated")):
        raise InvalidOutput()
    return {"model": MODEL_NAME, "answers": clean,
            "usage": {"input_tokens": usage["input_tokens"], "output_tokens": 0, "truncated": False}}


def infer_complete(model, state, questions, max_context):
    prepared = model.prepare(state, questions)
    if prepared.errors:
        if any(e.get("error") == "max_length_exceeded" for e in prepared.errors.values()):
            raise ContextExceeded()
        raise InvalidRequest()
    if prepared.images or prepared.videos or set(prepared.sequences) != set(questions):
        raise InvalidRequest()
    if any(len(tokens) > max_context for tokens in prepared.sequences.values()):
        raise ContextExceeded()
    probabilities, tokens = model.run(prepared)
    return safe_result(model.respond(prepared, probabilities, tokens), questions)


def create_app(model, max_context=8192, device="cpu"):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
    lock = threading.Lock()

    def result(body, status=200):
        return JSONResponse(body, status_code=status,
                            headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    @app.get("/health")
    def health():
        return result({"ok": True, "model": MODEL_NAME, "revision": REVISION,
                       "device": device, "max_context": max_context, "text_only": True})

    @app.get("/v1/models")
    def models():
        return result({"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})

    @app.post("/v1/systemone")
    async def system_one(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            return result({"error": "JSON requests are required."}, 415)
        try:
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > MAX_BODY_BYTES:
                    return result({"error": "Request exceeds the local size limit."}, 413)
            body = json.loads(raw)
            state, questions = validate_request(body)
        except Exception:
            return result({"error": "Invalid text-only SystemOne request."}, 422)
        if not lock.acquire(blocking=False):
            return result({"error": "The local model is busy."}, 409)
        try:
            response = await run_in_threadpool(infer_complete, model, state, questions, max_context)
            return result(response)
        except ContextExceeded:
            return result({"error": "Input exceeds the complete-input token limit; nothing was graded."}, 422)
        except InvalidRequest:
            return result({"error": "Invalid decision question; nothing was graded."}, 422)
        except Exception:
            # Upstream exceptions/error answers may echo input. Never return or
            # log their message, traceback, request, or partial response.
            return result({"error": "Local decision inference failed; no grade was assigned."}, 500)
        finally:
            lock.release()

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/d3-mini")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps" if sys.platform == "darwin" else "cpu")
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--port", type=int, default=1238)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535 or not 256 <= args.max_context <= 8192:
        parser.error("Use a port from 1024 to 65535 and a context budget from 256 to 8192.")
    configure_offline()
    print("Verifying and loading local d3-mini. Runtime and request output are disabled.", flush=True)
    try:
        with silence_process():
            model = load_runtime(args.model_dir, args.device, args.max_context)
            import uvicorn
            uvicorn.run(create_app(model, args.max_context, args.device), host="127.0.0.1",
                        port=args.port, workers=1, access_log=False, log_config=None)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        # The class alone is safe; the runtime message may contain input/path data.
        print(f"Local d3-mini stopped ({type(exc).__name__}). Check local files, dependencies, device, or memory.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
