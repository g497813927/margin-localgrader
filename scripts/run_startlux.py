#!/usr/bin/env python3
"""Run the pinned StartLux GGUF locally with its official decision readout.

This launcher never downloads anything. It verifies the weights and reviewed
runtime, starts llama.cpp on 127.0.0.1:1237, and serves /v1/systemone on :1236.
The official renderer, option ordering, per-type temperatures, and answer
mapping are retained. No submitted text is logged or written to a prompt cache.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c08b2c12095f8e743a2d90e02963485d65b379a5"
MODEL_NAME = "startlux-decision-4b-q8"
MODEL_ALIASES = frozenset({MODEL_NAME,
    "startlux-models/StartLux-Decision-v1-4B-Q8_0-GGUF", "StartLux-Decision-v1-4B-Q8_0"})
FILENAME = "StartLux-Decision-4B-Q8_0.gguf"
MODEL_SIZE = 4482402816
MODEL_SHA256 = "3de47f14ca2807813a670d697e70815fbecf020732ca4d7fc5f2e6c3898e34df"
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024 * 1024  # A full-vocabulary log-probability readout.
# Files reviewed from the pinned publisher snapshot; no remote code is trusted
# by Transformers. Import only these locally verified Python source files.
ASSET_HASHES = {
    "startlux_decision/__init__.py": "52c7fd27759098245b8fbe8fdb50846eefa06116b5500466621f6ff1139d61b0",
    "startlux_decision/model.py": "a902607fdd2b34b7007bc26763cd17f9e9157c54a45dc0f3a5ef3b3e4e38db62",
    "startlux_decision/jevfmt.py": "b0bd50b78e6391d91a8428db781949224012ee6b1ba8e6eda2889fda1ca4bfa7",
    "decision_config.json": "5ab7e2671a3fe113e89b5ebad2b808c64a6b751ab2663c025913922c0df065f7",
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
}


class SafeInputError(ValueError):
    """Only fixed, adapter-authored messages may cross the HTTP boundary."""


class SafeRuntimeError(RuntimeError):
    """Only fixed, adapter-authored messages may cross the HTTP boundary."""


def offline_environment(source=None):
    source = os.environ if source is None else source
    env = {key: value for key, value in source.items()
           if not key.startswith("LLAMA_") and key.upper() not in
           {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}}
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                "OTEL_SDK_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(directory):
    try:
        model = directory / FILENAME
        if not model.is_file() or model.stat().st_size != MODEL_SIZE:
            raise SafeRuntimeError("StartLux weights are missing or incomplete. Finish the pinned download before starting.")
        if file_hash(model) != MODEL_SHA256:
            raise SafeRuntimeError("StartLux weight checksum verification failed. No model was started.")
        for relative, expected in ASSET_HASHES.items():
            path = directory / relative
            if not path.is_file() or file_hash(path) != expected:
                raise SafeRuntimeError("StartLux runtime or tokenizer verification failed. Restore the complete pinned snapshot.")
    except OSError:
        raise SafeRuntimeError("Could not read the local StartLux files. Check the model directory and permissions.") from None


class CompleteTokenizer:
    """Keep the publisher's exact rendering while forbidding encode truncation."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def encode(self, text, **kwargs):
        kwargs["truncation"] = False
        return self.tokenizer.encode(text, **kwargs)


def render_jobs(renderer, tokenizer, rows, context_size):
    jobs = []
    try:
        for row in rows:
            order = [option["id"] for option in row["options"]]
            # llama.cpp reads one next-token distribution. Reserve its one token.
            ids, _ = renderer.render_ids(row, tokenizer, order, max_length=context_size - 1)
            if not ids or len(ids) >= context_size:
                raise SafeInputError("The complete prompt exceeds the local context limit. No input was truncated or graded.")
            jobs.append((ids, len(order)))
    except SafeInputError:
        raise
    except (ValueError, TypeError, KeyError):
        raise SafeInputError("The decision request is invalid or its complete prompt exceeds the local context limit. No input was truncated or graded.") from None
    return jobs


def bounded_json_response(response, limit=MAX_RESPONSE_BYTES):
    if response.status_code != 200:
        raise SafeRuntimeError("The local llama.cpp server rejected the request. No grade was assigned.")
    raw = bytearray()
    for chunk in response.iter_bytes(chunk_size=65536):
        if len(raw) + len(chunk) > limit:
            raise SafeRuntimeError("The local llama.cpp response exceeded the response size limit.")
        raw.extend(chunk)
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise SafeRuntimeError("The local llama.cpp server returned invalid JSON.") from None
    if not isinstance(result, dict):
        raise SafeRuntimeError("The local llama.cpp server returned an invalid response.")
    return result


class LocalCompletion:
    def __init__(self, port, letters, vocabulary, context_size):
        import httpx
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise SafeInputError("Choose a loopback port between 1024 and 65535.")
        self.url = f"http://127.0.0.1:{port}/completion"
        self.client = httpx.Client(trust_env=False, follow_redirects=False, timeout=900.0)
        self.letters, self.vocabulary, self.context_size = letters, vocabulary, context_size

    def close(self):
        self.client.close()

    def letter_logprobs(self, ids, count):
        import httpx
        if not 2 <= count <= len(self.letters) or not ids or len(ids) >= self.context_size:
            raise SafeInputError("The complete decision prompt does not fit the local runtime. No input was truncated or graded.")
        targets = self.letters[:count]
        # This is the publisher's readout: first top-256, then the full vocabulary
        # if an option letter is absent. Neither attempt changes the prompt.
        for number in (256, self.vocabulary):
            body = {"prompt": ids, "n_predict": 1, "temperature": -1,
                    "n_probs": number, "cache_prompt": False, "stream": False,
                    "response_fields": ["completion_probabilities", "probs", "truncated", "tokens_evaluated"]}
            try:
                with self.client.stream("POST", self.url, json=body) as response:
                    output = bounded_json_response(response)
            except httpx.HTTPError:
                raise SafeRuntimeError("Could not reach the local llama.cpp server. No grade was assigned.") from None
            if (output.get("truncated") is not False
                    or type(output.get("tokens_evaluated")) is not int
                    or output["tokens_evaluated"] != len(ids)):
                raise SafeRuntimeError("The local runtime did not confirm the complete prompt. No grade was assigned.")
            rows = output.get("completion_probabilities", output.get("probs"))
            try:
                if not isinstance(rows, list) or len(rows) != 1:
                    raise ValueError
                got = {token["id"]: token["logprob"] for token in rows[0]["top_logprobs"]}
                if all(token in got for token in targets):
                    values = [got[token] for token in targets]
                    if any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in values):
                        raise ValueError
                    return values
            except (ValueError, TypeError, KeyError, IndexError):
                raise SafeRuntimeError("The local runtime returned an invalid option distribution. No grade was assigned.") from None
        raise SafeRuntimeError("The local runtime omitted required option probabilities. No grade was assigned.")


def make_engine(directory, backend_port, context_size):
    """Load only the verified publisher code; no Torch model or CUDA graphs."""
    import torch
    from transformers import AutoTokenizer

    name = "_margin_startlux_runtime"
    package = directory / "startlux_decision"
    spec = importlib.util.spec_from_file_location(name, package / "__init__.py",
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    renderer = sys.modules[name + ".jevfmt"]
    config = json.loads((directory / "decision_config.json").read_text())
    metadata = json.loads((directory / "config.json").read_text())

    class LocalDecision(module.StartLuxDecision):
        def __init__(self):
            self.tok = CompleteTokenizer(AutoTokenizer.from_pretrained(
                str(directory), local_files_only=True, trust_remote_code=False))
            self.letters = renderer.check_tokenizer(self.tok)
            if self.letters != config["letter_token_ids"]:
                raise SafeRuntimeError("The local tokenizer does not match the verified decision configuration.")
            self.temperature = {key: float(value) for key, value in config["temperature_by_type"].items()}
            self.max_length, self.graphs = context_size - 1, {}
            self.completion = LocalCompletion(backend_port, self.letters,
                int(metadata.get("text_config", metadata)["vocab_size"]), context_size)

        def _logits(self, rows):
            jobs = render_jobs(renderer, self.tok, rows, context_size)
            logits = [torch.tensor(self.completion.letter_logprobs(ids, count), dtype=torch.float32)
                      for ids, count in jobs]
            return logits, sum(len(ids) for ids, count in jobs)

        def decide(self, state, questions):
            if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
                raise SafeInputError("Provide between 1 and 32 decision questions.")
            try:
                rows = [renderer.from_systemone(state, question) for question in questions.values()]
                # App rubrics have at most 12 levels. Keep each decision complete
                # in a single pass; no tournament approximation for wider lists.
                if any(not 2 <= len(row["options"]) <= 26 for row in rows):
                    raise SafeInputError("Each decision needs 2 to 26 options (2 to 10 for an ordinal score).")
                render_jobs(renderer, self.tok, rows, context_size)
            except SafeInputError:
                raise
            except (ValueError, TypeError, KeyError, AttributeError):
                raise SafeInputError("The decision question format is invalid. No grade was assigned.") from None
            return super().decide(state, questions)

    return LocalDecision()


def decision_response(engine, payload):
    if not isinstance(payload, dict) or "state" not in payload or "questions" not in payload:
        return 400, {"error": "Provide a state and decision questions as a JSON object."}
    # This process serves one verified checkpoint. Never label its output as
    # another requested model; an omitted model explicitly selects this default.
    requested_model = payload.get("model", MODEL_NAME)
    if not isinstance(requested_model, str) or requested_model not in MODEL_ALIASES:
        return 400, {"error": "This endpoint serves only the verified local StartLux Decision v1 4B Q8_0 model."}
    try:
        answers, usage = engine.decide(payload["state"], payload["questions"])
        return 200, {"answers": answers, "usage": usage, "model": MODEL_NAME}
    except SafeInputError as exc:
        return 422, {"error": str(exc)}
    except SafeRuntimeError as exc:
        return 503, {"error": str(exc)}
    except Exception:
        return 503, {"error": "Local StartLux inference failed. No grade was assigned."}


def serve_wrapper(directory, port, backend_port, context_size):
    # This process is launched with both output streams connected to DEVNULL.
    os.environ.clear()
    os.environ.update(offline_environment(_STARTUP_ENV))
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(ROOT))
    from grader.privacy import install_network_guard
    install_network_guard()
    verify_assets(directory)
    engine = make_engine(directory, backend_port, context_size)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass

        def send_json(self, status, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            authority = f"127.0.0.1:{port}"
            if (self.headers.get("Host") != authority
                    or self.headers.get("Origin") not in (None, f"http://{authority}")
                    or self.headers.get("Sec-Fetch-Site") == "cross-site"):
                self.send_json(403, {"error": "Only direct local requests are allowed."})
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            if self.path == "/health":
                self.send_json(200, {"status": "ok", "model": MODEL_NAME})
            else:
                self.send_json(404, {"error": "Not found."})

        def do_POST(self):
            if not self.allowed():
                return
            if self.path != "/v1/systemone":
                return self.send_json(404, {"error": "Not found."})
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length") is None:
                return self.send_json(411, {"error": "A bounded Content-Length is required."})
            try:
                size = int(self.headers["Content-Length"])
                if not 0 < size <= MAX_REQUEST_BYTES:
                    return self.send_json(413, {"error": "The local request exceeds the size limit."})
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError
                payload = json.loads(raw)
            except (ValueError, OSError):
                return self.send_json(400, {"error": "The local request is incomplete or contains invalid JSON."})
            status, payload = decision_response(engine, payload)
            self.send_json(status, payload)

    class QuietServer(HTTPServer):
        def handle_error(self, request, client_address):
            pass

    # One request at a time, matching the single llama slot and avoiding queues
    # that retain multiple submissions. No access or exception logging.
    try:
        with QuietServer(("127.0.0.1", port), Handler) as server:
            server.serve_forever()
    finally:
        engine.completion.close()


def backend_command(binary, model, port, context_size):
    return [str(binary), "--model", str(model), "--alias", MODEL_NAME,
            "--host", "127.0.0.1", "--port", str(port), "--offline", "--log-disable",
            "--no-webui", "--no-slots", "--cors-origins", "http://127.0.0.1",
            "--ctx-size", str(context_size), "--parallel", "1", "--n-gpu-layers", "99",
            "--batch-size", "2048", "--ubatch-size", "512", "--no-context-shift",
            "--cache-ram", "0", "--no-cache-idle-slots"]


def wait_ready(process, port, timeout=180):
    import httpx
    deadline = time.monotonic() + timeout
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=2.0) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise SafeRuntimeError("A local StartLux process stopped during startup. Check memory, ports, and local GPU access.")
            try:
                with client.stream("GET", f"http://127.0.0.1:{port}/health") as response:
                    if bounded_json_response(response, limit=65536).get("status") == "ok":
                        return
            except (httpx.HTTPError, SafeRuntimeError):
                pass
            time.sleep(.25)
    raise SafeRuntimeError("The local StartLux process did not become ready before the startup timeout.")


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def check_port(port):
    # Match HTTPServer/llama.cpp restart behavior: old TIME_WAIT connections
    # must not block preflight, while an active listener must still fail bind.
    # SO_REUSEPORT is deliberately not enabled.
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", port))


_STARTUP_ENV = dict(os.environ)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / MODEL_NAME)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--port", type=int, default=1236)
    parser.add_argument("--backend-port", type=int, default=1237)
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument("--check-only", action="store_true", help="Verify local files without starting either server.")
    parser.add_argument("--_serve-wrapper", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 1024 <= args.port <= 65535 or not 1024 <= args.backend_port <= 65535
            or args.port == args.backend_port):
        parser.error("Choose two different local ports between 1024 and 65535.")
    if not 256 <= args.ctx_size <= 65536:
        parser.error("Choose a context size between 256 and 65536 tokens.")
    directory = args.model_dir.expanduser().resolve()
    children = []
    stage = "verifying local files"
    try:
        if args._serve_wrapper:
            serve_wrapper(directory, args.port, args.backend_port, args.ctx_size)
            return
        print("Verifying the pinned local StartLux weights, runtime, and tokenizer…", flush=True)
        verify_assets(directory)
        if args.check_only:
            print("Verified. No model or server was started.", flush=True)
            return
        binary = (str(args.llama_server.expanduser()) if args.llama_server else
                  shutil.which("llama-server") or "/opt/homebrew/bin/llama-server")
        env = offline_environment()
        stage = "checking the installed llama.cpp version"
        inspected = subprocess.run([binary, "--version"], env=env, capture_output=True,
                                   text=True, timeout=15, check=True)
        match = re.search(r"version:\s*(\d+)\.(\d+)\.(\d+)", inspected.stdout + inspected.stderr)
        if not match or tuple(map(int, match.groups())) < (0, 6, 0):
            raise SafeRuntimeError("Use llama.cpp v0.6.0 or newer with Qwen3.5 support.")
        stage = "checking local ports"
        for port in (args.port, args.backend_port):
            check_port(port)
        stage = "starting the local llama.cpp backend"
        backend = subprocess.Popen(backend_command(binary, directory / FILENAME, args.backend_port, args.ctx_size),
                                   env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children.append(backend)
        stage = "waiting for the local llama.cpp backend"
        wait_ready(backend, args.backend_port)
        stage = "starting the local decision wrapper"
        wrapper = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--_serve-wrapper",
                                    "--model-dir", str(directory), "--port", str(args.port),
                                    "--backend-port", str(args.backend_port), "--ctx-size", str(args.ctx_size)],
                                   env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children.append(wrapper)
        stage = "waiting for the local decision wrapper"
        wait_ready(wrapper, args.port)
        print(f"Local StartLux is ready at http://127.0.0.1:{args.port}/v1/systemone", flush=True)
        print("Offline; runtime output and prompt caching disabled. Ctrl-C stops both owned processes.", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise SafeRuntimeError("A local StartLux process stopped. Both owned processes have been stopped.")
    except KeyboardInterrupt:
        print("\nStopping local StartLux.", flush=True)
    except SafeRuntimeError as exc:
        parser.exit(1, f"StartLux failed while {stage}: {exc}\n")
    except Exception as exc:
        # The stage and built-in error class help diagnose startup without ever
        # echoing arbitrary library messages, request content, or native output.
        kind = next((name for cls, name in ((OSError, "OS error"), (subprocess.SubprocessError, "subprocess error"),
                    (ImportError, "dependency import error"), (ValueError, "invalid value"), (TypeError, "type error"))
                    if isinstance(exc, cls)), "unexpected error")
        number = f", errno {exc.errno}" if isinstance(exc, OSError) and isinstance(exc.errno, int) else ""
        parser.exit(1, f"Could not start local StartLux while {stage} ({kind}{number}). Check local files, dependencies, ports, and GPU access.\n")
    finally:
        for child in reversed(children):
            stop_process(child)


if __name__ == "__main__":
    main()
