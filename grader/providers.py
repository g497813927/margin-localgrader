"""Offline Laya inference and an explicitly loopback-only SystemOne adapter.

Model setup is a separate command. Importing or calling this module never fetches
weights. The UI must not treat the uncalibrated model probabilities as accuracy.
"""
from __future__ import annotations

import ipaddress
import json
import math
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# Set before importing any Hugging Face or Transformers module, not with
# setdefault: a user's online shell settings must not enable network fallback.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "models/laya-multilingual"
DEFAULT_ENDPOINT = "http://127.0.0.1:1234/v1/systemone"
LAYA_VERSION = "0.4.3"
MAX_CONTEXT = 8192
_LOCK = threading.RLock()
_AGENT = None
_AGENT_KEY = None


class LocalInputError(ValueError):
    """Adapter-authored input guidance that is safe to show to the user."""


class LocalRuntimeError(RuntimeError):
    """Adapter-authored runtime guidance that is safe to show to the user."""


def validate_local_endpoint(url: str) -> str:
    """Reject all network destinations except literal loopback IPs/localhost.

    Resolve localhost ourselves to 127.0.0.1 instead of relying on DNS or the
    hosts file. Never follow redirects or environment-configured HTTP proxies.
    """
    if not isinstance(url, str) or not url or url != url.strip():
        raise LocalInputError("Enter a local SystemOne URL without surrounding spaces.")
    if any(ord(char) < 33 for char in url) or "\\" in url:
        raise LocalInputError("The local model URL contains invalid characters.")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise LocalInputError("The local model URL is invalid.") from exc
    if parts.scheme not in {"http", "https"}:
        raise LocalInputError("The local model URL must use http:// or https://.")
    if parts.username is not None or parts.password is not None or "?" in url or "#" in url:
        raise LocalInputError("Local model URLs cannot contain credentials, a query, or a fragment.")
    host = parts.hostname
    if host == "localhost":
        host = "127.0.0.1"
    else:
        try:
            address = ipaddress.ip_address(host or "")
        except ValueError as exc:
            raise LocalInputError("Only localhost or a literal loopback IP is allowed.") from exc
        if not address.is_loopback or "%" in (host or ""):
            raise LocalInputError("The model endpoint must be on this computer (a loopback IP).")
        host = str(address)
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority += f":{port}"
    if not parts.path or parts.path == "/":
        raise LocalInputError("Enter the complete local endpoint, including /v1/systemone.")
    return urlunsplit((parts.scheme, authority, parts.path, "", ""))


def _model_path(settings: dict) -> Path:
    model = settings.get("model") or DEFAULT_MODEL
    if model in {"multilingual", "laya-multilingual", "convaiinnovations/laya-multilingual"}:
        model = DEFAULT_MODEL
    path = Path(model).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise LocalRuntimeError("Local Laya weights are missing. Run .venv/bin/python scripts/setup_laya.py before opening student work.")
    required = ("rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json")
    if any(not (path / name).is_file() for name in required):
        raise LocalRuntimeError("The local Laya checkpoint is incomplete. Run scripts/setup_laya.py to finish setup.")
    return path


def _load_laya(settings: dict):
    global _AGENT, _AGENT_KEY
    path = _model_path(settings)
    device = settings.get("device") or "auto"
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise LocalInputError("Laya device must be auto, cpu, mps, or cuda.")
    key = (str(path), device)
    if _AGENT is not None and key == _AGENT_KEY:
        return _AGENT
    try:
        import laya
        from importlib.metadata import version
    except ImportError as exc:
        raise LocalRuntimeError("Laya is not installed. Run .venv/bin/python scripts/setup_laya.py --install-deps before opening student work.") from exc
    if version("laya") != LAYA_VERSION:
        raise LocalRuntimeError(f"This app requires laya=={LAYA_VERSION} so its complete-input checks match the inference tokenizer. Run setup_laya.py --install-deps.")
    config = json.loads((path / "rl_agent_config.json").read_text())
    encoder = json.loads((path / "encoder/config.json").read_text())
    if config.get("encoder") != "jhu-clsp/mmBERT-base" or encoder.get("max_position_embeddings", 0) < MAX_CONTEXT:
        raise LocalInputError("Use the Laya multilingual mmBERT checkpoint with its documented 8192-token context.")
    try:
        # Laya 0.4.3 does not expose local_files_only. Passing an existing local
        # directory bypasses snapshot_download entirely, and HF offline flags
        # prevent all nested tokenizer/config Hub lookups from using a network.
        agent = laya.load(str(path), device=None if device == "auto" else device)
    except Exception as exc:
        raise LocalRuntimeError(f"Could not load the local Laya checkpoint ({type(exc).__name__}). Check installed dependencies and checkpoint files.") from exc
    _AGENT, _AGENT_KEY = agent, key
    return agent


def _check_complete_input(agent, state: str, questions: dict, max_context: int) -> int:
    """Match the pinned runtime's tokenizer, including its 48-token option cap.

    Laya otherwise truncates options, instructions, and state independently.
    Enlarge the shared head budget to preserve instructions; reject long options
    and any per-question sequence that would discard submission tokens.
    """
    from laya.common import build_sequence, encode_text, render_options

    token_count = lambda text: len(encode_text(agent.tok, text, add_special_tokens=False, truncation=False)["input_ids"])
    internal = {}
    head_budget = 16
    mask = agent.tok.mask_token
    state_tokens = token_count(state.replace(mask, " "))
    for question_id, definition in questions.items():
        if not isinstance(definition, dict) or definition.get("type") not in {"choice", "score", "noul"}:
            raise LocalInputError(f"Question {question_id!r} must use a supported decision type.")
        definition = {"instructions": "", **definition}
        question = agent._to_internal(definition)
        internal[question_id] = question
        options = render_options(question)
        if not options:
            raise LocalInputError(f"Question {question_id!r} has no rubric levels.")
        option_tokens = []
        for number, option in enumerate(options, 1):
            size = token_count(" " + option.replace(mask, " "))
            if size > 48:
                raise LocalInputError(f"Rubric criterion {question_id!r}, level {number}, uses {size} tokens. Laya allows 48 tokens per level, including its label. Shorten this level; no text has been graded.")
            option_tokens.append(size + 1)  # One [MASK] per option.
        instruction_size = token_count(f"{question['t']} question: {str(question['ins']).replace(mask, ' ')}")
        head_budget = max(head_budget, sum(option_tokens) + max(16, instruction_size))
    for question_id, question in internal.items():
        sequence, markers, stats, truncation = build_sequence(
            agent.tok, state, question, max_len=max_context, head_max_len=head_budget,
            return_stats=True, return_truncation_stats=True,
        )
        if truncation["truncated"] or len(markers) != stats["options"]:
            available = truncation["state_tokens_used"]
            raise LocalInputError(f"This submission has {state_tokens} tokens; criterion {question_id!r} leaves room for {available} within Laya's {max_context}-token context. Shorten the submission or rubric. Nothing was silently truncated or graded.")
        if stats["tokens_per_option"] is not None:
            raise LocalInputError(f"Criterion {question_id!r} would truncate rubric levels. Use fewer or shorter levels.")
    return head_budget


def _native_predict(settings: dict, state: str, questions: dict) -> dict:
    with _LOCK:
        agent = _load_laya(settings)
        try:
            max_context = int(settings.get("max_context") or MAX_CONTEXT)
        except (TypeError, ValueError) as exc:
            raise LocalInputError("The Laya context limit must be an integer.") from exc
        if not 256 <= max_context <= MAX_CONTEXT:
            raise LocalInputError("The Laya context limit must be between 256 and 8192 tokens.")
        head_budget = _check_complete_input(agent, state, questions, max_context)
        normalized = {key: {"instructions": "", **value} for key, value in questions.items()}
        try:
            result = agent.predict(state, normalized, max_len=max_context, head_max_len=head_budget)
        except Exception as exc:
            raise LocalRuntimeError(f"Local Laya inference failed ({type(exc).__name__}). Try the CPU device or a shorter submission.") from exc
        if (result.get("usage") or {}).get("truncated"):
            raise LocalRuntimeError("Laya reported discarded input tokens. The grade has been rejected.")
        result["local_runtime"] = {"provider": "laya", "device": str(agent.device), "max_context": max_context}
        return result


def _systemone_predict(settings: dict, state: str, questions: dict) -> dict:
    import httpx

    endpoint = validate_local_endpoint(settings.get("endpoint") or DEFAULT_ENDPOINT)
    model = settings.get("model") or "Cloudflare/clef-flash"
    body = {"model": model, "state": state, "questions": questions}
    try:
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=180.0) as client:
            with client.stream("POST", endpoint, json=body) as response:
                if 300 <= response.status_code < 400:
                    raise LocalRuntimeError("The local model endpoint returned a redirect. Redirects are disabled to keep submissions local.")
                if response.status_code != 200:
                    raise LocalRuntimeError(f"The local model endpoint returned HTTP {response.status_code}. Check that a local SystemOne server is running.")
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 2_000_000:
                        raise LocalRuntimeError("The local model response exceeded the 2 MB limit.")
    except httpx.HTTPError as exc:
        raise LocalRuntimeError(f"Could not reach the local SystemOne model ({type(exc).__name__}). Check its local address and server process.") from exc
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise LocalRuntimeError("The local model did not return valid SystemOne JSON.") from exc
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise LocalRuntimeError("The local model response has no SystemOne answers object.")
    if (result.get("usage") or {}).get("truncated"):
        raise LocalRuntimeError("The local model reported discarded input tokens. The grade has been rejected.")
    return result


def predict(settings: dict, state: str, questions: dict) -> dict:
    """Return a SystemOne payload, including its ``answers`` mapping."""
    if not isinstance(state, str) or not state.strip():
        raise LocalInputError("Enter or import non-empty submission text first.")
    if not isinstance(questions, dict) or not questions:
        raise LocalInputError("Add at least one rubric criterion before grading.")
    provider = settings.get("provider") or "laya"
    if provider == "laya":
        return _native_predict(settings, state, questions)
    if provider in {"systemone", "clef"}:
        return _systemone_predict(settings, state, questions)
    raise LocalInputError("Choose the local Laya runtime or a local SystemOne server.")


def probe(settings: dict) -> dict:
    """Verify inference with synthetic text, never with an uploaded document."""
    try:
        result = predict(settings, "This is a local model connection test.", {
            "connection": {"type": "choice", "instructions": "What is the purpose of this text?", "criteria": {"test": "A connection test", "other": "Something else"}}
        })
        answer = result.get("answers", {}).get("connection")
        if not isinstance(answer, dict):
            raise LocalRuntimeError("The local server returned no answer to the connection test.")
        probabilities = answer.get("probabilities")
        if (not isinstance(probabilities, dict) or set(probabilities) != {"test", "other"}
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or not 0 <= value <= 1
                       for value in probabilities.values())
                or not 0.98 <= sum(probabilities.values()) <= 1.02):
            raise LocalRuntimeError("The local server returned invalid probabilities for the connection test.")
        runtime = result.get("local_runtime") or {}
        detail = f" on {runtime['device']}" if runtime.get("device") else ""
        return {"ok": True, "message": f"Local inference is ready{detail}. The connection test completed."}
    except (LocalInputError, LocalRuntimeError) as exc:
        return {"ok": False, "message": str(exc)}
    except Exception:
        return {"ok": False, "message": "The local model connection test failed. Check its setup and try again."}
