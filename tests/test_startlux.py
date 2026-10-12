"""StartLux transport/launcher tests, with no model weights or inference."""
import hashlib
import errno
import importlib.util
import json
import math
import socket
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest


SPEC = importlib.util.spec_from_file_location("run_startlux", Path(__file__).parents[1] / "scripts/run_startlux.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def output_for(ids, *, values=None):
    return {"truncated": False, "tokens_evaluated": len(ids),
            "completion_probabilities": [{"top_logprobs": [
                {"id": token, "logprob": value} for token, value in (values or {32: -2.0, 33: -1.0}).items()]}]}


def mock_completion(monkeypatch, handler):
    original = httpx.Client
    observed = {}
    def client(**kwargs):
        observed.update(kwargs)
        return original(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(httpx, "Client", client)
    return runtime.LocalCompletion(1237, list(range(32, 58)), 248320, 16384), observed


def test_offline_environment_and_backend_command_cannot_inherit_remote_settings():
    env = runtime.offline_environment({"LLAMA_ARG_HOST": "0.0.0.0", "LLAMA_ARG_MODEL_URL": "https://example.com",
                                       "HTTPS_PROXY": "http://example.com", "all_proxy": "socks5://example.com",
                                       "HF_HUB_OFFLINE": "0", "PATH": "/usr/bin"})
    assert not any(key.startswith("LLAMA_") or "proxy" in key.lower() for key in env)
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    command = runtime.backend_command("llama-server", Path("local.gguf"), 1237, 16384)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--cache-ram") + 1] == "0"
    assert {"--offline", "--log-disable", "--no-webui", "--no-slots", "--no-context-shift",
            "--no-cache-idle-slots"} <= set(command)


def test_weight_and_source_integrity_are_required(monkeypatch, tmp_path):
    model = tmp_path / runtime.FILENAME
    model.write_bytes(b"synthetic weights")
    source = tmp_path / "reviewed.py"
    source.write_bytes(b"reviewed source")
    monkeypatch.setattr(runtime, "MODEL_SIZE", model.stat().st_size)
    monkeypatch.setattr(runtime, "MODEL_SHA256", hashlib.sha256(model.read_bytes()).hexdigest())
    monkeypatch.setattr(runtime, "ASSET_HASHES", {"reviewed.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    runtime.verify_assets(tmp_path)
    source.write_bytes(b"unreviewed source")
    with pytest.raises(runtime.SafeRuntimeError, match="runtime or tokenizer verification failed"):
        runtime.verify_assets(tmp_path)
    model.write_bytes(b"partial")
    with pytest.raises(runtime.SafeRuntimeError, match="missing or incomplete"):
        runtime.verify_assets(tmp_path)


def test_transport_uses_same_full_prompt_for_missing_letter_fallback(monkeypatch):
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert str(request.url) == "http://127.0.0.1:1237/completion"
        values = {32: -2.0} if len(seen) == 1 else {32: -2.0, 33: -1.0}
        return httpx.Response(200, json=output_for(body["prompt"], values=values))
    client, options = mock_completion(monkeypatch, handler)
    with client.client:
        assert client.letter_logprobs([101, 102, 103], 2) == [-2.0, -1.0]
    assert options["trust_env"] is False and options["follow_redirects"] is False
    assert [body["n_probs"] for body in seen] == [256, 248320]
    assert all(body["prompt"] == [101, 102, 103] for body in seen)
    assert all(body["cache_prompt"] is False and body["n_predict"] == 1 and body["temperature"] == -1 for body in seen)


@pytest.mark.parametrize("update", [
    {"truncated": True}, {"truncated": None}, {"tokens_evaluated": 2}, {"tokens_evaluated": 3.0},
])
def test_unconfirmed_complete_prompt_is_rejected(monkeypatch, update):
    output = output_for([1, 2, 3])
    output.update(update)
    client, _ = mock_completion(monkeypatch, lambda request: httpx.Response(200, json=output))
    with client.client, pytest.raises(runtime.SafeRuntimeError, match="complete prompt"):
        client.letter_logprobs([1, 2, 3], 2)


@pytest.mark.parametrize("status", [307, 400, 500])
def test_redirects_and_backend_error_bodies_are_not_exposed(monkeypatch, capsys, status):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(status, text="PRIVATE_PROMPT_ECHO", headers={"location": "https://example.com"})
    client, _ = mock_completion(monkeypatch, handler)
    with client.client, pytest.raises(runtime.SafeRuntimeError) as error:
        client.letter_logprobs([1, 2, 3], 2)
    assert len(seen) == 1
    assert "PRIVATE_PROMPT_ECHO" not in str(error.value)
    captured = capsys.readouterr()
    assert "PRIVATE_PROMPT_ECHO" not in captured.out + captured.err


def test_response_and_input_limits_fail_before_unsafe_processing(monkeypatch):
    response = httpx.Response(200, content=b"x" * 1024)
    with pytest.raises(runtime.SafeRuntimeError, match="size limit"):
        runtime.bounded_json_response(response, limit=100)
    handler = Mock(side_effect=AssertionError("Oversize prompt must not reach the transport"))
    client, _ = mock_completion(monkeypatch, handler)
    client.context_size = 3
    with client.client, pytest.raises(runtime.SafeInputError, match="No input was truncated"):
        client.letter_logprobs([1, 2, 3], 2)
    handler.assert_not_called()


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, Exception])
def test_wrapper_sanitizes_every_untrusted_exception(monkeypatch, capsys, error_type):
    engine = SimpleNamespace(decide=Mock(side_effect=error_type("PRIVATE_PROMPT_ECHO")))
    status, payload = runtime.decision_response(engine, {"state": "PRIVATE_PROMPT_ECHO", "questions": {"q": {}}})
    assert status == 503
    assert "PRIVATE_PROMPT_ECHO" not in json.dumps(payload)
    assert "answers" not in payload
    captured = capsys.readouterr()
    assert "PRIVATE_PROMPT_ECHO" not in captured.out + captured.err


@pytest.mark.parametrize("model", ["clef-flash", "PRIVATE_MODEL_DETAIL", "", None, 123, {}, []])
def test_wrong_model_is_rejected_without_calling_engine(model):
    engine = SimpleNamespace(decide=Mock(side_effect=AssertionError("An unrecognized model must not run")))
    status, payload = runtime.decision_response(engine, {
        "model": model, "state": "Synthetic evidence.", "questions": {"q": {}}})
    assert status == 400
    assert payload == {"error": "This endpoint serves only the verified local StartLux Decision v1 4B Q8_0 model."}
    engine.decide.assert_not_called()


@pytest.mark.parametrize("model", [
    "startlux-decision-4b-q8", "startlux-models/StartLux-Decision-v1-4B-Q8_0-GGUF",
    "StartLux-Decision-v1-4B-Q8_0",
])
def test_supported_model_aliases_report_the_actual_checkpoint(model):
    engine = SimpleNamespace(decide=Mock(return_value=({"q": {"probabilities": {"a": 1, "b": 0}}}, {"input_tokens": 10})))
    status, payload = runtime.decision_response(engine, {
        "model": model, "state": "Synthetic evidence.", "questions": {"q": {}}})
    assert status == 200
    assert payload["model"] == "startlux-decision-4b-q8"
    engine.decide.assert_called_once_with("Synthetic evidence.", {"q": {}})


def test_omitted_model_uses_the_single_local_checkpoint():
    engine = SimpleNamespace(decide=Mock(return_value=({}, {"input_tokens": 10})))
    status, payload = runtime.decision_response(engine, {"state": "Synthetic evidence.", "questions": {"q": {}}})
    assert status == 200
    assert payload["model"] == "startlux-decision-4b-q8"
    engine.decide.assert_called_once()


def test_tokenizer_guard_explicitly_disables_truncation():
    base = SimpleNamespace(encode=Mock(return_value=[1, 2, 3]))
    assert runtime.CompleteTokenizer(base).encode("full text", truncation=True, add_special_tokens=False) == [1, 2, 3]
    base.encode.assert_called_once_with("full text", truncation=False, add_special_tokens=False)


def test_startup_failure_stops_both_owned_processes_without_output_leaks(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(runtime.sys, "argv", ["run_startlux.py", "--model-dir", str(tmp_path)])
    monkeypatch.setattr(runtime, "verify_assets", lambda directory: None)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="version: 0.6.0", stderr=""))
    fake_socket = Mock()
    fake_socket.__enter__ = Mock(return_value=fake_socket)
    fake_socket.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(runtime.socket, "socket", lambda: fake_socket)
    children, invocations = [], []
    def popen(command, **kwargs):
        child = Mock()
        child.poll.return_value = None
        children.append(child)
        invocations.append((command, kwargs))
        return child
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    def ready(child, port):
        if port == 1236:
            raise RuntimeError("PRIVATE_STARTUP_DETAIL")
    monkeypatch.setattr(runtime, "wait_ready", ready)
    with pytest.raises(SystemExit):
        runtime.main()
    assert len(children) == 2
    assert fake_socket.bind.call_args_list == [call(("127.0.0.1", 1236)), call(("127.0.0.1", 1237))]
    assert fake_socket.setsockopt.call_args_list == [call(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)] * 2
    assert all(kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL for command, kwargs in invocations)
    assert all(child.terminate.call_count == 1 for child in children)
    captured = capsys.readouterr()
    assert "PRIVATE_STARTUP_DETAIL" not in captured.out + captured.err


def test_port_preflight_reuses_time_wait_after_closed_loopback_connection():
    # An ephemeral test socket, never a bundled model port or model process.
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        address = listener.getsockname()
        with socket.create_connection(address, timeout=1) as client:
            with listener.accept()[0] as accepted:
                accepted.settimeout(1)
                accepted.shutdown(socket.SHUT_WR)  # Server actively closes first.
                assert client.recv(1) == b""
    with socket.socket() as plain:
        with pytest.raises(OSError) as raised:
            plain.bind(address)
    assert raised.value.errno == errno.EADDRINUSE
    runtime.check_port(address[1])


def test_port_preflight_still_rejects_an_active_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with pytest.raises(OSError) as raised:
            runtime.check_port(listener.getsockname()[1])
    assert raised.value.errno == errno.EADDRINUSE


PUBLISHER = runtime.ROOT / "models/startlux-decision-4b-q8"


@pytest.mark.skipif(not (PUBLISHER / "startlux_decision/model.py").is_file(), reason="Optional pinned publisher runtime is not downloaded")
def test_official_decision_readout_is_retained_without_loading_weights(monkeypatch):
    """Run publisher formatting and probability math only, with fake logits."""
    from transformers import AutoTokenizer

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return json.dumps(messages, ensure_ascii=False) + "<think>\n\n</think>\n\n"

        def encode(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return [ord(char) - ord("A") + 32 if "A" <= char <= "Z" else ord(char) + 1000 for char in text]

    tokenizer_loads = []
    def tokenizer(path, **kwargs):
        tokenizer_loads.append(kwargs)
        return Tokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer)
    completion = SimpleNamespace(letter_logprobs=Mock(side_effect=lambda ids, count: list(range(count))))
    monkeypatch.setattr(runtime, "LocalCompletion", lambda *args: completion)
    engine = runtime.make_engine(PUBLISHER, 1237, 4096)
    answers, usage = engine.decide("Synthetic evidence.", {
        "custom": {"type": "choice", "instructions": "Choose the supported description.",
                   "criteria": {"first-z": "First", "second-a": "Second"}},
        "ordinal": {"type": "score", "instructions": "Assess the level.", "criteria": ["Low", "Middle", "High"]},
    })
    assert tokenizer_loads == [{"local_files_only": True, "trust_remote_code": False}]
    assert answers["custom"]["choice"] == "second-a"
    assert set(answers["custom"]["probabilities"]) == {"first-z", "second-a"}
    assert answers["custom"]["probabilities"]["second-a"] == pytest.approx(1 / (1 + math.exp(-1 / 1.3872)))
    assert list(answers["ordinal"]["probabilities"]) == ["0", "1", "2"]
    expected = [math.exp(number / 1.4172) for number in range(3)]
    assert list(answers["ordinal"]["probabilities"].values()) == pytest.approx([value / sum(expected) for value in expected])
    assert usage["output_tokens"] == 0
    assert completion.letter_logprobs.call_count == 2
    completion.letter_logprobs.reset_mock()
    with pytest.raises(runtime.SafeInputError, match="No input was truncated"):
        engine.decide("x" * 5000, {"q": {"type": "choice", "instructions": "Choose.", "criteria": {"a": "One", "b": "Two"}}})
    completion.letter_logprobs.assert_not_called()
