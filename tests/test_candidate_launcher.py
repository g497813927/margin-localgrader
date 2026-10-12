"""Candidate launcher checks: synthetic files and mocked processes only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import urllib.error

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_explanation_candidate.py"
SPEC = importlib.util.spec_from_file_location("candidate_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def synthetic_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    data = b"GGUF synthetic comparison model"
    info = dict(launcher.CANDIDATES["pro"], size=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(launcher, "CANDIDATES", {"pro": info})
    model = tmp_path / info["path"]
    model.parent.mkdir(parents=True)
    model.write_bytes(data)
    return model


def startup_metadata():
    alias = launcher.CANDIDATES["pro"]["alias"]
    props = {"model_alias": alias, "total_slots": 1, "default_generation_settings": {"n_ctx": 16384},
             "ui": False, "cors_proxy_enabled": False, "modalities": {"vision": False, "video": False, "audio": False},
             "build_info": "0.6.0 (build 11429, commit d81235049)"}
    models = {"data": [{"id": alias, "meta": {"n_ctx": 16384}}]}
    return props, models


def test_hash_verification_reads_pinned_file_without_creating_cache(tmp_path, monkeypatch):
    model = synthetic_weights(tmp_path, monkeypatch)
    before = set(tmp_path.rglob("*"))
    assert launcher.verify_model("pro") == model
    assert set(tmp_path.rglob("*")) == before


def test_hash_rejects_same_size_modified_weights(tmp_path, monkeypatch):
    model = synthetic_weights(tmp_path, monkeypatch)
    model.write_bytes(b"x" * model.stat().st_size)
    with pytest.raises(launcher.LaunchError, match="checksum"):
        launcher.verify_model("pro")


@pytest.mark.parametrize("condition", ["missing", "partial", "symlink"])
def test_weights_fail_closed_before_launch(tmp_path, monkeypatch, condition):
    model = synthetic_weights(tmp_path, monkeypatch)
    if condition == "missing":
        model.unlink()
    elif condition == "partial":
        model.write_bytes(b"GGUF")
    else:
        target = tmp_path / "untrusted.gguf"
        model.rename(target)
        model.symlink_to(target)
    with pytest.raises(launcher.LaunchError):
        launcher.verify_model("pro")


def test_environment_discards_remote_tools_logging_injection_and_preserves_home():
    inherited = {"HOME": "/existing/home", "PATH": "/usr/bin", "LLAMA_ARG_MODEL_URL": "https://invalid/model",
                 "LLAMA_ARG_HOST": "0.0.0.0", "LLAMA_ARG_TOOLS": "all", "LLAMA_LOG_FILE": "private.log",
                 "HTTPS_PROXY": "http://remote.invalid", "DYLD_INSERT_LIBRARIES": "untrusted.dylib",
                 "HF_TOKEN": "synthetic", "OTEL_EXPORTER_OTLP_ENDPOINT": "https://remote.invalid"}
    env = launcher.offline_environment(inherited)
    assert env["HOME"] == inherited["HOME"]
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["OTEL_SDK_DISABLED"] == "true"
    assert not (set(inherited) - {"HOME", "PATH"}) & set(env)


def test_command_fixes_locality_no_logs_no_tools_no_cache_and_bounded_context():
    model = launcher.ROOT / launcher.CANDIDATES["pro"]["path"]
    cmd = launcher.build_command(Path("/installed/llama-server"), "pro", model, 11436)
    for flag, value in (("--host", "127.0.0.1"), ("--port", "11436"), ("--parallel", "1"),
                        ("--ctx-size", "16384"), ("--n-predict", "4096"), ("--cache-ram", "0"),
                        ("--reasoning", "on"), ("--reasoning-format", "deepseek")):
        assert cmd[cmd.index(flag) + 1] == value
    assert all(flag in cmd for flag in launcher.REQUIRED_FLAGS)
    assert not {"--model-url", "--hf-repo", "--rpc", "--tools", "--mcp-servers-json", "--slot-save-path", "--log-file"} & set(cmd)
    with pytest.raises(launcher.LaunchError):
        launcher.build_command(Path("/installed/server"), "pro", Path("/other/model.gguf"), 11436)


def test_qwen38_4b_pin_and_launch_keep_community_identity():
    info = launcher.candidate_spec("qwen38_4b")
    assert info == {
        "alias": "qwen38-distill-4b-q4",
        "repo": "empero-ai/Qwen3.8-4B-Distill-GGUF",
        "revision": "391fc7d103e3942a408def3e4f51c2f85d464417",
        "path": "models/qwen38-distill-4b-q4/Qwen3.8-4B-Q4_K_M.gguf",
        "size": 2783446304,
        "sha256": "dec96e8cf2e11b613bb46513dec485377f9ca5a351e71712ee0e244f287c6790",
    }
    cmd = launcher.build_command(Path("/installed/server"), "qwen38_4b", launcher.ROOT / info["path"], 11436)
    assert cmd[cmd.index("--alias") + 1] == "qwen38-distill-4b-q4"
    assert cmd[cmd.index("--reasoning") + 1] == "on"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_candidate_error_lists_all_pinned_choices():
    with pytest.raises(launcher.LaunchError, match="qwen36"):
        launcher.candidate_spec("unpublished-model")


def test_qwen36_moe_pin_and_launch_preserve_exact_release():
    info = launcher.candidate_spec("qwen36")
    assert info == {
        "alias": "qwen36-35b-a3b-ud-q4-k-xl",
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "revision": "a483e9e6cbd595906af30beda3187c2663a1118c",
        "path": "models/qwen36-35b-a3b-ud-q4-k-xl/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "size": 22360456160,
        "sha256": "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450",
    }
    cmd = launcher.build_command(Path("/installed/server"), "qwen36", launcher.ROOT / info["path"], 11436)
    assert cmd[cmd.index("--alias") + 1] == info["alias"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--parallel") + 1] == "1"
    assert cmd[cmd.index("--ctx-size") + 1] == "16384"
    assert {"--offline", "--log-disable", "--no-context-shift"} <= set(cmd)


@pytest.mark.parametrize("port", [11434, 11435, 80, 65536, True, "11436"])
def test_non_candidate_ports_rejected(port):
    with pytest.raises(launcher.LaunchError):
        launcher.validate_port(port)


def test_metadata_identifies_only_pinned_text_model_and_expected_context():
    props, models = startup_metadata()
    launcher.validate_metadata(props, models, "pro")
    for key, value in (("total_slots", 2), ("model_alias", "other"), ("ui", True),
                       ("cors_proxy_enabled", True), ("build_info", "unreviewed")):
        invalid = dict(props, **{key: value})
        with pytest.raises(launcher.LaunchError):
            launcher.validate_metadata(invalid, models, "pro")
    props["default_generation_settings"]["n_ctx"] = 8192
    with pytest.raises(launcher.LaunchError):
        launcher.validate_metadata(props, models, "pro")


def test_metadata_rejects_multimodal_missing_and_wrong_model():
    props, models = startup_metadata()
    props["modalities"]["vision"] = True
    with pytest.raises(launcher.LaunchError):
        launcher.validate_metadata(props, models, "pro")
    for p, m in (({}, models), (startup_metadata()[0], {"data": []}), (startup_metadata()[0], {"data": [{"id": "other"}]})):
        with pytest.raises(launcher.LaunchError):
            launcher.validate_metadata(p, m, "pro")


def test_redirects_and_non_metadata_requests_are_rejected():
    with pytest.raises(launcher.LaunchError, match="redirect"):
        launcher.NoRedirect().redirect_request(None)
    with pytest.raises(launcher.LaunchError, match="startup metadata"):
        launcher.metadata(None, 11436, "/v1/chat/completions")


def test_ready_client_disables_proxies_and_performs_only_metadata(monkeypatch):
    constructed = []
    seen = []
    client = object()
    monkeypatch.setattr(launcher.urllib.request, "build_opener", lambda *args: (constructed.extend(args), client)[1])
    props, models = startup_metadata()
    def metadata(actual_client, port, path):
        assert actual_client is client and port == 11436
        seen.append(path)
        return {"/health": {"status": "ok"}, "/props": props, "/v1/models": models}[path]
    monkeypatch.setattr(launcher, "metadata", metadata)
    launcher.wait_ready(SimpleNamespace(poll=lambda: None), "pro", 11436)
    assert constructed[0].proxies == {}
    assert isinstance(constructed[1], launcher.NoRedirect)
    assert seen == ["/health", "/props", "/v1/models"]


def test_runtime_errors_never_print_native_output(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "check_port", lambda port: None)
    monkeypatch.setattr(launcher, "find_binary", lambda path: Path("/installed/server"))
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "server", output="synthetic private prompt", stderr="synthetic private prompt")
    monkeypatch.setattr(launcher.subprocess, "run", fail)
    assert launcher.main(["--candidate", "pro"]) == 1
    assert "synthetic private prompt" not in "".join(capsys.readouterr())


def test_launch_nulls_all_native_streams_and_owns_process_group(tmp_path, monkeypatch):
    model = synthetic_weights(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "check_port", lambda port: None)
    monkeypatch.setattr(launcher, "find_binary", lambda path: Path("/installed/server"))
    monkeypatch.setattr(launcher, "check_runtime", lambda *args: None)
    monkeypatch.setattr(launcher, "wait_ready", lambda *args: None)
    calls = []
    process = SimpleNamespace(pid=123456)
    def wait():
        raise KeyboardInterrupt
    process.wait = wait
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    stopped = []
    monkeypatch.setattr(launcher, "stop_owned", stopped.append)
    assert launcher.main(["--candidate", "pro"]) == 0
    command, kwargs = calls[0]
    assert command[command.index("--model") + 1] == str(model)
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert stopped == [process]


def test_teardown_targets_only_owned_group(monkeypatch):
    sent = []
    waits = []
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    process = SimpleNamespace(pid=998877, wait=lambda **kw: waits.append(kw))
    launcher.stop_owned(process)
    assert sent == [(998877, signal.SIGTERM)]
    assert waits == [{"timeout": 10}]


def test_no_arbitrary_host_or_model_cli_override():
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--candidate", "pro", "--host", "0.0.0.0"])
    assert exc.value.code == 2


@pytest.mark.parametrize("termination_signal", [signal.SIGTERM, signal.SIGHUP])
def test_termination_signals_reap_owned_child_and_restore_handlers(tmp_path, monkeypatch, termination_signal):
    synthetic_weights(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "check_port", lambda port: None)
    monkeypatch.setattr(launcher, "find_binary", lambda path: Path("/installed/server"))
    monkeypatch.setattr(launcher, "check_runtime", lambda *args: None)
    monkeypatch.setattr(launcher, "wait_ready", lambda *args: None)
    previous = {item: signal.getsignal(item) for item in (signal.SIGTERM, signal.SIGHUP)}
    process = SimpleNamespace(pid=123456)
    def terminate_wait():
        signal.getsignal(termination_signal)(termination_signal, None)
    process.wait = terminate_wait
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: process)
    stopped = []
    monkeypatch.setattr(launcher, "stop_owned", stopped.append)
    assert launcher.main(["--candidate", "pro"]) == 0
    assert stopped == [process]
    assert all(signal.getsignal(item) == handler for item, handler in previous.items())


def test_cleanup_failure_never_emits_raw_native_error(tmp_path, monkeypatch, capsys):
    synthetic_weights(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "check_port", lambda port: None)
    monkeypatch.setattr(launcher, "find_binary", lambda path: Path("/installed/server"))
    monkeypatch.setattr(launcher, "check_runtime", lambda *args: None)
    def fail(*args):
        raise OSError("PRIVATE_NATIVE_DIAGNOSTIC")
    monkeypatch.setattr(launcher, "verify_model", fail)
    monkeypatch.setattr(launcher, "stop_owned", fail)
    assert launcher.main(["--candidate", "pro"]) == 1
    assert "PRIVATE_NATIVE_DIAGNOSTIC" not in "".join(capsys.readouterr())
