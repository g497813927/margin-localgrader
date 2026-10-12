"""Launcher privacy checks using synthetic files; never loads a model."""
import errno
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_ollama.py"
spec = importlib.util.spec_from_file_location("margin_ollama_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_offline_environment_removes_inherited_cloud_debug_proxy_and_native_trace(tmp_path):
    inherited = {"HOME": "/existing/home", "PATH": "/usr/bin", "TMPDIR": "/tmp",
                 "OLLAMA_NO_CLOUD": "0", "OLLAMA_DEBUG_LOG_REQUESTS": "true",
                 "OLLAMA_DEBUG": "2", "HTTPS_PROXY": "https://example.invalid",
                 "LLAMA_LOG_FILE": "/private/log", "OLLAMA_HOST": "0.0.0.0:11434",
                 "AWS_SECRET_ACCESS_KEY": "synthetic"}
    result = launcher.offline_environment(tmp_path, inherited)
    assert result["HOME"] == inherited["HOME"]
    assert result["OLLAMA_HOST"] == "127.0.0.1:11435"
    assert result["OLLAMA_MODELS"] == str(tmp_path)
    for key in ("OLLAMA_NO_CLOUD", "OLLAMA_NOHISTORY", "OLLAMA_NOPRUNE"):
        assert result[key] == "1"
    for key in ("OLLAMA_DEBUG", "OLLAMA_DEBUG_LOG_REQUESTS"):
        assert result[key] == "false"
    assert result["OLLAMA_KEEP_ALIVE"] == "0"
    assert result["OLLAMA_NUM_PARALLEL"] == result["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert not any(key in result for key in ("HTTPS_PROXY", "LLAMA_LOG_FILE", "AWS_SECRET_ACCESS_KEY"))


def test_port_preflight_reuses_time_wait_without_a_model_daemon(monkeypatch):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        address = listener.getsockname()
        with socket.create_connection(address, timeout=1) as client:
            with listener.accept()[0] as accepted:
                accepted.settimeout(1)
                accepted.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
    with socket.socket() as plain:
        with pytest.raises(OSError) as raised:
            plain.bind(address)
    assert raised.value.errno == errno.EADDRINUSE
    monkeypatch.setattr(launcher, "PORT", address[1])
    launcher.check_port()


def test_port_preflight_rejects_active_listener_without_reuse_port(monkeypatch):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        monkeypatch.setattr(launcher, "PORT", listener.getsockname()[1])
        with pytest.raises(launcher.LaunchError, match="already occupied"):
            launcher.check_port()


def test_fresh_clone_prepares_empty_store_without_personal_key_or_models(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher.Path, "home", classmethod(lambda cls: pytest.fail("No personal files inspected")))
    destination = tmp_path / "models/ollama-explanations"
    assert launcher.prepare_model_cache(destination) == destination
    assert list(destination.iterdir()) == [destination / "blobs"]
    assert list((destination / "blobs").iterdir()) == []
    assert not (tmp_path / ".ollama").exists()
    launcher.prepare_model_cache(destination)


@pytest.mark.parametrize("outside", ["../private", "models/../../private"])
def test_cache_cannot_escape_project(tmp_path, monkeypatch, outside):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    with pytest.raises(launcher.LaunchError, match="inside this project"):
        launcher.prepare_model_cache(tmp_path / outside)


@pytest.mark.parametrize("link_at", ["cache", "parent", "blobs"])
def test_cache_links_are_rejected_without_writing_target(tmp_path, monkeypatch, link_at):
    root, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    root.mkdir(); elsewhere.mkdir()
    monkeypatch.setattr(launcher, "ROOT", root)
    destination = root / "models/ollama-explanations"
    if link_at == "parent":
        (root / "models").symlink_to(elsewhere)
    else:
        (root / "models").mkdir()
        if link_at == "cache":
            destination.symlink_to(elsewhere)
        else:
            destination.mkdir(); (destination / "blobs").symlink_to(elsewhere)
    with pytest.raises(launcher.LaunchError, match="symbolic link"):
        launcher.prepare_model_cache(destination)
    assert list(elsewhere.iterdir()) == []


def test_version_check_accepts_client_output_without_daemon(tmp_path, monkeypatch):
    def run(command, **kwargs):
        assert command == [str(tmp_path / "ollama"), "--version"]
        assert kwargs["stdin"] == subprocess.DEVNULL
        return SimpleNamespace(stdout="", stderr="Warning: client version is 0.40.3\n")
    monkeypatch.setattr(launcher.subprocess, "run", run)
    launcher.check_version(tmp_path / "ollama", {})


def test_version_check_never_echoes_native_output(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="PRIVATE_NATIVE_OUTPUT", stderr="Warning: client version is 0.40.2"))
    with pytest.raises(launcher.LaunchError) as error:
        launcher.check_version(tmp_path / "ollama", {})
    assert "PRIVATE_NATIVE_OUTPUT" not in str(error.value)


def test_redirects_are_rejected():
    with pytest.raises(launcher.LaunchError, match="redirect"):
        launcher.NoRedirect().redirect_request(None, None, None, None, None)


def test_portable_copy_of_tiny_synthetic_file_has_independent_contents(tmp_path):
    source, destination = tmp_path / "source.txt", tmp_path / "copy.txt"
    source.write_bytes(b"synthetic original")
    launcher.copy_model_file(source, destination)
    assert source.stat().st_ino != destination.stat().st_ino
    destination.write_bytes(b"synthetic changed")
    assert source.read_bytes() == b"synthetic original"
    assert destination.stat().st_mode & 0o777 == 0o600


def test_main_launches_only_owned_quiet_serve_process(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=1234, wait=lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    stopped = []
    monkeypatch.setattr(launcher, "check_port", lambda: None)
    monkeypatch.setattr(launcher, "find_binary", lambda explicit: tmp_path / "ollama")
    monkeypatch.setattr(launcher, "check_version", lambda *args: None)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "wait_ready", lambda owned: None)
    monkeypatch.setattr(launcher, "stop_owned", lambda owned: stopped.append(owned))
    def popen(command, **kwargs):
        assert command == [str(tmp_path / "ollama"), "serve"]
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["env"]["OLLAMA_NO_CLOUD"] == "1"
        return process
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    assert launcher.main(["--model-cache", str(tmp_path / "private")]) == 0
    assert stopped == [process]


def test_main_sanitizes_unexpected_error_and_does_not_start_daemon(monkeypatch, capsys):
    def fail():
        raise RuntimeError("PRIVATE_NATIVE_OUTPUT")
    monkeypatch.setattr(launcher, "check_port", fail)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not launch"))
    assert launcher.main([]) == 1
    assert "PRIVATE_NATIVE_OUTPUT" not in capsys.readouterr().err


def test_portable_copy_never_replaces_an_existing_file(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"new bytes"); destination.write_bytes(b"existing bytes")
    with pytest.raises(launcher.LaunchError):
        launcher.copy_model_file(source, destination)
    assert destination.read_bytes() == b"existing bytes"


@pytest.mark.parametrize("cloud, permitted", [
    ({"disabled": True, "source": "env"}, True),
    ({"disabled": True, "source": "both"}, True),
    ({"disabled": False, "source": "none"}, False),
    ({"disabled": True, "source": "config"}, False),
])
def test_ready_requires_cloud_disabled_by_environment(monkeypatch, cloud, permitted):
    from io import BytesIO
    routes = []
    class Response(BytesIO):
        status = 200
    def open_url(url, timeout):
        routes.append(url)
        value = {"version": launcher.TESTED_VERSION} if url.endswith("/version") else {"cloud": cloud}
        return Response(json.dumps(value).encode())
    monkeypatch.setattr(launcher.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=open_url))
    process = SimpleNamespace(poll=lambda: None)
    if permitted:
        launcher.wait_ready(process)
    else:
        with pytest.raises(launcher.LaunchError, match="OLLAMA_NO_CLOUD"):
            launcher.wait_ready(process)
    assert routes == ["http://127.0.0.1:11435/api/version", "http://127.0.0.1:11435/api/status"]
