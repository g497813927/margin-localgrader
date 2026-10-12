"""Launcher privacy checks using synthetic files; never loads a model."""
import hashlib
import errno
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_ollama.py"
spec = importlib.util.spec_from_file_location("margin_ollama_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def fixture_cache(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    blobs = source / "blobs"
    blobs.mkdir()
    parts = []
    for content in (b'{"model_format":"gguf"}', b"GGUF synthetic weights"):
        digest = hashlib.sha256(content).hexdigest()
        (blobs / ("sha256-" + digest)).write_bytes(content)
        parts.append({"digest": "sha256:" + digest, "size": len(content)})
    raw = json.dumps({"config": parts[0], "layers": [parts[1]]}).encode()
    target = source / launcher.MANIFEST
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    monkeypatch.setattr(launcher, "MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(launcher, "clone_model_file", shutil.copyfile)
    return source, tmp_path / "private", parts


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


def test_cache_only_copies_selected_verified_manifest_and_preserves_source(tmp_path, monkeypatch):
    source, destination, parts = fixture_cache(tmp_path, monkeypatch)
    unrelated = source / "blobs/unrelated-model"
    unrelated.write_text("Do not copy")
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert launcher.prepare_model_cache(source, destination) == destination
    after = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert before == after
    assert not (destination / "blobs/unrelated-model").exists()
    assert (destination / launcher.MANIFEST).read_bytes() == (source / launcher.MANIFEST).read_bytes()
    assert len(list((destination / "blobs").iterdir())) == 2
    for part in parts:
        name = part["digest"].replace(":", "-")
        assert (source / "blobs" / name).stat().st_ino != (destination / "blobs" / name).stat().st_ino


def test_second_setup_reuses_existing_project_clones(tmp_path, monkeypatch):
    source, destination, _ = fixture_cache(tmp_path, monkeypatch)
    launcher.prepare_model_cache(source, destination)
    monkeypatch.setattr(launcher, "clone_model_file", lambda *args: pytest.fail("Existing files must not be recloned"))
    launcher.prepare_model_cache(source, destination)


def test_wrong_manifest_stops_before_creating_cache(tmp_path, monkeypatch):
    source, destination, _ = fixture_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "MANIFEST_SHA256", "0" * 64)
    with pytest.raises(launcher.LaunchError, match="manifest does not match"):
        launcher.prepare_model_cache(source, destination)
    assert not destination.exists()


def test_incomplete_source_stops_before_creating_cache(tmp_path, monkeypatch):
    source, destination, parts = fixture_cache(tmp_path, monkeypatch)
    (source / "blobs" / parts[-1]["digest"].replace(":", "-")).unlink()
    with pytest.raises(launcher.LaunchError, match="incomplete"):
        launcher.prepare_model_cache(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("relative", ["same", "nested", "parent"])
def test_original_cache_cannot_be_destination(tmp_path, monkeypatch, relative):
    source, destination, _ = fixture_cache(tmp_path, monkeypatch)
    destination = {"same": source, "nested": source / "private", "parent": tmp_path}[relative]
    with pytest.raises(launcher.LaunchError, match="separate project cache"):
        launcher.prepare_model_cache(source, destination)


def test_destination_blob_symlink_rejected_without_touching_source(tmp_path, monkeypatch):
    source, destination, parts = fixture_cache(tmp_path, monkeypatch)
    (destination / "blobs").mkdir(parents=True)
    name = parts[0]["digest"].replace(":", "-")
    target = destination / "blobs" / name
    target.symlink_to(source / "blobs" / name)
    with pytest.raises(launcher.LaunchError, match="unexpected file"):
        launcher.prepare_model_cache(source, destination)
    assert target.is_symlink()


def test_existing_key_check_never_creates_or_reads_key(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(launcher.LaunchError, match="key is absent"):
        launcher.check_existing_key()
    assert not (tmp_path / ".ollama").exists()
    (tmp_path / ".ollama").mkdir()
    (tmp_path / ".ollama/id_ed25519").write_bytes(b"synthetic key not read")
    launcher.check_existing_key()


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


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS APFS cloning")
def test_real_clone_of_tiny_synthetic_file_has_independent_contents(tmp_path):
    source, destination = tmp_path / "source.txt", tmp_path / "copy.txt"
    source.write_bytes(b"synthetic original")
    launcher.clone_model_file(source, destination)
    assert source.stat().st_ino != destination.stat().st_ino
    destination.write_bytes(b"synthetic changed")
    assert source.read_bytes() == b"synthetic original"
    assert destination.stat().st_mode & 0o777 == 0o600


def test_main_launches_only_owned_quiet_serve_process(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=1234, wait=lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    stopped = []
    monkeypatch.setattr(launcher, "check_existing_key", lambda: None)
    monkeypatch.setattr(launcher, "check_port", lambda: None)
    monkeypatch.setattr(launcher, "find_binary", lambda explicit: tmp_path / "ollama")
    monkeypatch.setattr(launcher, "check_version", lambda *args: None)
    monkeypatch.setattr(launcher, "prepare_model_cache", lambda source, destination: destination)
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
    monkeypatch.setattr(launcher, "check_existing_key", fail)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not launch"))
    assert launcher.main([]) == 1
    assert "PRIVATE_NATIVE_OUTPUT" not in capsys.readouterr().err
