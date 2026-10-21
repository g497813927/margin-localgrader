"""Setup-only checks use tiny metadata fixtures, never models or student data."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("margin_setup_vision", SCRIPTS / "setup_vision.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


@pytest.mark.parametrize("name", [setup.DEFAULT_MODEL, "qwen3.5:2b-q4_K_M", "local-example:q4"])
def test_explicit_local_model_tags(name):
    assert setup.model_name(name) == name


@pytest.mark.parametrize("name", ["openbmb/minicpm-v4.7", "openbmb/minicpm-v4.7:latest", "deepseek:cloud", "https://example.com/m:1b", "host:123/model:1b", "example.com/model:1b", "localhost/model:1b", "a/b/c:1b", "../evil:1b", "a:$(id)", "a:1b\n", ""])
def test_reject_implicit_large_defaults_remote_and_command_strings(name):
    with pytest.raises(argparse.ArgumentTypeError):
        setup.model_name(name)


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=False)


@pytest.mark.parametrize("version,cloud,valid", [
    ("0.40.3", {"disabled": True, "source": "env"}, True),
    ("0.40.5", {"disabled": True, "source": "env"}, True),
    ("0.40.4", {"disabled": True, "source": "env"}, False),
    ("0.40.3", {"disabled": True, "source": "both"}, True),
    ("0.40.2", {"disabled": True, "source": "env"}, False),
    ("0.40.3", {"disabled": False, "source": "none"}, False),
    ("0.40.3", {"disabled": True, "source": "config"}, False),
])
def test_daemon_version_and_cloud_disabled_required(version, cloud, valid):
    paths = []
    def handle(request):
        assert request.url.host == "127.0.0.1" and request.url.port == 11435
        paths.append(request.url.path)
        return httpx.Response(200, json={"version": version} if paths[-1] == "/api/version" else {"cloud": cloud})
    with client_for(handle) as client:
        if valid:
            setup.verify_daemon(client)
        else:
            with pytest.raises(setup.SetupError):
                setup.verify_daemon(client)
    assert all(path in {"/api/version", "/api/status"} for path in paths)


@pytest.mark.parametrize("status", [200, 404, 302])
def test_private_cache_proof_is_ephemeral_and_never_follows_redirect(tmp_path, monkeypatch, status):
    cache = tmp_path / "cache"
    (cache / "blobs").mkdir(parents=True)
    monkeypatch.setattr(setup, "prepare_model_cache", lambda path: cache)
    def handle(request):
        assert request.method == "HEAD"
        digest = request.url.path.split(":")[-1]
        assert (cache / "blobs" / ("sha256-" + digest)).stat().st_size == 32
        return httpx.Response(status, headers={"location": "https://example.invalid/private"})
    with client_for(handle) as client:
        if status == 200:
            setup.verify_cache(client)
        else:
            with pytest.raises(setup.SetupError):
                setup.verify_cache(client)
    assert list((cache / "blobs").iterdir()) == []


@pytest.mark.parametrize("value", [
    {"capabilities": ["completion"]},
    {"capabilities": ["vision"], "requires": "0.41.0"},
    {"capabilities": ["vision"], "requires": "PRIVATE_UNKNOWN"},
    {"capabilities": ["vision"], "remote_host": "https://example.invalid"},
    {"capabilities": ["vision"], "remote_model": "remote"},
])
def test_vision_metadata_rejects_text_only_or_remote(value):
    with client_for(lambda request: httpx.Response(200, json=value)) as client:
        with pytest.raises(setup.SetupError):
            setup.verify_model(client, setup.DEFAULT_MODEL)


def test_pull_uses_only_public_model_tag_and_bounded_sanitized_progress(capsys):
    def handle(request):
        assert request.method == "POST" and request.url.path == "/api/pull"
        assert json.loads(request.content) == {"model": setup.DEFAULT_MODEL, "stream": True}
        return httpx.Response(200, content=b'{"status":"PRIVATE_NATIVE_DETAIL","total":100,"completed":50}\n{"status":"success"}\n')
    with client_for(handle) as client:
        setup.pull_model(client, setup.DEFAULT_MODEL)
    output = capsys.readouterr().out
    assert "50%" in output and "PRIVATE" not in output


@pytest.mark.parametrize("response", [
    httpx.Response(302, headers={"location": "https://example.invalid"}),
    httpx.Response(200, content=b'{"error":"PRIVATE_NATIVE_DETAIL"}\n'),
    httpx.Response(200, content=b'x' * (setup.MAX_LINE + 1)),
    httpx.Response(200, content=b'{"status":"pulling"}\n'),
    httpx.Response(200, content=b'{"status":"success"}\ntruncated'),
])
def test_pull_never_accepts_redirect_or_incomplete_malformed_progress(response, capsys):
    with client_for(lambda request: response) as client:
        with pytest.raises(setup.SetupError) as error:
            setup.pull_model(client, setup.DEFAULT_MODEL)
    assert "PRIVATE" not in str(error.value) + str(capsys.readouterr())


def test_offline_checks_never_download_or_run_inference(monkeypatch, capsys):
    calls = []
    for name in ("verify_daemon", "verify_cache", "verify_model"):
        monkeypatch.setattr(setup, name, lambda *args, name=name: calls.append(name))
    monkeypatch.setattr(setup, "pull_model", lambda *args: pytest.fail("Offline check cannot download"))
    assert setup.main(["--offline"]) == 0
    assert calls == ["verify_daemon", "verify_cache", "verify_model"]
    assert "actual image recognition is checked when used locally" in capsys.readouterr().out


def test_download_not_started_when_cache_cannot_be_verified(monkeypatch):
    monkeypatch.setattr(setup, "verify_daemon", lambda *_: None)
    def fail(*_):
        raise setup.SetupError("Wrong private cache.")
    monkeypatch.setattr(setup, "verify_cache", fail)
    monkeypatch.setattr(setup, "pull_model", lambda *args: pytest.fail("No download before cache proof"))
    assert setup.main([]) == 1


def test_model_requires_is_compared_to_actual_reviewed_daemon_version():
    value = {"capabilities": ["vision"], "requires": "0.40.5"}
    with client_for(lambda request: httpx.Response(200, json=value)) as client:
        setup.verify_model(client, setup.DEFAULT_MODEL, "0.40.5")
        with pytest.raises(setup.SetupError, match="newer runtime"):
            setup.verify_model(client, setup.DEFAULT_MODEL, "0.40.3")
