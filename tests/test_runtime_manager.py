"""No real model/service processes or network requests are used in these tests."""
import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from grader import runtime_manager as runtime

HTTP_CLIENT = httpx.Client


def eventually(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.001)
    pytest.fail("The mocked lifecycle did not finish")


def snapshot(manager, runtime_id):
    with manager._lock:
        return manager._view(runtime_id)


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.waits = []
        self.wait_timeouts = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.wait_timeouts:
            self.wait_timeouts -= 1
            raise subprocess.TimeoutExpired("PRIVATE_COMMAND", timeout)
        self.returncode = 0
        return 0


@pytest.fixture
def harness(monkeypatch):
    class Harness:
        def __init__(self):
            self.states = {key: "absent" for key in runtime.SPECS}
            self.processes, self.calls, self.signals = [], [], []

        def popen(self, command, **options):
            key = next(key for key, spec in runtime.SPECS.items() if command[1].endswith('/' + spec.script))
            self.calls.append((key, command, options))
            process = FakeProcess(100000 + len(self.processes))
            self.processes.append(process)
            self.states[key] = "ready"
            return process

        def killpg(self, pid, signum):
            self.signals.append((pid, signum))
            for (key, _, _), process in zip(self.calls, self.processes):
                if process.pid == pid:
                    self.states[key] = "absent"

    fake = Harness()
    # A clean source checkout need not contain a project .venv. Process creation
    # remains fully mocked; use the test interpreter only for the file preflight.
    monkeypatch.setattr(runtime, "PYTHON", Path(sys.executable))
    monkeypatch.setattr(runtime, "_probe", lambda key: fake.states[key])
    monkeypatch.setattr(runtime.subprocess, "Popen", fake.popen)
    monkeypatch.setattr(runtime.os, "killpg", fake.killpg)
    monkeypatch.setattr(runtime, "POLL_SECONDS", .002)
    manager = runtime.RuntimeManager()
    yield manager, fake
    manager.shutdown()


@pytest.mark.parametrize("runtime_id", ["clef", "startlux", "d3", "ollama"])
def test_fixed_launcher_lifecycle_is_owned_and_quiet(runtime_id, harness):
    manager, fake = harness
    response = manager.start(runtime_id)
    assert response["state"] == "starting"
    eventually(lambda: snapshot(manager, runtime_id)["state"] == "running")
    view = snapshot(manager, runtime_id)
    assert view["owned"] and view["ready"] and view["can_stop"] and not view["can_start"]
    key, command, options = fake.calls[0]
    spec = runtime.SPECS[runtime_id]
    assert key == runtime_id
    assert command == [str(runtime.PYTHON), str(runtime.ROOT / "scripts" / spec.script), *spec.arguments]
    assert options["cwd"] == str(runtime.ROOT)
    assert options["start_new_session"] and options["close_fds"] and options["shell"] is False
    assert all(options[fd] == subprocess.DEVNULL for fd in ("stdin", "stdout", "stderr"))
    assert manager.stop(runtime_id)["state"] == "stopping"
    eventually(lambda: snapshot(manager, runtime_id)["state"] == "stopped")
    assert fake.signals == [(fake.processes[0].pid, signal.SIGINT)]
    assert fake.processes[0].waits == [runtime.STOP_SECONDS]
    assert not snapshot(manager, runtime_id)["owned"]


def test_environment_uses_allowlist_preserves_home_and_drops_secrets_tracing_and_proxies(monkeypatch):
    inherited = {"HOME": "/Users/test", "PATH": "/PRIVATE/bin", "LANG": "en_US.UTF-8",
                 "HTTPS_PROXY": "PRIVATE_PROXY", "http_proxy": "PRIVATE_PROXY", "HF_TOKEN": "PRIVATE_TOKEN",
                 "PYTHONPATH": "PRIVATE_IMPORT", "PYTHONHOME": "PRIVATE_IMPORT",
                 "DYLD_INSERT_LIBRARIES": "PRIVATE_TRACE", "LLAMA_RPC": "PRIVATE_REMOTE",
                 "OLLAMA_HOST": "PRIVATE_REMOTE", "OLLAMA_DEBUG_LOG_REQUESTS": "true",
                 "OTEL_EXPORTER_OTLP_ENDPOINT": "PRIVATE_REMOTE"}
    monkeypatch.setattr(runtime.os, "environ", inherited)
    environment = runtime.clean_environment()
    assert environment["HOME"] == inherited["HOME"]
    assert environment["PATH"].startswith(str(runtime.ROOT / ".venv/bin"))
    assert "PRIVATE" not in json.dumps(environment)
    assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["OLLAMA_NO_CLOUD"] == "1"
    assert environment["OLLAMA_DEBUG_LOG_REQUESTS"] == "false"
    assert environment["OTEL_SDK_DISABLED"] == "true"


@pytest.mark.parametrize("value", ["", "laya", "clef; echo PRIVATE", "../scripts/run.py", "http://127.0.0.1:1235", None, [], {}])
def test_unknown_ids_cannot_select_commands_or_processes(value, harness):
    manager, fake = harness
    for method in (manager.start, manager.stop):
        with pytest.raises(ValueError, match="^Unknown bundled runtime.$"):
            method(value)
    assert fake.calls == fake.signals == []


@pytest.mark.parametrize("probe", ["ready", "occupied", "unknown"])
def test_existing_or_unverifiable_services_are_never_started_adopted_or_stopped(probe, harness):
    manager, fake = harness
    fake.states["ollama"] = probe
    manager.start("ollama")
    eventually(lambda: snapshot(manager, "ollama")["state"] != "starting")
    result = manager.stop("ollama")
    assert result["state"] == ("error" if probe == "unknown" else "external")
    assert not result["owned"] and not result["can_start"] and not result["can_stop"]
    assert fake.calls == fake.signals == []


def test_simultaneous_start_is_idempotent_and_stop_during_preflight_prevents_spawn(harness, monkeypatch):
    manager, fake = harness
    entered, release = threading.Event(), threading.Event()
    def probe(_key):
        entered.set()
        release.wait(1)
        return "absent"
    monkeypatch.setattr(runtime, "_probe", probe)
    first = manager.start("clef")
    assert entered.wait(1)
    with runtime.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: manager.start("clef"), range(8)))
    assert all(result["state"] == first["state"] == "starting" for result in results)
    assert manager.stop("clef")["state"] == "stopping"
    release.set()
    eventually(lambda: snapshot(manager, "clef")["state"] == "stopped")
    assert fake.calls == fake.signals == []


def test_launch_exception_is_sanitized_and_retry_is_possible(harness, monkeypatch, capsys):
    manager, fake = harness
    def fail(*args, **kwargs):
        raise OSError("PRIVATE_SUBMISSION_OR_PATH")
    monkeypatch.setattr(runtime.subprocess, "Popen", fail)
    manager.start("d3")
    eventually(lambda: snapshot(manager, "d3")["state"] == "error")
    result = snapshot(manager, "d3")
    assert result["can_start"] and not result["owned"]
    assert "PRIVATE" not in json.dumps(result) + str(capsys.readouterr())
    monkeypatch.setattr(runtime.subprocess, "Popen", fake.popen)
    eventually(lambda: not manager._entries["d3"].worker.is_alive())
    manager.start("d3")
    eventually(lambda: snapshot(manager, "d3")["state"] == "running")


@pytest.mark.parametrize("missing", ["interpreter", "launcher"])
def test_missing_installation_files_reject_before_process_creation(harness, monkeypatch, tmp_path, missing):
    manager, fake = harness
    if missing == "interpreter":
        monkeypatch.setattr(runtime, "PYTHON", tmp_path / "missing-python")
    else:
        monkeypatch.setattr(runtime, "ROOT", tmp_path)
    manager.start("clef")
    eventually(lambda: snapshot(manager, "clef")["state"] == "error")
    assert fake.calls == fake.processes == fake.signals == []
    assert snapshot(manager, "clef")["message"] == runtime.MESSAGES["failed"]


def test_startup_timeout_cleans_only_owned_launcher_and_keeps_safe_error(harness, monkeypatch):
    manager, fake = harness
    monkeypatch.setattr(runtime, "STARTUP_SECONDS", .005)
    monkeypatch.setattr(runtime, "_probe", lambda _key: "absent")
    manager.start("startlux")
    eventually(lambda: snapshot(manager, "startlux")["state"] == "error")
    assert snapshot(manager, "startlux")["message"] == runtime.MESSAGES["timeout"]
    assert fake.signals == [(fake.processes[0].pid, signal.SIGINT)]


def test_force_kill_never_claims_nested_cleanup_or_signals_an_external_pid(harness):
    manager, fake = harness
    manager.start("ollama")
    eventually(lambda: snapshot(manager, "ollama")["state"] == "running")
    process = fake.processes[0]
    process.wait_timeouts = 1
    manager.stop("ollama")
    eventually(lambda: snapshot(manager, "ollama")["state"] == "error")
    assert fake.signals == [(process.pid, signal.SIGINT), (process.pid, signal.SIGKILL)]
    assert snapshot(manager, "ollama")["message"] == runtime.MESSAGES["cleanup"]
    assert not snapshot(manager, "ollama")["can_start"]
    manager.start("ollama")
    assert len(fake.calls) == 1


def test_exited_launcher_pid_is_never_signaled_and_surviving_listener_is_not_adopted(harness):
    manager, fake = harness
    manager.start("ollama")
    eventually(lambda: snapshot(manager, "ollama")["state"] == "running")
    fake.processes[0].returncode = 1
    eventually(lambda: snapshot(manager, "ollama")["state"] == "error")
    assert fake.signals == []
    manager._refresh()
    result = snapshot(manager, "ollama")
    assert result["state"] == "error" and not result["owned"] and not result["can_stop"]
    assert not result["can_start"]


def test_failed_signal_keeps_live_handle_for_explicit_owned_stop_retry(harness, monkeypatch):
    manager, fake = harness
    manager.start("ollama")
    eventually(lambda: snapshot(manager, "ollama")["state"] == "running")
    process = fake.processes[0]
    def denied(*_):
        raise PermissionError("PRIVATE_PROCESS_DETAIL")
    monkeypatch.setattr(runtime.os, "killpg", denied)
    manager.stop("ollama")
    eventually(lambda: snapshot(manager, "ollama")["state"] == "error")
    eventually(lambda: not manager._entries["ollama"].worker.is_alive())
    result = snapshot(manager, "ollama")
    assert result["owned"] and result["can_stop"] and not result["can_start"]
    manager.start("ollama")
    assert len(fake.calls) == 1
    monkeypatch.setattr(runtime.os, "killpg", fake.killpg)
    manager.stop("ollama")
    eventually(lambda: snapshot(manager, "ollama")["state"] == "stopped")
    assert fake.signals == [(process.pid, signal.SIGINT)]


def test_cached_status_returns_immediately_while_probes_are_blocked(harness, monkeypatch):
    manager, _ = harness
    release = threading.Event()
    monkeypatch.setattr(runtime, "_probe", lambda _key: (release.wait(1), "absent")[1])
    started = time.monotonic()
    result = manager.statuses()
    elapsed = time.monotonic() - started
    release.set()
    assert elapsed < .1 and set(result) == set(runtime.SPECS)
    eventually(lambda: not manager._refreshing)


@pytest.mark.parametrize("state,expected", [("starting", True), ("stopping", True),
    ("running", False), ("stopped", False), ("error", False), ("external", False)])
def test_transition_check_reads_only_cached_known_runtime_state(state, expected, monkeypatch):
    manager = runtime.RuntimeManager()
    monkeypatch.setattr(runtime, "_probe", lambda *_: pytest.fail("Transition checks must not probe"))
    with manager._lock:
        manager._entries["clef"].state = state
    assert manager.is_transitioning("clef") is expected
    assert manager.is_transitioning("ollama") is False
    with pytest.raises(ValueError, match="^Unknown bundled runtime.$"):
        manager.is_transitioning("PRIVATE_UNKNOWN")


def test_shutdown_waits_for_all_owned_launchers_and_never_touches_external_service(harness):
    manager, fake = harness
    fake.states["clef"] = "ready"
    manager.start("clef")
    for key in ("startlux", "d3", "ollama"):
        manager.start(key)
    eventually(lambda: all(snapshot(manager, key)["state"] == "running" for key in ("startlux", "d3", "ollama")))
    manager.shutdown()
    assert len(fake.signals) == 3
    assert all(not entry.worker or not entry.worker.is_alive() for entry in manager._entries.values())
    with pytest.raises(RuntimeError, match="shut down"):
        manager.start("d3")
    assert fake.states["clef"] == "ready"


HEALTH = {
    (1235, "/health"): {"status": "ok"},
    (1235, "/v1/models"): {"data": [{"id": "clef-flash"}]},
    (1236, "/health"): {"status": "ok", "model": "startlux-decision-4b-q8"},
    (1237, "/health"): {"status": "ok"},
    (1237, "/v1/models"): {"data": [{"id": "startlux-decision-4b-q8"}]},
    (1238, "/health"): {"ok": True, "model": "d3-mini", "revision": runtime.D3_REVISION,
                         "device": "mps", "text_only": True, "max_context": 8192},
    (11435, "/api/version"): {"version": "0.40.3"},
    (11435, "/api/status"): {"cloud": {"disabled": True, "source": "env"}},
}


def mock_health(monkeypatch, override=None):
    requests = []
    monkeypatch.setattr(runtime, "_port_state", lambda _port: "occupied")
    def handle(request):
        assert request.url.host == "127.0.0.1" and request.method == "GET" and not request.content
        key = (request.url.port, request.url.path)
        assert key in HEALTH
        requests.append(key)
        if override:
            return override(request)
        return httpx.Response(200, json=HEALTH[key])
    def factory(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        assert kwargs["timeout"].read <= .2
        return HTTP_CLIENT(transport=httpx.MockTransport(handle), **kwargs)
    monkeypatch.setattr(runtime.httpx, "Client", factory)
    return requests


@pytest.mark.parametrize("key", list(runtime.SPECS))
def test_health_verifies_fixed_service_identity_without_inference(key, monkeypatch):
    requests = mock_health(monkeypatch)
    assert runtime._probe(key) == "ready"
    assert requests and all(path not in {"/api/chat", "/v1/systemone"} for _, path in requests)


@pytest.mark.parametrize("response", [
    httpx.Response(302, headers={"Location": "https://example.com/PRIVATE"}),
    httpx.Response(200, json={"status": "ok", "data": [{"id": "PRIVATE_OTHER_MODEL"}]}),
    httpx.Response(200, content=b"x" * (runtime.MAX_METADATA + 1)),
    httpx.Response(200, content=b"PRIVATE_INVALID_JSON"),
    httpx.Response(500, content=b"PRIVATE_ERROR"),
])
def test_bad_or_redirected_health_is_occupied_never_available(response, monkeypatch, capsys):
    mock_health(monkeypatch, lambda _request: response)
    assert runtime._probe("clef") == "occupied"
    assert "PRIVATE" not in str(capsys.readouterr())


@pytest.mark.parametrize("key,address,changed", [
    ("d3", (1238, "/health"), {"revision": "PRIVATE_UNREVIEWED_REVISION"}),
    ("d3", (1238, "/health"), {"text_only": False}),
    ("startlux", (1237, "/v1/models"), {"data": [{"id": "clef-flash"}]}),
    ("ollama", (11435, "/api/version"), {"version": "0.40.2"}),
    ("ollama", (11435, "/api/status"), {"cloud": {"disabled": False, "source": "env"}}),
])
def test_metadata_must_match_reviewed_model_and_privacy_identity(key, address, changed, monkeypatch):
    def response(request):
        actual = (request.url.port, request.url.path)
        value = dict(HEALTH[actual])
        if actual == address:
            value.update(changed)
        return httpx.Response(200, json=value)
    mock_health(monkeypatch, response)
    assert runtime._probe(key) == "occupied"


def test_partial_startlux_ports_do_not_pass_preflight(monkeypatch):
    monkeypatch.setattr(runtime, "_port_state", lambda port: "occupied" if port == 1237 else "absent")
    monkeypatch.setattr(runtime.httpx, "Client", lambda **_: pytest.fail("Partial external service must not be queried further"))
    assert runtime._probe("startlux") == "occupied"


@pytest.mark.parametrize("number,expected", [(errno.ECONNREFUSED, "absent"), (errno.EACCES, "unknown"), (errno.ETIMEDOUT, "unknown")])
def test_only_confirmed_refusal_means_free_port(number, expected, monkeypatch):
    def fail(address, timeout):
        assert address == ("127.0.0.1", 1235) and timeout == .2
        raise OSError(number, "PRIVATE_OS_DETAIL")
    monkeypatch.setattr(runtime.socket, "create_connection", fail)
    assert runtime._port_state(1235) == expected
