"""Fixed local launchers, in-memory ownership, and content-free health checks.

Nothing here accepts a command, path, URL, environment, or process ID from the UI.
Existing services are never adopted. Application shutdown must call shutdown().
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from types import MappingProxyType

import httpx

ROOT = Path(__file__).resolve().parents[1]
IS_WINDOWS = os.name == "nt"
PYTHON = ROOT / (".venv/Scripts/python.exe" if IS_WINDOWS else ".venv/bin/python")
D3_DEVICE = "mps" if sys.platform == "darwin" else "cpu"
STARTUP_SECONDS = 240.0
POLL_SECONDS = 0.25
STOP_SECONDS = 45.0
KILL_SECONDS = 5.0
HEALTH_TIMEOUT = 0.2
MAX_METADATA = 65536
D3_REVISION = "a475367de0d8133a3c1a2a0dc12303d89cfa2ab4"


@dataclass(frozen=True)
class RuntimeSpec:
    script: str
    ports: tuple[int, ...]
    arguments: tuple[str, ...] = ()


SPECS = MappingProxyType({
    "clef": RuntimeSpec("run_clef.py", (1235,)),
    "startlux": RuntimeSpec("run_startlux.py", (1236, 1237)),
    "d3": RuntimeSpec("run_d3.py", (1238,), ("--device", D3_DEVICE)),
    "ollama": RuntimeSpec("run_ollama.py", (11435,)),
})
MESSAGES = {
    "checking": "Checking the fixed local server ports.",
    "stopped": "The bundled server is stopped.",
    "starting": "Starting the bundled local server and verifying its identity.",
    "running": "The bundled local server is ready.",
    "unready": "The owned server is running; readiness could not be confirmed.",
    "stopping": "Stopping the owned launcher and waiting for its child cleanup.",
    "external": "A service already occupies a bundled port. This page cannot stop or adopt it.",
    "unknown": "The fixed local ports could not be verified. No server was started.",
    "failed": "The bundled server could not start. Check installed files, dependencies, ports, and local GPU access.",
    "exited": "The owned launcher stopped unexpectedly. No runtime output was retained.",
    "timeout": "The bundled server did not become ready within the startup deadline.",
    "cleanup": "Owned launcher cleanup could not be confirmed. Check the local server outside this page before restarting.",
    "closed": "The application runtime manager has shut down.",
}


def clean_environment():
    environment = {key: os.environ[key] for key in ("HOME", "TMPDIR", "LANG", "LC_ALL",
        "SYSTEMROOT", "SystemRoot", "WINDIR", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP") if key in os.environ}
    if IS_WINDOWS:
        paths = [str(ROOT / ".venv/Scripts")]
        system_root = environment.get("SystemRoot") or environment.get("SYSTEMROOT") or environment.get("WINDIR")
        if system_root:
            paths.extend([str(Path(system_root) / "System32"), system_root])
        local = environment.get("LOCALAPPDATA")
        if local:
            paths.append(str(Path(local) / "Programs/Ollama"))
        search_path = ";".join(paths)
    else:
        search_path = f"{ROOT / '.venv/bin'}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    environment.update({
        "PATH": search_path,
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "OTEL_SDK_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "OLLAMA_NO_CLOUD": "1", "OLLAMA_DEBUG": "false",
        "OLLAMA_DEBUG_LOG_REQUESTS": "false", "OLLAMA_NOHISTORY": "1",
    })
    return environment


def _port_state(port):
    """Refusal alone proves absence; timeouts and permission errors are unknown."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=HEALTH_TIMEOUT):
            return "occupied"
    except OSError as exc:
        return "absent" if exc.errno == errno.ECONNREFUSED else "unknown"


def _metadata(client, port, path):
    started = time.monotonic()
    with client.stream("GET", f"http://127.0.0.1:{port}{path}") as response:
        if response.status_code != 200 or response.headers.get("content-encoding", "identity").lower() != "identity":
            raise ValueError
        raw = bytearray()
        for chunk in response.iter_bytes():
            raw.extend(chunk)
            if len(raw) > MAX_METADATA or time.monotonic() - started > HEALTH_TIMEOUT:
                raise ValueError
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError
        return value


def _sole_model(value, name):
    models = value.get("data")
    return isinstance(models, list) and len(models) == 1 and isinstance(models[0], dict) and models[0].get("id") == name


def _probe(runtime_id):
    """Return only absent, ready, occupied, or unknown; never native metadata."""
    spec = SPECS[runtime_id]
    states = [_port_state(port) for port in spec.ports]
    if all(state == "absent" for state in states):
        return "absent"
    if "unknown" in states:
        return "unknown"
    if "absent" in states:
        return "occupied"
    try:
        with httpx.Client(trust_env=False, follow_redirects=False,
                          timeout=httpx.Timeout(HEALTH_TIMEOUT), limits=httpx.Limits(max_connections=1)) as client:
            if runtime_id == "clef":
                valid = (_metadata(client, 1235, "/health").get("status") == "ok"
                         and _sole_model(_metadata(client, 1235, "/v1/models"), "clef-flash"))
            elif runtime_id == "startlux":
                health = _metadata(client, 1236, "/health")
                valid = (health.get("status") == "ok" and health.get("model") == "startlux-decision-4b-q8"
                         and _metadata(client, 1237, "/health").get("status") == "ok"
                         and _sole_model(_metadata(client, 1237, "/v1/models"), "startlux-decision-4b-q8"))
            elif runtime_id == "d3":
                health = _metadata(client, 1238, "/health")
                valid = (health.get("ok") is True and health.get("model") == "d3-mini"
                         and health.get("revision") == D3_REVISION and health.get("text_only") is True
                         and health.get("device") == D3_DEVICE and health.get("max_context") == 8192)
            else:
                version = _metadata(client, 11435, "/api/version")
                cloud = _metadata(client, 11435, "/api/status").get("cloud", {})
                valid = (version.get("version") in {"0.40.3", "0.40.5"} and isinstance(cloud, dict)
                         and cloud.get("disabled") is True and cloud.get("source") in {"env", "both"})
        return "ready" if valid else "occupied"
    except Exception:
        return "occupied"  # A known open port never becomes permission to start.


@dataclass
class _Entry:
    state: str = "stopped"
    message: str = MESSAGES["checking"]
    probe: str = "unknown"
    ready: bool = False
    process: subprocess.Popen | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None
    cleanup_blocked: bool = False


class RuntimeManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._entries = {key: _Entry() for key in SPECS}
        self._refreshing = False
        self._closed = False

    @staticmethod
    def _validate(runtime_id):
        if not isinstance(runtime_id, str) or runtime_id not in SPECS:
            raise ValueError("Unknown bundled runtime.")

    def _view(self, runtime_id):
        entry = self._entries[runtime_id]
        active = entry.worker is not None and entry.worker.is_alive()
        return {"id": runtime_id, "state": entry.state, "owned": entry.process is not None,
                "ready": entry.ready, "ports": list(SPECS[runtime_id].ports), "message": entry.message,
                "can_start": not self._closed and not active and entry.probe == "absent" and not entry.cleanup_blocked,
                "can_stop": not self._closed and ((active and entry.state in {"starting", "running"})
                                                 or (not active and entry.process is not None))}

    def statuses(self):
        """Immediate cached snapshots; bounded metadata refresh runs separately."""
        with self._lock:
            if not self._closed and not self._refreshing:
                self._refreshing = True
                threading.Thread(target=self._refresh, name="bundled-runtime-status", daemon=True).start()
            return {runtime_id: self._view(runtime_id) for runtime_id in SPECS}

    def is_transitioning(self, runtime_id):
        """Read lifecycle state without performing any network or process work."""
        self._validate(runtime_id)
        with self._lock:
            return self._entries[runtime_id].state in {"starting", "stopping"}

    def _refresh(self):
        try:
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="local-health") as pool:
                results = dict(zip(SPECS, pool.map(_probe, SPECS)))
            with self._lock:
                for runtime_id, probe in results.items():
                    entry = self._entries[runtime_id]
                    if entry.state in {"starting", "stopping"} or entry.cleanup_blocked:
                        continue
                    entry.probe, entry.ready = probe, probe == "ready"
                    if entry.process is not None:
                        entry.message = MESSAGES["running" if entry.ready else "unready"]
                    elif probe in {"ready", "occupied"}:
                        entry.state, entry.message = "external", MESSAGES["external"]
                    elif probe == "unknown":
                        entry.state, entry.message = "error", MESSAGES["unknown"]
                    elif entry.state != "error":
                        entry.state, entry.message = "stopped", MESSAGES["stopped"]
        except Exception:
            pass  # Keep the last safe snapshot; never print exception details.
        finally:
            with self._lock:
                self._refreshing = False

    def start(self, runtime_id):
        self._validate(runtime_id)
        with self._lock:
            if self._closed:
                raise RuntimeError(MESSAGES["closed"])
            entry = self._entries[runtime_id]
            if entry.cleanup_blocked or entry.process is not None or (entry.worker is not None and entry.worker.is_alive()):
                return self._view(runtime_id)
            entry = _Entry(state="starting", message=MESSAGES["starting"])
            self._entries[runtime_id] = entry
            entry.worker = threading.Thread(target=self._run, args=(runtime_id, entry),
                                            name=f"bundled-runtime-{runtime_id}", daemon=True)
            entry.worker.start()
            return self._view(runtime_id)

    def stop(self, runtime_id):
        self._validate(runtime_id)
        with self._lock:
            entry = self._entries[runtime_id]
            if entry.worker is not None and entry.worker.is_alive():
                entry.cancel.set()
                entry.state, entry.ready, entry.message = "stopping", False, MESSAGES["stopping"]
            elif entry.process is not None:
                self._retry_cleanup(runtime_id, entry)
            # No owned worker means no signal, even when an external service is ready.
            return self._view(runtime_id)

    def _run(self, runtime_id, entry):
        failure = None
        try:
            probe = _probe(runtime_id)
            with self._lock:
                entry.probe = probe
                if entry.cancel.is_set() or self._closed:
                    return
                if probe != "absent":
                    entry.state = "external" if probe in {"ready", "occupied"} else "error"
                    entry.ready = probe == "ready"
                    entry.message = MESSAGES["external" if entry.state == "external" else "unknown"]
                    return
                spec = SPECS[runtime_id]
                script = ROOT / "scripts" / spec.script
                if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK) or not script.is_file():
                    raise ValueError
                process_options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WINDOWS
                                   else {"start_new_session": True})
                entry.process = subprocess.Popen([str(PYTHON), str(script), *spec.arguments],
                    cwd=str(ROOT), env=clean_environment(), stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False,
                    close_fds=True, **process_options)
            deadline = time.monotonic() + STARTUP_SECONDS
            while not entry.cancel.is_set():
                if entry.process.poll() is not None:
                    failure = "exited"
                    break
                probe = _probe(runtime_id)
                with self._lock:
                    entry.probe = probe
                    if not entry.cancel.is_set() and probe == "ready":
                        entry.state, entry.ready, entry.message = "running", True, MESSAGES["running"]
                if entry.ready:
                    while not entry.cancel.wait(POLL_SECONDS):
                        if entry.process.poll() is not None:
                            failure = "exited"
                            break
                    break
                if time.monotonic() >= deadline:
                    failure = "timeout"
                    break
                entry.cancel.wait(POLL_SECONDS)
        except Exception:
            failure = "failed"
        finally:
            process = entry.process
            if process is not None:
                self._cleanup(runtime_id, entry, failure)
            elif failure:
                with self._lock:
                    entry.state, entry.ready, entry.message = "error", False, MESSAGES[failure]
            elif entry.cancel.is_set():
                with self._lock:
                    entry.state, entry.ready, entry.message = "stopped", False, MESSAGES["stopped"]

    def _cleanup(self, runtime_id, entry, failure=None):
        process = entry.process
        try:
            cleaned = self._stop_owned(process)
            alive = process.poll() is None
            probe = _probe(runtime_id)
        except Exception:
            cleaned, alive, probe = False, True, "unknown"
        with self._lock:
            entry.process, entry.ready, entry.probe = process if alive else None, False, probe
            entry.cleanup_blocked = not cleaned or alive or probe != "absent"
            if entry.cleanup_blocked:
                entry.state, entry.message = "error", MESSAGES["cleanup"]
            elif failure:
                entry.state, entry.message = "error", MESSAGES[failure]
            else:
                entry.state, entry.message = "stopped", MESSAGES["stopped"]

    def _retry_cleanup(self, runtime_id, entry):
        entry.state, entry.ready, entry.message = "stopping", False, MESSAGES["stopping"]
        entry.worker = threading.Thread(target=self._cleanup, args=(runtime_id, entry),
                                        name=f"bundled-runtime-{runtime_id}-cleanup", daemon=True)
        entry.worker.start()

    @staticmethod
    def _stop_owned(process):
        if process.poll() is not None:
            # A reaped PID/PGID can be reused. Never signal it; any remaining
            # listener is reported as unconfirmed cleanup by the caller.
            return True
        if IS_WINDOWS:
            system_root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR")
            if not system_root:
                return False
            try:
                # A live Popen handle establishes ownership. /T also stops the
                # launcher's children; no executable-name or listening-port kill.
                result = subprocess.run([str(Path(system_root) / "System32/taskkill.exe"),
                    "/PID", str(process.pid), "/T", "/F"], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False,
                    timeout=STOP_SECONDS, check=False)
                process.wait(timeout=KILL_SECONDS)
                return result.returncode == 0
            except Exception:
                return False
        try:
            # This group was created by our Popen. SIGINT invokes launcher finally
            # cleanup, including Ollama's separately grouped owned daemon.
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        except Exception:
            return False
        try:
            process.wait(timeout=STOP_SECONDS)
            return True
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=KILL_SECONDS)
            except Exception:
                return False
            # Forced launcher termination cannot prove nested child cleanup.
            return False
        except Exception:
            return False

    def shutdown(self):
        with self._lock:
            self._closed = True
            workers = []
            for runtime_id, entry in self._entries.items():
                if entry.process is not None and (entry.worker is None or not entry.worker.is_alive()):
                    self._retry_cleanup(runtime_id, entry)
                if entry.worker is not None and entry.worker.is_alive():
                    entry.cancel.set()
                    entry.state, entry.ready, entry.message = "stopping", False, MESSAGES["stopping"]
                    workers.append(entry.worker)
        deadline = time.monotonic() + STOP_SECONDS + KILL_SECONDS + 3
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
