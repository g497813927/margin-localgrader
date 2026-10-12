"""Application lifespan starts no models and cleans up only its manager."""
import importlib

import pytest
from fastapi.testclient import TestClient


application = importlib.import_module("grader.app")


class QuietManager:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1

    def start(self, *_args):
        pytest.fail("Opening the application must not start a model server")

    def statuses(self):
        pytest.fail("Application startup must not automatically probe servers")


def test_lifespan_is_idle_until_explicit_actions_and_cleans_up(monkeypatch):
    manager = QuietManager()
    monkeypatch.setattr(application, "runtime_manager", manager)
    with TestClient(application.app) as client:
        assert client.get("/api/health").status_code == 200
        catalog = client.get("/api/model-catalog")
        assert catalog.status_code == 200
        assert catalog.json()["max_models"] == 8
        assert manager.shutdown_calls == 0
    assert manager.shutdown_calls == 1


def test_lifespan_cleanup_also_runs_after_client_exception(monkeypatch):
    manager = QuietManager()
    monkeypatch.setattr(application, "runtime_manager", manager)
    with pytest.raises(RuntimeError, match="synthetic client interruption"):
        with TestClient(application.app):
            raise RuntimeError("synthetic client interruption")
    assert manager.shutdown_calls == 1
