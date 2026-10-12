import json
import os
import stat

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from grader.app import app, MODEL_LOCK
from grader import explanations, workspace_api
from grader.workspace import Workspace, WorkspaceStore, default_workspace, model_catalog, parse_workspace


@pytest.fixture
def store(tmp_path, monkeypatch):
    value = WorkspaceStore(tmp_path)
    monkeypatch.setattr(workspace_api, "workspace_store", value)
    return value


client = TestClient(app)


def test_default_load_is_read_only_and_four_models(store):
    response = client.get("/api/workspace")
    assert response.status_code == 200
    assert response.json()["saved"] is False
    assert len(response.json()["workspace"]["models"]) == 4
    assert not store.directory.exists()


def test_save_reload_private_settings_only(store):
    data = default_workspace().model_dump()
    data["models"] = data["models"][:2]
    data["models"][1]["settings"]["display_name"] = "My local Clef"
    result = client.put("/api/workspace", json=data)
    assert result.status_code == 200
    assert result.json() == {"workspace": data, "saved": True}
    assert client.get("/api/workspace").json() == result.json()
    path = store.directory / store.filename
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700
    assert json.loads(path.read_text()) == data
    assert len(list(store.directory.iterdir())) == 1


@pytest.mark.parametrize("field", ["text", "rubric", "assignment_details", "results", "submission_name", "grades", "history"])
def test_student_fields_rejected_without_write_or_echo(store, field):
    data = default_workspace().model_dump()
    data[field] = "SYNTHETIC_PRIVATE_SENTINEL"
    for route, method in [("/api/workspace", client.put), ("/api/workspace/validate", client.post)]:
        result = method(route, json=data)
        assert result.status_code == 422
        assert "SYNTHETIC_PRIVATE_SENTINEL" not in result.text
        assert field not in result.text or field == "models"
    assert not store.directory.exists()


def test_nested_unknown_fields_rejected(store):
    data = default_workspace().model_dump()
    data["models"][0]["settings"]["submission"] = "synthetic"
    assert client.put("/api/workspace", json=data).status_code == 422
    assert not store.directory.exists()


def test_import_validates_without_persisting(store):
    data = default_workspace().model_dump()
    data["models"] = []
    assert client.post("/api/workspace/validate", json=data).json() == {"workspace": data}
    assert not store.directory.exists()


@pytest.mark.parametrize("bad", [b'{"schema_version":1,"schema_version":1,"models":[]}', b'[]', b'null', b'{', b'"text"'])
def test_bad_json_is_fixed_error(store, bad):
    assert client.put("/api/workspace", content=bad).status_code == 422
    assert not store.directory.exists()


def test_workspace_size_bounded(store):
    result = client.put("/api/workspace", content=b" " * 65537)
    assert result.status_code == 413
    assert not store.directory.exists()


def test_duplicate_enabled_models_and_ids_rejected():
    data = default_workspace().model_dump()
    data["models"][1]["settings"] = data["models"][0]["settings"].copy()
    data["models"][1]["settings"]["display_name"] = "A new label"
    with pytest.raises(ValidationError):
        Workspace.model_validate(data)
    data["models"][1]["enabled"] = False
    Workspace.model_validate(data)
    data["models"][1]["id"] = data["models"][0]["id"]
    with pytest.raises(ValidationError):
        Workspace.model_validate(data)


@pytest.mark.parametrize("value", [True, 2, "1"])
def test_version_is_exact_integer(value):
    with pytest.raises(ValidationError):
        Workspace(schema_version=value, models=[])


@pytest.mark.parametrize("url", ["https://example.com/v1/systemone", "http://192.168.1.2/v1/systemone", "http://127.0.0.1:1235/v1/systemone?private=sentinel"])
def test_remote_and_unsafe_endpoints_rejected(store, url):
    data = default_workspace().model_dump()
    data["models"][1]["settings"]["endpoint"] = url
    result = client.put("/api/workspace", json=data)
    assert result.status_code == 422
    assert url not in result.text
    assert not store.directory.exists()


def test_corrupt_workspace_is_not_overwritten(store):
    store.directory.mkdir()
    path = store.directory / store.filename
    path.write_text("PRIVATE_BROKEN_JSON")
    result = client.get("/api/workspace")
    assert result.status_code == 409
    assert "PRIVATE_BROKEN_JSON" not in result.text
    assert path.read_text() == "PRIVATE_BROKEN_JSON"


@pytest.mark.parametrize("link_directory", [False, True])
def test_symlink_store_never_read_or_overwritten(store, tmp_path, link_directory):
    target = tmp_path / "other"
    if link_directory:
        target.mkdir()
        store.directory.symlink_to(target, target_is_directory=True)
        protected = target / store.filename
    else:
        store.directory.mkdir()
        protected = target
        (store.directory / store.filename).symlink_to(target)
    protected.write_text("PRIVATE_SENTINEL")
    assert client.get("/api/workspace").status_code == 409
    assert client.put("/api/workspace", json=default_workspace().model_dump()).status_code == 409
    assert protected.read_text() == "PRIVATE_SENTINEL"


def test_atomic_failure_preserves_previous_workspace(store, monkeypatch):
    original = default_workspace()
    store.save(original)
    def fail_replace(*args, **kwargs):
        raise OSError("synthetic failure")
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.save(Workspace(models=[]))
    assert store.load()[0] == original
    assert [p.name for p in store.directory.iterdir()] == [store.filename]


class FakeManager:
    def __init__(self):
        self.calls = []
    def statuses(self):
        return {"clef": {"id": "clef", "state": "external", "owned": False, "ready": True,
                         "ports": [1235], "message": "External server", "can_start": False, "can_stop": False}}
    def start(self, service):
        self.calls.append((service, "start"))
        return {"id": service, "state": "starting", "owned": True, "ready": False, "ports": [],
                "message": "Starting", "can_start": False, "can_stop": True}
    def stop(self, service):
        self.calls.append((service, "stop"))
        return {**self.start(service), "state": "stopping"}


def test_catalog_and_status_are_metadata_only(store, monkeypatch):
    manager = FakeManager()
    monkeypatch.setattr(workspace_api, "runtime_manager", manager)
    assert client.get("/api/model-catalog").json() == model_catalog()
    response = client.get("/api/runtimes").json()["services"][0]
    assert response["status"] == "external" and response["name"] == "Clef-Flash"
    assert manager.calls == []
    assert not store.directory.exists()


def test_fixed_runtime_actions_and_busy_guards(monkeypatch):
    manager = FakeManager()
    monkeypatch.setattr(workspace_api, "runtime_manager", manager)
    assert client.post("/api/runtimes/arbitrary/start").status_code == 404
    assert client.post("/api/runtimes/clef/start").json()["service"]["status"] == "starting"
    with MODEL_LOCK:
        assert client.post("/api/runtimes/clef/stop").status_code == 409
    with explanations._LOCK:
        assert client.post("/api/runtimes/ollama/start").status_code == 409
    assert manager.calls == [("clef", "start")]
    assert not MODEL_LOCK.locked()


def test_cross_origin_cannot_save_or_start(store, monkeypatch):
    manager = FakeManager()
    monkeypatch.setattr(workspace_api, "runtime_manager", manager)
    headers = {"origin": "https://external.example"}
    assert client.put("/api/workspace", json=default_workspace().model_dump(), headers=headers).status_code == 403
    assert client.post("/api/runtimes/clef/start", headers=headers).status_code == 403
    assert manager.calls == []
    assert not store.directory.exists()


def test_single_setup_page_available_offline():
    response = client.get("/setup")
    assert response.status_code == 200
    for anchor in ["laya", "clef", "startlux", "d3", "custom-models", "ollama"]:
        assert f'id="{anchor}"' in response.text
    assert "<script" not in response.text
    assert response.headers["cache-control"] == "no-store"
