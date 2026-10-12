"""Variable-size comparisons with synthetic inputs and no model execution."""
from copy import deepcopy
import importlib
import threading

from fastapi.testclient import TestClient
import pytest

from grader.schemas import Settings

app_module = importlib.import_module("grader.app")
client = TestClient(app_module.app)


def payload(count):
    return {
        "rubric": {"title": "Synthetic rubric", "criteria": [{
            "id": "reason", "name": "Reason", "description": "Explain a relevant reason.",
            "levels": [{"id": "full", "label": "Full", "points": 5},
                       {"id": "some", "label": "Some", "points": 2},
                       {"id": "none", "label": "None", "points": 0}],
        }]},
        "text": "Synthetic evidence for a reason.",
        "models": [Settings(provider="systemone", model=f"synthetic-{i}",
                            display_name=f"Saved model {i + 1}",
                            endpoint=f"http://127.0.0.1:{12000+i}/v1/systemone").model_dump()
                   for i in range(count)],
    }


def prediction(request, choice="some"):
    return {"answers": {criterion.id: {"probabilities": {
        level.id: 1 / len(criterion.levels) if choice == "withheld" else float(level.id == choice)
        for level in criterion.levels}} for criterion in request.rubric.criteria}}


@pytest.mark.parametrize("endpoint,runtime_id", [
    ("http://localhost:1235/v1/systemone", "clef"),
    ("http://127.0.0.1:1236/v1/systemone", "startlux"),
    ("http://127.0.0.1:1237/v1/systemone", "startlux"),
    ("http://127.0.0.1:1238/v1/systemone", "d3"),
])
@pytest.mark.parametrize("route", ["grade", "probe", "compare"])
def test_bundled_transition_rejects_new_work_before_inference(monkeypatch, endpoint, runtime_id, route):
    data = payload(1)
    data["models"][0]["endpoint"] = endpoint
    seen = []
    def transitioning(key):
        assert app_module.MODEL_LOCK.locked()
        seen.append(key)
        return True
    monkeypatch.setattr(app_module.runtime_manager, "is_transitioning", transitioning)
    monkeypatch.setattr(app_module, "predict_rubric", lambda *_: pytest.fail("No inference during transition"))
    monkeypatch.setattr(app_module.providers, "probe", lambda *_: pytest.fail("No probe during transition"))
    if route == "grade":
        data["settings"] = data.pop("models")[0]
    elif route == "probe":
        data = {"settings": data["models"][0]}
    response = client.post(f"/api/{route}", json=data)
    assert response.status_code == 409
    assert "starting or stopping" in response.json()["detail"]
    assert seen == [runtime_id]
    assert not app_module.MODEL_LOCK.locked()


@pytest.mark.parametrize("settings", [
    Settings(),
    Settings(provider="systemone", model="custom", endpoint="http://127.0.0.1:1250/v1/systemone"),
    Settings(provider="systemone", model="ipv6-custom", endpoint="http://[::1]:1235/v1/systemone"),
])
def test_unrelated_servers_and_native_laya_are_not_blocked_by_transitions(monkeypatch, settings):
    data = payload(1)
    data["models"] = [settings.model_dump()]
    monkeypatch.setattr(app_module.runtime_manager, "is_transitioning",
                        lambda *_: pytest.fail("Unrelated server state must not block inference"))
    monkeypatch.setattr(app_module, "predict_rubric", prediction)
    response = client.post("/api/compare", json=data)
    assert response.status_code == 200
    assert response.json()["results"][0]["result"]["total"] == 2


def test_stable_selected_runtime_ignores_other_transitions(monkeypatch):
    data = payload(1)
    data["models"][0]["endpoint"] = "http://127.0.0.1:1235/v1/systemone"
    seen = []
    def transitioning(key):
        seen.append(key)
        return key != "clef"
    monkeypatch.setattr(app_module.runtime_manager, "is_transitioning", transitioning)
    monkeypatch.setattr(app_module, "predict_rubric", prediction)
    response = client.post("/api/compare", json=data)
    assert response.status_code == 200
    assert seen == ["clef"]


@pytest.mark.parametrize("count", [1, 2, 5, 8])
def test_requested_models_return_exact_order_and_original_points(monkeypatch, count):
    data = payload(count)
    before = deepcopy(data)
    seen = []
    def predict(request):
        seen.append(request.settings.model)
        assert request.text == data["text"]
        return prediction(request, "full" if int(request.settings.model.rsplit("-", 1)[1]) % 2 else "some")
    monkeypatch.setattr(app_module, "predict_rubric", predict)
    response = client.post("/api/compare", json=data)
    assert response.status_code == 200
    result = response.json()
    expected_points = [5 if i % 2 else 2 for i in range(count)]
    assert [entry["label"] for entry in result["results"]] == [
        f"Model {chr(65+i)} · Saved model {i+1}" for i in range(count)]
    assert [entry["result"]["model"] for entry in result["results"]] == [f"synthetic-{i}" for i in range(count)]
    assert [entry["result"]["total"] for entry in result["results"]] == expected_points
    row = result["comparison"][0]
    assert row["model_points"] == expected_points and row["available_models"] == count
    assert row["model_level_ids"] == ["full" if i % 2 else "some" for i in range(count)]
    assert row["model_a_points"] == 2
    assert row["model_b_points"] == (5 if count > 1 else None)
    assert row["disagrees"] is (True if count > 1 else None)
    assert bool(row["comparison_receipt"]) is (count > 1)
    assert len(seen) == count and len(set(seen)) == count
    assert "total" not in result and data == before


def test_eight_models_use_at_most_four_workers_and_keep_result_order(monkeypatch):
    data = payload(8)
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = peak = 0
    worker_limits = []
    executor = app_module.ThreadPoolExecutor
    def bounded_executor(**kwargs):
        worker_limits.append(kwargs["max_workers"])
        return executor(**kwargs)
    monkeypatch.setattr(app_module, "ThreadPoolExecutor", bounded_executor)
    def predict(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if int(request.settings.model.rsplit("-", 1)[1]) < 4:
                barrier.wait(timeout=3)
            return prediction(request)
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(app_module, "predict_rubric", predict)
    response = client.post("/api/compare", json=data)
    assert response.status_code == 200
    assert worker_limits == [4] and peak == 4 and active == 0
    assert [entry["result"]["model"] for entry in response.json()["results"]] == [f"synthetic-{i}" for i in range(8)]


def test_partial_eight_model_results_keep_failed_and_withheld_scores_null(monkeypatch):
    data = payload(8)
    for settings in data["models"]:
        settings["scoring_mode"] = "strict_review"
    def predict(request):
        index = int(request.settings.model.rsplit("-", 1)[1])
        if index == 2:
            raise RuntimeError("The local runtime is unavailable.")
        return prediction(request, "withheld" if index == 3 else "some")
    monkeypatch.setattr(app_module, "predict_rubric", predict)
    result = client.post("/api/compare", json=data).json()
    row = result["comparison"][0]
    assert row["model_points"] == [2, 2, None, None, 2, 2, 2, 2]
    assert row["model_level_ids"] == ["some", "some", None, None, "some", "some", "some", "some"]
    assert row["available_models"] == 6 and row["disagrees"] is False
    assert row["comparison_receipt"] is None
    assert result["results"][2]["ok"] is False
    assert result["results"][3]["result"]["total"] is None
    assert not app_module.MODEL_LOCK.locked()


def test_display_labels_do_not_make_duplicate_models_distinct(monkeypatch):
    data = payload(2)
    data["models"][1] = {**data["models"][0], "display_name": "Different visible name",
                         "endpoint": data["models"][0]["endpoint"].replace("127.0.0.1", "localhost")}
    monkeypatch.setattr(app_module, "predict_rubric", lambda *_: pytest.fail("Duplicate configuration must not run"))
    response = client.post("/api/compare", json=data)
    assert response.status_code == 422 and "different local models" in response.text


@pytest.mark.parametrize("changes", [
    {"model": ""}, {"model": "   "}, {"display_name": "x" * 81},
    {"endpoint": "https://example.com/v1/systemone"},
    {"endpoint": "http://secret:credential@127.0.0.1:1234/v1/systemone"},
    {"endpoint": "http://127.0.0.1:1234/v1/systemone?secret=credential"},
])
def test_invalid_workspace_settings_fail_before_inference_without_echo(monkeypatch, changes):
    data = payload(1)
    data["models"][0].update(changes)
    monkeypatch.setattr(app_module, "predict_rubric", lambda *_: pytest.fail("Invalid configuration must not run"))
    response = client.post("/api/compare", json=data)
    assert response.status_code == 422 and "secret" not in response.text and "credential@" not in response.text


def test_default_settings_remain_backwards_compatible():
    settings = Settings()
    assert settings.provider == "laya" and settings.model == "models/laya-multilingual"
    assert settings.display_name == ""
    assert Settings(provider="systemone", endpoint="http://localhost:1234/v1/systemone").endpoint == "http://127.0.0.1:1234/v1/systemone"
