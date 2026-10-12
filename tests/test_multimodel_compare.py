"""Variable model counts with mocked providers; no model inference."""
from copy import deepcopy
import json
import threading

from fastapi.testclient import TestClient
import pytest

from grader import providers
from grader.app import app
from tests.rubric_fixtures import GENERIC_RUBRIC


client = TestClient(app)


def payload():
    return {"rubric": deepcopy(GENERIC_RUBRIC), "text": "Synthetic shared evidence.",
            "assignment_details": "Assess only the named criterion.", "grading_guidance": "Credit clear introductory understanding.",
            "models": client.get("/api/defaults").json()["comparison_models"]}


def answer(state, questions, high=False, uniform=False):
    answers = {}
    for identifier, question in questions.items():
        if question["type"] == "score":
            keys = [str(i) for i in range(len(question["criteria"]))]
            chosen = keys[-1 if high else 0]
        else:
            assert question["type"] == "choice"
            keys = list(question["criteria"])
            points = {rating["rating"]: rating["points"] for rating in json.loads(state)["rating_scale"]}
            chosen = (max if high else min)(keys, key=points.get)
        answers[identifier] = {"probabilities": {key: 1 / len(keys) if uniform else float(key == chosen) for key in keys}}
    return {"answers": answers}


def test_defaults_expose_four_actual_local_model_configurations_in_order():
    models = client.get("/api/defaults").json()["comparison_models"]
    assert [(model["provider"], model["model"]) for model in models] == [
        ("laya", "models/laya-multilingual"), ("systemone", "clef-flash"),
        ("systemone", "StartLux-Decision-v1-4B-Q8_0"), ("systemone", "d3-mini")]
    assert [model["endpoint"] for model in models[1:]] == [
        f"http://127.0.0.1:{port}/v1/systemone" for port in (1235, 1236, 1238)]


def test_four_models_run_concurrently_with_ordered_results_and_identical_inputs(monkeypatch):
    data = payload()
    model_names = [settings["model"] for settings in data["models"]]
    barrier = threading.Barrier(4)
    seen = []
    def predict(settings, state, questions):
        assert len(questions) == 1
        seen.append((settings["model"], state, questions))
        barrier.wait(timeout=3)
        return answer(state, questions, high=model_names.index(settings["model"]) % 2 == 1)
    monkeypatch.setattr(providers, "predict", predict)
    response = client.post("/api/compare", json=data)
    assert response.status_code == 200
    result = response.json()
    assert [entry["label"] for entry in result["results"]] == [
        "Model A · Laya", "Model B · clef-flash", "Model C · StartLux-Decision-v1-4B-Q8_0", "Model D · d3-mini"]
    assert [entry["result"]["total"] for entry in result["results"]] == [0, 10, 0, 10]
    assert "total" not in result
    assert len(seen) == 4 * len(GENERIC_RUBRIC["criteria"])
    by_model = {name: [(state, questions) for model, state, questions in seen if name == model] for name in model_names}
    assert all(calls == by_model[model_names[0]] for calls in by_model.values())
    for state, questions in by_model[model_names[0]]:
        context = json.loads(state)
        assert context["submission_text"] == data["text"]
        assert context["instructor_context"]["assignment_details"] == data["assignment_details"]
        assert context["instructor_context"]["grading_guidance"] == data["grading_guidance"]
    for criterion, row in zip(GENERIC_RUBRIC["criteria"], result["comparison"]):
        maximum = max(level["points"] for level in criterion["levels"])
        assert row["model_points"] == [0, maximum, 0, maximum]
        assert len(row["model_level_ids"]) == row["available_models"] == 4
        assert row["model_a_points"] == row["model_points"][0]
        assert row["model_b_points"] == row["model_points"][1]
        assert row["disagrees"] is True


def test_third_model_late_failure_preserves_all_other_results_and_releases_lock(monkeypatch):
    data = payload()
    calls = {model["model"]: 0 for model in data["models"]}
    def predict(settings, state, questions):
        name = settings["model"]
        calls[name] += 1
        if name == "StartLux-Decision-v1-4B-Q8_0" and calls[name] == 3:
            raise providers.LocalRuntimeError("The local model is unavailable.")
        return answer(state, questions, high=name != "clef-flash")
    monkeypatch.setattr(providers, "predict", predict)
    response = client.post("/api/compare", json=data)
    assert response.status_code == 200
    result = response.json()
    assert [entry["ok"] for entry in result["results"]] == [True, True, False, True]
    assert [entry.get("result", {}).get("total") for entry in result["results"]] == [10, 0, None, 10]
    assert "result" not in result["results"][2]
    assert list(calls.values()) == [3, 3, 3, 3]
    assert all(row["model_points"][2] is None and row["model_level_ids"][2] is None
               and row["available_models"] == 3 and row["disagrees"] is True for row in result["comparison"])
    monkeypatch.setattr(providers, "predict", lambda settings, state, questions: answer(state, questions))
    retry = client.post("/api/compare", json=data)
    assert retry.status_code == 200
    assert [entry["result"]["total"] for entry in retry.json()["results"]] == [0, 0, 0, 0]


@pytest.mark.parametrize("last_available", [True, False])
def test_failures_and_withheld_scores_remain_null_in_partial_four_model_comparison(monkeypatch, last_available):
    data = payload()
    for model in data["models"]:
        model["scoring_mode"] = "strict_review"
    def predict(settings, state, questions):
        if settings["model"] == "clef-flash" or settings["model"] == "d3-mini" and not last_available:
            raise providers.LocalRuntimeError("The local server is unavailable.")
        return answer(state, questions, uniform=settings["provider"] == "laya")
    monkeypatch.setattr(providers, "predict", predict)
    result = client.post("/api/compare", json=data).json()
    assert result["results"][0]["ok"] is True
    assert result["results"][0]["result"]["total"] is None
    assert result["results"][2]["result"]["total"] == 0
    for row in result["comparison"]:
        assert row["model_points"] == [None, None, 0, 0 if last_available else None]
        assert row["model_level_ids"][:2] == [None, None]
        assert row["available_models"] == (2 if last_available else 1)
        assert row["disagrees"] is (False if last_available else None)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, Exception])
def test_fourth_model_failure_never_echoes_private_exception_content(monkeypatch, capsys, caplog, error_type):
    def predict(settings, state, questions):
        if settings["model"] == "d3-mini":
            raise error_type("PRIVATE_STUDENT_RUNTIME_ECHO")
        return answer(state, questions)
    monkeypatch.setattr(providers, "predict", predict)
    response = client.post("/api/compare", json=payload())
    assert response.status_code == 200
    assert [entry["ok"] for entry in response.json()["results"]] == [True, True, True, False]
    assert "PRIVATE_STUDENT_RUNTIME_ECHO" not in response.text
    captured = capsys.readouterr()
    assert "PRIVATE_STUDENT_RUNTIME_ECHO" not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("count", [0, 9])
def test_comparison_count_is_bounded_before_inference(monkeypatch, count):
    def predict(*args):
        pytest.fail("Invalid model count must be rejected before inference")
    monkeypatch.setattr(providers, "predict", predict)
    data = payload()
    data["models"] = [{**data["models"][1], "model": f"synthetic-{i}"} for i in range(count)]
    assert client.post("/api/compare", json=data).status_code == 422


@pytest.mark.parametrize("source,destination", [(0, 2), (1, 3), (2, 3)])
def test_duplicates_anywhere_in_the_four_model_list_are_rejected(monkeypatch, source, destination):
    def predict(*args):
        pytest.fail("Duplicate models must be rejected before inference")
    monkeypatch.setattr(providers, "predict", predict)
    data = payload()
    data["models"][destination] = deepcopy(data["models"][source])
    if source == 0:
        data["models"][destination]["endpoint"] = "http://127.0.0.1:9999/v1/systemone"
    response = client.post("/api/compare", json=data)
    assert response.status_code == 422
    assert "different local models" in response.json()["detail"]


def test_three_models_are_supported_without_padding_results(monkeypatch):
    monkeypatch.setattr(providers, "predict", lambda settings, state, questions: answer(state, questions))
    data = payload()
    data["models"] = data["models"][:3]
    response = client.post("/api/compare", json=data)
    assert response.status_code == 200
    result = response.json()
    assert len(result["results"]) == 3
    assert all(len(row["model_points"]) == len(row["model_level_ids"]) == row["available_models"] == 3
               for row in result["comparison"])


def test_equal_point_ratings_with_distinct_ids_still_show_rating_disagreement(monkeypatch):
    data = payload()
    data["rubric"] = {"title": "Alternative demonstrations", "criteria": [{
        "id": "evidence", "name": "Evidence", "description": "Choose the demonstrated form.", "levels": [
            {"id": "diagram", "label": "Diagram", "points": 5},
            {"id": "prose", "label": "Prose", "points": 5},
        ]}]}
    def predict(settings, state, questions):
        chosen = "r0" if settings["provider"] == "laya" else "r1"
        return {"answers": {"evidence": {"probabilities": {"r0": float(chosen == "r0"), "r1": float(chosen == "r1")}}}}
    monkeypatch.setattr(providers, "predict", predict)
    result = client.post("/api/compare", json=data).json()
    assert result["comparison"][0]["model_points"] == [5, 5, 5, 5]
    assert result["comparison"][0]["model_level_ids"] == ["diagram", "prose", "prose", "prose"]
    assert result["comparison"][0]["disagrees"] is True
