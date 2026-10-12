import json
import threading
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from grader.app import app
from grader import providers
from tests.rubric_fixtures import GENERIC_RUBRIC
from grader.schemas import Settings

client = TestClient(app)


def payload():
    return {"rubric": deepcopy(GENERIC_RUBRIC), "text": "A synthetic explanation of urban tree cooling.",
            "models": [Settings().model_dump(), Settings(provider="systemone", model="Cloudflare/clef-flash").model_dump()]}


def answer(state, questions, highest=True):
    answers = {}
    for key, question in questions.items():
        if question["type"] == "score":
            # Ordinal model probabilities are keyed by zero-based rating index.
            keys = [str(index) for index in range(len(question["criteria"]))]
            chosen = keys[-1 if highest else 0]
        else:
            assert question["type"] == "choice"
            keys = list(question["criteria"])
            scale = json.loads(state).get("rating_scale")
            points = ({level["rating"]: level["points"] for level in scale} if scale else
                      {level["id"]: level["points"] for criterion in GENERIC_RUBRIC["criteria"]
                       if criterion["id"] == key for level in criterion["levels"]})
            chosen = (max if highest else min)(keys, key=points.get)
        answers[key] = {"probabilities": {level: float(level == chosen) for level in keys}}
    return {"answers": answers}


def test_both_models_run_concurrently_and_disagreement_is_visible(monkeypatch):
    barrier = threading.Barrier(2)
    seen = []
    def predict(settings, state, questions):
        assert len(questions) == 1
        seen.append((settings["provider"], state, questions))
        barrier.wait(timeout=3)
        return answer(state, questions, highest=settings["provider"] == "laya")
    monkeypatch.setattr(providers, "predict", predict)
    response = client.post("/api/compare", json=payload())
    assert response.status_code == 200
    result = response.json()
    assert len(seen) == 2 * len(GENERIC_RUBRIC["criteria"])
    by_model = {provider: [(state, questions) for name, state, questions in seen if name == provider]
                for provider in ("laya", "systemone")}
    assert by_model["laya"] == by_model["systemone"]
    assert [next(iter(questions)) for state, questions in by_model["laya"]] == [
        criterion["id"] for criterion in GENERIC_RUBRIC["criteria"]]
    assert [entry["result"]["total"] for entry in result["results"]] == [10, 0]
    assert all(row["disagrees"] for row in result["comparison"])
    assert "total" not in result  # No unrequested averaging or automated final grade.


def test_single_model_failure_does_not_hide_other_model_or_fabricate_grade(monkeypatch):
    def predict(settings, state, questions):
        if settings["provider"] == "systemone":
            raise providers.LocalRuntimeError("Local server unavailable.")
        return answer(state, questions)
    monkeypatch.setattr(providers, "predict", predict)
    result = client.post("/api/compare", json=payload()).json()
    assert result["results"][0]["ok"] is True
    assert result["results"][1] == {"label": "Model B · Cloudflare/clef-flash", "ok": False, "error": "Local server unavailable."}
    assert all(row["model_b_points"] is None and row["disagrees"] is None for row in result["comparison"])


def test_custom_servers_are_identified_without_calling_them_clef(monkeypatch):
    monkeypatch.setattr(providers, "predict", lambda settings, state, questions: answer(state, questions))
    request = payload()
    request["models"] = [Settings(provider="systemone", model=name,
                                  endpoint=f"http://127.0.0.1:{port}/v1/systemone").model_dump()
                         for name, port in (("StartLux-Decision-v1-4B-Q8_0", 1236), ("d3-mini", 1238))]
    result = client.post("/api/compare", json=request).json()
    assert [entry["label"] for entry in result["results"]] == [
        "Model A · StartLux-Decision-v1-4B-Q8_0", "Model B · d3-mini"]


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, Exception])
def test_unexpected_runtime_error_does_not_echo_student_content(monkeypatch, capsys, caplog, error_type):
    def predict(settings, state, questions):
        raise error_type("SECRET_STUDENT_INFORMATION")
    monkeypatch.setattr(providers, "predict", predict)
    response = client.post("/api/compare", json=payload())
    assert "SECRET_STUDENT_INFORMATION" not in response.text
    assert all(not entry["ok"] for entry in response.json()["results"])
    captured = capsys.readouterr()
    assert "SECRET_STUDENT_INFORMATION" not in captured.out + captured.err + caplog.text


def test_later_criterion_failure_is_atomic_per_model_and_releases_lock(monkeypatch):
    calls = {"laya": 0, "systemone": 0}
    def predict(settings, state, questions):
        provider = settings["provider"]
        calls[provider] += 1
        if provider == "systemone" and calls[provider] == 3:
            raise providers.LocalRuntimeError("Local server unavailable.")
        return answer(state, questions)
    monkeypatch.setattr(providers, "predict", predict)
    response = client.post("/api/compare", json=payload())
    assert response.status_code == 200
    result = response.json()
    assert calls == {"laya": 3, "systemone": 3}
    assert result["results"][0]["result"]["total"] == 10
    assert result["results"][1]["ok"] is False
    assert "result" not in result["results"][1]
    assert all(row["model_b_points"] is None and row["disagrees"] is None for row in result["comparison"])
    # A subsequent request must have neither stale partial answers nor a held lock.
    monkeypatch.setattr(providers, "predict", lambda settings, state, questions: answer(state, questions, False))
    next_response = client.post("/api/compare", json=payload())
    assert next_response.status_code == 200
    assert [entry["result"]["total"] for entry in next_response.json()["results"]] == [0, 0]


def test_duplicate_models_rejected():
    data = payload()
    data["models"][1] = data["models"][0]
    response = client.post("/api/compare", json=data)
    assert response.status_code == 422


def test_withheld_scores_are_not_compared_as_zero_or_agreement(monkeypatch):
    def predict(settings, state, questions):
        result = answer(state, questions, False)
        if settings["provider"] == "laya" and "understanding" in questions:
            assert questions["understanding"]["type"] == "score"
            result["answers"]["understanding"]["probabilities"] = {"2": .25, "1": .35, "0": .4}
        return result
    monkeypatch.setattr(providers, "predict", predict)
    data = payload()
    for model in data["models"]:
        model["scoring_mode"] = "strict_review"
    result = client.post("/api/compare", json=data).json()
    assert result["comparison"][0]["model_a_points"] is None
    assert result["comparison"][0]["model_b_points"] == 0
    assert result["comparison"][0]["disagrees"] is None
    assert result["results"][0]["result"]["total"] is None
    assert result["results"][1]["result"]["total"] == 0


def test_old_browser_cannot_render_withheld_scores_as_zero(monkeypatch):
    def predict(*args):
        raise AssertionError("Old pages must not trigger inference")
    monkeypatch.setattr(providers, "predict", predict)
    response = client.post("/api/compare", json=payload(), headers={"sec-fetch-site": "same-origin"})
    assert response.status_code == 409
    assert "older scoring display" in response.json()["detail"]
    assert "results" not in response.json()


def test_updated_browser_accepts_withheld_score_contract(monkeypatch):
    monkeypatch.setattr(providers, "predict", lambda settings, state, questions: answer(state, questions))
    response = client.post("/api/compare", json=payload(), headers={
        "sec-fetch-site": "same-origin", "x-margin-scoring-version": "3"})
    assert response.status_code == 200
    assert response.json()["results"][0]["ok"] is True


def test_blank_rating_descriptors_and_supplied_points_are_preserved(monkeypatch):
    data = payload()
    for criterion in data["rubric"]["criteria"]:
        for level in criterion["levels"]:
            level["description"] = ""
    before = deepcopy(data["rubric"])
    monkeypatch.setattr(providers, "predict", lambda settings, state, questions: answer(state, questions))
    response = client.post("/api/compare", json=data)
    assert response.status_code == 200
    for entry in response.json()["results"]:
        assert entry["result"]["rubric"] == before
        assert entry["result"]["total"] == 10


def test_assignment_context_reaches_both_models(monkeypatch):
    seen = []
    def predict(settings, state, questions):
        seen.append((settings["provider"], json.loads(state), questions))
        return answer(state, questions)
    monkeypatch.setattr(providers, "predict", predict)
    data = payload()
    data["assignment_details"] = "Explain two ways trees reduce urban heat and give a concrete example."
    result = client.post("/api/compare", json=data).json()
    assert len(seen) == 2 * len(GENERIC_RUBRIC["criteria"])
    assert {provider for provider, state, questions in seen} == {"laya", "systemone"}
    assert all(state["instructor_context"]["assignment_details"] == data["assignment_details"]
               and state["submission_text"] == data["text"] for provider, state, questions in seen)
    assert all(entry["result"]["assignment_details"] == data["assignment_details"] for entry in result["results"])
