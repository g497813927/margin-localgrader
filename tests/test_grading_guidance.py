"""Empty production defaults and optional user guidance on synthetic rubrics."""
from copy import deepcopy
import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from grader import providers
from grader.app import app
from grader.defaults import DEFAULT_GRADING_GUIDANCE, DEFAULT_RUBRIC
from grader.harness import build_plan
from grader.schemas import CompareRequest, GradeRequest, Settings
from grader.scoring import score_response
from tests.rubric_fixtures import GENERIC_RUBRIC


client = TestClient(app)
# Frozen from the generic synthetic fixture, independent of production defaults.
BASELINE_PLAN_SHA256 = "9490ef75f35c1d33e425b9fe2fa84f62d0440c5b1953f84fb91edc8e01f1ed63"


def base_payload():
    return {"rubric": deepcopy(GENERIC_RUBRIC), "settings": {},
            "text": "Synthetic unchanged baseline.\nUnicode: 树木.",
            "assignment_details": "Use evidence from this excerpt."}


def comparison_payload(data):
    data = deepcopy(data)
    data.pop("settings")
    data["models"] = [Settings().model_dump(), Settings(provider="systemone", model="clef-flash").model_dump()]
    return data


@pytest.mark.parametrize("extra", [{}, {"grading_guidance": ""}, {"grading_guidance": " \n\t "}])
def test_empty_or_omitted_guidance_preserves_frozen_model_payload_bytes(extra):
    request = GradeRequest.model_validate({**base_payload(), **extra})
    assert request.grading_guidance == ""
    plan = build_plan(request)
    serialized = json.dumps(plan, ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(serialized).hexdigest() == BASELINE_PLAN_SHA256
    assert all("grading_guidance" not in json.loads(state)["instructor_context"]
               for state, questions, mappings in plan)
    compare_request = CompareRequest.model_validate(comparison_payload({**base_payload(), **extra}))
    assert compare_request.grading_guidance == ""


def test_guidance_changes_only_instructor_context_and_is_not_truncated():
    request = GradeRequest.model_validate(base_payload())
    suffix = "GUIDANCE_END"
    guidance = 'Policy with "quotes", 中文, and a newline.\n'
    guidance += "g" * (12000 - len(guidance) - len(suffix)) + suffix
    guided = GradeRequest.model_validate({**base_payload(), "grading_guidance": guidance})
    original = deepcopy(guided.model_dump())
    for (state, questions, mappings), (guided_state, guided_questions, guided_mappings) in zip(
            build_plan(request), build_plan(guided)):
        context = json.loads(guided_state)
        assert context["instructor_context"].pop("grading_guidance") == guidance
        assert json.dumps(context, ensure_ascii=False) == state
        assert guided_questions == questions
        assert guided_mappings == mappings
        assert context["submission_text"] == request.text
    assert guided.model_dump() == original
    assert guided.rubric == request.rubric


def test_default_guidance_and_rubric_are_empty():
    response = client.get("/api/defaults")
    assert response.status_code == 200
    data = response.json()
    assert data["grading_guidance"] == DEFAULT_GRADING_GUIDANCE == ""
    assert data["rubric"] == DEFAULT_RUBRIC == {"title": "", "instructions": "", "criteria": []}
    assert GradeRequest.model_validate(base_payload()).grading_guidance == ""


@pytest.mark.parametrize("route", ["/api/grade", "/api/compare"])
def test_empty_default_rubric_cannot_grade_before_user_defines_criteria(monkeypatch, route):
    monkeypatch.setattr(providers, "predict", lambda *_: pytest.fail("An empty rubric must not reach inference"))
    data = {**base_payload(), "rubric": deepcopy(DEFAULT_RUBRIC)}
    if route == "/api/compare":
        data = comparison_payload(data)
    response = client.post(route, json=data)
    assert response.status_code == 422
    assert "criteria" in response.json()["detail"]
    assert "total" not in response.json() and "results" not in response.json()


@pytest.mark.parametrize("route", ["/api/grade", "/api/compare"])
def test_guidance_reaches_every_model_criterion_and_is_exported(monkeypatch, route):
    seen = []
    guidance = "Accept relevant examples from any city. Keep the rubric's stated requirements and point values."
    def predict(settings, state, questions):
        seen.append((settings["provider"], json.loads(state)))
        answers = {}
        for key, question in questions.items():
            keys = ([str(i) for i in range(len(question["criteria"]))]
                    if question["type"] == "score" else list(question["criteria"]))
            answers[key] = {"probabilities": {token: float(token == keys[0]) for token in keys}}
        return {"answers": answers}
    monkeypatch.setattr(providers, "predict", predict)
    data = {**base_payload(), "grading_guidance": guidance}
    if route == "/api/compare":
        data = comparison_payload(data)
    response = client.post(route, json=data)
    assert response.status_code == 200
    results = ([entry["result"] for entry in response.json()["results"]]
               if route == "/api/compare" else [response.json()])
    assert len(seen) == len(GENERIC_RUBRIC["criteria"]) * len(results)
    assert {provider for provider, context in seen} == ({"laya", "systemone"} if len(results) == 2 else {"laya"})
    assert all(context["instructor_context"]["grading_guidance"] == guidance
               and context["submission_text"] == data["text"] for provider, context in seen)
    assert all(result["grading_guidance"] == guidance and result["rubric"] == data["rubric"] for result in results)


@pytest.mark.parametrize("route", ["/api/grade", "/api/compare"])
@pytest.mark.parametrize("guidance", ["PRIVATE_GUIDANCE" + "g" * 12000, {"private": "PRIVATE_GUIDANCE"}, None])
def test_invalid_guidance_is_rejected_before_inference_without_echo(monkeypatch, capsys, route, guidance):
    def predict(*args):
        pytest.fail("Invalid guidance must be rejected before model invocation")
    monkeypatch.setattr(providers, "predict", predict)
    data = {**base_payload(), "grading_guidance": guidance}
    if route == "/api/compare":
        data = comparison_payload(data)
    response = client.post(route, json=data)
    assert response.status_code == 422
    assert "PRIVATE_GUIDANCE" not in response.text
    captured = capsys.readouterr()
    assert "PRIVATE_GUIDANCE" not in captured.out + captured.err


@pytest.mark.parametrize("choose_highest", [False, True])
def test_guidance_does_not_apply_a_boost_or_override_the_selected_level(choose_highest):
    request = GradeRequest.model_validate(base_payload())
    guidance = "Accept clear paraphrases while keeping all defined level requirements and points."
    guided = request.model_copy(update={"grading_guidance": guidance})
    answers = {}
    for criterion in request.rubric.criteria:
        chosen = (max if choose_highest else min)(criterion.levels, key=lambda level: level.points)
        answers[criterion.id] = {"probabilities": {level.id: float(level.id == chosen.id) for level in criterion.levels}}
    original_result = score_response(request, {"answers": answers})
    guided_result = score_response(guided, {"answers": answers})
    assert original_result["grading_guidance"] == ""
    assert guided_result["grading_guidance"] == guidance
    assert guided_result["total"] == (10 if choose_highest else 0)
    for result in (original_result, guided_result):
        result.pop("created_at")
        result.pop("grading_guidance")
    assert guided_result == original_result
