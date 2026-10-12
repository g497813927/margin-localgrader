import socket

import pytest
from fastapi.testclient import TestClient

from grader.app import app
from tests.rubric_fixtures import GENERIC_RUBRIC
from grader import providers
from grader.schemas import Settings


client = TestClient(app)


def payload():
    return {"rubric": GENERIC_RUBRIC, "settings": Settings().model_dump(), "text": "Synthetic submission."}


def test_live_api_scoring_contract(monkeypatch):
    calls = []
    def predict(settings, state, questions):
        calls.append(questions)
        assert len(questions) == 1
        answers = {}
        for key, question in questions.items():
            assert question["type"] == "score"
            assert len(question["criteria"]) == 3
            answers[key] = {"probabilities": {"2": 0.8, "1": 0.15, "0": 0.05}, "score": 0}
        return {"answers": answers}
    monkeypatch.setattr(providers, "predict", predict)
    result = client.post("/api/grade", json=payload())
    assert result.status_code == 200
    assert result.json()["total"] == 10
    assert len(calls) == len(GENERIC_RUBRIC["criteria"])
    assert all(row["level_id"] == "strong" for row in result.json()["criteria"])
    assert result.json()["rubric"] == GENERIC_RUBRIC
    assert result.json()["harness"] == "rubric-v2"
    assert result.headers["cache-control"] == "no-store"
    assert "connect-src 'self'" in result.headers["content-security-policy"]


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, Exception])
def test_single_grade_sanitizes_untrusted_errors(monkeypatch, capsys, caplog, error_type):
    def predict(*args):
        raise error_type("SYNTHETIC_PRIVATE_RUNTIME_DETAIL")
    monkeypatch.setattr(providers, "predict", predict)
    result = client.post("/api/grade", json=payload())
    assert result.status_code == 503
    assert "SYNTHETIC_PRIVATE_RUNTIME_DETAIL" not in result.text
    assert "total" not in result.json()
    captured = capsys.readouterr()
    assert "SYNTHETIC_PRIVATE_RUNTIME_DETAIL" not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("error,status", [
    (providers.LocalInputError("Rubric exceeds the safe input limit."), 422),
    (providers.LocalRuntimeError("The local checkpoint is unavailable."), 503),
])
def test_single_grade_preserves_trusted_provider_guidance(monkeypatch, error, status):
    def predict(*args):
        raise error
    monkeypatch.setattr(providers, "predict", predict)
    result = client.post("/api/grade", json=payload())
    assert result.status_code == status
    assert result.json() == {"detail": str(error)}


def test_single_grade_rejects_truncated_response_without_partial_scores(monkeypatch):
    calls = []
    def predict(settings, state, questions):
        calls.append(questions)
        return {"answers": {}, "usage": {"truncated": True}}
    monkeypatch.setattr(providers, "predict", predict)
    result = client.post("/api/grade", json=payload())
    assert result.status_code == 422
    assert "discarded input tokens" in result.json()["detail"]
    assert len(calls) == 1
    assert "total" not in result.json()


def test_document_upload_extracts_text():
    result = client.post("/api/extract", files={"file": ("synthetic.txt", b"Trees cool streets.", "text/plain")})
    assert result.status_code == 200
    assert result.json()["text"] == "Trees cool streets."


def test_cross_origin_and_host_are_blocked():
    assert client.post("/api/grade", json=payload(), headers={"Origin": "https://example.com"}).status_code == 403
    assert client.get("/api/defaults", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/api/defaults", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_validation_does_not_echo_student_data():
    data = payload()
    data["text"] = {"secret_student_data": "never_echo_this"}
    result = client.post("/api/grade", json=data)
    assert result.status_code == 422
    assert "never_echo_this" not in result.text


def test_oversize_is_explicit_and_no_inference(monkeypatch):
    def fail(*args):
        pytest.fail("Must reject before inference")
    monkeypatch.setattr(providers, "predict", fail)
    data = payload()
    data["settings"]["max_input_chars"] = 100
    data["text"] = "x" * 101
    result = client.post("/api/grade", json=data)
    assert result.status_code == 422
    assert "never silently truncated" in result.text


def test_external_network_and_dns_blocked():
    with socket.socket() as sock:
        with pytest.raises(PermissionError, match="Local-only"):
            sock.connect(("1.1.1.1", 443))
    with pytest.raises(PermissionError, match="Local-only"):
        socket.getaddrinfo("example.com", 443)
