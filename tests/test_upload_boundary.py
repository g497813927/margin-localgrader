"""Exercise the HTTP byte boundary before any multipart parser side effects."""

import asyncio
import json
import tempfile

import pytest
from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient
from starlette.formparsers import MultiPartParser

from grader.upload_boundary import BodySizeLimitMiddleware


@pytest.fixture(autouse=True)
def preserve_parser_settings(monkeypatch):
    monkeypatch.setattr(MultiPartParser, "spool_max_size", MultiPartParser.spool_max_size)


def run_body(chunks, limit=8, headers=(), disconnect=False):
    messages = [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1 or disconnect}
        for index, chunk in enumerate(chunks)
    ]
    if disconnect:
        messages.append({"type": "http.disconnect"})
    sent = []
    consumed = []
    receive_count = 0

    async def receive():
        nonlocal receive_count
        receive_count += 1
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async def downstream(scope, receive, send):
        message = await receive()
        consumed.append(message)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = BodySizeLimitMiddleware(downstream, max_body_bytes=limit)
    asyncio.run(middleware({"type": "http", "headers": list(headers)}, receive, send))
    return sent, consumed, receive_count


def test_exact_limit_is_replayed_as_complete_body():
    sent, consumed, _ = run_body([b"123", b"45678"])
    assert sent[0]["status"] == 200
    assert consumed == [{"type": "http.request", "body": b"12345678", "more_body": False}]


def test_oversized_chunked_body_never_reaches_parser():
    sent, consumed, received = run_body([b"1234", b"5678", b"9", b"do not drain"])
    assert sent[0]["status"] == 413
    assert "too large" in json.loads(sent[1]["body"])["detail"]
    assert consumed == []
    assert received == 3


def test_understated_content_length_cannot_bypass_actual_byte_count():
    sent, consumed, _ = run_body([b"123456789"], headers=[(b"content-length", b"1")])
    assert sent[0]["status"] == 413
    assert not consumed


def test_declared_oversize_is_rejected_without_reading_body():
    sent, consumed, received = run_body([b"unused"], headers=[(b"content-length", b"9")])
    assert sent[0]["status"] == 413
    assert not consumed
    assert received == 0


@pytest.mark.parametrize("headers", [
    [(b"content-length", b"-1")],
    [(b"content-length", b"invalid")],
    [(b"content-length", b"2"), (b"content-length", b"3")],
])
def test_invalid_lengths_get_json_error(headers):
    sent, consumed, _ = run_body([b"unused"], headers=headers)
    assert sent[0]["status"] == 400
    assert json.loads(sent[1]["body"])["detail"] == "Invalid request size."
    assert not consumed


def test_disconnect_discards_partial_body():
    sent, consumed, _ = run_body([b"123"], disconnect=True)
    assert not sent
    assert not consumed


def test_empty_body_is_forwarded():
    sent, consumed, _ = run_body([b""])
    assert sent[0]["status"] == 200
    assert consumed[0]["body"] == b""


def test_non_http_scope_passes_through():
    seen = []

    async def downstream(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(BodySizeLimitMiddleware(downstream)({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]


def test_large_accepted_multipart_upload_never_spools_to_disk(monkeypatch):
    def forbidden_rollover(self):
        raise AssertionError("An accepted upload attempted to spool to disk")

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", forbidden_rollover)
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware)

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        try:
            data = await file.read()
            return {"size": len(data), "in_memory": not file.file._rolled, "name": file.filename}
        finally:
            await file.close()

    payload = b"x" * (2 * 1024 * 1024)
    with TestClient(app) as client:
        response = client.post("/upload", files={"file": ("answer.txt", payload, "text/plain")})
    assert response.status_code == 200
    assert response.json() == {"size": len(payload), "in_memory": True, "name": "answer.txt"}


def test_oversized_multipart_is_rejected_before_route_and_spool(monkeypatch):
    def forbidden_rollover(self):
        raise AssertionError("A rejected upload attempted to spool to disk")

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", forbidden_rollover)
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=1024)

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        pytest.fail("Oversized upload reached the route")

    boundary = "test-upload-boundary"
    prefix = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="answer.txt"\r\nContent-Type: text/plain\r\n\r\n'.encode()
    chunks = iter([prefix, b"x" * 1024, f"\r\n--{boundary}--\r\n".encode()])
    with TestClient(app) as client:
        response = client.post("/upload", content=chunks, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert response.status_code == 413
    assert response.headers["content-type"] == "application/json"
    assert "too large" in response.json()["detail"]
