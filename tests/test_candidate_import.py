"""Pinned public-weight imports use synthetic files and mocked loopback HTTP."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import import_explanation_candidate as importer


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    cache = tmp_path / "models/ollama-explanations"
    (cache / "blobs").mkdir(parents=True)
    source = tmp_path / "models/source.gguf"
    data = b"GGUF public synthetic model bytes"
    source.write_bytes(data)
    spec = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "path": "models/source.gguf"}
    verified = []
    monkeypatch.setattr(importer, "ROOT", tmp_path)
    monkeypatch.setattr(importer, "CACHE", cache)
    monkeypatch.setattr(importer, "CANDIDATES", {key: spec for key in importer.MODEL_NAMES})
    monkeypatch.setattr(importer, "verify_model", lambda candidate: verified.append(candidate) or source)
    monkeypatch.setattr(importer, "copy_model_file", shutil.copyfile)
    return {"root": tmp_path, "cache": cache, "source": source, "data": data, "spec": spec, "verified": verified}


def store_metadata(prepared, value):
    raw = json.dumps(value, separators=(",", ":")).encode()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    (prepared["cache"] / "blobs" / digest.replace(":", "-")).write_bytes(raw)
    return {"digest": digest, "size": len(raw)}


def store_model(prepared, name, *, wrong_weights=False, extra_layer=False, context=16384):
    spec = prepared["spec"]
    blob = prepared["cache"] / "blobs" / ("sha256-" + spec["sha256"])
    if not blob.exists():
        blob.write_bytes(prepared["data"])
    config = store_metadata(prepared, {"model_format": "gguf", "model_family": "qwen35"})
    params = store_metadata(prepared, {"num_ctx": context})
    manifest = {"schemaVersion": 2, "config": config, "layers": [
        {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + ("0"*64 if wrong_weights else spec["sha256"]), "size": spec["size"]},
        {"mediaType": "application/vnd.ollama.image.params", **params},
    ]}
    if extra_layer:
        manifest["layers"].append({"mediaType": "application/vnd.ollama.image.system", "digest": "sha256:" + "0"*64})
    descriptor = store_metadata(prepared, manifest)
    path = importer.manifest_paths(name)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to("../../../../blobs/" + descriptor["digest"].replace(":", "-"))
    return {"name": name, "digest": descriptor["digest"]}


@contextmanager
def server(prepared, *, candidate="qwen38_4b", existing=None, version="0.40.3", cloud=None,
           show_changes=None, route_override=None, create_success=True, wrong_cache=False):
    name = importer.MODEL_NAMES[candidate]
    models = [existing] if existing else []
    requests = []
    def handler(request):
        requests.append(request)
        assert str(request.url).startswith("http://127.0.0.1:11435/api/")
        assert "authorization" not in request.headers
        route = request.url.path
        if route_override:
            override = route_override(request)
            if override is not None:
                return override
        if route == "/api/version": return httpx.Response(200, json={"version": version})
        if route == "/api/status": return httpx.Response(200, json={"cloud": cloud if cloud is not None else {"disabled": True, "source": "env"}})
        if route == "/api/tags": return httpx.Response(200, json={"models": models})
        if route.startswith("/api/blobs/"):
            assert request.method == "HEAD"
            blob = prepared["cache"] / "blobs" / route.rsplit("/", 1)[1].replace(":", "-")
            return httpx.Response(200 if blob.is_file() and not wrong_cache else 404)
        if route == "/api/create":
            payload = json.loads(request.content)
            assert payload == {"model": name, "files": {prepared["source"].name: "sha256:" + prepared["spec"]["sha256"]}, "parameters": {"num_ctx": 16384}, "stream": False}
            assert not models
            if create_success:
                models.append(store_model(prepared, name))
            return httpx.Response(200, json={"status": "success" if create_success else "incomplete"})
        if route == "/api/show":
            assert json.loads(request.content) == {"model": name}
            quantization = "Q4_K_M" if candidate == "qwen36" else name.split(":")[1].upper()
            family = "qwen35moe" if candidate == "qwen36" else "qwen35"
            value = {"details": {"format": "gguf", "family": family, "quantization_level": quantization}, "capabilities": ["completion", "thinking"], "parameters": "num_ctx 16384", "template": "synthetic local template"}
            value.update(show_changes or {})
            return httpx.Response(200, json=value)
        pytest.fail("Unexpected route (no generation, pull, or remote endpoints permitted): " + route)
    with httpx.Client(transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=False) as client:
        yield client, requests


@pytest.mark.parametrize("candidate", list(importer.MODEL_NAMES))
def test_all_pins_import_only_the_local_digest_then_verify(prepared, candidate):
    with server(prepared, candidate=candidate) as (client, requests):
        result = importer.import_candidate(candidate, client=client)
    assert result["model"] == importer.MODEL_NAMES[candidate]
    assert result["sha256"] == prepared["spec"]["sha256"]
    assert prepared["verified"] == [candidate]
    assert sum(request.url.path == "/api/create" for request in requests) == 1
    assert prepared["source"].read_bytes() == prepared["data"]


def test_qwen36_descriptive_tag_uses_verified_base_quantization(prepared):
    assert importer.MODEL_NAMES["qwen36"] == "qwen36-35b-a3b:ud_q4_k_xl"
    with server(prepared, candidate="qwen36") as (client, requests):
        result = importer.import_candidate("qwen36", client=client)
    assert result["details"]["quantization_level"] == "Q4_K_M"
    assert result["details"]["family"] == "qwen35moe"
    assert all(r.url.host == "127.0.0.1" and r.url.port == 11435 for r in requests)


def test_qwen36_wrong_base_quantization_is_rejected_without_recreation(prepared):
    name = importer.MODEL_NAMES["qwen36"]
    existing = store_model(prepared, name)
    changes = {"details": {"format": "gguf", "family": "qwen35moe", "quantization_level": "Q8_0"}}
    with server(prepared, candidate="qwen36", existing=existing, show_changes=changes) as (client, requests):
        with pytest.raises(importer.ImportError, match="metadata differs"):
            importer.import_candidate("qwen36", client=client)
    assert not any(r.url.path == "/api/create" for r in requests)


@pytest.mark.parametrize("bare_digest", [True, False])
def test_existing_import_is_verified_without_recreation_or_clone(prepared, monkeypatch, bare_digest):
    name = importer.MODEL_NAMES["qwen38_4b"]
    existing = store_model(prepared, name)
    if bare_digest:
        existing["digest"] = existing["digest"].removeprefix("sha256:")
    monkeypatch.setattr(importer, "copy_model_file", lambda *args: pytest.fail("No duplicate clone"))
    with server(prepared, existing=existing) as (client, requests):
        result = importer.import_candidate("qwen38_4b", client=client)
    assert result["model"] == name
    assert not any(r.url.path == "/api/create" for r in requests)


@pytest.mark.parametrize("changes", [{"wrong_weights": True}, {"extra_layer": True}, {"context": 8192}])
def test_unknown_existing_model_is_never_overwritten(prepared, changes):
    existing = store_model(prepared, importer.MODEL_NAMES["qwen38_4b"], **changes)
    with server(prepared, existing=existing) as (client, requests):
        with pytest.raises(importer.ImportError, match="manifest differs"):
            importer.import_candidate("qwen38_4b", client=client)
    assert not any(r.url.path == "/api/create" for r in requests)


def test_endpoint_manifest_identity_must_match_project(prepared):
    existing = store_model(prepared, importer.MODEL_NAMES["qwen38_4b"])
    existing["digest"] = "sha256:" + "f"*64
    with server(prepared, existing=existing) as (client, _):
        with pytest.raises(importer.ImportError, match="identities differ"):
            importer.import_candidate("qwen38_4b", client=client)


@pytest.mark.parametrize("cloud", [{"disabled": False, "source": "none"}, {"disabled": True, "source": "config"}, {}])
def test_requires_daemon_cloud_disabled_by_environment_before_writes(prepared, cloud):
    with server(prepared, cloud=cloud) as (client, requests):
        with pytest.raises(importer.ImportError, match="OLLAMA_NO_CLOUD"):
            importer.import_candidate("qwen38_4b", client=client)
    assert list((prepared["cache"] / "blobs").iterdir()) == []
    assert not any(r.method == "POST" for r in requests)


def test_rejects_wrong_runtime_before_writes(prepared):
    with server(prepared, version="0.40.2") as (client, requests):
        with pytest.raises(importer.ImportError, match="0.40.3"):
            importer.import_candidate("qwen38_4b", client=client)
    assert len(requests) == 1


def test_incomplete_pin_stops_before_any_request(prepared, monkeypatch):
    def fail(candidate): raise RuntimeError("missing candidate")
    monkeypatch.setattr(importer, "verify_model", fail)
    with server(prepared) as (client, requests):
        with pytest.raises(RuntimeError): importer.import_candidate("pro", client=client)
    assert not requests


def test_wrong_existing_blob_is_left_untouched(prepared):
    blob = prepared["cache"] / "blobs" / ("sha256-" + prepared["spec"]["sha256"])
    blob.write_bytes(b"x" * prepared["spec"]["size"])
    with server(prepared) as (client, requests):
        with pytest.raises(importer.ImportError, match="checksum"):
            importer.import_candidate("qwen38_4b", client=client)
    assert blob.read_bytes() == b"x" * prepared["spec"]["size"]
    assert not any(r.url.path == "/api/create" for r in requests)


def test_unknown_local_manifest_is_left_untouched(prepared):
    path = importer.manifest_paths(importer.MODEL_NAMES["qwen38_4b"])[1]
    path.parent.mkdir(parents=True); path.write_text("keep")
    with server(prepared) as (client, requests):
        with pytest.raises(importer.ImportError, match="unrecognized local"):
            importer.import_candidate("qwen38_4b", client=client)
    assert path.read_text() == "keep" and not any(r.url.path == "/api/create" for r in requests)


def test_wrong_daemon_cache_rejected_before_any_import_write(prepared):
    before = sorted(str(p.relative_to(prepared["cache"])) for p in prepared["cache"].rglob("*"))
    with server(prepared, wrong_cache=True) as (client, requests):
        with pytest.raises(importer.ImportError, match="different model cache"):
            importer.import_candidate("qwen38_4b", client=client)
    assert before == sorted(str(p.relative_to(prepared["cache"])) for p in prepared["cache"].rglob("*"))
    assert not any(r.method == "POST" for r in requests)
    assert any(r.method == "HEAD" for r in requests)


def test_cache_probe_is_removed_when_daemon_request_fails(prepared):
    def fail_probe(request):
        if request.method == "HEAD":
            raise httpx.ConnectError("private error details")
    with server(prepared, route_override=fail_probe) as (client, requests):
        with pytest.raises(importer.ImportError, match="Could not verify"):
            importer.import_candidate("qwen38_4b", client=client)
    assert list((prepared["cache"] / "blobs").iterdir()) == []
    assert not any(r.method == "POST" for r in requests)


@pytest.mark.parametrize("changes", [{"remote_host": "https://example.invalid"}, {"remote_model": "cloud-model"}, {"system": "Unexpected instructions"}, {"parameters": "num_ctx 8192"}, {"template": ""}, {"capabilities": []}])
def test_rejects_unsafe_or_wrong_model_metadata(prepared, changes):
    existing = store_model(prepared, importer.MODEL_NAMES["qwen38_4b"])
    with server(prepared, existing=existing, show_changes=changes) as (client, requests):
        with pytest.raises(importer.ImportError, match="metadata differs"):
            importer.import_candidate("qwen38_4b", client=client)
    assert not any(r.url.path == "/api/create" for r in requests)


def test_redirect_is_not_followed_or_logged(prepared):
    def override(request):
        return httpx.Response(302, headers={"location": "https://example.invalid/secret"})
    with server(prepared, route_override=override) as (client, requests):
        with pytest.raises(importer.ImportError, match="rejected"):
            importer.import_candidate("qwen38_4b", client=client)
    assert len(requests) == 1


def test_metadata_size_is_bounded(prepared):
    def override(request): return httpx.Response(200, content=b" " * (importer.MAX_METADATA + 1))
    with server(prepared, route_override=override) as (client, _):
        with pytest.raises(importer.ImportError, match="oversized"):
            importer.import_candidate("qwen38_4b", client=client)


def test_import_lock_prevents_two_writers(prepared):
    with importer.import_lock():
        with pytest.raises(importer.ImportError, match="in progress"):
            with importer.import_lock(): pytest.fail("Acquired second lock")


def test_owned_http_client_disables_proxies_redirects_and_is_closed(prepared, monkeypatch):
    with server(prepared) as (client, requests):
        def factory(**kwargs):
            assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
            return client
        monkeypatch.setattr(importer.httpx, "Client", factory)
        importer.import_candidate("qwen38_4b")
        assert client.is_closed


def test_main_hides_unexpected_native_errors(monkeypatch, capsys):
    def fail(candidate): raise RuntimeError("PRIVATE_NATIVE_DATA")
    monkeypatch.setattr(importer, "import_candidate", fail)
    assert importer.main(["--candidate", "qwen38_4b"]) == 1
    assert "PRIVATE_NATIVE_DATA" not in capsys.readouterr().err
