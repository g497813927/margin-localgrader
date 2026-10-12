"""No weights loaded: verify the local d3 boundary using synthetic fake results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from fastapi.testclient import TestClient

from scripts import run_d3


PAYLOAD = {
    "model": "d3-mini", "state": "A synthetic readable test sentence.",
    "questions": {"quality": {"type": "choice", "instructions": "Choose the fitting level.",
                              "criteria": {"none": "Missing", "full": "Complete"}}},
}


class FakeRuntime:
    def __init__(self):
        self.prepare_calls = 0
        self.run_calls = 0
        self.prepare_errors = {}
        self.sequence_length = 20
        self.failure = None
        self.answer = {"type": "choice", "choice": "full",
                       "probabilities": {"none": 0.1, "full": 0.9}, "confidence": 0.53}

    def prepare(self, state, questions):
        self.prepare_calls += 1
        return SimpleNamespace(errors=self.prepare_errors, images=[], videos=[],
                               sequences={key: [1] * self.sequence_length for key in questions})

    def run(self, prepared):
        self.run_calls += 1
        if self.failure:
            raise self.failure
        return {}, 20

    def respond(self, prepared, probabilities, tokens):
        return {"model": "d3-mini", "answers": {"quality": self.answer},
                "usage": {"input_tokens": tokens, "output_tokens": 0},
                "untrusted_extra": "SYNTHETIC_DO_NOT_ECHO"}


class D3BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeRuntime()
        self.client = TestClient(run_d3.create_app(self.model, max_context=256),
                                 base_url="http://127.0.0.1")

    def tearDown(self):
        self.client.close()

    def post(self, payload=None):
        return self.client.post("/v1/systemone", json=PAYLOAD if payload is None else payload)

    def test_reference_probabilities_preserved_without_echo_or_cache(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answers"]["quality"], self.model.answer)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("SYNTHETIC_DO_NOT_ECHO", response.text)
        self.assertFalse(response.json()["usage"]["truncated"])
        self.assertEqual(self.model.run_calls, 1)

    def test_repeated_requests_run_independently(self):
        self.assertEqual(self.post().status_code, 200)
        self.assertEqual(self.post().status_code, 200)
        self.assertEqual(self.model.run_calls, 2)

    def test_images_videos_extra_fields_and_nontext_state_rejected_before_runtime(self):
        for change in ({"images": ["https://example.invalid/image.png"]},
                       {"videos": []}, {"options": {}}, {"state": {"image": "x"}},
                       {"model": "a-remote-model"}):
            with self.subTest(change=change):
                self.assertEqual(self.post({**PAYLOAD, **change}).status_code, 422)
        self.assertEqual(self.model.prepare_calls, 0)

    def test_invalid_schema_never_echoes_request_or_validation_detail(self):
        payload = deepcopy(PAYLOAD)
        payload["questions"]["quality"]["type"] = "SYNTHETIC_DO_NOT_ECHO"
        response = self.post(payload)
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("SYNTHETIC_DO_NOT_ECHO", response.text)
        self.assertEqual(self.model.prepare_calls, 0)

    def test_content_type_and_body_limit(self):
        response = self.client.post("/v1/systemone", content=json.dumps(PAYLOAD))
        self.assertEqual(response.status_code, 415)
        response = self.client.post("/v1/systemone", content=b"x" * (run_d3.MAX_BODY_BYTES + 1),
                                    headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.model.prepare_calls, 0)

    def test_context_overflow_is_rejected_before_any_forward_pass(self):
        self.model.prepare_errors = {"quality": {"error": "max_length_exceeded",
                                                 "message": "SYNTHETIC_DO_NOT_ECHO"}}
        response = self.post()
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("SYNTHETIC_DO_NOT_ECHO", response.text)
        self.assertEqual(self.model.run_calls, 0)
        self.model.prepare_errors = {}
        self.model.sequence_length = 257
        self.assertEqual(self.post().status_code, 422)
        self.assertEqual(self.model.run_calls, 0)

    def test_runtime_exception_is_sanitized(self):
        self.model.failure = RuntimeError("SYNTHETIC_DO_NOT_ECHO")
        response = self.post()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("SYNTHETIC_DO_NOT_ECHO", response.text)

    def test_partial_or_invalid_model_answers_fail_closed(self):
        invalid = [
            {"type": "choice", "error": "invalid_model_output", "message": "SYNTHETIC_DO_NOT_ECHO"},
            {"type": "choice", "choice": "full", "probabilities": {"none": 0.1}},
            {"type": "choice", "choice": "full", "probabilities": {"none": 0.9, "full": 0.9}},
            {"type": "choice", "choice": "full", "probabilities": {"none": float("nan"), "full": 0.9}},
        ]
        for answer in invalid:
            with self.subTest(answer=answer):
                self.model.answer = answer
                response = self.post()
                self.assertEqual(response.status_code, 500)
                self.assertNotIn("SYNTHETIC_DO_NOT_ECHO", response.text)

    def test_ordinal_index_and_probabilities_remain_native(self):
        payload = deepcopy(PAYLOAD)
        payload["questions"]["quality"] = {
            "type": "score", "instructions": "Rate the response.", "criteria": ["Incomplete", "Complete"]}
        self.model.answer = {"type": "score", "score": 0.8,
                             "probabilities": {"0": 0.2, "1": 0.8}, "confidence": 0.28}
        response = self.post(payload)
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answers"]["quality"]
        self.assertEqual(answer["score"], 0.8)
        self.assertEqual(answer["probabilities"], {"0": 0.2, "1": 0.8})
        self.assertEqual(answer["legend"], {"0": "Incomplete", "1": "Complete"})

    def test_noul_probability_preserved(self):
        payload = deepcopy(PAYLOAD)
        payload["questions"]["quality"] = {"type": "noul", "instructions": "Is the text readable?"}
        self.model.answer = {"type": "noul", "noul": 0.73}
        self.assertEqual(self.post(payload).json()["answers"]["quality"], self.model.answer)

    def test_single_inference_at_a_time(self):
        entered, release = threading.Event(), threading.Event()
        original = self.model.run

        def slow_run(prepared):
            entered.set()
            release.wait(5)
            return original(prepared)

        self.model.run = slow_run
        first = []
        thread = threading.Thread(target=lambda: first.append(self.post()))
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(self.post().status_code, 409)
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(first[0].status_code, 200)

    def test_unknown_host_rejected_and_health_contains_no_inputs(self):
        response = self.client.get("/health", headers={"host": "not-local.example"})
        self.assertEqual(response.status_code, 400)
        health = self.client.get("/health").json()
        self.assertEqual(health["revision"], run_d3.REVISION)
        self.assertTrue(health["text_only"])


class D3SourceTests(unittest.TestCase):
    def test_vendor_source_matches_reviewed_pin(self):
        for name, digest in run_d3.VENDOR_HASHES.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((run_d3.ROOT / "vendor/d3" / name).read_bytes()).hexdigest(), digest)

    def test_missing_or_unpinned_checkpoint_rejected_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_d3.verify_local_sources(Path(directory) / "missing")
            (Path(directory) / "MODEL_MANIFEST.json").write_text("{}")
            with self.assertRaises(ValueError):
                run_d3.verify_local_sources(directory)


if __name__ == "__main__":
    unittest.main()
