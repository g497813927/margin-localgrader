import importlib.util
import json
import unittest
from unittest.mock import Mock, patch

from grader import providers


class TrustedErrorTests(unittest.TestCase):
    def test_trusted_errors_preserve_existing_exception_contract(self):
        self.assertTrue(issubclass(providers.LocalInputError, ValueError))
        self.assertTrue(issubclass(providers.LocalRuntimeError, RuntimeError))
        with self.assertRaises(providers.LocalInputError):
            providers.validate_local_endpoint("https://example.com/v1/systemone")
        with self.assertRaises(providers.LocalRuntimeError):
            providers._model_path({"model": "models/does-not-exist-for-test"})

    def test_input_validation_happens_before_provider_invocation(self):
        with patch.object(providers, "_native_predict") as native:
            with self.assertRaises(providers.LocalInputError):
                providers.predict({}, "   ", {"quality": {}})
            with self.assertRaises(providers.LocalInputError):
                providers.predict({}, "Synthetic text", {})
        native.assert_not_called()

    def test_unexpected_preflight_error_is_not_marked_trusted(self):
        failure = ValueError("SYNTHETIC_SECRET_PREFLIGHT")
        agent = Mock()
        with patch.object(providers, "_load_laya", return_value=agent), \
                patch.object(providers, "_check_complete_input", side_effect=failure):
            with self.assertRaises(ValueError) as error:
                providers.predict({}, "Synthetic text", {"quality": {}})
        self.assertIs(error.exception, failure)
        self.assertNotIsInstance(error.exception, providers.LocalInputError)
        agent.predict.assert_not_called()

    def test_inference_failure_exposes_only_safe_adapter_guidance(self):
        agent = Mock()
        agent.predict.side_effect = RuntimeError("SYNTHETIC_SECRET_INFERENCE")
        with patch.object(providers, "_load_laya", return_value=agent), \
                patch.object(providers, "_check_complete_input", return_value=16):
            with self.assertRaises(providers.LocalRuntimeError) as error:
                providers.predict({}, "Synthetic text", {"quality": {}})
        self.assertIn("Local Laya inference failed", str(error.exception))
        self.assertNotIn("SYNTHETIC_SECRET_INFERENCE", str(error.exception))

    def test_probe_preserves_trusted_guidance_and_sanitizes_unknown_errors(self):
        for failure in (providers.LocalInputError("Safe input guidance"),
                        providers.LocalRuntimeError("Safe runtime guidance")):
            with self.subTest(failure=failure), patch.object(providers, "predict", side_effect=failure):
                self.assertEqual(providers.probe({}), {"ok": False, "message": str(failure)})
        for error_type in (ValueError, RuntimeError, Exception):
            with self.subTest(error_type=error_type), \
                    patch.object(providers, "predict", side_effect=error_type("SYNTHETIC_SECRET_PROBE")):
                result = providers.probe({})
                self.assertFalse(result["ok"])
                self.assertNotIn("SYNTHETIC_SECRET_PROBE", result["message"])


class EndpointValidationTests(unittest.TestCase):
    def test_localhost_is_pinned_to_literal_loopback(self):
        self.assertEqual(providers.validate_local_endpoint("http://localhost:1234/v1/systemone"),
                         "http://127.0.0.1:1234/v1/systemone")
        self.assertEqual(providers.validate_local_endpoint("http://[::1]:8000/v1/systemone"),
                         "http://[::1]:8000/v1/systemone")

    def test_nonlocal_and_ambiguous_destinations_rejected(self):
        urls = ["https://api.cloudflare.com/v1/systemone", "http://192.168.1.1/v1/systemone",
                "http://0.0.0.0/v1/systemone", "http://127.0.0.1.evil.example/v1/systemone",
                "http://2130706433/v1/systemone", "http://127.1/v1/systemone",
                "http://user:pass@127.0.0.1/v1/systemone", "http://localhost/v1/systemone?x=y",
                "http://localhost/v1/systemone?", "http://localhost/v1/systemone#",
                "http://localhost/v1/systemone#part", "http://[::1%eth0]/v1/systemone",
                "file:///tmp/model", "http://localhost", " http://localhost/v1/systemone",
                "http://localhost\\@evil.example/v1/systemone"]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                providers.validate_local_endpoint(url)

    def test_missing_checkpoint_fails_without_download(self):
        with self.assertRaisesRegex(RuntimeError, "weights are missing"):
            providers._model_path({"model": "models/does-not-exist-for-test"})


@unittest.skipUnless(importlib.util.find_spec("httpx"), "httpx is not installed")
class SystemOneTests(unittest.TestCase):
    def run_request(self, response):
        import httpx
        original_client = httpx.Client
        observed = {}

        def handler(request):
            observed["request"] = json.loads(request.content)
            observed["url"] = str(request.url)
            return response

        def client(**kwargs):
            observed["client_options"] = kwargs
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        questions = {"quality": {"type": "choice", "instructions": "Which level fits?",
                                 "criteria": {"zero": "Missing", "full": "Complete"}}}
        with patch.object(httpx, "Client", side_effect=client):
            result = providers.predict({"provider": "systemone", "endpoint": "http://localhost:1234/v1/systemone",
                                        "model": "test-model"}, "Synthetic text only", questions)
        return result, observed

    def test_request_contract_disables_proxy_and_redirects(self):
        import httpx
        result, observed = self.run_request(httpx.Response(200, json={"answers": {
            "quality": {"type": "choice", "choice": "full", "probabilities": {"zero": 0.1, "full": 0.9}}
        }}))
        self.assertFalse(observed["client_options"]["trust_env"])
        self.assertFalse(observed["client_options"]["follow_redirects"])
        self.assertEqual(observed["url"], "http://127.0.0.1:1234/v1/systemone")
        self.assertEqual(observed["request"]["state"], "Synthetic text only")
        self.assertEqual(observed["request"]["model"], "test-model")
        self.assertEqual(result["answers"]["quality"]["probabilities"]["full"], 0.9)

    def test_redirect_never_followed(self):
        import httpx
        with self.assertRaisesRegex(RuntimeError, "Redirects are disabled"):
            self.run_request(httpx.Response(307, headers={"location": "https://example.com"}))

    def test_errors_never_echo_response_body(self):
        import httpx
        with self.assertRaises(providers.LocalRuntimeError) as error:
            self.run_request(httpx.Response(401, text="SYNTHETIC_SECRET_ECHO"))
        self.assertIn("HTTP 401", str(error.exception))
        self.assertNotIn("SYNTHETIC_SECRET_ECHO", str(error.exception))

    def test_malformed_and_truncated_responses_rejected(self):
        import httpx
        for response in [httpx.Response(200, text="not json"), httpx.Response(200, json={}),
                         httpx.Response(200, json={"answers": {}, "usage": {"truncated": True}})]:
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                self.run_request(response)

    def test_probe_requires_valid_decision_probabilities(self):
        for answer in [None, {}, {"probabilities": {"test": 9, "other": -8}},
                       {"probabilities": {"test": 0.1, "other": 0.1}}]:
            with self.subTest(answer=answer), patch.object(providers, "predict", return_value={"answers": {"connection": answer}}):
                self.assertFalse(providers.probe({"provider": "systemone"})["ok"])


@unittest.skipUnless(importlib.util.find_spec("laya"), "optional Laya runtime is not installed")
class LayaCompleteInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from laya.agent import Agent
        backend = Tokenizer(WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[PAD]": 4}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]",
                                           sep_token="[SEP]", mask_token="[MASK]", pad_token="[PAD]")
        cls.agent = type("TokenizerOnlyAgent", (), {"tok": tokenizer, "_to_internal": staticmethod(Agent._to_internal)})()

    def test_long_level_is_rejected_before_laya_truncates_it(self):
        questions = {"quality": {"type": "choice", "instructions": "Assess quality",
                                 "criteria": {"zero": "Missing", "full": "evidence " * 60}}}
        with self.assertRaisesRegex(providers.LocalInputError, "48 tokens per level"):
            providers._check_complete_input(self.agent, "Short synthetic submission", questions, 8192)

    def test_question_tokens_count_toward_context(self):
        questions = {"quality": {"type": "choice", "instructions": "instruction " * 200,
                                 "criteria": {"zero": "Missing", "full": "Complete"}}}
        with self.assertRaisesRegex(providers.LocalInputError, "Nothing was silently truncated"):
            providers._check_complete_input(self.agent, "evidence " * 100, questions, 256)

    def test_long_instructions_are_preserved_by_expanding_head(self):
        from laya.common import build_head, encode_text
        questions = {"quality": {"type": "choice", "instructions": "instruction " * 400,
                                 "criteria": {"zero": "Missing", "full": "Complete"}}}
        budget = providers._check_complete_input(self.agent, "Short synthetic submission", questions, 8192)
        internal = self.agent._to_internal(questions["quality"])
        head, markers, stats = build_head(self.agent.tok, internal, budget)
        instruction_tokens = encode_text(self.agent.tok, "choice question: " + questions["quality"]["instructions"], add_special_tokens=False)["input_ids"]
        # The first separator follows every original instruction token.
        self.assertEqual(head[1:1 + len(instruction_tokens)], instruction_tokens)
        self.assertEqual(head[1 + len(instruction_tokens)], self.agent.tok.sep_token_id)
        self.assertIsNone(stats["tokens_per_option"])


if __name__ == "__main__":
    unittest.main()
