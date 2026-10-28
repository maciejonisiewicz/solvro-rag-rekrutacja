"""Offline tests of Streamlit->agent->LLM wiring. No API key, network or Streamlit needed."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from llm_synthesis import SynthesisResult
from streamlit_llm_bridge import run_pipeline


class FakeAssistant:
    def __init__(self, *, has_data=True):
        self.top_k = 4
        self.similarity_threshold = 0.38
        self.calls = []
        self.has_data = has_data

    def run_agent(self, query, conference=None):
        self.calls.append((query, conference))
        return SimpleNamespace(
            has_sufficient_data=self.has_data,
            constructed_prompt="EVIDENCE: [SOURCE 1] ..." if self.has_data else None,
        )


class StreamlitBridgeTests(unittest.TestCase):
    def test_offline_never_calls_llm(self):
        a = FakeAssistant()
        llm = Mock(side_effect=AssertionError("No API calls in offline mode"))
        payload, result = run_pipeline(a, "  review question  ", use_gemini=False, synthesis_fn=llm)
        self.assertTrue(payload.has_sufficient_data)
        self.assertEqual(result.status, "retrieval_only")
        self.assertEqual(a.calls, [("review question", None)])
        llm.assert_not_called()

    def test_opt_in_invokes_synthesis_once_with_same_payload(self):
        a = FakeAssistant()
        llm = Mock(return_value=SynthesisResult(status="generated", answer="Citation [SOURCE 1]"))
        payload, result = run_pipeline(
            a, "question", conference="ICLR 2024", top_k=5,
            similarity_threshold=0.42, use_gemini=True,
            model_name="google:gemini-3.5-flash-lite", synthesis_fn=llm,
        )
        self.assertEqual(result.status, "generated")
        self.assertEqual(a.top_k, 5)
        self.assertEqual(a.similarity_threshold, 0.42)
        self.assertEqual(a.calls, [("question", "ICLR 2024")])
        llm.assert_called_once_with(payload, model_name="google:gemini-3.5-flash-lite")

    def test_abstaining_agent_never_calls_model_even_when_enabled(self):
        a = FakeAssistant(has_data=False)
        llm = Mock(side_effect=AssertionError("Abstention must block calls"))
        payload, result = run_pipeline(a, "Impossible question", use_gemini=True, synthesis_fn=llm)
        self.assertFalse(payload.has_sufficient_data)
        self.assertEqual(result.status, "insufficient_evidence")
        llm.assert_not_called()

    def test_missing_prompt_falls_back_at_synthesis_layer(self):
        a = FakeAssistant()
        res = SynthesisResult(status="retrieval_only", reason="no key")
        llm = Mock(return_value=res)
        _, actual = run_pipeline(a, "question", use_gemini=True, synthesis_fn=llm)
        self.assertEqual(actual.status, "retrieval_only")

    def test_query_validation_does_not_touch_agent(self):
        a = FakeAssistant()
        with self.assertRaises(ValueError):
            run_pipeline(a, "  ")
        self.assertEqual(a.calls, [])

    def test_invalid_slider_settings_do_not_touch_agent(self):
        a = FakeAssistant()
        with self.assertRaises(ValueError):
            run_pipeline(a, "q", top_k=0)
        with self.assertRaises(ValueError):
            run_pipeline(a, "q", similarity_threshold=0)
        self.assertEqual(a.calls, [])


if __name__ == "__main__":
    unittest.main()
