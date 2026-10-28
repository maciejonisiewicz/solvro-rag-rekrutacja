"""Synthetic regression tests; no data download or model inference."""
import tempfile
import unittest
from pathlib import Path

from evidence_gate import has_joint_model_support, required_model_entities
# Import the assistant with a lightweight rag_system stand-in. The tests
# exercise only the decision logic, so no ChromaDB or model is needed.
import importlib.util
import sys
import types
from unittest.mock import patch

_fake_rag = types.ModuleType("rag_system")
_fake_rag.LiteratureReviewRAG = object
_spec = importlib.util.spec_from_file_location(
    "agent_rag_evidence_test", Path(__file__).parent / "agent_rag.py"
)
_mod = importlib.util.module_from_spec(_spec)
with patch.dict(sys.modules, {"rag_system": _fake_rag, "agent_rag_evidence_test": _mod}):
    _spec.loader.exec_module(_mod)
    _mod.RAGQueryPayload.model_rebuild(_types_namespace=vars(_mod))
LiteratureReviewAssistant = _mod.LiteratureReviewAssistant


def result(text: str, score: float = 0.9, review_id: str = "R1") -> dict:
    return {
        "score": score,
        "metadata": {
            "review_id": review_id, "paper_id": "P" + review_id,
            "conference": "ICLR 2024", "title": "An ML paper",
        },
        "text_snippet": text,
    }


class FakeRAG:
    def __init__(self, rows):
        self.rows = rows

    def search_vector(self, query, top_k, conference_filter=None):
        return self.rows[:top_k]

    def search_hybrid(self, query, top_k, conference_filter=None):
        return self.rows[:top_k]

    def get_reviews_for_paper(self, paper_id):
        return []


class EvidenceGateTests(unittest.TestCase):
    def test_palm8b_without_space(self):
        self.assertEqual(required_model_entities("PaLM 8B"), ("palm 8b",))
        self.assertTrue(has_joint_model_support("Compare results on PaLM 8B", [result("Only PaLM8B was used.")]))

    def test_versioned_models_normalize_and_do_not_confuse_versions(self):
        self.assertEqual(required_model_entities("Compare GPT-6 with Claude 5"),
                         ("gpt 6", "claude 5"))
        self.assertFalse(has_joint_model_support(
            "Compare GPT-6 with Claude 5", [result("GPT - 4 and Claude 5")]))

    def test_both_models_must_occur_in_same_source(self):
        query = "Compare GPT-4 and Claude 3"
        self.assertFalse(has_joint_model_support(query, [
            result("GPT-4 was used.", review_id="R1"),
            result("Claude 3 was used.", review_id="R2"),
        ]))
        self.assertTrue(has_joint_model_support(query, [
            result("Claude 3 and GPT - 4 were compared in an ablation.")]))

    def test_query_without_explicit_model_version_remains_unchanged(self):
        self.assertTrue(has_joint_model_support("ROUGE-L in Equation 3", []))

    def test_agent_abstains_despite_high_cosine_without_named_models(self):
        query = "Which paper compares GPT-6 and Claude 5 energy use?"
        with tempfile.TemporaryDirectory() as tmp:
            assistant = LiteratureReviewAssistant(
                rag=FakeRAG([result("GPT-4 is a widely used baseline", score=0.85)]),
                data_dir=Path(tmp), top_k=3,
            )
            payload = assistant.run_agent(query)
            self.assertFalse(payload.has_sufficient_data)
            self.assertEqual(payload.sources, [])
            self.assertIsNone(payload.constructed_prompt)
            self.assertIn("Abstain", payload.agent_trace[-1].decision)
            direct = assistant.process_query(query)
            self.assertFalse(direct.has_sufficient_data)

    def test_agent_accepts_synthetic_joint_model_evidence(self):
        query = "Which paper compares GPT-4 and Claude 3?"
        with tempfile.TemporaryDirectory() as tmp:
            assistant = LiteratureReviewAssistant(
                rag=FakeRAG([result("Experimental comparison: GPT-4 vs Claude 3.")]),
                data_dir=Path(tmp), top_k=3,
            )
            payload = assistant.run_agent(query)
            self.assertTrue(payload.has_sufficient_data)
            self.assertEqual(len(payload.sources), 1)


if __name__ == "__main__":
    unittest.main()
