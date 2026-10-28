import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_synthesis import synthesize


def source(abstract_only=False):
    return SimpleNamespace(
        title="A paper", paper_url="https://openreview.net/forum?id=p1",
        review_url="" if abstract_only else "https://openreview.net/forum?id=r1",
        is_abstract_only=abstract_only,
    )


def payload(*, has_data=True, sources=None, supplementary=None):
    return SimpleNamespace(
        has_sufficient_data=has_data,
        constructed_prompt="CONTEXT [SOURCE 1] ... USER QUESTION ...",
        sources=[source()] if sources is None else sources,
        supplementary_sources=[] if supplementary is None else supplementary,
    )


class SynthesisTests(unittest.TestCase):
    def test_agent_abstention_never_calls_model(self):
        def forbidden(_):
            raise AssertionError("Model must not be called")
        res = synthesize(payload(has_data=False), model_call=forbidden)
        self.assertEqual(res.status, "insufficient_evidence")

    def test_no_key_falls_back_without_importing_pydantic_ai(self):
        with patch.dict(os.environ, {}, clear=True):
            res = synthesize(payload())
        self.assertEqual(res.status, "retrieval_only")
        self.assertIn("CONTEXT", res.prompt)

    def test_author_claim_cites_paper_url_from_metadata(self):
        res = synthesize(payload(), model_call=lambda _: {
            "cannot_answer": False, "claims": [
                {"text": "The authors make a claim.", "origin": "author", "source_numbers": [1]}
            ]
        })
        self.assertEqual(res.status, "generated")
        self.assertIn("[SOURCE 1]", res.answer)
        self.assertIn("https://openreview.net/forum?id=p1", res.answer)
        self.assertEqual(res.validated_source_numbers, [1])

    def test_reviewer_claim_cites_review_url(self):
        res = synthesize(payload(), model_call=lambda _: {
            "cannot_answer": False, "claims": [
                {"text": "A reviewer notes a limitation.", "origin": "reviewer", "source_numbers": [1]}
            ]
        })
        self.assertEqual(res.status, "generated")
        self.assertIn("https://openreview.net/forum?id=r1", res.answer)

    def test_invalid_source_number_rejects_entire_answer(self):
        res = synthesize(payload(), model_call=lambda _: {
            "cannot_answer": False, "claims": [
                {"text": "Unsupported.", "origin": "reviewer", "source_numbers": [999]}
            ]
        })
        self.assertEqual(res.status, "invalid_citations")
        self.assertIsNone(res.answer)

    def test_abstract_only_cannot_support_reviewer_claim(self):
        res = synthesize(payload(sources=[source(abstract_only=True)]), model_call=lambda _: {
            "cannot_answer": False, "claims": [
                {"text": "Reviewer states.", "origin": "reviewer", "source_numbers": [1]}
            ]
        })
        self.assertEqual(res.status, "invalid_citations")

    def test_supplementary_source_numbering_is_preserved(self):
        res = synthesize(payload(supplementary=[source()]), model_call=lambda _: {
            "cannot_answer": False, "claims": [
                {"text": "Extra review mentions it.", "origin": "reviewer", "source_numbers": [2]}
            ]
        })
        self.assertEqual(res.status, "generated")
        self.assertEqual(res.validated_source_numbers, [2])

    def test_llm_abstention_does_not_fabricate_claim(self):
        res = synthesize(payload(), model_call=lambda _: {
            "cannot_answer": True, "reason": "No support in provided snippets.", "claims": []
        })
        self.assertEqual(res.status, "insufficient_evidence")
        self.assertIn("No support", res.answer)

    def test_inconsistent_abstention_is_rejected(self):
        res = synthesize(payload(), model_call=lambda _: {
            "cannot_answer": True, "reason": "No support", "claims": [
                {"text": "Yet a fact.", "origin": "author", "source_numbers": [1]}
            ]
        })
        self.assertEqual(res.status, "invalid_citations")

    def test_provider_exception_keeps_retrieval_available(self):
        def broken(_):
            raise ConnectionError("private-secret")
        res = synthesize(payload(), model_call=broken)
        self.assertEqual(res.status, "llm_error")
        self.assertNotIn("private-secret", res.reason)

    def test_review_citation_does_not_satisfy_author_link(self):
        bad = source()
        bad.paper_url = ""
        res = synthesize(payload(sources=[bad]), model_call=lambda _: {
            "cannot_answer": False, "claims": [
                {"text": "Author claims this.", "origin": "author", "source_numbers": [1]}
            ]
        })
        self.assertEqual(res.status, "invalid_citations")


if __name__ == "__main__":
    unittest.main()
