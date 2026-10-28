"""Offline regressions for reviewer evidence and the abstract-only ablation.

Run from the repository root:
  python -m unittest -v test_context_integrity

Uses the fake model and Chroma objects from test_rag_index: no network needed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_rag_index import FakeBM25, FakeClient, FakeSentenceTransformer, load_project_rag


class TestContextIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag_module = load_project_rag()
        # agent_rag imports rag_system at module load. Supply the same fake-backed
        # module so no real Chroma or HuggingFace dependency is imported here.
        agent_path = Path(__file__).resolve().parent / "agent_rag.py"
        spec = importlib.util.spec_from_file_location("agent_rag_context_test", agent_path)
        cls.agent_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"rag_system": cls.rag_module}):
            spec.loader.exec_module(cls.agent_module)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        long_criticism = "critical weakness " + "very long detailed objection " * 40
        self.criticism = long_criticism
        papers = [
            {
                "paper_id": "paper-a", "title": "Alpha Model", "abstract": "Abstract alpha fact",
                "conference": "ICLR 2024", "url": "https://example.test/paper-a", "keywords": ["alpha"],
            },
            {
                "paper_id": "paper-b", "title": "Beta Model", "abstract": "Abstract beta fact",
                "conference": "NeurIPS 2024", "url": "https://example.test/paper-b", "keywords": ["beta"],
            },
        ]
        reviews = [
            {
                "review_id": "review-a1", "paper_id": "paper-a", "conference": "ICLR 2024",
                "forum_url": "https://example.test/review-a1", "scores": {"rating": 6},
                "text_fields": {
                    "weaknesses": long_criticism,
                    "strengths": "clear mathematical formulation",
                    "summary": "summary unique evidence",
                    "questions": "question unique evidence about sample size",
                    "limitations": "limitations unique evidence",
                },
            },
            {
                "review_id": "review-a2", "paper_id": "paper-a", "conference": "ICLR 2024",
                "forum_url": "https://example.test/review-a2", "scores": {},
                "text_fields": {"weaknesses": "other criticism"},
            },
            {
                "review_id": "review-b1", "paper_id": "paper-b", "conference": "NeurIPS 2024",
                "forum_url": "https://example.test/review-b1", "scores": {},
                "text_fields": {"summary": "beta review"},
            },
        ]
        (directory / "papers.json").write_text(json.dumps(papers), encoding="utf-8")
        (directory / "reviews.json").write_text(json.dumps(reviews), encoding="utf-8")
        self.chroma = FakeClient()
        self.addCleanup(patch.stopall)
        patch.object(self.rag_module.chromadb, "PersistentClient", return_value=self.chroma).start()
        self.engine = self.rag_module.LiteratureReviewRAG(data_dir=directory)
        self.assistant = self.agent_module.LiteratureReviewAssistant(rag=self.engine, data_dir=directory)

    def test_reviewer_evidence_preserved_without_truncation(self):
        first = self.engine.corpus_metadata[0]
        self.assertEqual(first["reviewer_weaknesses"], self.criticism)
        self.assertEqual(first["reviewer_summary"], "summary unique evidence")
        self.assertEqual(first["reviewer_questions"], "question unique evidence about sample size")
        self.assertEqual(first["reviewer_limitations"], "limitations unique evidence")
        sources = self.assistant._to_sources([
            {"score": 0.7, "metadata": first, "score_type": "cosine_similarity"}
        ])
        prompt = self.assistant.build_prompt("Explain the criticism", sources)
        for expected in (
            self.criticism, "summary unique evidence",
            "question unique evidence about sample size", "limitations unique evidence",
            "https://example.test/review-a1", "Abstract alpha fact",
        ):
            self.assertIn(expected, prompt)

    def test_one_abstract_per_paper_and_no_reviewer_metadata(self):
        self.assertEqual(len(self.engine.corpus_metadata), 3)
        self.assertEqual(len(self.engine.corpus_abstracts_only), 2)
        self.assertEqual(len(self.engine.abstract_metadata), 2)
        # This score stub is sufficient to exercise filtering and formatting.
        with patch.object(FakeBM25, "get_scores", return_value=[10.0, 1.0], create=True):
            found = self.engine.search_abstracts_only("alpha", top_k=5)
        self.assertEqual(len(found), 2)
        self.assertEqual([item["metadata"]["paper_id"] for item in found], ["paper-a", "paper-b"])
        self.assertEqual(found[0]["score_type"], "bm25_abstract")
        self.assertNotIn("review_id", found[0]["metadata"])
        self.assertNotIn("reviewer_weaknesses", found[0]["metadata"])
        self.assertNotIn("review_url", found[0]["metadata"])
        sources = self.assistant._to_sources(found)
        self.assertTrue(all(source.is_abstract_only for source in sources))
        self.assertTrue(all(source.review_id == "" for source in sources))
        prompt = self.assistant.build_prompt("alpha", sources)
        self.assertIn("Abstract alpha fact", prompt)
        for forbidden in (
            "critical weakness", "summary unique evidence", "Reviewer Rating:",
            "Reviewer Weaknesses:", "Review Link:", "Review ID:",
            "https://example.test/review-a1",
        ):
            self.assertNotIn(forbidden, prompt)

    def test_abstract_conference_filter(self):
        with patch.object(FakeBM25, "get_scores", return_value=[10.0, 1.0], create=True):
            results = self.engine.search_abstracts_only("alpha", conference_filter="NeurIPS 2024")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["paper_id"], "paper-b")

    def test_report_rankings_still_expose_review_ids(self):
        # The normal review retrieval still needs review IDs for Hit@5/MRR.
        # No search call needed: test the unchanged source conversion contract.
        results = self.assistant._to_sources([
            {"score": 0.55, "metadata": self.engine.corpus_metadata[0],
             "score_type": "rrf"}
        ])
        self.assertEqual(results[0].review_id, "review-a1")
        self.assertEqual(results[0].score_type, "rrf")
        self.assertFalse(results[0].is_abstract_only)


if __name__ == "__main__":
    unittest.main()
