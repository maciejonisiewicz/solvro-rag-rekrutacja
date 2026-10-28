"""Offline tests for review chunking and review-level vector aggregation.

Run: python -m unittest -v test_chunking
Requires test_rag_index.py from the project (fake HF/Chroma components).
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_rag_index import FakeClient, FakeSentenceTransformer, load_project_rag


class TestReviewChunking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag_module = load_project_rag()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data_dir = Path(self.temporary.name)
        papers = [
            {"paper_id": "p1", "title": "Paper one", "abstract": "An author claim",
             "keywords": [], "url": "https://example.test/p1"},
            {"paper_id": "p2", "title": "Paper two", "abstract": "Another author claim",
             "keywords": [], "url": "https://example.test/p2"},
        ]
        reviews = [
            {"review_id": "r1", "paper_id": "p1", "conference": "ICLR 2024",
             "forum_url": "https://example.test/r1", "scores": {},
             "text_fields": {
                 "weaknesses": "longcriticism " * 700,
                 "questions": "final distinctive question about reproducibility",
             }},
            {"review_id": "r2", "paper_id": "p2", "conference": "NeurIPS 2024",
             "forum_url": "https://example.test/r2", "scores": {},
             "text_fields": {"weaknesses": "short criticism"}},
        ]
        (self.data_dir / "papers.json").write_text(json.dumps(papers), encoding="utf-8")
        (self.data_dir / "reviews.json").write_text(json.dumps(reviews), encoding="utf-8")
        self.client = FakeClient()
        self.patcher = patch.object(self.rag_module.chromadb, "PersistentClient", return_value=self.client)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        FakeSentenceTransformer.fail_next = False
        self.engine = self.rag_module.LiteratureReviewRAG(data_dir=self.data_dir)

    def test_long_review_is_chunked_and_attribution_is_preserved(self):
        self.assertEqual(len(self.engine.corpus_metadata), 2)
        self.assertGreater(len(self.engine.chunk_documents), 2)
        self.assertEqual(len(self.engine.chunk_documents), self.engine.collection.count())
        self.assertEqual(len(set(self.engine.chunk_ids)), len(self.engine.chunk_ids))
        self.assertTrue(all(len(chunk.split()) <= 192 for chunk in self.engine.chunk_documents))
        first_review = [
            (text, meta)
            for text, meta in zip(self.engine.chunk_documents, self.engine.chunk_metadata)
            if meta["review_id"] == "r1"
        ]
        self.assertGreater(len(first_review), 1)
        self.assertTrue(any("reproducibility" in text for text, _ in first_review))
        self.assertTrue(all(meta["paper_id"] == "p1" for _, meta in first_review))
        self.assertTrue(all(meta["chunk_count"] == len(first_review) for _, meta in first_review))
        self.assertEqual(len(self.engine.get_reviews_for_paper("p1")), 1)

    def test_vector_returns_distinct_reviews_and_best_chunk(self):
        first = [m for m in self.engine.chunk_metadata if m["review_id"] == "r1"]
        second = next(m for m in self.engine.chunk_metadata if m["review_id"] == "r2")
        self.assertGreaterEqual(len(first), 2)
        fake_query = Mock(return_value={
            "ids": [[first[0]["chunk_id"], first[1]["chunk_id"], second["chunk_id"]]],
            "metadatas": [[first[0], first[1], second]],
            "documents": [["first r1 chunk", "best r1 chunk", "r2 chunk"]],
            "distances": [[0.2, 0.1, 0.3]],
        })
        self.engine.collection.query = fake_query
        result = self.engine.search_vector("query", top_k=2, conference_filter="ICLR 2024")
        self.assertEqual([item["metadata"]["review_id"] for item in result], ["r1", "r2"])
        self.assertEqual(result[0]["text_snippet"], "best r1 chunk")
        self.assertAlmostEqual(result[0]["score"], 0.9)
        self.assertEqual(fake_query.call_args.kwargs["where"], {"conference": "ICLR 2024"})


if __name__ == "__main__":
    unittest.main()
