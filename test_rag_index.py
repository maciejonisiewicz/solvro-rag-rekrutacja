"""Offline index-freshness regression tests. No HF/Chroma downloads required."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class FakeEmbeddings(list):
    def tolist(self):
        return list(self)


class FakeSentenceTransformer:
    models = []
    fail_next = False

    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = 0
        self.models.append(self)

    def encode(self, documents, show_progress_bar=False):
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Simulated model failure")
        return FakeEmbeddings([[float(i), 0.5] for i in range(len(documents))])


class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus


class FakeCollection:
    def __init__(self, name, metadata=None):
        self.name = name
        self.metadata = metadata
        self.ids = []
        self.records = []

    def count(self):
        return len(self.ids)

    def add(self, ids, documents, embeddings, metadatas):
        self.ids = list(ids)
        self.records = list(zip(ids, documents, embeddings, metadatas))


class FakeClient:
    def __init__(self):
        self.collections = {}
        self.deletions = 0

    def get_or_create_collection(self, name, metadata=None):
        if name not in self.collections:
            self.collections[name] = FakeCollection(name, metadata)
        return self.collections[name]

    def delete_collection(self, name):
        del self.collections[name]
        self.deletions += 1

    def create_collection(self, name, metadata=None):
        if name in self.collections:
            raise RuntimeError("Already exists")
        self.collections[name] = FakeCollection(name, metadata)
        return self.collections[name]


def load_project_rag():
    chroma = types.ModuleType("chromadb")
    bm25 = types.ModuleType("rank_bm25")
    st = types.ModuleType("sentence_transformers")
    chroma.PersistentClient = lambda path: None
    bm25.BM25Okapi = FakeBM25
    st.SentenceTransformer = FakeSentenceTransformer
    deps = {"chromadb": chroma, "rank_bm25": bm25, "sentence_transformers": st}
    path = Path(__file__).resolve().parent / "rag_system.py"
    spec = importlib.util.spec_from_file_location("rag_system_index_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, deps):
        spec.loader.exec_module(module)
    return module


class TestIndexFreshness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = load_project_rag()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name)
        self.client = FakeClient()
        self.addCleanup(patch.stopall)
        patch.object(self.rag.chromadb, "PersistentClient", return_value=self.client).start()
        FakeSentenceTransformer.models.clear()
        FakeSentenceTransformer.fail_next = False
        self.write_data()

    def write_data(self, abstract="Original author abstract", weakness="Original criticism"):
        papers = [{
            "paper_id": "paper-1", "title": "A test paper", "abstract": abstract,
            "keywords": ["test"], "url": "https://openreview.net/forum?id=paper-1",
        }]
        reviews = [{
            "paper_id": "paper-1", "review_id": "review-1", "conference": "ICLR 2024",
            "forum_url": "https://openreview.net/forum?id=paper-1",
            "text_fields": {"weaknesses": weakness, "strengths": "Some strengths"},
            "scores": {"rating": "5"},
        }]
        (self.data_dir / "papers.json").write_text(json.dumps(papers), encoding="utf8")
        (self.data_dir / "reviews.json").write_text(json.dumps(reviews), encoding="utf8")

    def initialize(self, model="all-MiniLM-L6-v2"):
        return self.rag.LiteratureReviewRAG(data_dir=self.data_dir, embedding_model_name=model)

    def test_initial_index_created(self):
        engine = self.initialize()
        self.assertEqual(engine.collection.count(), 1)
        self.assertEqual(len(engine.collection.metadata["corpus_fingerprint"]), 64)
        self.assertEqual(engine.collection.metadata["hnsw:space"], "cosine")
        self.assertEqual(FakeSentenceTransformer.models[-1].calls, 1)

    def test_unchanged_data_uses_existing_index(self):
        self.initialize()
        deleted_before = self.client.deletions
        engine = self.initialize()
        self.assertEqual(self.client.deletions, deleted_before)
        self.assertEqual(FakeSentenceTransformer.models[-1].calls, 0)
        self.assertEqual(engine.collection.count(), 1)

    def test_review_change_forces_rebuild(self):
        old = self.initialize().collection.metadata["corpus_fingerprint"]
        self.write_data(weakness="Updated criticism")
        new_engine = self.initialize()
        self.assertNotEqual(new_engine.collection.metadata["corpus_fingerprint"], old)
        self.assertEqual(self.client.deletions, 2)  # initial empty collection + replaced one

    def test_abstract_change_forces_rebuild(self):
        old = self.initialize().collection.metadata["corpus_fingerprint"]
        self.write_data(abstract="Changed author abstract")
        self.assertNotEqual(self.initialize().collection.metadata["corpus_fingerprint"], old)

    def test_embedding_model_name_change_forces_rebuild(self):
        old = self.initialize().collection.metadata["corpus_fingerprint"]
        self.assertNotEqual(self.initialize(model="another-model").collection.metadata["corpus_fingerprint"], old)

    def test_legacy_collection_without_fingerprint_is_rebuilt(self):
        existing = self.client.create_collection("openreview_reviews_v2", {"hnsw:space": "cosine"})
        existing.add(["review-1"], ["stale"], [[0.2]], [{"paper_id": "paper-1"}])
        fresh = self.initialize()
        self.assertIn("corpus_fingerprint", fresh.collection.metadata)
        self.assertEqual(fresh.collection.count(), 1)
        self.assertEqual(self.client.deletions, 1)

    def test_embedding_failure_preserves_existing_index(self):
        existing = self.initialize().collection
        self.write_data(weakness="New criticism")
        FakeSentenceTransformer.fail_next = True
        with self.assertRaisesRegex(RuntimeError, "Simulated model failure"):
            self.initialize()
        self.assertIs(self.client.collections["openreview_reviews_v2"], existing)
        self.assertEqual(existing.count(), 1)


if __name__ == "__main__":
    unittest.main()
