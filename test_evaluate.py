"""Lightweight evaluation regression tests (no model download or network).

Run: python -m unittest -v test_evaluate.py
"""

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate


def result(review, paper, score=0.7, conference="ICLR 2024"):
    return {
        "score": score,
        "metadata": {
            "review_id": review,
            "paper_id": paper,
            "title": f"Paper {paper}",
            "conference": conference,
        },
    }


class TestPureMetrics(unittest.TestCase):
    def test_target_at_first_rank(self):
        self.assertEqual(evaluate.compute_ndcg_at_k([result("R1", "P1")], "P1", "R1"), 1)

    def test_target_at_second_rank(self):
        data = [result("R0", "P2"), result("R1", "P1")]
        self.assertAlmostEqual(
            evaluate.compute_ndcg_at_k(data, "P1", "R1"),
            1 / math.log2(3),
        )
        self.assertEqual(evaluate.find_rank_of_target(data, "P1", "R1"), (2, 2))

    def test_other_reviews_of_same_paper_are_not_auto_relevant(self):
        data = [result("OTHER", "P1"), result("R1", "P1")]
        self.assertAlmostEqual(
            evaluate.compute_ndcg_at_k(
                data, "P1", "R1", relevant_review_ids={"OTHER", "R1"}
            ),
            1 / math.log2(3),
        )

    def test_no_target_and_outside_top5(self):
        data = [result(f"R{i}", f"P{i}") for i in range(6)]
        self.assertEqual(evaluate.compute_ndcg_at_k(data, "P5", "R5", k=5), 0)
        self.assertIsNone(evaluate.compute_ndcg_at_k(data, "P5", None))

    def test_duplicate_review_counted_once(self):
        data = [result("R0", "P0"), result("R0", "P0"), result("R1", "P1")]
        self.assertEqual(len(evaluate._unique_review_results(data)), 2)
        self.assertEqual(evaluate.find_rank_of_target(data, "P1", "R1"), (2, 2))
        self.assertAlmostEqual(
            evaluate.compute_ndcg_at_k(data, "P1", "R1"), 1 / math.log2(3)
        )

    def test_review_must_belong_to_target_paper(self):
        self.assertEqual(
            evaluate.compute_ndcg_at_k([result("R1", "OTHER")], "P1", "R1"), 0
        )

    def test_abstract_only_paper_rank_no_review(self):
        self.assertEqual(
            evaluate.find_rank_of_target([result("R2", "P2")], "P2", None),
            (None, 1),
        )

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            evaluate.compute_ndcg_at_k([], "P1", "R1", k=0)

    def test_ground_truth_rejects_invalid_ids_and_empty_justification(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "ground_truth.json"
            questions = [
                dict(id=1, query="something", has_data=True,
                     target_paper_id="P1", target_review_id="R1", justification="")
            ]
            file.write_text(json.dumps(questions), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "justification"):
                evaluate.load_ground_truth_dataset(file)


class FakeRAG:
    def __init__(self):
        self.papers = {"P1": {}, "P2": {}}
        self.reviews = [
            {"paper_id": "P1", "review_id": "R1"},
            {"paper_id": "P2", "review_id": "R2"},
        ]

    def search_vector(self, query, top_k):
        if query == "out of scope":
            return [result("R2", "P2", score=0.2)]
        if query == "question one":
            return [result("R2", "P2", score=0.9), result("R1", "P1", score=0.8)]
        return [result("R2", "P2", score=0.8), result("R1", "P1", score=0.5)]

    def search_bm25(self, query, top_k):
        return [result("R1", "P1"), result("R2", "P2")] if query == "question one" \
            else [result("R2", "P2"), result("R1", "P1")]

    def search_hybrid(self, query, top_k):
        return self.search_bm25(query, top_k)

    def search_abstracts_only(self, query, top_k):
        return self.search_bm25(query, top_k)


class FakeAssistant:
    def __init__(self, **kwargs):
        self.context_log_path = None
        self.trace_log_path = None

    def run_agent(self, query):
        ordered = [result("R1", "P1"), result("R2", "P2")] if query == "question one" \
            else [result("R2", "P2"), result("R1", "P1")]
        sources = [SimpleNamespace(
            similarity_score=item["score"],
            review_id=item["metadata"]["review_id"],
            paper_id=item["metadata"]["paper_id"],
            conference=item["metadata"]["conference"],
            title=item["metadata"]["title"],
        ) for item in ordered]
        return SimpleNamespace(sources=sources)


class TestIntegrationWithFakeRetrieval(unittest.TestCase):
    def test_report_schema_ground_truth_immutability_and_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            questions_path = folder / "questions.json"
            report_path = folder / "results.json"
            qs = [
                dict(id=1, query="question one", has_data=True,
                     target_paper_id="P1", target_review_id="R1", justification="manual R1"),
                dict(id=2, query="question two", has_data=True,
                     target_paper_id="P2", target_review_id=None, justification="manual P2"),
                dict(id=3, query="out of scope", has_data=False,
                     target_paper_id=None, target_review_id=None, justification="no corpus support"),
            ]
            questions_path.write_text(json.dumps(qs), encoding="utf-8")
            before = questions_path.read_bytes()
            fake_agent = types.ModuleType("agent_rag")
            fake_agent.LiteratureReviewAssistant = FakeAssistant
            fake_rag = types.ModuleType("rag_system")
            fake_rag.LiteratureReviewRAG = FakeRAG
            with patch.dict(sys.modules, {"agent_rag": fake_agent, "rag_system": fake_rag}):
                report = evaluate.run_evaluation(
                    FakeRAG(), questions_path=questions_path,
                    report_path=report_path, save_report=True
                )

            self.assertEqual(before, questions_path.read_bytes())
            self.assertTrue(report_path.exists())
            self.assertEqual(report.evaluation_version, 3)
            self.assertEqual(report.metrics.vector_review_hit_at_5, 1)
            self.assertAlmostEqual(report.metrics.vector_review_mrr, 0.5)
            self.assertAlmostEqual(report.metrics.vector_ndcg_at_5, 0.631)
            self.assertEqual(report.metrics.vector_threshold_answerable_acceptance, 1)
            self.assertEqual(report.metrics.unanswerable_rejection_accuracy, 1)
            self.assertEqual(report.metrics.vector_threshold_balanced_accuracy, 1)
            self.assertIsNone(report.method_comparison[-1].ndcg_at_5)
            self.assertIsNone(report.questions_results[1].vector_review_hit_at_5)
            self.assertEqual(report.questions_results[0].vector_review_rank, 2)
            self.assertEqual(report.metrics.abstracts_only_paper_hit_at_5, 1)
            self.assertEqual(len(report.ground_truth_sha256 or ""), 64)
            evaluate.FullEvaluationReport.model_validate_json(report_path.read_text())


if __name__ == "__main__":
    unittest.main()
