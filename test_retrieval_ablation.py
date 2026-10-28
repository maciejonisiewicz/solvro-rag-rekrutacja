"""Lightweight offline tests for retrieval_ablation.py. No ML downloads."""

import unittest
from types import SimpleNamespace

from retrieval_ablation import evaluate_rankings, ranked_reviews_from_scores


def meta(review, paper):
    return {"review_id": review, "paper_id": paper, "conference": "ICLR 2024"}


class TestFairAblation(unittest.TestCase):
    def test_best_chunk_wins_but_review_appears_once(self):
        scores = [0.2, 0.8, 0.7, 0.1]
        metadata = [meta("R1", "P1"), meta("R1", "P1"), meta("R2", "P2"), meta("R2", "P2")]
        ranked = ranked_reviews_from_scores(scores, metadata)
        self.assertEqual(len(ranked), 2)
        self.assertEqual([v["metadata"]["review_id"] for v in ranked], ["R1", "R2"])
        self.assertEqual(ranked[0]["score"], 0.8)

    def test_ties_and_filter_are_stable(self):
        a, b = meta("R1", "P1"), meta("R2", "P2")
        b["conference"] = "NeurIPS 2024"
        ranked = ranked_reviews_from_scores([0.0, 0.0], [a, b])
        self.assertEqual([x["metadata"]["review_id"] for x in ranked], ["R1", "R2"])
        only_one = ranked_reviews_from_scores([0, 0], [a, b], conference_filter="ICLR 2024")
        self.assertEqual(len(only_one), 1)

    def test_different_review_and_paper_denominators(self):
        questions = [
            SimpleNamespace(id=1, has_data=True, target_paper_id="P1", target_review_id="R1", query="q1"),
            SimpleNamespace(id=2, has_data=True, target_paper_id="P2", target_review_id=None, query="q2"),
        ]
        methods = {"example": lambda q: [
            {"metadata": meta("R0", "P2")},
            {"metadata": meta("R1", "P1")},
        ] if q == "q1" else [{"metadata": meta("R0", "P2")}]}
        metrics = evaluate_rankings(questions, methods)["method_comparison"]["example"]
        self.assertEqual(metrics["paper_hit_at_5"], 1.0)
        self.assertEqual(metrics["review_hit_at_5"], 1.0)
        self.assertEqual(metrics["review_mrr_at_5"], 0.5)
        self.assertEqual(metrics["review_ndcg_at_5"], 0.631)

    def test_mismatched_lengths_fail(self):
        with self.assertRaises(ValueError):
            ranked_reviews_from_scores([0.1], [meta("R1", "P1"), meta("R2", "P2")])


if __name__ == "__main__":
    unittest.main()
