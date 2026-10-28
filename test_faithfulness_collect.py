import unittest
from types import SimpleNamespace

from faithfulness_collect import build_audit, compute_summary, extract_claims


def fake_source():
    return SimpleNamespace(
        paper_id="p1", review_id="r1", title="Paper", conference="ICLR 2024",
        paper_url="https://openreview.net/forum?id=p1",
        review_url="https://openreview.net/forum?id=p1&noteId=r1",
        is_abstract_only=False, is_supplementary=False,
        author_abstract="Authors claim improvement", reviewer_summary="Review text",
        reviewer_strengths="Strong methods", reviewer_weaknesses="Reviewer flags ROUGE-L",
        reviewer_limitations="", reviewer_questions="",
    )


class FakeAssistant:
    def run_agent(self, query):
        return SimpleNamespace(query=query, has_sufficient_data=True,
                               constructed_prompt="Source and query", sources=[fake_source()],
                               supplementary_sources=[])


def fake_synthesis(payload, *, model_name):
    return SimpleNamespace(
        status="generated", reason="", model_name=model_name,
        answer="The reviewer criticizes ROUGE-L.  [SOURCE 1] (https://openreview.net/forum?id=p1&noteId=r1)",
        validated_source_numbers=[1],
    )


class TestFaithfulnessCollect(unittest.TestCase):
    def test_split_multiple_claims(self):
        parsed = extract_claims(
            "Claim A.  [SOURCE 1] (https://a)\n\n"
            "Claim B.  [SOURCE 2] (https://b); [SOURCE 1] (https://a)"
        )
        self.assertEqual([c["text"] for c in parsed], ["Claim A.", "Claim B."])
        self.assertEqual(parsed[1]["cited_source_numbers"], [2, 1])
        self.assertIsNone(parsed[0]["verdict"])

    def test_reject_missing_citation(self):
        with self.assertRaises(ValueError):
            extract_claims("No citation.")

    def test_only_answerable_questions_and_full_evidence(self):
        data = [
            {"id": 1, "query": "question 1", "has_data": True},
            {"id": 2, "query": "question 2", "has_data": False},
            {"id": 3, "query": "question 3", "has_data": True},
        ]
        doc = build_audit(data, assistant=FakeAssistant(), synthesis_fn=fake_synthesis,
                          model_name="fake", limit=1)
        self.assertEqual(len(doc["cases"]), 1)
        self.assertEqual(doc["cases"][0]["question_id"], 1)
        self.assertEqual(doc["cases"][0]["sources"][0]["reviewer_weaknesses"], "Reviewer flags ROUGE-L")
        self.assertEqual(doc["cases"][0]["claims"][0]["cited_source_numbers"], [1])

    def test_validate_bad_citation(self):
        def bad(payload, *, model_name):
            obj = fake_synthesis(payload, model_name=model_name)
            obj.validated_source_numbers = []
            return obj
        with self.assertRaises(ValueError):
            build_audit([{"id": 1, "query": "x", "has_data": True}],
                        assistant=FakeAssistant(), synthesis_fn=bad,
                        model_name="fake")

    def test_summary_counts_unreviewed(self):
        doc = build_audit([{"id": 1, "query": "x", "has_data": True}],
                          assistant=FakeAssistant(), synthesis_fn=fake_synthesis,
                          model_name="fake")
        self.assertIsNone(compute_summary(doc)["faithfulness_fraction"])
        doc["cases"][0]["claims"][0]["verdict"] = "supported"
        result = compute_summary(doc)
        self.assertEqual(result["faithfulness_fraction"], 1)
        self.assertEqual(result["supported_claims"], 1)
        self.assertEqual(result["reviewed_claims"], 1)

    def test_unrecognized_judgment_refused(self):
        doc = build_audit([{"id": 1, "query": "x", "has_data": True}],
                          assistant=FakeAssistant(), synthesis_fn=fake_synthesis,
                          model_name="fake")
        doc["cases"][0]["claims"][0]["verdict"] = "yes"
        with self.assertRaises(ValueError):
            compute_summary(doc)


if __name__ == "__main__":
    unittest.main()
