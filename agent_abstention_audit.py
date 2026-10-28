"""Audit agent acceptance separately from the vector-score threshold.

Q106 was used to motivate the gate and is not an untouched holdout for it.
This tool prints diagnostics only and does not replace the v3 evaluation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from evaluate import load_ground_truth_dataset
from agent_rag import LiteratureReviewAssistant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/eval_questions.json")
    args = parser.parse_args()
    questions = load_ground_truth_dataset(args.questions)
    assistant = LiteratureReviewAssistant(top_k=5, max_steps=3)
    folder = Path("data/experiments")
    folder.mkdir(exist_ok=True, parents=True)
    assistant.context_log_path = folder / "agent_audit_last_context.json"
    assistant.trace_log_path = folder / "agent_audit_last_trace.json"
    positive = [q for q in questions if q.has_data]
    negative = [q for q in questions if not q.has_data]
    accepted_positive = rejected_negative = 0
    for q in questions:
        p = assistant.run_agent(q.query)
        got = bool(p.has_sufficient_data)
        accepted_positive += int(got and q.has_data)
        rejected_negative += int((not got) and (not q.has_data))
        print(f"Q{q.id:03d} expected={'YES' if q.has_data else 'NO'} "
              f"agent={'YES' if got else 'NO'} "
              f"{'OK' if got == q.has_data else 'MISMATCH'}")
    print(f"Answerable accepted: {accepted_positive}/{len(positive)}")
    print(f"Unanswerable rejected: {rejected_negative}/{len(negative)}")
    print("This is an agent decision metric, distinct from the vector-score threshold.")
    print("Q106 is a development diagnostic, NOT an independent holdout for this gate.")


if __name__ == "__main__":
    main()
