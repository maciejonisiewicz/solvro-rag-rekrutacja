"""Controlled, read-only ablation of lexical and vector retrieval.

Run in the project root (with .venv active):
    python retrieval_ablation.py

The default input is the existing PRIMARY (development) question set. This script does not modify the agent, question labels, or the main/holdout
evaluation reports. It creates a separate experiment JSON file. RAG init may
rebuild Chroma only if the index fingerprint is stale (normal project behavior).

Treat results as descriptive on a small, previously inspected set; don't use
frozen holdout outcomes for tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


def tokenize_bm25(text: str) -> list[str]:
    """Match the existing project's BM25 tokenization exactly."""
    return text.lower().split()


def ranked_reviews_from_scores(
    scores: Sequence[float],
    metadata: Sequence[dict[str, Any]],
    *,
    text_snippets: Sequence[str] | None = None,
    conference_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate scores at review level (MAX over chunks), then rank reviews.

    Ties are deterministically broken by first review occurrence in the corpus.
    Ranking is independent of how many chunks one review has.
    """
    if len(scores) != len(metadata):
        raise ValueError("Scores and metadata have different lengths")
    if text_snippets is not None and len(text_snippets) != len(metadata):
        raise ValueError("Snippets and metadata have different lengths")

    best: dict[str, dict[str, Any]] = {}
    for index, (score, meta) in enumerate(zip(scores, metadata)):
        if conference_filter and meta.get("conference") != conference_filter:
            continue
        review_id = str(meta["review_id"])
        value = float(score)
        if math.isnan(value):
            raise ValueError(f"NaN score for review {review_id}")
        if review_id not in best or value > best[review_id]["score"]:
            first_index = best[review_id]["first_index"] if review_id in best else index
            best[review_id] = {
                "score": value,
                "first_index": first_index,
                "metadata": meta,
                "text_snippet": text_snippets[index] if text_snippets is not None else "",
            }

    items = sorted(best.values(), key=lambda item: (-item["score"], item["first_index"]))
    return [
        {"score": item["score"], "metadata": item["metadata"], "text_snippet": item["text_snippet"]}
        for item in items
    ]


def bm25_results(
    bm25_model: Any,
    query: str,
    metadata: Sequence[dict[str, Any]],
    texts: Sequence[str],
) -> list[dict[str, Any]]:
    return ranked_reviews_from_scores(
        bm25_model.get_scores(tokenize_bm25(query)), metadata, text_snippets=texts
    )


def evaluate_rankings(
    questions: Sequence[Any],
    methods: dict[str, Any],
) -> dict[str, Any]:
    """Compute exactly the same rank-based metrics as evaluate.py."""
    from evaluate import compute_ndcg_at_k, find_rank_of_target

    answerable = [q for q in questions if q.has_data]
    annotated = [q for q in answerable if q.target_review_id is not None]
    if not answerable or not annotated:
        raise ValueError("Need answerable questions with manually labelled reviews")

    stats = {
        name: {"paper_hits": 0, "review_hits": 0, "reciprocal_rank": 0.0, "ndcg": 0.0}
        for name in methods
    }
    by_question: list[dict[str, Any]] = []
    for question in answerable:
        row: dict[str, Any] = {
            "id": question.id,
            "target_paper_id": question.target_paper_id,
            "target_review_id": question.target_review_id,
            "methods": {},
        }
        for name, retrieve in methods.items():
            ranked = retrieve(question.query)
            review_rank, paper_rank = find_rank_of_target(
                ranked, question.target_paper_id, question.target_review_id
            )
            top5_paper = paper_rank is not None and paper_rank <= 5
            top5_review = review_rank is not None and review_rank <= 5
            stats[name]["paper_hits"] += int(top5_paper)
            if question.target_review_id is not None:
                stats[name]["review_hits"] += int(top5_review)
                if top5_review:
                    stats[name]["reciprocal_rank"] += 1.0 / review_rank
                stats[name]["ndcg"] += compute_ndcg_at_k(
                    ranked, question.target_paper_id, question.target_review_id, k=5
                )
            row["methods"][name] = {
                "paper_rank": paper_rank,
                "review_rank": review_rank,
            }
        by_question.append(row)

    summary = {
        name: {
            "paper_hit_at_5": round(s["paper_hits"] / len(answerable), 3),
            "review_hit_at_5": round(s["review_hits"] / len(annotated), 3),
            "review_mrr_at_5": round(s["reciprocal_rank"] / len(annotated), 3),
            "review_ndcg_at_5": round(s["ndcg"] / len(annotated), 3),
        }
        for name, s in stats.items()
    }
    return {"method_comparison": summary, "questions_results": by_question}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair-corpus retrieval ablation (review / chunk text)")
    parser.add_argument("--questions", default="data/eval_questions.json")
    parser.add_argument("--output", default="data/experiments/fair_corpus_ablation.json")
    args = parser.parse_args()

    # Import heavier dependencies only during actual experiment execution.
    from rank_bm25 import BM25Okapi
    from evaluate import load_ground_truth_dataset, validate_ground_truth
    from rag_system import LiteratureReviewRAG

    questions_file = Path(args.questions)
    report_file = Path(args.output)
    if questions_file.resolve() == report_file.resolve():
        raise SystemExit("Input questions and output must be different files")
    rag = LiteratureReviewRAG()
    questions = load_ground_truth_dataset(questions_file)
    validate_ground_truth(questions, rag)

    # A. Existing baseline: whole reviews + author abstracts (not a fair data match).
    # B. Review-only baseline: whole reviews, no author abstracts.
    # C. Chunk-level BM25: exactly the same 1344 text chunks indexed by Chroma.
    # D. Existing vector retrieval on those same chunks, aggregated to reviews.
    bm25_review = BM25Okapi([tokenize_bm25(t) for t in rag.corpus_for_embedding])
    bm25_chunk = BM25Okapi([tokenize_bm25(t) for t in rag.chunk_documents])
    all_reviews = len(rag.corpus_metadata)

    methods = {
        "bm25_review_plus_abstract": lambda q: rag.search_bm25(q, top_k=all_reviews),
        "bm25_review_only": lambda q: bm25_results(
            bm25_review, q, rag.corpus_metadata, rag.corpus_for_embedding
        ),
        "bm25_same_chunks": lambda q: bm25_results(
            bm25_chunk, q, rag.chunk_metadata, rag.chunk_documents
        ),
        "vector_same_chunks": lambda q: rag.search_vector(q, top_k=all_reviews),
    }
    report = evaluate_rankings(questions, methods)
    report.update({
        "protocol": (
            "Development-set ablation, same manually annotated target reviews as evaluate.py; "
            "BM25 and vector_same_chunks index identical chunk texts. BM25 uses "
            "whitespace lexical tokenization, vector uses SentenceTransformer. "
            "Each chunk's score is merged to its review by MAX before ranking; "
            "one annotated target review per question, other reviews unjudged. "
            "Vector relies on Chroma approximate nearest neighbor retrieval; "
            "results need not match BM25 ranking depth exactly. "
            "Do not infer answer faithfulness or tune against the frozen holdout."
        ),
        "ground_truth_sha256": hashlib.sha256(questions_file.read_bytes()).hexdigest(),
        "answerable_questions": sum(q.has_data for q in questions),
        "review_annotated_questions": sum(q.has_data and q.target_review_id is not None for q in questions),
        "reviews": all_reviews,
        "chunks": len(rag.chunk_documents),
        "text_corpora": {
            "bm25_review_plus_abstract": "corpus_for_bm25 (review+abstract, review-level)",
            "bm25_review_only": "corpus_for_embedding (review only, review-level)",
            "bm25_same_chunks": "chunk_documents (review only, chunk-level, MAX by review)",
            "vector_same_chunks": "chunk_documents (review only, chunk-level, MAX by review)",
        },
    })
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== FAIR-CORPUS ABLATION — ZBIOR GLOWNY ===")
    print(f"Reviews: {report['reviews']} | Chunks: {report['chunks']}")
    print("Methods                         Paper Hit@5   Review Hit@5   MRR@5   nDCG@5")
    for name, value in report["method_comparison"].items():
        print(
            f"{name:<32} {value['paper_hit_at_5']:>7.1%}"
            f"      {value['review_hit_at_5']:>7.1%}"
            f"      {value['review_mrr_at_5']:.3f}    {value['review_ndcg_at_5']:.3f}"
        )
    print(f"\nSaved: {report_file}")
    print("Note: this is a controlled ablation on the known main set, not a new holdout.")


if __name__ == "__main__":
    main()
