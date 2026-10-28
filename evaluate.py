"""Reproducible retrieval-only evaluation for the OpenReview literature RAG.

Paper Hit@5 uses the five *distinct* papers ranked from retrieval results.
Review Hit@5 / MRR@5 / binary nDCG@5 use one manually annotated target review
per question. Other reviews have not been judged; they are not silently granted
relevance credit merely for belonging to the same paper.

The ground truth in data/eval_questions.json is read-only. No LLM-generated
answer quality or claim faithfulness is measured here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from rag_system import LiteratureReviewRAG

SIMILARITY_THRESHOLD = 0.38
EVALUATION_VERSION = 3


class GroundTruthQuestion(BaseModel):
    id: int
    query: str
    has_data: bool
    target_paper_id: str | None
    target_review_id: str | None
    justification: str


class MethodMetrics(BaseModel):
    method_name: str
    paper_hit_at_5: float
    # None means that review-level evaluation is not meaningful for this method.
    review_hit_at_5: float | None
    review_mrr: float | None
    ndcg_at_5: float | None


class DiversityAnalysis(BaseModel):
    total_papers_in_db: int
    unique_papers_recommended_top5: int
    catalog_coverage_ratio: float
    iclr_2024_share_in_top5: float
    neurips_2024_share_in_top5: float
    avg_unique_papers_per_query_top5: float


class QuestionEvalDetail(BaseModel):
    id: int
    query: str
    target_paper_id: str
    target_review_id: str | None
    vector_paper_rank: int | None = None
    bm25_paper_rank: int | None = None
    hybrid_paper_rank: int | None = None
    agent_paper_rank: int | None = None
    abstract_only_paper_rank: int | None = None
    vector_review_rank: int | None
    bm25_review_rank: int | None
    hybrid_review_rank: int | None
    agent_review_rank: int | None
    abstract_only_review_rank: int | None
    vector_review_hit_at_5: bool | None
    bm25_review_hit_at_5: bool | None
    hybrid_review_hit_at_5: bool | None
    agent_review_hit_at_5: bool | None
    abstract_only_hit_at_5: bool | None


class UnanswerableEvalDetail(BaseModel):
    id: int
    query: str
    max_similarity_score: float
    threshold: float
    correctly_rejected: bool


class FailureCaseDetail(BaseModel):
    question_id: int
    query: str
    expected_paper_id: str
    expected_review_id: str
    vector_review_rank: int | None
    bm25_review_rank: int | None
    hybrid_review_rank: int | None
    agent_review_rank: int | None
    vector_top1_title: str | None
    bm25_top1_title: str | None
    probable_cause: str


class EvaluationMetrics(BaseModel):
    evaluation_unit: str
    vector_paper_hit_at_5: float
    vector_review_hit_at_5: float
    vector_review_mrr: float
    vector_ndcg_at_5: float
    bm25_paper_hit_at_5: float
    bm25_review_hit_at_5: float
    bm25_review_mrr: float
    bm25_ndcg_at_5: float
    hybrid_review_hit_at_5: float
    hybrid_review_mrr: float
    hybrid_ndcg_at_5: float
    agent_review_hit_at_5: float
    agent_review_mrr: float
    agent_ndcg_at_5: float
    unanswerable_rejection_accuracy: float
    # Diagnostic for the fixed *vector-score* threshold (not an agent metric).
    vector_threshold_answerable_acceptance: float | None = None
    vector_threshold_balanced_accuracy: float | None = None
    abstracts_only_paper_hit_at_5: float = 0.0
    # Review metrics for abstract-only retrieval are deliberately N/A.
    abstracts_only_review_hit_at_5: float | None = None
    abstracts_only_review_mrr: float | None = None
    abstracts_only_ndcg_at_5: float | None = None


class FullEvaluationReport(BaseModel):
    evaluation_version: int = EVALUATION_VERSION
    metrics: EvaluationMetrics
    method_comparison: list[MethodMetrics]
    diversity_analysis: DiversityAnalysis
    questions_results: list[QuestionEvalDetail]
    unanswerable_results: list[UnanswerableEvalDetail]
    failure_cases: list[FailureCaseDetail]
    # Additive, backward-compatible fields for auditing the evaluation protocol.
    ground_truth_sha256: str | None = None
    evaluation_protocol: str = (
        "Binary review nDCG@5: only the manually annotated target review is "
        "relevant; unjudged reviews receive zero *evaluation* credit, not a "
        "judgment of irrelevance. With one relevant review, its nDCG@5 is "
        "1/log2(rank+1) if rank<=5, otherwise 0. Paper Hit@5 counts "
        "five distinct papers. Reject/accept diagnostics use the top vector "
        "cosine score and a fixed threshold, not the full agent's decision."
    )


def load_ground_truth_dataset(
    path: str | Path = "data/eval_questions.json",
) -> list[GroundTruthQuestion]:
    """Load and validate the fixed, manually annotated evaluation set."""
    questions_path = Path(path)
    if not questions_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {questions_path}. "
            "Create and manually annotate the dataset before running evaluation."
        )

    raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(raw_questions, list):
        raise ValueError("Evaluation dataset must be a JSON list of questions.")
    questions = [GroundTruthQuestion.model_validate(item) for item in raw_questions]

    if not questions:
        raise ValueError("Evaluation dataset is empty.")

    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation dataset contains duplicate question IDs.")

    if any(question.has_data and not question.target_paper_id for question in questions):
        raise ValueError("Every answerable question must define target_paper_id.")

    for question in questions:
        if not question.query.strip():
            raise ValueError(f"Q{question.id:02d} has an empty query.")
        if question.has_data and not question.justification.strip():
            raise ValueError(f"Q{question.id:02d} needs a manual relevance justification.")

    if any(not question.has_data and question.target_paper_id for question in questions):
        raise ValueError(
            "Questions marked has_data=False must not define target_paper_id."
        )

    return questions


def validate_ground_truth(
    questions: list[GroundTruthQuestion], rag: LiteratureReviewRAG
) -> None:
    """Fail early when manually annotated IDs do not match the local corpus."""
    paper_ids = set(rag.papers)
    review_by_id = {review["review_id"]: review for review in rag.reviews}

    for question in questions:
        if not question.has_data:
            if question.target_review_id is not None:
                raise ValueError(
                    f"Q{question.id:02d} is unanswerable but defines target_review_id."
                )
            continue

        if question.target_paper_id not in paper_ids:
            raise ValueError(
                f"Q{question.id:02d}: unknown target_paper_id "
                f"{question.target_paper_id!r}."
            )

        if question.target_review_id is not None:
            target_review = review_by_id.get(question.target_review_id)
            if target_review is None:
                raise ValueError(
                    f"Q{question.id:02d}: unknown target_review_id "
                    f"{question.target_review_id!r}."
                )
            if target_review["paper_id"] != question.target_paper_id:
                raise ValueError(
                    f"Q{question.id:02d}: target_review_id belongs to a different paper."
                )


def _unique_review_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove repeated fragments of a review, preserving first-seen rank.

    The current index has one document per review; this guard prevents future
    chunking changes from counting the same review multiple times.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in results:
        review_id = str(item["metadata"].get("review_id") or "")
        if not review_id:
            raise ValueError("Review retrieval result is missing metadata.review_id.")
        if review_id not in seen:
            seen.add(review_id)
            unique.append(item)
    return unique


def find_rank_of_target(
    results: list[dict[str, Any]],
    target_paper_id: str,
    target_review_id: str | None,
) -> tuple[int | None, int | None]:
    """Return (review rank, distinct-paper rank), both one-based.

    Review rank treats duplicate chunks as one review. Paper rank likewise
    counts unique papers rather than review chunks.
    """
    review_rank: int | None = None
    seen_reviews: set[str] = set()
    seen_papers: set[str] = set()
    paper_rank: int | None = None
    for item in results:
        meta = item["metadata"]
        paper_id = meta["paper_id"]
        review_id = meta.get("review_id")

        if paper_id not in seen_papers:
            seen_papers.add(paper_id)
            if paper_id == target_paper_id:
                paper_rank = len(seen_papers)

        if review_id and review_id not in seen_reviews:
            seen_reviews.add(review_id)
            if review_id == target_review_id and paper_id == target_paper_id:
                review_rank = len(seen_reviews)

        if paper_rank is not None and (review_rank is not None or target_review_id is None):
            break

    return review_rank, paper_rank


def compute_ndcg_at_k(
    results: list[dict[str, Any]],
    target_paper_id: str,
    target_review_id: str | None,
    relevant_review_ids: set[str] | None = None,
    k: int = 5,
) -> float | None:
    """Binary nDCG@k against the *manually annotated* target review only.

    Each question has at most one manually annotated relevant review. Thus the
    ideal DCG is 1; nDCG is 1/log2(rank+1) if the target is within the first k
    distinct reviews, otherwise zero. All other reviews are **unjudged**; the
    zero assigned for scoring is not a claim that they are irrelevant.

    ``relevant_review_ids`` is ignored: kept only for compatibility with old
    call sites that passed all reviews of a paper as automatic relevance.
    """
    if k < 1:
        raise ValueError("k must be a positive integer")
    if target_review_id is None:
        return None

    review_rank, _ = find_rank_of_target(results, target_paper_id, target_review_id)
    if review_rank is None or review_rank > k:
        return 0.0
    return 1.0 / math.log2(review_rank + 1)


def _safe_ratio(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _review_metric_or_none(value: float, denominator: int) -> float | None:
    return round(value / denominator, 3) if denominator else None


def run_evaluation(
    rag: LiteratureReviewRAG | None = None,
    *,
    questions_path: str | Path = "data/eval_questions.json",
    report_path: str | Path = "data/eval_results.json",
    save_report: bool = True,
) -> FullEvaluationReport:
    """Run fixed-ground-truth retrieval evaluation and optionally save its report.

    Paths are configurable for unit tests; default calls are compatible with
    the existing Streamlit UI and the command ``python evaluate.py``.
    """
    # Heavy ML imports occur only when evaluation is run, so the pure metrics
    # can be unit-tested without loading sentence transformers / ChromaDB.
    from agent_rag import LiteratureReviewAssistant
    from rag_system import LiteratureReviewRAG

    questions = load_ground_truth_dataset(questions_path)
    rag_engine = rag if rag is not None else LiteratureReviewRAG()
    validate_ground_truth(questions, rag_engine)

    answerable = [question for question in questions if question.has_data]
    unanswerable = [question for question in questions if not question.has_data]
    review_questions = [
        question for question in answerable if question.target_review_id is not None
    ]
    paper_denominator = len(answerable)
    review_denominator = len(review_questions)

    if paper_denominator == 0:
        raise ValueError("Evaluation requires at least one answerable question.")

    stats: dict[str, dict[str, float]] = {
        method: {"paper_hits": 0, "review_hits": 0, "rr_sum": 0.0, "ndcg_sum": 0.0}
        for method in ("vector", "bm25", "hybrid", "agent", "abstracts")
    }

    recommended_papers: set[str] = set()
    conference_counts = {"ICLR 2024": 0, "NeurIPS 2024": 0}
    unique_papers_per_query: list[int] = []
    failure_cases: list[FailureCaseDetail] = []
    detailed_results: list[QuestionEvalDetail] = []
    answerable_threshold_accepted = 0

    # Reuse the already initialized RAG engine; do not load a second embedding model.
    assistant = LiteratureReviewAssistant(
        similarity_threshold=SIMILARITY_THRESHOLD,
        top_k=5,
        max_steps=3,
        rag=rag_engine,
    )
    # Keep evaluation traces separate from a user's latest interactive query.
    trace_dir = Path(report_path).parent
    assistant.context_log_path = trace_dir / "evaluation_last_query_context.json"
    assistant.trace_log_path = trace_dir / "evaluation_agent_trace_log.json"

    for question in answerable:
        assert question.target_paper_id is not None
        target_paper_id = question.target_paper_id
        target_review_id = question.target_review_id

        # Retrieve the full small corpus so paper-level deduplication and ranks
        # are not distorted by an arbitrary 20-document cutoff.
        retrieval_k = max(len(rag_engine.reviews), len(rag_engine.papers), 1)
        vector_results = _unique_review_results(
            rag_engine.search_vector(question.query, top_k=retrieval_k)
        )
        if vector_results and float(vector_results[0]["score"]) >= SIMILARITY_THRESHOLD:
            answerable_threshold_accepted += 1
        bm25_results = _unique_review_results(
            rag_engine.search_bm25(question.query, top_k=retrieval_k)
        )
        hybrid_results = _unique_review_results(
            rag_engine.search_hybrid(question.query, top_k=retrieval_k)
        )
        abstract_results = rag_engine.search_abstracts_only(question.query, top_k=retrieval_k)

        agent_payload = assistant.run_agent(question.query)
        agent_results = _unique_review_results([
            {
                "score": source.similarity_score,
                "metadata": {
                    "review_id": source.review_id,
                    "paper_id": source.paper_id,
                    "conference": source.conference,
                    "title": source.title,
                },
            }
            for source in agent_payload.sources
        ])

        vector_top5 = vector_results[:5]
        unique_for_query = {item["metadata"]["paper_id"] for item in vector_top5}
        unique_papers_per_query.append(len(unique_for_query))
        for item in vector_top5:
            metadata = item["metadata"]
            recommended_papers.add(metadata["paper_id"])
            conference = metadata["conference"]
            if conference in conference_counts:
                conference_counts[conference] += 1

        method_outputs: dict[str, list[dict[str, Any]]] = {
            "vector": vector_results,
            "bm25": bm25_results,
            "hybrid": hybrid_results,
            "agent": agent_results,
            "abstracts": abstract_results,
        }

        ranks: dict[str, tuple[int | None, int | None]] = {}
        hits: dict[str, tuple[bool | None, bool]] = {}

        for method, results in method_outputs.items():
            # Abstract-only documents represent papers; their inherited review IDs
            # are arbitrary metadata and must never be scored as review retrieval.
            method_target_review = (
                None if method == "abstracts" else target_review_id
            )
            review_rank, paper_rank = find_rank_of_target(
                results, target_paper_id, method_target_review
            )
            ranks[method] = (review_rank, paper_rank)

            review_hit: bool | None
            if method == "abstracts" or target_review_id is None:
                review_hit = None
            else:
                review_hit = review_rank is not None and review_rank <= 5

            paper_hit = paper_rank is not None and paper_rank <= 5
            hits[method] = (review_hit, paper_hit)

            if paper_hit:
                stats[method]["paper_hits"] += 1

            if review_hit is not None:
                if review_hit:
                    stats[method]["review_hits"] += 1
                    # MRR@5: a target outside the first five contributes zero.
                    stats[method]["rr_sum"] += 1.0 / review_rank
                ndcg = compute_ndcg_at_k(
                    results,
                    target_paper_id,
                    target_review_id,
                    k=5,
                )
                if ndcg is not None:
                    stats[method]["ndcg_sum"] += ndcg

        # Failure analysis is only meaningful when a specific review was annotated.
        if target_review_id is not None and (
            hits["vector"][0] is False or hits["bm25"][0] is False
        ):
            if hits["vector"][0] is False and hits["bm25"][0] is True:
                probable_cause = (
                    "The exact annotated review is outside the vector Top-5 but "
                    "inside the BM25 Top-5. Rare technical terms or exact wording "
                    "may be better captured lexically; inspect the source text."
                )
            elif hits["bm25"][0] is False and hits["vector"][0] is True:
                probable_cause = (
                    "The exact annotated review is outside the BM25 Top-5 but "
                    "inside the vector Top-5. Vocabulary mismatch may reduce "
                    "lexical matching; inspect the source text."
                )
            else:
                probable_cause = (
                    "Neither baseline retrieved the annotated review in Top-5. "
                    "Inspect query specificity, review text, and source rankings."
                )

            failure_cases.append(
                FailureCaseDetail(
                    question_id=question.id,
                    query=question.query,
                    expected_paper_id=target_paper_id,
                    expected_review_id=target_review_id,
                    vector_review_rank=ranks["vector"][0],
                    bm25_review_rank=ranks["bm25"][0],
                    hybrid_review_rank=ranks["hybrid"][0],
                    agent_review_rank=ranks["agent"][0],
                    vector_top1_title=(
                        vector_results[0]["metadata"]["title"] if vector_results else None
                    ),
                    bm25_top1_title=(
                        bm25_results[0]["metadata"]["title"] if bm25_results else None
                    ),
                    probable_cause=probable_cause,
                )
            )

        detailed_results.append(
            QuestionEvalDetail(
                id=question.id,
                query=question.query,
                target_paper_id=target_paper_id,
                target_review_id=target_review_id,
                vector_paper_rank=ranks["vector"][1],
                bm25_paper_rank=ranks["bm25"][1],
                hybrid_paper_rank=ranks["hybrid"][1],
                agent_paper_rank=ranks["agent"][1],
                abstract_only_paper_rank=ranks["abstracts"][1],
                vector_review_rank=ranks["vector"][0],
                bm25_review_rank=ranks["bm25"][0],
                hybrid_review_rank=ranks["hybrid"][0],
                agent_review_rank=ranks["agent"][0],
                abstract_only_review_rank=None,
                vector_review_hit_at_5=hits["vector"][0],
                bm25_review_hit_at_5=hits["bm25"][0],
                hybrid_review_hit_at_5=hits["hybrid"][0],
                agent_review_hit_at_5=hits["agent"][0],
                abstract_only_hit_at_5=None,
            )
        )

    unanswerable_results: list[UnanswerableEvalDetail] = []
    unanswerable_correct = 0
    for question in unanswerable:
        vector_results = rag_engine.search_vector(question.query, top_k=1)
        raw_top_score = float(vector_results[0]["score"]) if vector_results else 0.0
        rejected = raw_top_score < SIMILARITY_THRESHOLD
        unanswerable_correct += int(rejected)
        unanswerable_results.append(
            UnanswerableEvalDetail(
                id=question.id,
                query=question.query,
                max_similarity_score=round(raw_top_score, 3),
                threshold=SIMILARITY_THRESHOLD,
                correctly_rejected=rejected,
            )
        )

    total_top5_slots = sum(conference_counts.values())
    total_papers_in_db = len(rag_engine.papers)
    diversity = DiversityAnalysis(
        total_papers_in_db=total_papers_in_db,
        unique_papers_recommended_top5=len(recommended_papers),
        catalog_coverage_ratio=round(
            _safe_ratio(len(recommended_papers), total_papers_in_db), 3
        ),
        iclr_2024_share_in_top5=round(
            _safe_ratio(conference_counts["ICLR 2024"], total_top5_slots), 3
        ),
        neurips_2024_share_in_top5=round(
            _safe_ratio(conference_counts["NeurIPS 2024"], total_top5_slots), 3
        ),
        avg_unique_papers_per_query_top5=round(
            _safe_ratio(sum(unique_papers_per_query), len(unique_papers_per_query)), 2
        ),
    )

    metrics = EvaluationMetrics(
        evaluation_unit=(
            f"Paper-level Hit@5 uses all {paper_denominator} answerable questions; "
            f"review-level Hit@5/MRR/nDCG@5 use only {review_denominator} questions "
            "with a manually annotated target review (binary nDCG: one relevant "
            "review only). Abstract-only review metrics are N/A."
        ),
        vector_paper_hit_at_5=_safe_ratio(stats["vector"]["paper_hits"], paper_denominator),
        vector_review_hit_at_5=_safe_ratio(stats["vector"]["review_hits"], review_denominator),
        vector_review_mrr=round(_safe_ratio(stats["vector"]["rr_sum"], review_denominator), 3),
        vector_ndcg_at_5=round(_safe_ratio(stats["vector"]["ndcg_sum"], review_denominator), 3),
        bm25_paper_hit_at_5=_safe_ratio(stats["bm25"]["paper_hits"], paper_denominator),
        bm25_review_hit_at_5=_safe_ratio(stats["bm25"]["review_hits"], review_denominator),
        bm25_review_mrr=round(_safe_ratio(stats["bm25"]["rr_sum"], review_denominator), 3),
        bm25_ndcg_at_5=round(_safe_ratio(stats["bm25"]["ndcg_sum"], review_denominator), 3),
        hybrid_review_hit_at_5=_safe_ratio(stats["hybrid"]["review_hits"], review_denominator),
        hybrid_review_mrr=round(_safe_ratio(stats["hybrid"]["rr_sum"], review_denominator), 3),
        hybrid_ndcg_at_5=round(_safe_ratio(stats["hybrid"]["ndcg_sum"], review_denominator), 3),
        agent_review_hit_at_5=_safe_ratio(stats["agent"]["review_hits"], review_denominator),
        agent_review_mrr=round(_safe_ratio(stats["agent"]["rr_sum"], review_denominator), 3),
        agent_ndcg_at_5=round(_safe_ratio(stats["agent"]["ndcg_sum"], review_denominator), 3),
        unanswerable_rejection_accuracy=_safe_ratio(
            unanswerable_correct, len(unanswerable)
        ),
        vector_threshold_answerable_acceptance=_safe_ratio(
            answerable_threshold_accepted, paper_denominator
        ),
        vector_threshold_balanced_accuracy=(
            round(
                (
                    _safe_ratio(answerable_threshold_accepted, paper_denominator)
                    + _safe_ratio(unanswerable_correct, len(unanswerable))
                ) / 2,
                3,
            )
            if unanswerable else None
        ),
        abstracts_only_paper_hit_at_5=_safe_ratio(
            stats["abstracts"]["paper_hits"], paper_denominator
        ),
        abstracts_only_review_hit_at_5=None,
        abstracts_only_review_mrr=None,
        abstracts_only_ndcg_at_5=None,
    )

    def method_metrics(name: str, label: str, has_review_metric: bool = True) -> MethodMetrics:
        return MethodMetrics(
            method_name=label,
            paper_hit_at_5=_safe_ratio(stats[name]["paper_hits"], paper_denominator),
            review_hit_at_5=(
                _review_metric_or_none(stats[name]["review_hits"], review_denominator)
                if has_review_metric else None
            ),
            review_mrr=(
                _review_metric_or_none(stats[name]["rr_sum"], review_denominator)
                if has_review_metric else None
            ),
            ndcg_at_5=(
                _review_metric_or_none(stats[name]["ndcg_sum"], review_denominator)
                if has_review_metric else None
            ),
        )

    method_comparison = [
        method_metrics("vector", "1. Wyszukiwanie Wektorowe (ChromaDB)"),
        method_metrics("bm25", "2. Baseline (BM25 - Abstrakty + Recenzje)"),
        method_metrics("hybrid", "3. Wyszukiwanie Hybrydowe (RRF: Wektor + BM25)"),
        method_metrics("agent", "4. Agent deterministyczny z narzędziami (limit 3 kroki)"),
        method_metrics("abstracts", "5. Tylko Abstrakty (bez recenzji)", has_review_metric=False),
    ]

    report = FullEvaluationReport(
        evaluation_version=EVALUATION_VERSION,
        metrics=metrics,
        method_comparison=method_comparison,
        diversity_analysis=diversity,
        questions_results=detailed_results,
        unanswerable_results=unanswerable_results,
        failure_cases=failure_cases,
        ground_truth_sha256=hashlib.sha256(
            Path(questions_path).read_bytes()
        ).hexdigest(),
    )

    if save_report:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def print_console_report(report: FullEvaluationReport) -> None:
    def percentage(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:5.1f}%"

    def decimal(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    print("\n" + "=" * 100)
    print(" RETRIEVAL EVALUATION (paper-level and review-level metrics)")
    print("=" * 100) 
    print(report.metrics.evaluation_unit)
    for method in report.method_comparison:
        print(
            f" * {method.method_name:<50} | "
            f"Paper Hit@5: {percentage(method.paper_hit_at_5)} | "
            f"Review Hit@5: {percentage(method.review_hit_at_5)} | "
            f"MRR@5: {decimal(method.review_mrr)} | nDCG@5: {decimal(method.ndcg_at_5)}"
        )

    metrics = report.metrics
    print(
        "\nUnanswerable rejection accuracy: "
        f"{percentage(metrics.unanswerable_rejection_accuracy)}"
    )
    print(
        "Vector threshold answerable acceptance: "
        f"{percentage(metrics.vector_threshold_answerable_acceptance)} "
        f"(threshold={SIMILARITY_THRESHOLD:.2f}, {len(report.questions_results)} questions)"
    )
    print(
        "Vector threshold balanced accuracy: "
        f"{percentage(metrics.vector_threshold_balanced_accuracy)} "
        "(diagnostic only; threshold not independently validated)"
    )
    print("Note: one annotated review per query; unjudged reviews get no nDCG credit.")
    print("\nUnanswerable query details:")
    for result in report.unanswerable_results:
        status = "REJECTED" if result.correctly_rejected else "NOT REJECTED"
        print(
            f" Q{result.id:02d}: score={result.max_similarity_score:.3f}, "
            f"threshold={result.threshold:.2f} -> {status}"
        )

    if report.failure_cases:
        print("\nReview-level failure cases:")
        for failure in report.failure_cases:
            print(
                f" Q{failure.question_id:02d}: expected review {failure.expected_review_id}; "
                f"vector rank={failure.vector_review_rank}, "
                f"BM25 rank={failure.bm25_review_rank}, "
                f"hybrid rank={failure.hybrid_review_rank}, "
                f"agent rank={failure.agent_review_rank}"
            )
            print(f"  Possible cause: {failure.probable_cause}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG retrieval methods on fixed, manually labelled sources."
    )
    parser.add_argument(
        "--questions", default="data/eval_questions.json",
        help="Path to an existing, manually annotated JSON question set.",
    )
    parser.add_argument(
        "--output", default="data/eval_results.json",
        help="Destination for the evaluation report (JSON).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print evaluation without rewriting the JSON report.",
    )
    args = parser.parse_args()
    report = run_evaluation(
        questions_path=args.questions,
        report_path=args.output,
        save_report=not args.no_save,
    )
    print_console_report(report)


if __name__ == "__main__":
    main()
