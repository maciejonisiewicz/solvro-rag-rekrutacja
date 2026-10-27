"""Evaluation utilities for the OpenReview literature-review RAG.

The ground-truth dataset is read from data/eval_questions.json and is never
regenerated as a side effect of running evaluation.
"""

import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_rag import LiteratureReviewAssistant
from rag_system import LiteratureReviewRAG

SIMILARITY_THRESHOLD = 0.38
EVALUATION_VERSION = 2


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
    questions = [GroundTruthQuestion.model_validate(item) for item in raw_questions]

    if not questions:
        raise ValueError("Evaluation dataset is empty.")

    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation dataset contains duplicate question IDs.")

    if any(question.has_data and not question.target_paper_id for question in questions):
        raise ValueError("Every answerable question must define target_paper_id.")

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


def find_rank_of_target(
    results: list[dict[str, Any]],
    target_paper_id: str,
    target_review_id: str | None,
) -> tuple[int | None, int | None]:
    """Return (review rank, deduplicated paper rank); ranks are 1-based."""
    review_rank = None
    if target_review_id is not None:
        for rank, result in enumerate(results, start=1):
            if result["metadata"]["review_id"] == target_review_id:
                review_rank = rank
                break

    seen_papers: set[str] = set()
    paper_rank = None
    for result in results:
        paper_id = result["metadata"]["paper_id"]
        if paper_id in seen_papers:
            continue
        seen_papers.add(paper_id)
        if paper_id == target_paper_id:
            paper_rank = len(seen_papers)
            break

    return review_rank, paper_rank


def compute_ndcg_at_k(
    results: list[dict[str, Any]],
    target_paper_id: str,
    target_review_id: str | None,
    relevant_review_ids: set[str],
    k: int = 5,
) -> float | None:
    """Compute review-level nDCG with qrels derived from annotated paper/review IDs.

    Relevance policy: target review = 2; other reviews of the target paper = 1;
    all other documents = 0. This reflects the current evaluation protocol and
    should be revisited if each individual review is manually labelled.
    """
    if target_review_id is None:
        return None

    qrels = set(relevant_review_ids)
    qrels.add(target_review_id)

    dcg = 0.0
    for rank, result in enumerate(results[:k], start=1):
        metadata = result["metadata"]
        review_id = metadata["review_id"]
        paper_id = metadata["paper_id"]

        if review_id == target_review_id and paper_id == target_paper_id:
            relevance = 2
        elif review_id in qrels and paper_id == target_paper_id:
            relevance = 1
        else:
            relevance = 0

        if relevance:
            dcg += (2**relevance - 1) / math.log2(rank + 1)

    ideal_relevances = sorted(
        [2 if review_id == target_review_id else 1 for review_id in qrels],
        reverse=True,
    )[:k]
    idcg = sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )

    return dcg / idcg if idcg else 0.0


def _safe_ratio(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _review_metric_or_none(value: float, denominator: int) -> float | None:
    return round(value / denominator, 3) if denominator else None


def run_evaluation(rag: LiteratureReviewRAG | None = None) -> FullEvaluationReport:
    questions = load_ground_truth_dataset()
    rag_engine = rag or LiteratureReviewRAG()
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

    reviews_by_paper: dict[str, set[str]] = {}
    for review in rag_engine.reviews:
        reviews_by_paper.setdefault(review["paper_id"], set()).add(review["review_id"])

    recommended_papers: set[str] = set()
    conference_counts = {"ICLR 2024": 0, "NeurIPS 2024": 0}
    unique_papers_per_query: list[int] = []
    failure_cases: list[FailureCaseDetail] = []
    detailed_results: list[QuestionEvalDetail] = []

    # Reuse the already initialized RAG engine; do not load a second embedding model.
    assistant = LiteratureReviewAssistant(
        similarity_threshold=SIMILARITY_THRESHOLD,
        top_k=5,
        max_steps=3,
        rag=rag_engine,
    )
    # Keep evaluation traces separate from a user's latest interactive query.
    assistant.context_log_path = Path("data/evaluation_last_query_context.json")
    assistant.trace_log_path = Path("data/evaluation_agent_trace_log.json")

    for question in answerable:
        assert question.target_paper_id is not None
        target_paper_id = question.target_paper_id
        target_review_id = question.target_review_id

        # Retrieve the full small corpus so paper-level deduplication and ranks
        # are not distorted by an arbitrary 20-document cutoff.
        retrieval_k = max(len(rag_engine.reviews), len(rag_engine.papers), 1)
        vector_results = rag_engine.search_vector(question.query, top_k=retrieval_k)
        bm25_results = rag_engine.search_bm25(question.query, top_k=retrieval_k)
        hybrid_results = rag_engine.search_hybrid(question.query, top_k=retrieval_k)
        abstract_results = rag_engine.search_abstracts_only(question.query, top_k=retrieval_k)

        agent_payload = assistant.run_agent(question.query)
        agent_results = [
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
        ]

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
                relevant_ids = reviews_by_paper.get(target_paper_id, set())
                ndcg = compute_ndcg_at_k(
                    results,
                    target_paper_id,
                    target_review_id,
                    relevant_review_ids=relevant_ids,
                    k=5,
                )
                if ndcg is not None:
                    stats[method]["ndcg_sum"] += ndcg

        # Failure analysis is only meaningful when a specific review was annotated.
        if target_review_id is not None and (
            hits["vector"][0] is False or hits["bm25"][0] is False
        ):
            if question.id == 4:
                probable_cause = (
                    "Q04 asks for a specific technical criticism (ROUGE-L in Equation 3). "
                    "Dense retrieval can under-rank a rare exact term; compare the lexical "
                    "and hybrid rankings rather than assuming the hybrid method fixed it."
                )
            else:
                probable_cause = (
                    "The target review was not in the Top-5 for at least one baseline. "
                    "Inspect the query wording, relevant review fields, and the ranks from "
                    "the vector, BM25, hybrid, and agent methods."
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
        top_score = round(float(vector_results[0]["score"]), 3) if vector_results else 0.0
        rejected = top_score < SIMILARITY_THRESHOLD
        unanswerable_correct += int(rejected)
        unanswerable_results.append(
            UnanswerableEvalDetail(
                id=question.id,
                query=question.query,
                max_similarity_score=top_score,
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
            "with a manually annotated target review. Abstract-only review metrics are N/A."
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
    )

    report_path = Path("data/eval_results.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
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


if __name__ == "__main__":
    print_console_report(run_evaluation())
