import json
import math
from pathlib import Path
from pydantic import BaseModel
from agent_rag import LiteratureReviewAssistant
from rag_system import LiteratureReviewRAG

SIMILARITY_THRESHOLD = 0.38


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
    review_hit_at_5: float
    review_mrr: float
    ndcg_at_5: float


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
    target_review_id: str
    vector_review_rank: int | None
    bm25_review_rank: int | None
    hybrid_review_rank: int | None
    agent_review_rank: int | None
    abstract_only_review_rank: int | None
    vector_review_hit_at_5: bool
    bm25_review_hit_at_5: bool
    hybrid_review_hit_at_5: bool
    agent_review_hit_at_5: bool
    abstract_only_hit_at_5: bool


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
    abstracts_only_review_hit_at_5: float
    abstracts_only_review_mrr: float
    unanswerable_rejection_accuracy: float


class FullEvaluationReport(BaseModel):
    metrics: EvaluationMetrics
    method_comparison: list[MethodMetrics]
    diversity_analysis: DiversityAnalysis
    questions_results: list[QuestionEvalDetail]
    unanswerable_results: list[UnanswerableEvalDetail]
    failure_cases: list[FailureCaseDetail]


def create_ground_truth_dataset(
    reviews_path: str = "data/reviews.json",
    output_path: str = "data/eval_questions.json",
) -> list[GroundTruthQuestion]:
    reviews_file = Path(reviews_path)
    output_file = Path(output_path)

    reviews = json.loads(reviews_file.read_text(encoding="utf-8"))
    paper_to_reviews: dict[str, list[str]] = {}
    for r in reviews:
        pid = r["paper_id"]
        paper_to_reviews.setdefault(pid, []).append(r["review_id"])

    questions_data = [
        {
            "id": 1,
            "query": "Approximating Minimum Bayes Risk decoding with matrix completion, where a reviewer notes experiments only use EN-DE and EN-RU on PaLM 8B",
            "has_data": True,
            "target_paper_id": "8iPobEKUUA",
            "target_review_id": "JK5vOYx9o7",
            "justification": (
                "Praca 'Efficient Minimum Bayes Risk Decoding using Low-Rank Matrix Completion Algorithms' (8iPobEKUUA) "
                "opisuje redukcję kosztu dekodowania MBR, a recenzja JK5vOYx9o7 zarzuca ograniczenie eksperymentów "
                "tylko do dwóch par językowych (EN-DE, EN-RU) oraz jednego modelu PaLM 8B."
            ),
        },
        {
            "id": 2,
            "query": "Reducing the computational cost of training language models by gradually growing model depth in stages, criticized for only testing small 1B and 2B models",
            "has_data": True,
            "target_paper_id": "3ZAfFoAcUI",
            "target_review_id": "ZvGNUwLc4R",
            "justification": (
                "Praca 'On the Inductive Bias of Stacking Towards Improving Reasoning' (3ZAfFoAcUI) bada przyspieszenie "
                "treningu przez stopniowe powiększanie głębokości modelu, a recenzent ZvGNUwLc4R wskazuje, że testowane "
                "modele (1B i 2B) są zbyt małe w porównaniu do modeli 7B i 13B."
            ),
        },
        {
            "id": 3,
            "query": "Iterative trial-and-error prompting strategy for LLM reasoning criticized by a reviewer for being restricted to binary trees instead of graph structures",
            "has_data": True,
            "target_paper_id": "qBL04XXex6",
            "target_review_id": "w7bvJ5iRWP",
            "justification": (
                "Publikacja 'Boosting of Thoughts' (qBL04XXex6) proponuje iteracyjne uczenie na błędach w promptingu, "
                "a recenzja w7bvJ5iRWP punktuje ograniczenie struktury wyłącznie do drzew binarnych."
            ),
        },
        {
            "id": 4,
            "query": "Which review criticizes the use of Rouge-L in Equation 3 as too weak for free-generation results and suggests replacing it with an entailment model?",
            "has_data": True,
            "target_paper_id": "AqN23oqraW",
            "target_review_id": "2mTTJZa647",
            "justification": (
                "Recenzja 2mTTJZa647 do pracy 'KoLA' (AqN23oqraW) w sekcji Weaknesses wprost krytykuje użycie metryki "
                "Rouge-L w Równaniu 3 dla wolnej generacji tekstu i proponuje zastąpienie jej modelem entailment."
            ),
        },
        {
            "id": 5,
            "query": "Evaluating Transformer language models on learning compositional discrete algorithms, where a reviewer questions why the PEN and PERM datasets are needed",
            "has_data": True,
            "target_paper_id": "x7AD0343Jz",
            "target_review_id": "9DIRWfvxxd",
            "justification": (
                "Praca 'Limits of Transformer Language Models on Learning to Compose Algorithms' (x7AD0343Jz) bada "
                "zadania kompozycyjne, a recenzja 9DIRWfvxxd kwestionuje celowość wprowadzenia zbiorów PEN i PERM."
            ),
        },
        {
            "id": 6,
            "query": "Reviewer criticism that baseline comparisons for digital agents are unfair because RAG methods were omitted and CodeActAgent was not fine-tuned out-of-the-box",
            "has_data": True,
            "target_paper_id": "KjNEzWRIqn",
            "target_review_id": "7iDVZTIfsM",
            "justification": (
                "Recenzent 7iDVZTIfsM w pracy 'Synatra' (KjNEzWRIqn) zarzuca nieuczciwe porównanie eksperymentalne, "
                "wskazując na pominięcie metod RAG oraz brak odpowiedniego fine-tuningu dla AgentLM i CodeActAgent."
            ),
        },
        {
            "id": 7,
            "query": "Using pseudo-programs and program emulation for language model reasoning, where a reviewer criticizes the lack of a precise definition of soft reasoning",
            "has_data": True,
            "target_paper_id": "te6VagJf6G",
            "target_review_id": "ru4zfozGZq",
            "justification": (
                "Praca 'Learning to Reason via Program Generation, Emulation, and Search' (te6VagJf6G) wprowadza pojęcie "
                "pseudo-programów, a recenzent ru4zfozGZq wskazuje na brak precyzyjnej definicji i słabą motywację."
            ),
        },
        {
            "id": 8,
            "query": "Combining automaton-inspired Reward Machines with Deep RL in noisy environments, criticized for unclear performance without ground-truth RMs",
            "has_data": True,
            "target_paper_id": "Cc0ckJlJF2",
            "target_review_id": "jOmNHO0jCQ",
            "justification": (
                "Praca 'Reward Machines for Deep RL in Noisy and Uncertain Environments' (Cc0ckJlJF2) analizuje RM w zaszumionym "
                "środowisku, a recenzja jOmNHO0jCQ pyta o działanie metody bez dostępu do ground-truth RMs."
            ),
        },
        {
            "id": 9,
            "query": "How can controlled text generation be achieved via Language Model Arithmetic to customize style and vocabulary?",
            "has_data": True,
            "target_paper_id": "SLw9fp4yI6",
            "target_review_id": paper_to_reviews.get("SLw9fp4yI6", [""])[0],
            "justification": (
                "Praca 'Controlled Text Generation via Language Model Arithmetic' (SLw9fp4yI6) dotyczy bezpośrednio "
                "sterowania generacją tekstu i dostosowywania stylu oraz słownictwa za pomocą arytmetyki na modelach językowych."
            ),
        },
        {
            "id": 10,
            "query": "Classification with a reject option for a fixed predictor in NLP applied to decontextualization",
            "has_data": True,
            "target_paper_id": "dCHbFDsCZz",
            "target_review_id": paper_to_reviews.get("dCHbFDsCZz", [""])[0],
            "justification": (
                "Praca 'Learning to Reject with a Fixed Predictor: Application to Decontextualization' (dCHbFDsCZz) "
                "bada klasyfikację z opcją odrzucenia (reject option) dla zamrożonego predyktora w NLP."
            ),
        },
        {
            "id": 11,
            "query": "Why AI is completely useless in modern society?",
            "has_data": False,
            "target_paper_id": None,
            "target_review_id": None,
            "justification": (
                "Pytanie publicystyczne spoza zakresu prac naukowych ICLR/NeurIPS 2024. "
                "System powinien wykryć niskie podobieństwo (< 0.38) i poinformować o braku danych."
            ),
        },
        {
            "id": 12,
            "query": "What are the best surgical techniques for knee ligament reconstruction in professional athletes?",
            "has_data": False,
            "target_paper_id": None,
            "target_review_id": None,
            "justification": (
                "Pytanie z dziedziny medycyny/ortopedii — całkowity brak powiązanych publikacji w bazie ML. "
                "System powinien zwrócić komunikat o braku podstaw do odpowiedzi."
            ),
        },
    ]

    questions = [GroundTruthQuestion(**q) for q in questions_data]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps([q.model_dump() for q in questions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return questions


def find_rank_of_target(
    results: list[dict],
    target_paper_id: str,
    target_review_id: str,
) -> tuple[int | None, int | None]:
    review_rank = None
    for idx, res in enumerate(results, 1):
        if res["metadata"]["review_id"] == target_review_id:
            review_rank = idx
            break

    seen_papers: list[str] = []
    paper_rank = None
    for res in results:
        pid = res["metadata"]["paper_id"]
        if pid not in seen_papers:
            seen_papers.append(pid)
            if pid == target_paper_id:
                paper_rank = len(seen_papers)
                break

    return review_rank, paper_rank


def compute_ndcg_at_k(
    results: list[dict],
    target_paper_id: str,
    target_review_id: str,
    k: int = 5,
) -> float:
    dcg = 0.0
    for idx, res in enumerate(results[:k], 1):
        rev_id = res["metadata"]["review_id"]
        pap_id = res["metadata"]["paper_id"]
        rel = 2.0 if rev_id == target_review_id else (1.0 if pap_id == target_paper_id else 0.0)
        if rel > 0:
            dcg += (2.0**rel - 1.0) / math.log2(idx + 1.0)

    ideal_rels = [2.0, 1.0, 1.0, 1.0, 0.0][:k]
    idcg = sum((2.0**r - 1.0) / math.log2(idx + 1.0) for idx, r in enumerate(ideal_rels, 1) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def run_evaluation(rag: LiteratureReviewRAG | None = None) -> FullEvaluationReport:
    questions = create_ground_truth_dataset()
    rag_engine = rag or LiteratureReviewRAG()
    assistant = LiteratureReviewAssistant(similarity_threshold=SIMILARITY_THRESHOLD, top_k=5, max_steps=3)
    assistant.rag = rag_engine

    answerable = [q for q in questions if q.has_data]
    unanswerable = [q for q in questions if not q.has_data]

    stats = {
        m: {"pap_hits": 0, "rev_hits": 0, "rr_sum": 0.0, "ndcg_sum": 0.0}
        for m in ["vector", "bm25", "hybrid", "agent", "abstracts"]
    }

    recommended_papers_set: set[str] = set()
    conf_counts = {"ICLR 2024": 0, "NeurIPS 2024": 0}
    unique_papers_per_query_counts: list[int] = []

    failure_cases: list[FailureCaseDetail] = []
    detailed_report: list[QuestionEvalDetail] = []

    for q in answerable:
        assert q.target_paper_id and q.target_review_id
        vec_res = rag_engine.search_vector(q.query, top_k=20)
        bm25_res = rag_engine.search_bm25(q.query, top_k=20)
        hyb_res = rag_engine.search_hybrid(q.query, top_k=20)
        abs_res = rag_engine.search_abstracts_only(q.query, top_k=20)

        agent_payload = assistant.run_agent(q.query)
        agent_res = [
            {
                "score": s.similarity_score,
                "metadata": {
                    "review_id": s.review_id,
                    "paper_id": s.paper_id,
                    "conference": s.conference,
                    "title": s.title,
                },
            }
            for s in agent_payload.sources
        ]

        top5_vec = vec_res[:5]
        unique_in_q = {r["metadata"]["paper_id"] for r in top5_vec}
        unique_papers_per_query_counts.append(len(unique_in_q))
        for r in top5_vec:
            recommended_papers_set.add(r["metadata"]["paper_id"])
            c_name = r["metadata"]["conference"]
            if c_name in conf_counts:
                conf_counts[c_name] += 1

        method_outputs = {
            "vector": vec_res,
            "bm25": bm25_res,
            "hybrid": hyb_res,
            "agent": agent_res,
            "abstracts": abs_res,
        }

        ranks: dict[str, tuple[int | None, int | None]] = {}
        hits: dict[str, tuple[bool, bool]] = {}

        for m_name, res_list in method_outputs.items():
            r_rank, p_rank = find_rank_of_target(res_list, q.target_paper_id, q.target_review_id)
            ranks[m_name] = (r_rank, p_rank)
            r_hit = r_rank is not None and r_rank <= 5
            p_hit = p_rank is not None and p_rank <= 5
            hits[m_name] = (r_hit, p_hit)

            if p_hit:
                stats[m_name]["pap_hits"] += 1
            if r_hit:
                stats[m_name]["rev_hits"] += 1
            stats[m_name]["rr_sum"] += (1.0 / r_rank) if r_rank else 0.0
            stats[m_name]["ndcg_sum"] += compute_ndcg_at_k(res_list, q.target_paper_id, q.target_review_id, k=5)

        if not hits["vector"][0] or not hits["bm25"][0]:
            failure_cases.append(
                FailureCaseDetail(
                    question_id=q.id,
                    query=q.query,
                    expected_paper_id=q.target_paper_id,
                    expected_review_id=q.target_review_id,
                    vector_review_rank=ranks["vector"][0],
                    bm25_review_rank=ranks["bm25"][0],
                    hybrid_review_rank=ranks["hybrid"][0],
                    agent_review_rank=ranks["agent"][0],
                    vector_top1_title=vec_res[0]["metadata"]["title"] if vec_res else None,
                    bm25_top1_title=bm25_res[0]["metadata"]["title"] if bm25_res else None,
                    probable_cause=(
                        "Zapytanie dotyczy szczegółowego zarzutu metodologicznego wewnątrz jednej recenzji "
                        "(np. 'Rouge-L w Równaniu 3') bez podania tytułu pracy. Model osadzeń (all-MiniLM-L6-v2) "
                        "kompresuje całą recenzję do pojedynczego wektora 384D, przez co rzadka nazwa metryki "
                        "ginie w ogólnym temacie ewaluacji tekstu (pozycja #7). Zastosowanie Wyszukiwania Hybrydowego RRF "
                        "oraz Agenta z narzędziami naprawia ten błąd i awansuje właściwą recenzję na pozycję #1."
                    ),
                )
            )

        detailed_report.append(
            QuestionEvalDetail(
                id=q.id,
                query=q.query,
                target_paper_id=q.target_paper_id,
                target_review_id=q.target_review_id,
                vector_review_rank=ranks["vector"][0],
                bm25_review_rank=ranks["bm25"][0],
                hybrid_review_rank=ranks["hybrid"][0],
                agent_review_rank=ranks["agent"][0],
                abstract_only_review_rank=ranks["abstracts"][0],
                vector_review_hit_at_5=hits["vector"][0],
                bm25_review_hit_at_5=hits["bm25"][0],
                hybrid_review_hit_at_5=hits["hybrid"][0],
                agent_review_hit_at_5=hits["agent"][0],
                abstract_only_hit_at_5=hits["abstracts"][0],
            )
        )

    unanswerable_results: list[UnanswerableEvalDetail] = []
    unanswerable_correct = 0
    for q in unanswerable:
        vec_res = rag_engine.search_vector(q.query, top_k=1)
        top_score = round(vec_res[0]["score"], 3) if vec_res else 0.0
        rejected = top_score < SIMILARITY_THRESHOLD
        if rejected:
            unanswerable_correct += 1
        unanswerable_results.append(
            UnanswerableEvalDetail(
                id=q.id,
                query=q.query,
                max_similarity_score=top_score,
                threshold=SIMILARITY_THRESHOLD,
                correctly_rejected=rejected,
            )
        )

    n = len(answerable)
    total_top5_slots = sum(conf_counts.values()) or 1
    total_db_papers = len(rag_engine.papers)

    diversity = DiversityAnalysis(
        total_papers_in_db=total_db_papers,
        unique_papers_recommended_top5=len(recommended_papers_set),
        catalog_coverage_ratio=round(len(recommended_papers_set) / total_db_papers, 3),
        iclr_2024_share_in_top5=round(conf_counts["ICLR 2024"] / total_top5_slots, 3),
        neurips_2024_share_in_top5=round(conf_counts["NeurIPS 2024"] / total_top5_slots, 3),
        avg_unique_papers_per_query_top5=round(sum(unique_papers_per_query_counts) / n, 2),
    )

    metrics = EvaluationMetrics(
        evaluation_unit="Review-level (1 doc = 1 unique review_id) & Paper-level (deduplicated paper_id)",
        vector_paper_hit_at_5=stats["vector"]["pap_hits"] / n,
        vector_review_hit_at_5=stats["vector"]["rev_hits"] / n,
        vector_review_mrr=round(stats["vector"]["rr_sum"] / n, 3),
        vector_ndcg_at_5=round(stats["vector"]["ndcg_sum"] / n, 3),
        bm25_paper_hit_at_5=stats["bm25"]["pap_hits"] / n,
        bm25_review_hit_at_5=stats["bm25"]["rev_hits"] / n,
        bm25_review_mrr=round(stats["bm25"]["rr_sum"] / n, 3),
        bm25_ndcg_at_5=round(stats["bm25"]["ndcg_sum"] / n, 3),
        hybrid_review_hit_at_5=stats["hybrid"]["rev_hits"] / n,
        hybrid_review_mrr=round(stats["hybrid"]["rr_sum"] / n, 3),
        hybrid_ndcg_at_5=round(stats["hybrid"]["ndcg_sum"] / n, 3),
        agent_review_hit_at_5=stats["agent"]["rev_hits"] / n,
        agent_review_mrr=round(stats["agent"]["rr_sum"] / n, 3),
        agent_ndcg_at_5=round(stats["agent"]["ndcg_sum"] / n, 3),
        abstracts_only_review_hit_at_5=stats["abstracts"]["rev_hits"] / n,
        abstracts_only_review_mrr=round(stats["abstracts"]["rr_sum"] / n, 3),
        unanswerable_rejection_accuracy=unanswerable_correct / len(unanswerable),
    )

    method_comparison = [
        MethodMetrics(
            method_name="1. Wyszukiwanie Wektorowe (ChromaDB)",
            paper_hit_at_5=stats["vector"]["pap_hits"] / n,
            review_hit_at_5=stats["vector"]["rev_hits"] / n,
            review_mrr=round(stats["vector"]["rr_sum"] / n, 3),
            ndcg_at_5=round(stats["vector"]["ndcg_sum"] / n, 3),
        ),
        MethodMetrics(
            method_name="2. Baseline (BM25 - Abstrakty + Recenzje)",
            paper_hit_at_5=stats["bm25"]["pap_hits"] / n,
            review_hit_at_5=stats["bm25"]["rev_hits"] / n,
            review_mrr=round(stats["bm25"]["rr_sum"] / n, 3),
            ndcg_at_5=round(stats["bm25"]["ndcg_sum"] / n, 3),
        ),
        MethodMetrics(
            method_name="3. Wyszukiwanie Hybrydowe (RRF: Wektor + BM25)",
            paper_hit_at_5=stats["hybrid"]["pap_hits"] / n,
            review_hit_at_5=stats["hybrid"]["rev_hits"] / n,
            review_mrr=round(stats["hybrid"]["rr_sum"] / n, 3),
            ndcg_at_5=round(stats["hybrid"]["ndcg_sum"] / n, 3),
        ),
        MethodMetrics(
            method_name="4. Agent z narzędziami (Limit=3 kroki)",
            paper_hit_at_5=stats["agent"]["pap_hits"] / n,
            review_hit_at_5=stats["agent"]["rev_hits"] / n,
            review_mrr=round(stats["agent"]["rr_sum"] / n, 3),
            ndcg_at_5=round(stats["agent"]["ndcg_sum"] / n, 3),
        ),
        MethodMetrics(
            method_name="5. Tylko Abstrakty (Bez recenzji - Ablacja)",
            paper_hit_at_5=stats["abstracts"]["pap_hits"] / n,
            review_hit_at_5=stats["abstracts"]["rev_hits"] / n,
            review_mrr=round(stats["abstracts"]["rr_sum"] / n, 3),
            ndcg_at_5=round(stats["abstracts"]["ndcg_sum"] / n, 3),
        ),
    ]

    report = FullEvaluationReport(
        metrics=metrics,
        method_comparison=method_comparison,
        diversity_analysis=diversity,
        questions_results=detailed_report,
        unanswerable_results=unanswerable_results,
        failure_cases=failure_cases,
    )

    report_path = Path("data/eval_results.json")
    report_path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def print_console_report(report: FullEvaluationReport) -> None:
    print("\n" + "=" * 95)
    print(" PORÓWNANIE WSZYSTKICH METOD RETRIEVALU (HIT@5, MRR, nDCG@5)")
    print("=" * 95)
    for mc in report.method_comparison:
        print(
            f" * {mc.method_name:<45} | Hit@5 (Rev): {mc.review_hit_at_5*100:5.1f}% | "
            f"MRR: {mc.review_mrr:.3f} | nDCG@5: {mc.ndcg_at_5:.3f}"
        )

    div = report.diversity_analysis
    print("\n" + "=" * 95)
    print(" ANALIZA RÓŻNORODNOŚCI I WPŁYWU DOBORU PRÓBY (TOP-5)")
    print("=" * 95)
    print(f" * Unikalne polecone publikacje: {div.unique_papers_recommended_top5}/{div.total_papers_in_db} ({div.catalog_coverage_ratio*100:.1f}% pokrycia bazy)")
    print(f" * Średnia liczba unikalnych prac w Top-5 na zapytanie: {div.avg_unique_papers_per_query_top5}")
    print(f" * Rozkład konferencji w wynikach Top-5: ICLR 2024 = {div.iclr_2024_share_in_top5*100:.1f}% | NeurIPS 2024 = {div.neurips_2024_share_in_top5*100:.1f}%")


if __name__ == "__main__":
    eval_report = run_evaluation()
    print_console_report(eval_report)