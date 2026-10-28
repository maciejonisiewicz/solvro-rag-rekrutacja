import json
from pathlib import Path
from typing import Literal

import streamlit as st

from agent_rag import AgentStep, LiteratureReviewAssistant, RetrievedSource
from rag_system import LiteratureReviewRAG
from evaluate import FullEvaluationReport, run_evaluation, EVALUATION_VERSION

RetrievalMode = Literal[
    "Agent Wieloetapowy z Narzędziami (Limit 3 kroki)",
    "Wyszukiwanie Hybrydowe (RRF: Wektor + BM25)",
    "Wyszukiwanie Wektorowe (ChromaDB)",
    "Baseline (BM25)",
    "Tylko Abstrakty (Bez recenzji)",
]


def display_percent(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value * 100:.{digits}f}%"


def display_decimal(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


@st.cache_resource
def get_rag() -> LiteratureReviewRAG:
    # Cache expensive index/model only; assistant settings are per session.
    return LiteratureReviewRAG()


def get_assistant() -> LiteratureReviewAssistant:
    return LiteratureReviewAssistant(
        similarity_threshold=0.38, top_k=4, max_steps=3, rag=get_rag()
    )


def render_sidebar() -> tuple[RetrievalMode, str | None, int, float]:
    st.sidebar.header("⚙️ Konfiguracja RAG")
    retrieval_method: RetrievalMode = st.sidebar.selectbox(
        "Tryb pracy / Silnik wyszukiwania",
        [
            "Agent Wieloetapowy z Narzędziami (Limit 3 kroki)",
            "Wyszukiwanie Hybrydowe (RRF: Wektor + BM25)",
            "Wyszukiwanie Wektorowe (ChromaDB)",
            "Baseline (BM25)",
            "Tylko Abstrakty (Bez recenzji)",
        ],
    )
    conf_option = st.sidebar.selectbox(
        "Konferencja",
        ["Wszystkie", "ICLR 2024", "NeurIPS 2024"],
    )
    top_k = st.sidebar.slider("Liczba źródeł (Top K)", min_value=1, max_value=8, value=4)
    threshold = st.sidebar.slider(
        "Próg podobieństwa (Cosine Threshold)",
        min_value=0.10,
        max_value=0.70,
        value=0.38,
        step=0.02,
    )
    conference_filter = None if conf_option == "Wszystkie" else conf_option
    return retrieval_method, conference_filter, top_k, threshold


def render_source_card(index: int, source: RetrievedSource) -> None:
    """Show a paper-only card in the ablation; otherwise show review evidence."""
    if source.is_abstract_only:
        header = (
            f"[{index}] {source.title} ({source.conference}) — "
            f"Abstrakt publikacji | {source.score_type}: {source.similarity_score:.3f}"
        )
        with st.expander(header, expanded=(index == 1)):
            st.markdown(f"📄 **[Strona publikacji ({source.paper_id})]({source.paper_url})**")
            st.markdown("#### Deklaracje autorów — abstrakt")
            st.markdown(source.author_abstract or "Brak abstraktu w zapisanych danych.")
            st.caption("Tryb tylko z abstraktami: opinie recenzentów są celowo pominięte.")
        return

    header = (
        f"[{index}] {source.title} ({source.conference}) — "
        f"{source.score_type}: {source.similarity_score:.3f} | Ocena: {source.rating}"
    )
    with st.expander(header, expanded=(index == 1)):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"🔗 **[Oficjalna recenzja ({source.review_id})]({source.review_url})**")
        with col2:
            st.markdown(f"📄 **[Strona publikacji ({source.paper_id})]({source.paper_url})**")
        with col3:
            st.markdown(f"🏛️ **{source.conference}**")
        st.divider()
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("#### Deklaracje autorów — abstrakt")
            st.markdown(source.author_abstract or "Brak abstraktu w zapisanych danych.")
        with right_col:
            st.markdown("#### Opinia recenzenta")
            reviewer_fields = (
                ("Podsumowanie recenzenta", source.reviewer_summary),
                ("Mocne strony wskazane przez recenzenta", source.reviewer_strengths),
                ("Zastrzeżenia wskazane przez recenzenta", source.reviewer_weaknesses),
                ("Ograniczenia wskazane przez recenzenta", source.reviewer_limitations),
                ("Pytania recenzenta", source.reviewer_questions),
            )
            for label, content in reviewer_fields:
                if content:
                    st.markdown(f"**{label}**")
                    st.markdown(content)
            if not any(content for _, content in reviewer_fields):
                st.caption("Brak tych pól w zapisanych danych recenzji.")


def execute_search(
    assistant: LiteratureReviewAssistant,
    query: str,
    mode: RetrievalMode,
    conference: str | None,
    top_k: int,
    threshold: float,
) -> tuple[list[RetrievedSource], str | None, bool, list[AgentStep]]:
    assistant.top_k = top_k
    assistant.similarity_threshold = threshold

    if mode == "Agent Wieloetapowy z Narzędziami (Limit 3 kroki)":
        payload = assistant.run_agent(query=query, conference=conference)
        return payload.sources, payload.constructed_prompt, payload.has_sufficient_data, payload.agent_trace

    if mode == "Wyszukiwanie Wektorowe (ChromaDB)":
        payload = assistant.process_query(query=query, conference=conference)
        return payload.sources, payload.constructed_prompt, payload.has_sufficient_data, payload.agent_trace

    if mode == "Wyszukiwanie Hybrydowe (RRF: Wektor + BM25)":
        raw_results = assistant.rag.search_hybrid(
            query=query, top_k=top_k, conference_filter=conference
        )
        tool_label = "hybrid_rrf_search"
    elif mode == "Tylko Abstrakty (Bez recenzji)":
        raw_results = assistant.rag.search_abstracts_only(
            query=query, top_k=top_k, conference_filter=conference
        )
        tool_label = "abstracts_only_search"
    else:
        raw_results = assistant.rag.search_bm25(
            query=query, top_k=top_k, conference_filter=conference
        )
        tool_label = "bm25_search"

    sources = assistant._to_sources(raw_results)
    vector_check = assistant.rag.search_vector(
        query=query, top_k=1, conference_filter=conference
    )
    in_domain = bool(vector_check) and float(vector_check[0]["score"]) >= threshold
    has_data = bool(sources) and sources[0].similarity_score > 0.0 and in_domain
    prompt = assistant.build_prompt(query, sources) if has_data else None
    step = AgentStep(
        step_number=1,
        tool_name=tool_label,
        tool_input=query,
        observation=(
            f"Zwrócono {len(sources)} wyników; kontrola domeny: "
            f"{'PASS' if in_domain else 'FAIL'}."
        ),
        decision="Zakończono pojedyncze wyszukiwanie." if has_data else "Odrzucono zapytanie jako spoza zakresu.",
    )
    return sources, prompt, has_data, [step]


def render_evaluation_section(assistant: LiteratureReviewAssistant) -> None:
    st.subheader("📊 Ewaluacja jakości retrieval")
    st.caption(
        "Paper-level Hit@5 jest liczony na wszystkich pytaniach z odpowiedzią. "
        "Metryki recenzji są liczone wyłącznie dla pytań z ręcznie oznaczoną recenzją. "
        "Dla wariantu tylko z abstraktami metryki recenzji nie mają zastosowania."
    )

    results_path = Path("data/eval_results.json")
    report: FullEvaluationReport | None = None

    if st.button("▶️ Uruchom ponownie ewaluację", type="primary"):
        with st.spinner("Porównywanie retrieval, agenta i pytań bez odpowiedzi..."):
            report = run_evaluation(rag=assistant.rag)
    elif results_path.exists():
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
            if data.get("evaluation_version") == EVALUATION_VERSION:
                report = FullEvaluationReport.model_validate(data)
            else:
                st.warning(
                    "Zapisany raport pochodzi ze starszej wersji ewaluacji. "
                    "Uruchom ewaluację ponownie, aby nie wyświetlać nieporównywalnych wyników."
                )
        except (json.JSONDecodeError, ValueError) as exc:
            st.warning(f"Nie udało się wczytać zapisanego raportu: {exc}")
    else:
        st.info("Nie ma jeszcze aktualnego raportu. Kliknij przycisk, aby uruchomić ewaluację.")

    if report is None:
        return

    metrics = report.metrics
    st.markdown("#### Wyniki główne")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Agent — Review Hit@5",
        display_percent(metrics.agent_review_hit_at_5, 0),
        f"MRR@5: {display_decimal(metrics.agent_review_mrr)}",
    )
    c2.metric(
        "Hybryda — Review Hit@5",
        display_percent(metrics.hybrid_review_hit_at_5, 0),
        f"nDCG@5: {display_decimal(metrics.hybrid_ndcg_at_5)}",
    )
    c3.metric(
        "Wektory — Review Hit@5",
        display_percent(metrics.vector_review_hit_at_5, 0),
        f"MRR@5: {display_decimal(metrics.vector_review_mrr)}",
    )
    c4.metric(
        "Tylko abstrakty — Paper Hit@5",
        display_percent(metrics.abstracts_only_paper_hit_at_5, 0),
        "Ocena na poziomie publikacji",
    )

    st.caption(metrics.evaluation_unit)

    st.markdown("#### 1. Porównanie metod")
    comparison_rows = [
        {
            "Metoda": item.method_name,
            "Paper Hit@5": display_percent(item.paper_hit_at_5),
            "Review Hit@5": display_percent(item.review_hit_at_5),
            "MRR@5 (recenzje)": display_decimal(item.review_mrr),
            "nDCG@5 (recenzje)": display_decimal(item.ndcg_at_5),
        }
        for item in report.method_comparison
    ]
    st.dataframe(comparison_rows, width="stretch", hide_index=True)

    st.markdown("#### 2. Różnorodność wyników wyszukiwania")
    diversity = report.diversity_analysis
    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Unikalne publikacje w Top-5",
        f"{diversity.unique_papers_recommended_top5}/{diversity.total_papers_in_db}",
        f"Pokrycie katalogu: {diversity.catalog_coverage_ratio * 100:.1f}%",
    )
    d2.metric(
        "Średnia liczba unikalnych prac na zapytanie",
        f"{diversity.avg_unique_papers_per_query_top5:.2f} / 5",
    )
    d3.metric(
        "Udział konferencji w wynikach",
        f"ICLR {diversity.iclr_2024_share_in_top5 * 100:.0f}% · "
        f"NeurIPS {diversity.neurips_2024_share_in_top5 * 100:.0f}%",
    )

    st.markdown("#### 3. Wyniki dla poszczególnych pytań")
    question_rows = []
    for result in report.questions_results:
        question_rows.append(
            {
                "ID": f"Q{result.id:02d}",
                "Pytanie": result.query,
                "Target paper": result.target_paper_id,
                "Target review": result.target_review_id or "N/A — paper-level",
                "Wektor — paper rank": str(result.vector_paper_rank or "—"),
                "BM25 — paper rank": str(result.bm25_paper_rank or "—"),
                "Hybryda — paper rank": str(result.hybrid_paper_rank or "—"),
                "Agent — paper rank": str(result.agent_paper_rank or "—"),
                "Abstrakty — paper rank": str(result.abstract_only_paper_rank or "—"),
                "Wektor — review rank": str(result.vector_review_rank or "—"),
                "BM25 — review rank": str(result.bm25_review_rank or "—"),
                "Hybryda — review rank": str(result.hybrid_review_rank or "—"),
                "Agent — review rank": str(result.agent_review_rank or "—"),
            }
        )
    st.dataframe(question_rows, width="stretch", hide_index=True)

    st.markdown("#### 4. Pytania bez wystarczających danych")
    unanswerable_rows = [
        {
            "ID": f"Q{item.id:02d}",
            "Pytanie": item.query,
            "Najwyższe podobieństwo cosinusowe": f"{item.max_similarity_score:.3f}",
            "Próg": f"{item.threshold:.2f}",
            "Poprawnie odrzucono": "Tak" if item.correctly_rejected else "Nie",
        }
        for item in report.unanswerable_results
    ]
    st.dataframe(unanswerable_rows, width="stretch", hide_index=True)

    if report.failure_cases:
        st.markdown("#### 5. Analiza przypadków nietrafnych")
        for failure in report.failure_cases:
            st.warning(
                f"**Q{failure.question_id:02d}:** {failure.query}\n\n"
                f"- **Oczekiwana recenzja:** `{failure.expected_review_id}` "
                f"(publikacja `{failure.expected_paper_id}`)\n"
                f"- **Wektor:** {failure.vector_review_rank or 'poza wynikami'}\n"
                f"- **BM25:** {failure.bm25_review_rank or 'poza wynikami'}\n"
                f"- **Hybryda:** {failure.hybrid_review_rank or 'poza wynikami'}\n"
                f"- **Agent:** {failure.agent_review_rank or 'poza wynikami'}\n\n"
                f"**Możliwa przyczyna:** {failure.probable_cause}"
            )


def main() -> None:
    st.set_page_config(
        page_title="Asystent Przeglądu Literatury RAG",
        page_icon="📚",
        layout="wide",
    )
    assistant = get_assistant()
    mode, conference, top_k, threshold = render_sidebar()

    st.title("📚 Asystent Przeglądu Literatury RAG")
    st.caption(
        "OpenReview 2024 · ICLR i NeurIPS · porównanie abstraktów autorów z opiniami recenzentów"
    )
    st.info(
        "Tryb bez generacji LLM: aplikacja wyszukuje źródła i buduje prompt z kontekstem, "
        "ale nie generuje jeszcze końcowej syntezy odpowiedzi."
    )

    search_tab, evaluation_tab = st.tabs(["🔍 Wyszukiwarka RAG", "📊 Ewaluacja"])

    with search_tab:
        with st.form("literature_search_form"):
            query = st.text_input(
                "Pytanie badawcze lub zagadnienie",
                placeholder="np. Which review criticizes ROUGE-L in Equation 3?",
            )
            submitted = st.form_submit_button("Szukaj źródeł", type="primary")

        if submitted:
            if not query.strip():
                st.warning("Wpisz pytanie badawcze.")
            else:
                with st.spinner("Wyszukiwanie źródeł..."):
                    sources, prompt, has_data, trace = execute_search(
                        assistant=assistant,
                        query=query.strip(),
                        mode=mode,
                        conference=conference,
                        top_k=top_k,
                        threshold=threshold,
                    )
                st.session_state["last_search"] = {
                    "query": query.strip(),
                    "sources": [source.model_dump() for source in sources],
                    "prompt": prompt,
                    "has_data": has_data,
                    "trace": [step.model_dump() for step in trace],
                }

        result = st.session_state.get("last_search")
        if result:
            st.markdown(f"### Wyniki dla: {result['query']}")
            with st.expander("🛠️ Przebieg wyszukiwania", expanded=False):
                for step in result["trace"]:
                    st.markdown(
                        f"**Krok {step['step_number']} — narzędzie:** `{step['tool_name']}`  \n"
                        f"**Wejście:** `{step['tool_input']}`  \n"
                        f"**Obserwacja:** {step['observation']}  \n"
                        f"**Decyzja:** {step['decision']}"
                    )

            if not result["has_data"]:
                st.warning(
                    f"Nie znaleziono wystarczających podstaw w bazie przy progu {threshold:.2f}. "
                    "Spróbuj innego sformułowania lub zmień filtr konferencji."
                )
            else:
                source_tab, prompt_tab, json_tab = st.tabs(
                    [f"📑 Źródła ({len(result['sources'])})", "🤖 Prompt z kontekstem", "🗂️ Kontekst JSON"]
                )
                with source_tab:
                    for index, raw_source in enumerate(result["sources"], 1):
                        render_source_card(index, RetrievedSource.model_validate(raw_source))
                with prompt_tab:
                    st.code(result["prompt"] or "", language="markdown")
                with json_tab:
                    st.json(result["sources"])

    with evaluation_tab:
        render_evaluation_section(assistant)


if __name__ == "__main__":
    main()
