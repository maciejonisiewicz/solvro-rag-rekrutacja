import json
from pathlib import Path
from typing import Literal
import streamlit as st
from agent_rag import AgentStep, LiteratureReviewAssistant, RetrievedSource
from evaluate import FullEvaluationReport, run_evaluation

RetrievalMode = Literal[
    "Agent Wieloetapowy z Narzędziami (Limit 3 kroki)",
    "Wyszukiwanie Hybrydowe (RRF: Wektor + BM25)",
    "Wyszukiwanie Wektorowe (ChromaDB)",
    "Baseline (BM25)",
    "Tylko Abstrakty (Bez recenzji)",
]


@st.cache_resource
def get_assistant() -> LiteratureReviewAssistant:
    return LiteratureReviewAssistant(similarity_threshold=0.38, top_k=4, max_steps=3)


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
    header = (
        f"[{index}] {source.title} ({source.conference}) — "
        f"Score: {source.similarity_score:.3f} | Rating: {source.rating}"
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
            st.markdown("#### 📝 Deklaracje Autorów (Abstrakt)")
            st.info(source.author_abstract)
        with right_col:
            st.markdown("#### 💬 Opinia Recenzenta")
            st.success(f"**Strengths:**\n\n{source.reviewer_strengths}")
            st.error(f"**Weaknesses & Limitations:**\n\n{source.reviewer_weaknesses}")


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
        raw_results = assistant.rag.search_hybrid(query=query, top_k=top_k, conference_filter=conference)
        tool_label = "hybrid_rrf_search"
    elif mode == "Tylko Abstrakty (Bez recenzji)":
        raw_results = assistant.rag.search_abstracts_only(query=query, top_k=top_k, conference_filter=conference)
        tool_label = "abstracts_only_search"
    else:
        raw_results = assistant.rag.search_bm25(query=query, top_k=top_k, conference_filter=conference)
        tool_label = "bm25_search"

    sources = assistant._to_sources(raw_results)
    vec_check = assistant.rag.search_vector(query=query, top_k=1, conference_filter=conference)
    in_domain = bool(vec_check) and float(vec_check[0]["score"]) >= threshold

    has_data = bool(sources) and sources[0].similarity_score > 0.0 and in_domain
    prompt = assistant.build_prompt(query, sources) if has_data else None
    step = AgentStep(
        step_number=1,
        tool_name=tool_label,
        tool_input=query,
        observation=f"Returned {len(sources)} results (domain check: {'PASS' if in_domain else 'FAIL'}).",
        decision="Single-step search completed." if has_data else "Rejected as out-of-domain query.",
    )
    return sources, prompt, has_data, [step]


def render_evaluation_section(assistant: LiteratureReviewAssistant) -> None:
    st.subheader("📊 Kompleksowa Ewaluacja Jakości i Rozszerzeń")

    if st.button("▶️ Przelicz pełną ewaluację (Wszystkie 5 kierunków rozszerzeń)", type="primary"):
        with st.spinner("Wykonywanie testów Hit@5, MRR, nDCG@5, Agenta oraz Ablacji..."):
            report = run_evaluation(rag=assistant.rag)
    else:
        results_path = Path("data/eval_results.json")
        if results_path.exists():
            data = json.loads(results_path.read_text(encoding="utf-8"))
            if "method_comparison" in data:
                report = FullEvaluationReport(**data)
            else:
                report = run_evaluation(rag=assistant.rag)
        else:
            report = run_evaluation(rag=assistant.rag)

    m = report.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agent z Narzędziami (Hit@5)", f"{m.agent_review_hit_at_5*100:.0f}%", f"MRR: {m.agent_review_mrr:.3f}")
    c2.metric("Hybryda RRF (Hit@5)", f"{m.hybrid_review_hit_at_5*100:.0f}%", f"nDCG@5: {m.hybrid_ndcg_at_5:.3f}")
    c3.metric("ChromaDB Wektor (Hit@5)", f"{m.vector_review_hit_at_5*100:.0f}%", f"MRR: {m.vector_review_mrr:.3f}")
    c4.metric("Tylko Abstrakty (MRR)", f"{m.abstracts_only_review_mrr:.3f}", "Spadek bez recenzji")

    st.markdown("#### 1. Porównanie wszystkich wariantów (Kierunki 1, 2, 3 i 4)")
    comp_rows = [
        {
            "Metoda / Eksperyment": mc.method_name,
            "Hit@5 (Publikacje)": f"{mc.paper_hit_at_5*100:.1f}%",
            "Hit@5 (Recenzje)": f"{mc.review_hit_at_5*100:.1f}%",
            "MRR (Recenzje)": f"{mc.review_mrr:.3f}",
            "nDCG@5 (Stopniowana trafność)": f"{mc.ndcg_at_5:.3f}",
        }
        for mc in report.method_comparison
    ]
    st.dataframe(comp_rows, use_container_width=True, hide_index=True)

    st.markdown("#### 2. Analiza wpływu doboru próby na różnorodność (Kierunek 5)")
    div = report.diversity_analysis
    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Pokrycie bazy w Top-5 (Catalog Coverage)",
        f"{div.unique_papers_recommended_top5}/{div.total_papers_in_db} prac",
        f"{div.catalog_coverage_ratio*100:.1f}% całej próby",
    )
    d2.metric(
        "Różnorodność wewnątrz zapytania",
        f"{div.avg_unique_papers_per_query_top5:.2f} / 5",
        "Unikalnych publikacji na Top-5",
    )
    d3.metric(
        "Balans konferencji (ICLR vs NeurIPS)",
        f"{div.iclr_2024_share_in_top5*100:.0f}% vs {div.neurips_2024_share_in_top5*100:.0f}%",
        "Udział w polecanych wynikach",
    )

    st.markdown("#### 3. Wyniki szczegółowe na poszczególnych pytaniach (Pozycja docelowej recenzji)")
    detail_rows = [
        {
            "ID": f"Q{r.id:02d}",
            "Pytanie": r.query,
            "Cel (Recenzja)": r.target_review_id,
            "Wektor": f"#{r.vector_review_rank}",
            "BM25": f"#{r.bm25_review_rank}",
            "Hybryda RRF": f"#{r.hybrid_review_rank}",
            "Agent (3 kroki)": f"#{r.agent_review_rank}",
            "Same Abstrakty": f"#{r.abstract_only_review_rank}",
        }
        for r in report.questions_results
    ]
    st.dataframe(detail_rows, use_container_width=True, hide_index=True)

    st.markdown("#### 4. Analiza przypadku nietrafnego i naprawy przez Agenta / RRF")
    for fc in report.failure_cases:
        st.warning(
            f"**Pytanie Q{fc.question_id:02d}:** *{fc.query}*\n\n"
            f"- **Oczekiwana recenzja:** `{fc.expected_review_id}` (Praca: `{fc.expected_paper_id}`)\n"
            f"- **Pojedynczy Wektor:** pozycja `#{fc.vector_review_rank}` (MISS w Top-5)\n"
            f"- **Po uruchomieniu Hybrydy RRF / Agenta:** pozycja `#{fc.hybrid_review_rank}` (**HIT #1**)\n"
            f"- **Wniosek:** {fc.probable_cause}"
        )


def main() -> None:
    st.set_page_config(
        page_title="Asystent Przeglądu Literatury RAG",
        page_icon="📚",
        layout="wide",
    )

    assistant = get_assistant()
    mode, conference, top_k, threshold = render_sidebar()

    st.title("📚 Asystent Przeglądu Literatury RAG (OpenReview 2024)")
    st.caption("Zestawienie deklaracji autorów z recenzjami eksperckimi (ICLR 2024 & NeurIPS 2024)")

    main_tab_search, main_tab_eval = st.tabs(
        ["🔍 Wyszukiwarka i Agent RAG", "📊 Ewaluacja Jakości (Wszystkie 5 kierunków rozszerzeń)"]
    )

    with main_tab_search:
        query = st.text_input(
            "Pytanie badawcze lub zagadnienie:",
            placeholder="np. Which review criticizes the use of Rouge-L in Equation 3?",
        )

        if query:
            sources, prompt, has_data, trace = execute_search(
                assistant=assistant,
                query=query,
                mode=mode,
                conference=conference,
                top_k=top_k,
                threshold=threshold,
            )

            with st.expander(f"🛠️ Przebieg rozumowania Agenta (Kroki: {len(trace)}/{assistant.max_steps})", expanded=True):
                for step in trace:
                    st.markdown(
                        f"**Krok {step.step_number} — Narzędzie:** `{step.tool_name}` | **Wejście:** `{step.tool_input}`  \n"
                        f"↳ *Obserwacja:* {step.observation}  \n"
                        f"↳ *Decyzja:* **{step.decision}**"
                    )

            if not has_data:
                st.warning(
                    f"Brak wystarczających danych w bazie (najwyższe podobieństwo poniżej progu {threshold:.2f})."
                )
            else:
                tab_sources, tab_prompt, tab_json = st.tabs(
                    [
                        f"📑 Powiązane źródła ({len(sources)})",
                        "🤖 Zbudowany Prompt dla LLM",
                        "🗂️ Kontekst JSON (Pydantic)",
                    ]
                )

                with tab_sources:
                    for idx, src in enumerate(sources, 1):
                        render_source_card(idx, src)

                with tab_prompt:
                    st.code(prompt or "", language="markdown")

                with tab_json:
                    st.json([src.model_dump() for src in sources])

    with main_tab_eval:
        render_evaluation_section(assistant)


if __name__ == "__main__":
    main()