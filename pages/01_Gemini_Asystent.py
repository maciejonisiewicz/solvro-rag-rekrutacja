"""Optional Gemini + PydanticAI page of the existing Streamlit RAG application.

Run the original entry point: python -m streamlit run app.py.
Streamlit discovers pages/ automatically; app.py is intentionally unchanged.
"""
from __future__ import annotations

import os

import streamlit as st

from agent_rag import RAGQueryPayload
from app import get_assistant, render_source_card
from llm_synthesis import SynthesisResult
from streamlit_llm_bridge import run_pipeline


st.set_page_config(page_title="Solvro | Synteza Gemini", page_icon="📚", layout="wide")
st.title("📚 Synteza literatury z Gemini")
st.caption("OpenReview · ICLR 2024 i NeurIPS 2024 · agent RAG + opcjonalna synteza PydanticAI")
st.info(
    "Najpierw agent wyszukuje recenzje i sprawdza dostępność dowodów. "
    "Gemini jest uruchamiany tylko po wybraniu trybu generowania. "
    "Cytowania wskazują źródła, ale ich obecność nie dowodzi automatycznie zgodności twierdzenia z tekstem."
)

with st.sidebar:
    st.subheader("Synteza — ustawienia")
    generation_mode = st.radio(
        "Tryb odpowiedzi",
        options=["Tylko źródła (offline)", "Gemini — odpowiedź z cytowaniami"],
        index=0,
        help="Tryb offline nigdy nie wywołuje API Gemini.",
    )
    use_gemini = generation_mode.startswith("Gemini")
    model_name = st.selectbox(
        "Model Gemini",
        options=["google:gemini-3.5-flash-lite"],
        help="Model zweryfikowany w tej konfiguracji. Inne można przetestować osobno w CLI.",
        disabled=not use_gemini,
    )
    conf_label = st.selectbox("Konferencja", ["Wszystkie", "ICLR 2024", "NeurIPS 2024"])
    conference = None if conf_label == "Wszystkie" else conf_label
    top_k = st.slider("Liczba recenzji", 1, 8, 4)
    threshold = st.slider("Próg cosine (agent)", 0.10, 0.70, 0.38, 0.02)
    if use_gemini and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        st.warning(
            "Brak klucza Gemini w środowisku uruchomionej aplikacji. "
            "Wynik automatycznie przejdzie w tryb retrieval-only."
        )
    st.caption("Kluczy API nie wpisuj do formularza ani nie zapisuj w Git.")

with st.form("gemini_literature_query_form"):
    query = st.text_input(
        "Pytanie do publikacji i recenzji",
        value="",
        placeholder="Which review criticizes ROUGE-L in Equation 3?",
    )
    submitted = st.form_submit_button("Analizuj źródła", type="primary")

if submitted:
    if not query.strip():
        st.warning("Wpisz pytanie badawcze.")
    else:
        with st.spinner("Agent wyszukuje źródła i przygotowuje odpowiedź..."):
            # Use the same cached RAG index as the original app's retrieval page.
            assistant = get_assistant()
            payload, synthesis = run_pipeline(
                assistant,
                query,
                conference=conference,
                top_k=top_k,
                similarity_threshold=threshold,
                use_gemini=use_gemini,
                model_name=model_name,
            )
        # Keep the chosen settings and outputs together so they don't change
        # accidentally when Streamlit reruns because of a sidebar interaction.
        st.session_state["solvro_gemini_last"] = {
            "query": query.strip(),
            "payload": payload.model_dump(),
            "synthesis": synthesis.model_dump(),
            "mode": generation_mode,
            "model": model_name if use_gemini else "offline",
        }

last = st.session_state.get("solvro_gemini_last")
if last:
    payload = RAGQueryPayload.model_validate(last["payload"])
    synthesis = SynthesisResult.model_validate(last["synthesis"])

    st.subheader("Wynik")
    st.caption(f"Zapytanie: {last['query']} · tryb: {last['mode']}")
    a, b, c = st.columns(3)
    a.metric("Wyniki rankingowe", len(payload.sources))
    b.metric("Recenzje uzupełniające", len(payload.supplementary_sources))
    c.metric("Kroki agenta", f"{payload.steps_taken}/{payload.max_steps_limit}")

    if synthesis.status == "generated":
        st.success(f"Odpowiedź wygenerowana — {last['model']}")
        st.markdown(synthesis.answer or "")
        st.caption(
            "Zwalidowano numery źródeł i typy cytowań. "
            "Nie wykonano automatycznej weryfikacji semantycznej (faithfulness)."
        )
    elif synthesis.status == "retrieval_only":
        st.info(synthesis.reason)
    elif synthesis.status == "insufficient_evidence":
        st.warning(synthesis.answer or synthesis.reason)
    elif synthesis.status == "invalid_citations":
        st.error("Odrzucono odpowiedź: " + synthesis.reason)
    else:
        st.warning(synthesis.reason or "Generowanie nie powiodło się; źródła są dostępne poniżej.")

    with st.expander("🛠️ Przebieg pracy agenta", expanded=False):
        for step in payload.agent_trace:
            st.markdown(f"**Krok {step.step_number} — {step.tool_name}**")
            st.write("Wejście:", step.tool_input)
            st.write("Obserwacja:", step.observation)
            st.write("Decyzja:", step.decision)

    sources_tab, prompt_tab, data_tab = st.tabs(
        ["📑 Materiały OpenReview", "🧠 Kontekst dla modelu", "🗂️ JSON"]
    )
    with sources_tab:
        if not payload.sources:
            st.info("Brak wystarczających źródeł dla tego zapytania.")
        for i, source in enumerate(payload.sources, 1):
            render_source_card(i, source)
        if payload.supplementary_sources:
            st.markdown("#### Recenzje uzupełniające — poza rankingiem")
            for i, source in enumerate(payload.supplementary_sources, len(payload.sources) + 1):
                render_source_card(i, source)
    with prompt_tab:
        st.code(payload.constructed_prompt or "Brak promptu: agent odrzucił zapytanie.", language="markdown")
    with data_tab:
        st.json({
            "agent": payload.model_dump(),
            "synthesis": synthesis.model_dump(),
        })
