"""Pure integration layer: existing RAG agent -> optional structured synthesis.

The Streamlit page imports this helper; it never changes retrieval evaluation and
never sends a request to an LLM unless the user explicitly opts in.
"""
from __future__ import annotations

from typing import Any, Callable

from llm_synthesis import SynthesisResult, synthesize


def run_pipeline(
    assistant: Any,
    query: str,
    *,
    conference: str | None = None,
    top_k: int = 4,
    similarity_threshold: float = 0.38,
    use_gemini: bool = False,
    model_name: str = "google:gemini-3.5-flash-lite",
    synthesis_fn: Callable[..., SynthesisResult] = synthesize,
) -> tuple[Any, SynthesisResult]:
    """Return the exact agent payload and a safe optional generation result.

    The original agent retains responsibility for evidence checks, trace and
    writing the source context. A user choice (not mere presence of a key)
    governs whether a request can reach the provider.
    """
    query = query.strip()
    if not query:
        raise ValueError("Pytanie nie moze byc puste")
    if not 1 <= top_k <= 8:
        raise ValueError("top_k musi nalezec do zakresu 1–8")
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("Nieprawidlowy prog podobienstwa")

    assistant.top_k = top_k
    assistant.similarity_threshold = similarity_threshold
    payload = assistant.run_agent(query=query, conference=conference)

    if not payload.has_sufficient_data:
        return payload, SynthesisResult(
            status="insufficient_evidence",
            reason="Agent nie znalazl wystarczajacych dowodow w bazie OpenReview.",
            prompt=payload.constructed_prompt,
        )
    if not use_gemini:
        return payload, SynthesisResult(
            status="retrieval_only",
            reason="Wybrano tryb offline. Zrodla i prompt sa dostepne; nie generowano odpowiedzi.",
            prompt=payload.constructed_prompt,
        )
    return payload, synthesis_fn(payload, model_name=model_name)
