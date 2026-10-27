from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag_system import LiteratureReviewRAG


STOPWORDS = {
    # English stopwords
    "which", "what", "where", "when", "how", "does", "do", "is", "are",
    "the", "a", "an", "in", "on", "of", "for", "to", "with", "and",
    "or", "by", "from", "that", "this", "why", "can", "be", "as", "too",
    "it", "its", "their", "about", "using", "use", "based", "paper",
    "review", "reviews", "reviewer", "reviewers", "method", "methods",
    # Common Polish stopwords (the corpus itself is primarily English)
    "jak", "jaki", "jaka", "jakie", "czy", "oraz", "dla", "jest", "są",
    "się", "na", "do", "z", "w", "i", "lub", "to", "że", "który",
    "która", "które", "oraz", "nad", "pod", "przez",
}


class RetrievedSource(BaseModel):
    review_id: str
    paper_id: str
    conference: str
    title: str
    paper_url: str
    review_url: str
    rating: str
    # Kept for compatibility with the existing Streamlit UI. This is the
    # retrieval method's score, not always cosine similarity (e.g. for RRF).
    similarity_score: float
    author_abstract: str
    reviewer_strengths: str
    reviewer_weaknesses: str
    score_type: str = "cosine_similarity"
    is_supplementary: bool = False


class AgentStep(BaseModel):
    step_number: int
    tool_name: str
    tool_input: str
    observation: str
    decision: str


class RAGQueryPayload(BaseModel):
    query: str
    conference_filter: str | None
    has_sufficient_data: bool
    similarity_threshold: float
    max_steps_limit: int
    steps_taken: int
    agent_trace: list[AgentStep]
    sources: list[RetrievedSource]
    constructed_prompt: str | None
    # Supplementary reviews are kept separate from ranked retrieval results,
    # so they cannot artificially improve or worsen review ranking metrics.
    supplementary_sources: list[RetrievedSource] = Field(default_factory=list)


class LiteratureReviewAssistant:
    def __init__(
        self,
        similarity_threshold: float = 0.38,
        top_k: int = 4,
        max_steps: int = 3,
        rag: LiteratureReviewRAG | None = None,
        data_dir: str | Path = "data",
    ) -> None:
        self.rag = rag if rag is not None else LiteratureReviewRAG(data_dir=data_dir)
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.max_steps = max(1, max_steps)
        self.context_log_path = Path(data_dir) / "last_query_context.json"
        self.trace_log_path = Path(data_dir) / "agent_trace_log.json"

    def _to_sources(
        self,
        raw_results: list[dict[str, Any]],
        *,
        score_type: str = "cosine_similarity",
        supplementary: bool = False,
    ) -> list[RetrievedSource]:
        sources: list[RetrievedSource] = []
        for result in raw_results:
            metadata = result.get("metadata", {})
            sources.append(
                RetrievedSource(
                    review_id=str(metadata.get("review_id", "")),
                    paper_id=str(metadata.get("paper_id", "")),
                    conference=str(metadata.get("conference", "Unknown")),
                    title=str(metadata.get("title", "Untitled")),
                    paper_url=str(metadata.get("paper_url", "")),
                    review_url=str(metadata.get("review_url", "")),
                    rating=str(metadata.get("rating", "N/A")),
                    similarity_score=round(float(result.get("score", 0.0)), 4),
                    author_abstract=str(metadata.get("author_abstract", "")),
                    reviewer_strengths=str(metadata.get("reviewer_strengths", "")),
                    reviewer_weaknesses=str(metadata.get("reviewer_weaknesses", "")),
                    score_type=str(result.get("score_type", score_type)),
                    is_supplementary=bool(supplementary),
                )
            )
        return sources

    def _refine_query_keywords(self, query: str) -> str:
        # Keep technical terms such as ROUGE-L, KV-cache, PaLM 8B and Unicode
        # letters; remove only low-information tokens.
        tokens = re.findall(r"[^\W_]+(?:-[^\W_]+)*", query, flags=re.UNICODE)
        kept = [
            token
            for token in tokens
            if token.casefold() not in STOPWORDS and len(token) > 1
        ]
        return " ".join(kept) if kept else query

    @staticmethod
    def _technical_terms(query: str) -> list[str]:
        patterns = (
            r"\b[A-Z]{3,}(?:-[A-Z0-9]+)*\b",      # ROUGE-L, LLM, BM25
            r"\b[A-Za-z0-9]+-[A-Za-z0-9-]+\b",   # KV-cache, fine-tuning
            r"\b[A-Za-z]*\d+[A-Za-z0-9]*\b",    # 8B, PaLM8B, 2024
            r"\b(?:equation|eq\.?|figure|fig\.?|table|tab\.?)\s*\d+\b",
        )
        found: list[str] = []
        for pattern in patterns:
            found.extend(re.findall(pattern, query, flags=re.IGNORECASE if "equation" in pattern else 0))
        # Preserve order and avoid duplicate tokens.
        return list(dict.fromkeys(term.strip() for term in found if term.strip()))

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        """Normalize punctuation so ROUGE-L and free-generation match consistently."""
        return re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE).strip()

    def _exact_term_matches(
        self,
        query: str,
        result: dict[str, Any],
    ) -> list[str]:
        """Return distinctive technical query terms found in one retrieved review."""
        terms = [term for term in self._technical_terms(query) if not term.isdigit()]
        metadata = result.get("metadata", {})
        searchable_text = self._normalize_match_text(
            " ".join(
                [
                    str(metadata.get("title", "")),
                    str(result.get("text_snippet", "")),
                ]
            )
        )
        padded_text = f" {searchable_text} "
        matched: list[str] = []
        for term in terms:
            normalized_term = self._normalize_match_text(term)
            if normalized_term and f" {normalized_term} " in padded_text:
                matched.append(term)
        return matched

    def _rerank_by_exact_terms(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Move documents matching exact technical terms above generic semantic matches.

        The original retrieval score is preserved. Original order breaks ties, so
        this is a deterministic reranking heuristic, not a calibrated score.
        """
        if not results or not self._technical_terms(query):
            return results

        decorated = [
            (index, result, len(self._exact_term_matches(query, result)))
            for index, result in enumerate(results)
        ]
        if not any(match_count > 0 for _, _, match_count in decorated):
            return results

        decorated.sort(key=lambda item: (-item[2], item[0]))
        return [result for _, result, _ in decorated]

    def _has_exact_technical_match(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> bool:
        # A lexical fallback is accepted only when a distinctive term appears
        # in at least one retrieved review, not merely in unrelated metadata.
        return any(self._exact_term_matches(query, item) for item in results)

    def build_prompt(
        self,
        query: str,
        sources: list[RetrievedSource],
        supplementary_sources: list[RetrievedSource] | None = None,
    ) -> str:
        supplementary_sources = supplementary_sources or []
        all_sources = [(source, False) for source in sources] + [
            (source, True) for source in supplementary_sources
        ]

        context_blocks: list[str] = []
        for idx, (source, supplementary) in enumerate(all_sources, 1):
            source_kind = (
                "SUPPLEMENTARY REVIEW — retrieved because it belongs to a cited paper; "
                "not ranked by query relevance"
                if supplementary
                else f"RETRIEVAL RESULT — score type: {source.score_type}"
            )
            context_blocks.append(
                f"[SOURCE {idx}] {source_kind}\n"
                f"Paper Title: {source.title} ({source.conference})\n"
                f"Paper ID: {source.paper_id} | Review ID: {source.review_id}\n"
                f"Paper Link: {source.paper_url}\n"
                f"Review Link: {source.review_url}\n"
                f"Reviewer Rating (original field value): {source.rating}\n"
                f"Author Claims (Abstract): {source.author_abstract}\n"
                f"Reviewer Strengths: {source.reviewer_strengths}\n"
                f"Reviewer Weaknesses & Limitations: {source.reviewer_weaknesses}"
            )

        joined_context = "\n\n".join(context_blocks) if context_blocks else "[NO SOURCES RETRIEVED]"
        return (
            "You are a machine-learning literature review assistant. "
            "Answer strictly from the provided OpenReview context.\n"
            "Rules:\n"
            "1. Clearly distinguish author claims (abstracts) from reviewer opinions.\n"
            "2. Cite every factual claim with [SOURCE X] and include the paper/review link.\n"
            "3. Do not infer details absent from the source text.\n"
            "4. If sources disagree, describe the disagreement rather than choosing a side.\n"
            "5. A supplementary review provides context about the same paper but was not ranked by query relevance.\n"
            "6. If the evidence is insufficient, explicitly say so.\n\n"
            f"CONTEXT:\n{joined_context}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            "ANSWER:"
        )

    def process_query(
        self,
        query: str,
        conference: str | None = None,
    ) -> RAGQueryPayload:
        raw_results = self.rag.search_vector(
            query=query,
            top_k=self.top_k,
            conference_filter=conference,
        )
        top_score = float(raw_results[0]["score"]) if raw_results else 0.0
        has_data = bool(raw_results) and top_score >= self.similarity_threshold

        sources = self._to_sources(raw_results[: self.top_k], score_type="cosine_similarity") if has_data else []
        prompt = self.build_prompt(query, sources) if has_data else None
        step = AgentStep(
            step_number=1,
            tool_name="vector_search",
            tool_input=query,
            observation=(
                f"Returned {len(raw_results)} results; top cosine similarity={top_score:.4f}."
            ),
            decision=(
                "Accept semantic retrieval and build source-grounded prompt."
                if has_data
                else "Insufficient semantic similarity; report missing evidence."
            ),
        )

        payload = RAGQueryPayload(
            query=query,
            conference_filter=conference,
            has_sufficient_data=has_data,
            similarity_threshold=self.similarity_threshold,
            max_steps_limit=1,
            steps_taken=1,
            agent_trace=[step],
            sources=sources,
            constructed_prompt=prompt,
        )
        self._save_logs(payload)
        return payload

    def run_agent(
        self,
        query: str,
        conference: str | None = None,
    ) -> RAGQueryPayload:
        trace: list[AgentStep] = []
        supplementary_sources: list[RetrievedSource] = []
        steps_taken = 0
        candidate_k = max(self.top_k * 4, 20)

        # Step 1: semantic retrieval.
        steps_taken += 1
        vector_results = self.rag.search_vector(
            query=query,
            top_k=candidate_k,
            conference_filter=conference,
        )
        top_vector_score = float(vector_results[0]["score"]) if vector_results else 0.0
        technical_terms = self._technical_terms(query)
        needs_hybrid = (
            not vector_results
            or top_vector_score < max(0.55, self.similarity_threshold)
            or bool(technical_terms)
        )
        trace.append(
            AgentStep(
                step_number=steps_taken,
                tool_name="vector_search",
                tool_input=query,
                observation=(
                    f"Returned {len(vector_results)} semantic candidates; "
                    f"top cosine similarity={top_vector_score:.4f}."
                ),
                decision=(
                    "Run hybrid retrieval because the semantic score is modest or the query contains exact technical terms."
                    if needs_hybrid and steps_taken < self.max_steps
                    else (
                        "Step limit reached; continue with the available semantic candidates."
                        if needs_hybrid
                        else "Semantic retrieval is strong enough; hybrid search is not needed."
                    )
                ),
            )
        )

        current_results = vector_results
        exact_technical_match = False
        if needs_hybrid and steps_taken < self.max_steps:
            # Step 2: refine the query while preserving technical terminology,
            # then fuse semantic and lexical rankings using the RAG engine.
            steps_taken += 1
            refined_query = self._refine_query_keywords(query)
            hybrid_results = self.rag.search_hybrid(
                query=refined_query,
                top_k=candidate_k,
                conference_filter=conference,
            )
            exact_technical_match = self._has_exact_technical_match(query, hybrid_results)
            if hybrid_results:
                current_results = self._rerank_by_exact_terms(query, hybrid_results)

            top_after_rerank = current_results[0] if current_results else None
            top_after_rerank_matches = (
                self._exact_term_matches(query, top_after_rerank)
                if top_after_rerank is not None
                else []
            )
            top_after_rerank_title = (
                str(top_after_rerank.get("metadata", {}).get("title", "Unknown"))
                if top_after_rerank is not None
                else "no result"
            )

            trace.append(
                AgentStep(
                    step_number=steps_taken,
                    tool_name="refine_and_hybrid_search",
                    tool_input=refined_query,
                    observation=(
                        f"Hybrid search returned {len(hybrid_results)} candidates; "
                        f"exact technical-term match={'yes' if exact_technical_match else 'no'}; "
                        f"top after exact-term reranking='{top_after_rerank_title}' "
                        f"({len(top_after_rerank_matches)} technical-term matches)."
                    ),
                    decision=(
                        "Continue with source-grounded context."
                        if top_vector_score >= self.similarity_threshold or exact_technical_match
                        else "No adequate semantic score or distinctive exact-term match; abstain if no other evidence is available."
                    ),
                )
            )

        has_data = bool(current_results) and (
            top_vector_score >= self.similarity_threshold or exact_technical_match
        )

        ranked_results = current_results[: self.top_k] if has_data else []
        sources = self._to_sources(
            ranked_results,
            score_type="rrf" if needs_hybrid and len(trace) >= 2 else "cosine_similarity",
        )

        # Step 3: optionally attach another official review of the leading paper.
        # It is stored separately and never inserted into the ranked list, so it
        # cannot fabricate a retrieval score or alter Hit@5/MRR/nDCG rankings.
        if has_data and sources and steps_taken < self.max_steps:
            # If the query contains distinctive technical terms, only enrich the
            # context when the leading ranked review matches at least one of them.
            # This avoids fetching another review from an obviously unrelated top paper.
            top_ranked_raw = current_results[0] if current_results else {}
            top_matches_terms = bool(self._exact_term_matches(query, top_ranked_raw))
            allow_supplementary = not technical_terms or top_matches_terms

            top_paper_id = sources[0].paper_id
            current_review_ids = {source.review_id for source in sources}
            paper_reviews = self.rag.get_reviews_for_paper(top_paper_id) if allow_supplementary else []
            extra_reviews = [
                result
                for result in paper_reviews
                if result.get("metadata", {}).get("review_id") not in current_review_ids
            ]

            if technical_terms and not top_matches_terms and trace:
                # Do not spend the last agent step enriching an unrelated paper.
                trace[-1].decision = (
                    "The leading result lacks the query's distinctive technical terms; "
                    "skip supplementary retrieval and keep the ranked results."
                )

            if extra_reviews:
                steps_taken += 1
                selected_extra = extra_reviews[0]
                selected_extra = {**selected_extra, "score": 0.0, "score_type": "supplementary"}
                supplementary_sources = self._to_sources(
                    [selected_extra],
                    score_type="supplementary",
                    supplementary=True,
                )
                trace.append(
                    AgentStep(
                        step_number=steps_taken,
                        tool_name="fetch_additional_paper_review",
                        tool_input=top_paper_id,
                        observation=(
                            f"Attached review {supplementary_sources[0].review_id} from the leading paper "
                            "as supplementary context; it is not part of the ranked retrieval results."
                        ),
                        decision="Keep supplementary evidence separate and finalize the prompt.",
                    )
                )

        prompt = (
            self.build_prompt(query, sources, supplementary_sources)
            if has_data
            else None
        )
        payload = RAGQueryPayload(
            query=query,
            conference_filter=conference,
            has_sufficient_data=has_data,
            similarity_threshold=self.similarity_threshold,
            max_steps_limit=self.max_steps,
            steps_taken=steps_taken,
            agent_trace=trace,
            sources=sources,
            constructed_prompt=prompt,
            supplementary_sources=supplementary_sources,
        )
        self._save_logs(payload)
        return payload

    def _save_logs(self, payload: RAGQueryPayload) -> None:
        self.context_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.context_log_path.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.trace_log_path.write_text(
            json.dumps(
                [step.model_dump() for step in payload.agent_trace],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def answer_query(self, query: str, conference: str | None = None) -> None:
        payload = self.run_agent(query, conference)
        print(f"\nPRZEBIEG AGENTA (Kroki: {payload.steps_taken}/{payload.max_steps_limit}):")
        for step in payload.agent_trace:
            print(f"  [Krok {step.step_number}] Narzędzie: {step.tool_name}('{step.tool_input}')")
            print(f"           Obserwacja: {step.observation}")
            print(f"           Decyzja:    {step.decision}")

        if not payload.has_sufficient_data:
            print("\n[BRAK WYSTARCZAJĄCYCH DANYCH W BAZIE]")
            return

        print("\nZESTAWIENIE: DEKLARACJE AUTORÓW VS OPINIE RECENZENTÓW")
        print("=" * 80)
        for idx, source in enumerate(payload.sources, 1):
            print(
                f"ŹRÓDŁO [{idx}] | {source.conference} | "
                f"{source.score_type}: {source.similarity_score:.4f}"
            )
            print(f"Tytuł: {source.title}")
            print(f"Link do recenzji (ID: {source.review_id}): {source.review_url}")
            print(f"Ocena recenzenta: {source.rating}")
            print("-" * 80)
            print(f"  [AUTORZY - Abstrakt]: {source.author_abstract[:250]}...")
            print(f"  [RECENZENT - Zalety]: {source.reviewer_strengths}")
            print(f"  [RECENZENT - Zarzuty/Ograniczenia]: {source.reviewer_weaknesses}")
            print("=" * 80)

        if payload.supplementary_sources:
            print("\nRECENZJE UZUPEŁNIAJĄCE (nie są pozycjonowane według trafności zapytania):")
            for source in payload.supplementary_sources:
                print(f"- {source.title} | review_id={source.review_id} | {source.review_url}")

        print("\nPrompt z kontekstem został zapisany w data/last_query_context.json.")


if __name__ == "__main__":
    assistant = LiteratureReviewAssistant()
    while True:
        try:
            user_query = input("\nPodaj pytanie badawcze ('exit' aby zakończyć): ").strip()
            if user_query.casefold() in {"exit", "quit"}:
                break
            if user_query:
                assistant.answer_query(user_query)
        except (KeyboardInterrupt, EOFError):
            print("\nZamykanie asystenta.")
            break
