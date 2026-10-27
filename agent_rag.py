import json
import re
from pathlib import Path
from pydantic import BaseModel
from rag_system import LiteratureReviewRAG


STOPWORDS = {
    "which", "what", "where", "when", "how", "does", "do", "is", "are", "the",
    "a", "an", "in", "on", "of", "for", "to", "with", "and", "or", "by", "from",
    "that", "this", "why", "can", "be", "as", "too", "it", "its", "their", "about",
}


class RetrievedSource(BaseModel):
    review_id: str
    paper_id: str
    conference: str
    title: str
    paper_url: str
    review_url: str
    rating: str
    similarity_score: float
    author_abstract: str
    reviewer_strengths: str
    reviewer_weaknesses: str


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


class LiteratureReviewAssistant:
    def __init__(
        self,
        similarity_threshold: float = 0.38,
        top_k: int = 4,
        max_steps: int = 3,
    ) -> None:
        self.rag = LiteratureReviewRAG()
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.max_steps = max_steps
        self.context_log_path = Path("data/last_query_context.json")
        self.trace_log_path = Path("data/agent_trace_log.json")

    def _to_sources(self, raw_results: list[dict]) -> list[RetrievedSource]:
        return [
            RetrievedSource(
                review_id=res["metadata"]["review_id"],
                paper_id=res["metadata"]["paper_id"],
                conference=res["metadata"]["conference"],
                title=res["metadata"]["title"],
                paper_url=res["metadata"]["paper_url"],
                review_url=res["metadata"]["review_url"],
                rating=str(res["metadata"]["rating"]),
                similarity_score=round(float(res["score"]), 4),
                author_abstract=res["metadata"]["author_abstract"],
                reviewer_strengths=res["metadata"]["reviewer_strengths"],
                reviewer_weaknesses=res["metadata"]["reviewer_weaknesses"],
            )
            for res in raw_results
        ]

    def _refine_query_keywords(self, query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9\-]+", query)
        kept = [t for t in tokens if t.lower() not in STOPWORDS and len(t) > 1]
        return " ".join(kept) if kept else query

    def build_prompt(self, query: str, sources: list[RetrievedSource]) -> str:
        context_blocks = [
            (
                f"[SOURCE {idx}]\n"
                f"Paper Title: {src.title} ({src.conference})\n"
                f"Paper ID: {src.paper_id} | Review ID: {src.review_id}\n"
                f"Review Link: {src.review_url}\n"
                f"Reviewer Rating: {src.rating}\n"
                f"Author Claims (Abstract): {src.author_abstract}\n"
                f"Reviewer Strengths: {src.reviewer_strengths}\n"
                f"Reviewer Weaknesses & Limitations: {src.reviewer_weaknesses}"
            )
            for idx, src in enumerate(sources, 1)
        ]
        joined_context = "\n\n".join(context_blocks)
        return (
            "You are a machine learning literature review assistant. "
            "Answer the user question strictly based on the provided OpenReview sources.\n"
            "Rules:\n"
            "1. Explicitly distinguish between author claims (from abstracts) and reviewer opinions.\n"
            "2. Cite every claim with [SOURCE X], including the paper title and direct OpenReview link.\n"
            "3. Do not extrapolate beyond the provided text. If sources conflict, highlight the discrepancy.\n\n"
            f"CONTEXT:\n{joined_context}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            "ANSWER:"
        )

    def process_query(self, query: str, conference: str | None = None) -> RAGQueryPayload:
        raw_results = self.rag.search_vector(
            query=query,
            top_k=self.top_k,
            conference_filter=conference,
        )
        sources = self._to_sources(raw_results)
        has_data = bool(sources) and sources[0].similarity_score >= self.similarity_threshold
        prompt = self.build_prompt(query, sources) if has_data else None

        step = AgentStep(
            step_number=1,
            tool_name="vector_search",
            tool_input=query,
            observation=f"Returned {len(sources)} results; top similarity={sources[0].similarity_score if sources else 0.0}",
            decision="Single-step search completed.",
        )

        payload = RAGQueryPayload(
            query=query,
            conference_filter=conference,
            has_sufficient_data=has_data,
            similarity_threshold=self.similarity_threshold,
            max_steps_limit=1,
            steps_taken=1,
            agent_trace=[step],
            sources=sources if has_data else [],
            constructed_prompt=prompt,
        )
        self._save_logs(payload)
        return payload

    def run_agent(self, query: str, conference: str | None = None) -> RAGQueryPayload:
        trace: list[AgentStep] = []
        steps_taken = 0

        steps_taken += 1
        vec_raw = self.rag.search_vector(query=query, top_k=self.top_k, conference_filter=conference)
        top_vec_score = round(float(vec_raw[0]["score"]), 4) if vec_raw else 0.0

        if top_vec_score < self.similarity_threshold:
            trace.append(
                AgentStep(
                    step_number=steps_taken,
                    tool_name="vector_search",
                    tool_input=query,
                    observation=f"Top semantic similarity ({top_vec_score}) is below threshold ({self.similarity_threshold}).",
                    decision="Abort and report insufficient data in database.",
                )
            )
            payload = RAGQueryPayload(
                query=query,
                conference_filter=conference,
                has_sufficient_data=False,
                similarity_threshold=self.similarity_threshold,
                max_steps_limit=self.max_steps,
                steps_taken=steps_taken,
                agent_trace=trace,
                sources=[],
                constructed_prompt=None,
            )
            self._save_logs(payload)
            return payload

        needs_refinement = top_vec_score < 0.55 or any(char.isdigit() or "-" in char for char in query)
        trace.append(
            AgentStep(
                step_number=steps_taken,
                tool_name="vector_search",
                tool_input=query,
                observation=f"Found {len(vec_raw)} semantic candidates (top score={top_vec_score}).",
                decision=(
                    "Refine query keywords and run hybrid RRF search to capture exact technical terms."
                    if needs_refinement and steps_taken < self.max_steps
                    else "Semantic confidence high; proceed with vector results."
                ),
            )
        )

        current_raw = vec_raw

        if needs_refinement and steps_taken < self.max_steps:
            steps_taken += 1
            refined_query = self._refine_query_keywords(query)
            hybrid_raw = self.rag.search_hybrid(
                query=refined_query,
                top_k=self.top_k,
                conference_filter=conference,
            )
            merged_by_id: dict[str, dict] = {}
            for item in hybrid_raw + vec_raw:
                rid = item["metadata"]["review_id"]
                if rid not in merged_by_id:
                    merged_by_id[rid] = item
            current_raw = list(merged_by_id.values())[: self.top_k]

            trace.append(
                AgentStep(
                    step_number=steps_taken,
                    tool_name="refine_and_hybrid_search",
                    tool_input=refined_query,
                    observation=f"Hybrid RRF re-ranked candidates; top paper is '{current_raw[0]['metadata']['title']}'.",
                    decision=(
                        "Check if additional reviews for the top paper should be fetched."
                        if steps_taken < self.max_steps
                        else "Step limit reached; finalize context."
                    ),
                )
            )

        if steps_taken < self.max_steps and current_raw:
            top_paper_id = current_raw[0]["metadata"]["paper_id"]
            existing_rev_ids = {r["metadata"]["review_id"] for r in current_raw}
            paper_reviews = self.rag.get_reviews_for_paper(top_paper_id)
            extra_reviews = [r for r in paper_reviews if r["metadata"]["review_id"] not in existing_rev_ids]

            if extra_reviews:
                steps_taken += 1
                supplementary = extra_reviews[0]
                supplementary["score"] = current_raw[0]["score"]
                current_raw = current_raw[: max(1, self.top_k - 1)] + [supplementary]
                trace.append(
                    AgentStep(
                        step_number=steps_taken,
                        tool_name="fetch_additional_paper_reviews",
                        tool_input=top_paper_id,
                        observation=f"Fetched supplementary review {supplementary['metadata']['review_id']} for paper {top_paper_id}.",
                        decision="Context enriched with multi-reviewer perspective; finish execution.",
                    )
                )

        sources = self._to_sources(current_raw)
        if sources and sources[0].similarity_score < self.similarity_threshold:
            sources[0].similarity_score = top_vec_score

        prompt = self.build_prompt(query, sources)
        payload = RAGQueryPayload(
            query=query,
            conference_filter=conference,
            has_sufficient_data=True,
            similarity_threshold=self.similarity_threshold,
            max_steps_limit=self.max_steps,
            steps_taken=steps_taken,
            agent_trace=trace,
            sources=sources,
            constructed_prompt=prompt,
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
            json.dumps([step.model_dump() for step in payload.agent_trace], ensure_ascii=False, indent=2),
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
        for idx, src in enumerate(payload.sources, 1):
            print(f"ŹRÓDŁO [{idx}] | {src.conference} | Score: {src.similarity_score:.3f}")
            print(f"Tytuł: {src.title}")
            print(f"Link do recenzji (ID: {src.review_id}): {src.review_url}")
            print(f"Ocena recenzenta: {src.rating}")
            print("-" * 80)
            print(f"  [AUTORZY - Abstrakt]: {src.author_abstract[:250]}...")
            print(f"  [RECENZENT - Zalety]: {src.reviewer_strengths}")
            print(f"  [RECENZENT - Zarzuty/Ograniczenia]: {src.reviewer_weaknesses}")
            print("=" * 80)


if __name__ == "__main__":
    assistant = LiteratureReviewAssistant()
    while True:
        try:
            user_query = input("\nPodaj pytanie badawcze ('exit' aby zakończyć): ").strip()
            if user_query.lower() in {"exit", "quit"}:
                break
            if user_query:
                assistant.answer_query(user_query)
        except KeyboardInterrupt:
            break