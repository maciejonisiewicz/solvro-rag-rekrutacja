"""Capture real RAG/LLM claims with their exact cited OpenReview evidence.

This tool PREPARES a manual faithfulness audit. It intentionally does not grade
claim entailment automatically: correct citation indices are not proof of support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable


SOURCE_PATTERN = re.compile(r"\[SOURCE (\d+)\]")
VERDICTS = {"supported", "unsupported", "unclear"}


def extract_claims(answer: str) -> list[dict[str, Any]]:
    """Parse the *validated* paragraphs from llm_synthesis.render_synthesis."""
    if not answer or not answer.strip():
        raise ValueError("No generated answer to annotate")
    claims = []
    for paragraph in answer.strip().split("\n\n"):
        marker = paragraph.find("  [SOURCE ")
        if marker < 1:
            raise ValueError("Expected a paragraph with a source citation")
        text = paragraph[:marker].strip()
        source_ids = list(dict.fromkeys(int(x) for x in SOURCE_PATTERN.findall(paragraph[marker:])))
        if not text or not source_ids:
            raise ValueError("Claim without text or citation")
        claims.append({
            "claim_id": len(claims) + 1,
            "text": text,
            "cited_source_numbers": source_ids,
            "verdict": None,
            "supporting_quote": "",
            "reviewer_note": "",
        })
    return claims


def _source_snapshot(source: Any, number: int) -> dict[str, Any]:
    """Preserve FULL evidence: never shorten reviewer text for manual judging."""
    fields = (
        "paper_id", "review_id", "title", "conference", "paper_url",
        "review_url", "is_abstract_only", "is_supplementary", "author_abstract",
        "reviewer_summary", "reviewer_strengths", "reviewer_weaknesses",
        "reviewer_limitations", "reviewer_questions",
    )
    result: dict[str, Any] = {"source_number": number}
    for field in fields:
        result[field] = getattr(source, field, None)
    return result


def build_audit(
    questions: list[dict[str, Any]],
    *,
    assistant: Any,
    synthesis_fn: Callable[..., Any],
    model_name: str,
    limit: int = 5,
) -> dict[str, Any]:
    """Collect outputs from answerable questions, without modifying gold labels."""
    if limit < 1:
        raise ValueError("limit must be positive")
    selected = [q for q in questions if q.get("has_data") is True][:limit]
    if not selected:
        raise ValueError("No answerable questions in selected set")
    cases = []
    for q in selected:
        query = str(q["query"])
        payload = assistant.run_agent(query)
        result = synthesis_fn(payload, model_name=model_name)
        sources = list(payload.sources) + list(getattr(payload, "supplementary_sources", []))
        case = {
            "question_id": q["id"],
            "query": query,
            "agent_has_sufficient_data": bool(payload.has_sufficient_data),
            "generation_status": result.status,
            "generation_reason": result.reason,
            "model": result.model_name or model_name,
            "rendered_answer": result.answer,
            "context_sha256": hashlib.sha256(
                (payload.constructed_prompt or "").encode("utf-8")
            ).hexdigest(),
            "context_sent_to_model": payload.constructed_prompt,
            "claims": [],
            "sources": [_source_snapshot(src, i) for i, src in enumerate(sources, 1)],
        }
        if result.status == "generated":
            case["claims"] = extract_claims(result.answer)
            valid = set(result.validated_source_numbers)
            for claim in case["claims"]:
                if not set(claim["cited_source_numbers"]).issubset(valid):
                    raise ValueError("Inconsistent rendered citation and validated numbers")
                if max(claim["cited_source_numbers"]) > len(sources):
                    raise ValueError("Claim references missing source")
        cases.append(case)
    return {
        "schema": "solvro-manual-faithfulness-v1",
        "purpose": "Human annotation of model-generated claims using only cited OpenReview fields",
        "instructions": (
            "For each atomic, checkable claim: verdict=supported only if the cited "
            "author/reviewer text explicitly supports ALL its factual content. "
            "Otherwise unsupported; unclear if ambiguous. Quote the exact supporting "
            "passage. For compound claims, note that all subclaims must be supported. "
            "Do not infer faithfulness from a valid citation number or URL."
        ),
        "model": model_name,
        "sample": "Known answerable evaluation questions, NOT an independent holdout",
        "cases": cases,
    }


def compute_summary(document: dict[str, Any]) -> dict[str, int | float | None]:
    all_claims = [claim for case in document["cases"] for claim in case["claims"]]
    annotated = [claim for claim in all_claims if claim.get("verdict") in VERDICTS]
    invalid = [claim for claim in all_claims if claim.get("verdict") is not None and claim.get("verdict") not in VERDICTS]
    if invalid:
        raise ValueError("Unknown annotation verdict; use supported/unsupported/unclear/null")
    n_supported = sum(claim["verdict"] == "supported" for claim in annotated)
    return {
        "generated_cases": sum(case["generation_status"] == "generated" for case in document["cases"]),
        "total_cases": len(document["cases"]),
        "total_claims": len(all_claims),
        "reviewed_claims": len(annotated),
        "supported_claims": n_supported,
        "unsupported_claims": sum(claim["verdict"] == "unsupported" for claim in annotated),
        "unclear_claims": sum(claim["verdict"] == "unclear" for claim in annotated),
        "faithfulness_fraction": (n_supported / len(annotated)) if annotated else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect", "score"])
    parser.add_argument("--questions", default="data/eval_questions.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="google:gemini-3.5-flash-lite")
    parser.add_argument("--file", default="data/experiments/faithfulness_manual_audit.json")
    args = parser.parse_args()
    path = Path(args.file)
    if args.action == "score":
        document = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(compute_summary(document), indent=2, ensure_ascii=False))
        return

    if path.exists():
        raise SystemExit(f"Plik juz istnieje: {path}. Nie nadpisuje recznych ocen. Wybierz --file.")
    if not __import__("os").getenv("GOOGLE_API_KEY"):
        raise SystemExit("Brak GOOGLE_API_KEY. Uruchom w terminalu z poprawnym kluczem.")

    from agent_rag import LiteratureReviewAssistant
    from llm_synthesis import synthesize

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise SystemExit("Pytania musza byc lista w pliku JSON")
    assistant = LiteratureReviewAssistant()
    print(f"Przygotowuje maksymalnie {args.limit} odpowiedzi. Kazda moze wywolac platne API Gemini.")
    document = build_audit(
        questions, assistant=assistant, synthesis_fn=synthesize,
        model_name=args.model, limit=args.limit,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for case in document["cases"]:
        print(f"Q{case['question_id']:03d}: {case['generation_status']}; claims={len(case['claims'])}")
    summary = compute_summary(document)
    print(f"Claims to review: {summary['total_claims']}. Verified so far: 0.")
    print(f"Zapisano: {path}")
    print("To NIE jest automatyczny wynik faithfulness. Twierdzenia wymagaja recznej oceny.")


if __name__ == "__main__":
    main()
