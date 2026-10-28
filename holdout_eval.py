"""Review and run an independent, frozen holdout for the OpenReview RAG.

Usage, in the repository root:
  python holdout_eval.py --verify
  python holdout_eval.py --run

This tool never edits the primary data/eval_questions.json or data/eval_results.json.
Candidate qrels are PROPOSALS until a human checks the linked review text and
confirms every label *before* retrieval results are displayed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CANDIDATES = DATA / "eval_holdout_candidates.json"
VERIFIED = DATA / "eval_holdout_verified.json"
REPORT = DATA / "eval_holdout_results.json"
ORIGINAL = DATA / "eval_questions.json"
EXPECTED_ANSWERABLE = 4
EXPECTED_UNANSWERABLE = 2


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_candidates(questions: list[dict], originals: list[dict], reviews: list[dict]) -> dict:
    if not isinstance(questions, list):
        raise ValueError("Kandydaci muszą być listą JSON.")
    if sum(bool(q.get("has_data")) for q in questions) != EXPECTED_ANSWERABLE:
        raise ValueError("Holdout musi mieć 4 pytania z odpowiedzią.")
    if sum(not bool(q.get("has_data")) for q in questions) != EXPECTED_UNANSWERABLE:
        raise ValueError("Holdout musi mieć 2 pytania bez odpowiedzi.")

    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Powtarzające się ID pytań.")
    old_queries = {q["query"].strip().casefold() for q in originals}
    old_papers = {q["target_paper_id"] for q in originals if q.get("target_paper_id")}
    by_review = {r["review_id"]: r for r in reviews}
    seen_papers = set()
    for q in questions:
        if q["query"].strip().casefold() in old_queries:
            raise ValueError(f"H{q['id']}: pytanie powtórzone ze starej ewaluacji.")
        if not q["query"].strip():
            raise ValueError(f"H{q['id']}: puste pytanie.")
        if q["has_data"]:
            paper_id = q["target_paper_id"]
            review_id = q["target_review_id"]
            if paper_id in old_papers:
                raise ValueError(f"H{q['id']}: publikacja była w starej ewaluacji.")
            if paper_id in seen_papers:
                raise ValueError(f"H{q['id']}: dwukrotnie użyta publikacja.")
            seen_papers.add(paper_id)
            review = by_review.get(review_id)
            if review is None or review["paper_id"] != paper_id:
                raise ValueError(f"H{q['id']}: błędne powiązanie review-paper.")
            if not q["justification"].strip():
                raise ValueError(f"H{q['id']}: brak uzasadnienia.")
        elif q.get("target_review_id") is not None or q.get("target_paper_id") is not None:
            raise ValueError(f"H{q['id']}: pytanie bez danych nie może mieć targetów.")
    return by_review


def verify() -> None:
    candidates = _load_json(CANDIDATES)
    originals = _load_json(ORIGINAL)
    reviews = _load_json(DATA / "reviews.json")
    by_review = _validate_candidates(candidates, originals, reviews)
    if VERIFIED.exists():
        raise FileExistsError(
            f"{VERIFIED} już istnieje. Nie wolno nadpisywać ground truth po zobaczeniu rankingu."
        )

    print("\nNIEZALEŻNY TEST HOLDOUT: 4 nowe publikacje + 2 pytania bez danych.")
    print("Oceń proponowane etykiety na podstawie recenzji PRZED uruchomieniem retrievalu.")
    print("Odpowiedz 't' tylko po samodzielnym sprawdzeniu źródła, 'n' przerywa cały test.\n")
    for q in candidates:
        print("=" * 88)
        print(f"H{q['id']}: {q['query']}")
        print(f"Uzasadnienie: {q['justification']}")
        if q["has_data"]:
            review = by_review[q["target_review_id"]]
            print(f"paper_id: {q['target_paper_id']} | review_id: {q['target_review_id']}")
            print(f"OpenReview: {review.get('forum_url', '')}")
            field = q.get("evidence_field", "weaknesses")
            excerpt = str(review.get("text_fields", {}).get(field, ""))
            print(f"\nTREŚĆ ŹRÓDŁA [{field}]:\n{excerpt[:2300]}")
            if len(excerpt) > 2300:
                print("[dalsza część pominięta — w razie potrzeby otwórz link OpenReview]")
            print("\nCzy WŁASNORĘCZNIE potwierdzasz, że recenzja uzasadnia to pytanie?")
        else:
            print("\nCzy potwierdzasz, że w tym korpusie BRAKUJE danych do odpowiedzi?")
            print("Uwaga: sam brak dopasowania słów nie jest wystarczającym uzasadnieniem.")
        reply = input("Potwierdź [t/N]: ").strip().casefold()
        if reply not in {"t", "tak", "y", "yes"}:
            raise SystemExit("Weryfikacja przerwana. Nie zapisano żadnych etykiet.")

    from evaluate import GroundTruthQuestion

    verified = [
        GroundTruthQuestion.model_validate({
            key: q[key]
            for key in ("id", "query", "has_data", "target_paper_id", "target_review_id", "justification")
        }).model_dump()
        for q in candidates
    ]
    VERIFIED.write_text(json.dumps(verified, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nZweryfikowane etykiety ZAPISANE: {VERIFIED}")
    print("Teraz uruchom: python holdout_eval.py --run")


def run() -> None:
    if not VERIFIED.exists():
        raise SystemExit("Najpierw wykonaj ręczną weryfikację: python holdout_eval.py --verify")
    questions = _load_json(VERIFIED)
    original = _load_json(ORIGINAL)
    reviews = _load_json(DATA / "reviews.json")
    _validate_candidates(questions, original, reviews)

    from evaluate import run_evaluation, print_console_report

    print("\nUruchamiam HOLDOUT. Pierwotny raport i ground truth pozostaną bez zmian.")
    report = run_evaluation(questions_path=VERIFIED, report_path=REPORT)
    print_console_report(report)
    print("\nWAŻNE: jest to tylko 4-pytaniowy holdout; nie dopasowuj do niego heurystyk.")
    print(f"Raport: {REPORT}")
    print(f"Etykiety SHA-256: {report.ground_truth_sha256}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual-review holdout benchmark")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true", help="Zatwierdź źródła PRZED wyszukiwaniem")
    group.add_argument("--run", action="store_true", help="Uruchom zamrożony holdout")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        run()


if __name__ == "__main__":
    main()
