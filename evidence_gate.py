"""Conservative consistency check for explicitly versioned ML model names.

This is a *necessary condition* for named-model queries, not an entailment
checker or a guarantee that evidence answers the question. A model/version
mentioned in a document may be unrelated to the requested comparison.

The design was motivated by Q106, so Q106 is NOT a new, untouched holdout
for evaluating this gate. Evaluate generalization on fresh questions.
"""

from __future__ import annotations

import re
from typing import Any

# Fixed model-family vocabulary; not a list of forbidden model versions.
# Queries without a recognized versioned model are unaffected by this gate.
_MODEL_NAMES = re.compile(
    r"\b(?:gpt|claude|gemini|llama|mistral|qwen|palm|phi|deepseek)"
    r"\s+\d+[a-z]?(?:\s+\d+[a-z]?)?\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Normalize GPT-4, GPT - 4 and GPT 4 to the same surface form."""
    text = re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE)
    return re.sub(r"(?<=[a-z])(?=\d)", " ", text).strip()


def required_model_entities(query: str) -> tuple[str, ...]:
    normalized = _normalize(query)
    return tuple(dict.fromkeys(m.group(0) for m in _MODEL_NAMES.finditer(normalized)))


def _evidence_text(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    # A model mentioned only in a separate paper's review must not be
    # combined with models in other papers to fabricate a comparison.
    fields = (
        "title", "author_abstract", "reviewer_strengths", "reviewer_weaknesses",
        "reviewer_limitations", "reviewer_summary", "reviewer_questions",
    )
    return _normalize(" ".join(
        [str(metadata.get(name) or "") for name in fields]
        + [str(result.get("text_snippet") or "")]
    ))


def has_joint_model_support(query: str, results: list[dict[str, Any]]) -> bool:
    """Require all explicitly versioned models in at least ONE source.

    Text coverage is not proof that the requested relation (e.g., energy
    comparison) is supported. Such checks need a separate claim verifier.
    """
    required = required_model_entities(query)
    if not required:
        return True
    for result in results:
        padded = f" {_evidence_text(result)} "
        if all(f" {term} " in padded for term in required):
            return True
    return False
