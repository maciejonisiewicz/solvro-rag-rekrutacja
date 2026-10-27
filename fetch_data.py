import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import openreview
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)


class ConferenceConfig(BaseModel):
    venue_id: str
    short_name: str
    year: int


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    abstract: str
    keywords: list[str]
    primary_area: str
    conference: str
    venue_id: str
    year: int
    url: str
    review_count: int


class ReviewRecord(BaseModel):
    review_id: str
    paper_id: str
    conference: str
    year: int
    forum_url: str
    text_fields: dict[str, str]
    scores: dict[str, Any]
    available_fields: list[str]


class ConferenceSamplingStats(BaseModel):
    venue_id: str
    papers_count: int
    reviews_count: int
    paper_ids: list[str]
    filtered_out_non_review_notes: dict[str, int]


class SamplingMetadata(BaseModel):
    download_timestamp_utc: str
    sampling_method: str
    topic_keywords: list[str]
    total_papers: int
    total_reviews: int
    conferences: dict[str, ConferenceSamplingStats]


CONFERENCES: list[ConferenceConfig] = [
    ConferenceConfig(venue_id="ICLR.cc/2024/Conference", short_name="ICLR 2024", year=2024),
    ConferenceConfig(venue_id="NeurIPS.cc/2024/Conference", short_name="NeurIPS 2024", year=2024),
]

TARGET_PAPERS_PER_CONF = 40
FETCH_BATCH_SIZE = 200

TOPIC_KEYWORDS: list[str] = [
    "language model", "llm", "inference", "transformer", "attention",
    "quantization", "pruning", "efficient", "efficiency", "compression",
    "acceleration", "decoding", "speculative", "kv cache", "token",
    "fine-tuning", "finetuning", "reasoning", "retrieval", "context",
]

TEXT_FIELDS: list[str] = ["summary", "strengths", "weaknesses", "questions", "limitations"]
SCORE_FIELDS: list[str] = ["rating", "confidence", "soundness", "presentation", "contribution"]


def matches_topic(title: str, abstract: str, keywords: list[str]) -> bool:
    combined_text = f"{title} {abstract} {' '.join(keywords)}".lower()
    return any(kw in combined_text for kw in TOPIC_KEYWORDS)


def extract_official_reviews(
    replies: list[dict[str, Any]],
    paper_id: str,
    conf_name: str,
    year: int,
) -> tuple[list[ReviewRecord], dict[str, int]]:
    reviews: list[ReviewRecord] = []
    skipped_types: dict[str, int] = {}

    for reply in replies:
        invs = reply.get("invitations", [])
        inv_types = [inv.split("/")[-1] for inv in invs]

        if "Official_Review" not in inv_types:
            for t in inv_types:
                skipped_types[t] = skipped_types.get(t, 0) + 1
            continue

        content = reply.get("content", {})
        review_id = str(reply.get("id", ""))

        text_data = {
            field: str(content[field]["value"]).strip()
            for field in TEXT_FIELDS
            if field in content and content[field].get("value")
        }

        scores_data = {
            field: content[field]["value"]
            for field in SCORE_FIELDS
            if field in content and content[field].get("value") is not None
        }

        if not text_data or not review_id:
            continue

        reviews.append(
            ReviewRecord(
                review_id=review_id,
                paper_id=paper_id,
                conference=conf_name,
                year=year,
                forum_url=f"https://openreview.net/forum?id={paper_id}&noteId={review_id}",
                text_fields=text_data,
                scores=scores_data,
                available_fields=list(content.keys()),
            )
        )

    return reviews, skipped_types


def fetch_and_save_dataset(data_dir: Path = Path("data")) -> None:
    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=os.getenv("OPENREVIEW_USERNAME"),
        password=os.getenv("OPENREVIEW_PASSWORD"),
    )

    data_dir.mkdir(parents=True, exist_ok=True)

    all_papers: list[PaperRecord] = []
    all_reviews: list[ReviewRecord] = []
    sampling_stats: dict[str, ConferenceSamplingStats] = {}

    for conf in CONFERENCES:
        submissions = client.get_notes(
            content={"venueid": conf.venue_id},
            details="replies",
            limit=FETCH_BATCH_SIZE,
        )

        selected_papers: list[PaperRecord] = []
        conf_skipped_notes: dict[str, int] = {}

        for sub in submissions:
            if len(selected_papers) >= TARGET_PAPERS_PER_CONF:
                break

            content = sub.content
            title = str(content.get("title", {}).get("value", "")).strip()
            abstract = str(content.get("abstract", {}).get("value", "")).strip()
            keywords = list(content.get("keywords", {}).get("value", []))

            if not title or not abstract or not matches_topic(title, abstract, keywords):
                continue

            replies = sub.details.get("replies", []) if sub.details else []
            reviews, skipped = extract_official_reviews(replies, sub.id, conf.short_name, conf.year)

            if not reviews:
                continue

            for k, v in skipped.items():
                conf_skipped_notes[k] = conf_skipped_notes.get(k, 0) + v

            paper_record = PaperRecord(
                paper_id=sub.id,
                title=title,
                abstract=abstract,
                keywords=keywords,
                primary_area=str(content.get("primary_area", {}).get("value", "")),
                conference=conf.short_name,
                venue_id=conf.venue_id,
                year=conf.year,
                url=f"https://openreview.net/forum?id={sub.id}",
                review_count=len(reviews),
            )

            selected_papers.append(paper_record)
            all_papers.append(paper_record)
            all_reviews.extend(reviews)

        sampling_stats[conf.short_name] = ConferenceSamplingStats(
            venue_id=conf.venue_id,
            papers_count=len(selected_papers),
            reviews_count=sum(p.review_count for p in selected_papers),
            paper_ids=[p.paper_id for p in selected_papers],
            filtered_out_non_review_notes=conf_skipped_notes,
        )

    metadata = SamplingMetadata(
        download_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        sampling_method=(
            f"Fetched top {FETCH_BATCH_SIZE} accepted submissions per venue and selected "
            f"{TARGET_PAPERS_PER_CONF} papers matching LLM/inference keywords with public Official_Review notes."
        ),
        topic_keywords=TOPIC_KEYWORDS,
        total_papers=len(all_papers),
        total_reviews=len(all_reviews),
        conferences=sampling_stats,
    )

    (data_dir / "papers.json").write_text(
        json.dumps([p.model_dump() for p in all_papers], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (data_dir / "reviews.json").write_text(
        json.dumps([r.model_dump() for r in all_reviews], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (data_dir / "sampling_metadata.json").write_text(
        json.dumps(metadata.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Zapisano {len(all_papers)} publikacji oraz {len(all_reviews)} recenzji w katalogu '{data_dir}/'.")


if __name__ == "__main__":
    fetch_and_save_dataset()