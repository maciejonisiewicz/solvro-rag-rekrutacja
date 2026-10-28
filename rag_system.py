import hashlib
import json
from pathlib import Path
from typing import Any
import chromadb
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class ReviewDocumentMetadata(BaseModel):
    review_id: str
    paper_id: str
    conference: str
    title: str
    paper_url: str
    review_url: str
    rating: str
    author_abstract: str
    reviewer_strengths: str
    reviewer_weaknesses: str


class LiteratureReviewRAG:
    def __init__(
        self,
        data_dir: str | Path = "data",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "openreview_reviews_v2",
    ) -> None:
        self.data_path = Path(data_dir)
        papers_raw = json.loads((self.data_path / "papers.json").read_text(encoding="utf-8"))
        self.papers: dict[str, dict[str, Any]] = {p["paper_id"]: p for p in papers_raw}
        self.reviews: list[dict[str, Any]] = json.loads(
            (self.data_path / "reviews.json").read_text(encoding="utf-8")
        )

        self.corpus_for_embedding: list[str] = []
        self.corpus_for_bm25: list[str] = []
        self.corpus_abstracts_only: list[str] = []
        self.corpus_metadata: list[dict[str, Any]] = []

        self._prepare_corpus()

        tokenized_corpus = [doc.lower().split() for doc in self.corpus_for_bm25]
        self.bm25 = BM25Okapi(tokenized_corpus)

        tokenized_abstracts = [doc.lower().split() for doc in self.corpus_abstracts_only]
        self.bm25_abstracts = BM25Okapi(tokenized_abstracts)

        self.chroma_client = chromadb.PersistentClient(path=str(self.data_path / "chroma_db"))
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._ensure_index(embedding_model_name, collection_name)

    def _prepare_corpus(self) -> None:
        for rev in self.reviews:
            paper = self.papers.get(rev["paper_id"])
            if not paper:
                continue

            tf = rev.get("text_fields", {})
            weaknesses = tf.get("weaknesses", "")
            limitations = tf.get("limitations", "")
            strengths = tf.get("strengths", "")
            summary = tf.get("summary", "")
            questions = tf.get("questions", "")

            review_only_text = (
                f"Paper Title: {paper['title']}\n"
                f"Keywords: {', '.join(paper.get('keywords', []))}\n"
                f"Reviewer Weaknesses & Criticisms: {weaknesses}\n"
                f"Reviewer Limitations: {limitations}\n"
                f"Reviewer Strengths: {strengths}\n"
                f"Reviewer Summary: {summary}\n"
                f"Reviewer Questions: {questions}"
            )

            abstract_only_text = (
                f"Paper Title: {paper['title']}\n"
                f"Keywords: {', '.join(paper.get('keywords', []))}\n"
                f"Author Abstract: {paper['abstract']}"
            )

            full_bm25_text = f"{review_only_text}\nAuthor Abstract: {paper['abstract']}"
            rating = str(rev.get("scores", {}).get("rating", "N/A"))
            combined_weaknesses = f"{weaknesses} {limitations}".strip()

            meta_model = ReviewDocumentMetadata(
                review_id=rev["review_id"],
                paper_id=paper["paper_id"],
                conference=rev["conference"],
                title=paper["title"],
                paper_url=paper["url"],
                review_url=rev["forum_url"],
                rating=rating,
                author_abstract=paper["abstract"],
                reviewer_strengths=strengths[:400] + ("..." if len(strengths) > 400 else ""),
                reviewer_weaknesses=combined_weaknesses[:500]
                + ("..." if len(combined_weaknesses) > 500 else ""),
            )

            self.corpus_for_embedding.append(review_only_text)
            self.corpus_for_bm25.append(full_bm25_text)
            self.corpus_abstracts_only.append(abstract_only_text)
            self.corpus_metadata.append(meta_model.model_dump())

    def _ensure_index(self, embedding_model_name: str, collection_name: str) -> None:
        """Rebuild an outdated Chroma collection when corpus or model changes.

        The fingerprint is deterministic and covers every indexed text, its metadata,
        and the embedding model name. Existing collections without a fingerprint
        are rebuilt once on first run after upgrading.
        """
        payload = {
            "embedding_model": embedding_model_name,
            "documents": self.corpus_for_embedding,
            "metadata": self.corpus_metadata,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        previous_metadata = self.collection.metadata or {}
        is_current = (
            self.collection.count() == len(self.corpus_metadata)
            and previous_metadata.get("corpus_fingerprint") == fingerprint
        )
        if is_current:
            return

        if not self.corpus_metadata:
            raise ValueError("Cannot build Chroma index: no reviews in the corpus.")

        # Calculate embeddings BEFORE replacing an existing collection. If model
        # encoding fails, the original index is preserved.
        embeddings = self.embedding_model.encode(
            self.corpus_for_embedding, show_progress_bar=True
        ).tolist()
        self.chroma_client.delete_collection(name=collection_name)
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "corpus_fingerprint": fingerprint},
        )
        self._index_documents(embeddings=embeddings)

    def _index_documents(self, embeddings: list[list[float]] | None = None) -> None:
        ids = [m["review_id"] for m in self.corpus_metadata]
        if embeddings is None:
            embeddings = self.embedding_model.encode(
                self.corpus_for_embedding,
                show_progress_bar=True,
            ).tolist()
        self.collection.add(
            ids=ids,
            documents=self.corpus_for_embedding,
            embeddings=embeddings,
            metadatas=self.corpus_metadata,
        )


    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        conference_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[dict[str, Any]] = []
        for idx in ranked_indices:
            meta = self.corpus_metadata[idx]
            if conference_filter and meta["conference"] != conference_filter:
                continue
            results.append(
                {
                    "score": float(scores[idx]),
                    "metadata": meta,
                    "text_snippet": self.corpus_for_bm25[idx],
                }
            )
            if len(results) >= top_k:
                break
        return results

    def search_abstracts_only(
        self,
        query: str,
        top_k: int = 5,
        conference_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        tokenized_query = query.lower().split()
        scores = self.bm25_abstracts.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[dict[str, Any]] = []
        seen_papers: set[str] = set()
        for idx in ranked_indices:
            meta = self.corpus_metadata[idx]
            if conference_filter and meta["conference"] != conference_filter:
                continue
            if meta["paper_id"] in seen_papers:
                continue
            seen_papers.add(meta["paper_id"])
            results.append(
                {
                    "score": float(scores[idx]),
                    "metadata": meta,
                    "text_snippet": self.corpus_abstracts_only[idx],
                }
            )
            if len(results) >= top_k:
                break
        return results

    def search_vector(
        self,
        query: str,
        top_k: int = 5,
        conference_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query]).tolist()
        where_filter = {"conference": conference_filter} if conference_filter else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k * 3, len(self.corpus_metadata)),
            where=where_filter,
        )

        formatted_results: list[dict[str, Any]] = []
        ids_list = results.get("ids", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]
        documents_list = results.get("documents", [[]])[0]

        for i in range(len(ids_list)):
            similarity = 1.0 - float(distances_list[i])
            formatted_results.append(
                {
                    "score": similarity,
                    "metadata": metadatas_list[i],
                    "text_snippet": documents_list[i],
                }
            )
            if len(formatted_results) >= top_k:
                break

        return formatted_results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        conference_filter: str | None = None,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        pool_size = max(top_k * 4, 25)
        vec_results = self.search_vector(query, top_k=pool_size, conference_filter=conference_filter)
        bm25_results = self.search_bm25(query, top_k=pool_size, conference_filter=conference_filter)

        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(vec_results, 1):
            rev_id = item["metadata"]["review_id"]
            rrf_scores[rev_id] = rrf_scores.get(rev_id, 0.0) + (1.0 / (rrf_k + rank))
            doc_map[rev_id] = item

        for rank, item in enumerate(bm25_results, 1):
            rev_id = item["metadata"]["review_id"]
            if item["score"] > 0:
                rrf_scores[rev_id] = rrf_scores.get(rev_id, 0.0) + (1.0 / (rrf_k + rank))
                if rev_id not in doc_map:
                    doc_map[rev_id] = item

        sorted_rev_ids = sorted(rrf_scores.keys(), key=lambda rid: rrf_scores[rid], reverse=True)

        fused_results: list[dict[str, Any]] = []
        for rid in sorted_rev_ids[:top_k]:
            base_item = doc_map[rid]
            fused_results.append(
                {
                    "score": round(rrf_scores[rid] * 30.0, 4),
                    "metadata": base_item["metadata"],
                    "text_snippet": base_item["text_snippet"],
                }
            )
        return fused_results

    def get_reviews_for_paper(self, paper_id: str) -> list[dict[str, Any]]:
        matching: list[dict[str, Any]] = []
        for idx, meta in enumerate(self.corpus_metadata):
            if meta["paper_id"] == paper_id:
                matching.append(
                    {
                        "score": 1.0,
                        "metadata": meta,
                        "text_snippet": self.corpus_for_embedding[idx],
                    }
                )
        return matching