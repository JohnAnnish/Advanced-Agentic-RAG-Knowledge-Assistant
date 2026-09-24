from collections import defaultdict

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import settings


class HybridRetriever:

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.bm25 = None
        self.documents = []
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-base"
        )

    def build_bm25(self, documents):
        self.documents = documents
        tokenized = [
            d.page_content.lower().split()
            for d in documents
        ]
        self.bm25 = BM25Okapi(tokenized)

    def bm25_search(self, query):
        if not self.bm25:
            return []

        scores = self.bm25.get_scores(
            query.lower().split()
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:settings.top_k_bm25]

        return [
            self.documents[index]
            for index, _ in ranked
        ]

    def vector_search(self, query):
        return self.vector_store.similarity_search(
            query,
            k=settings.top_k_vector,
        )

    def rrf(self, ranked_lists, k=60):
        scores = defaultdict(float)
        documents = {}

        for ranked_docs in ranked_lists:
            for rank, document in enumerate(
                ranked_docs,
                start=1,
            ):
                key = (
                    document.metadata.get("source", ""),
                    document.metadata.get("page", -1),
                    document.page_content[:100],
                )
                scores[key] += 1 / (k + rank)
                documents[key] = document

        ordered = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            documents[key]
            for key, _ in ordered
        ]

    def retrieve(self, query):
        candidates = self.rrf([
            self.bm25_search(query),
            self.vector_search(query),
        ])
        return candidates[:settings.top_k_vector]

    def rerank(self, query, documents):
        if not documents:
            return []

        pairs = [
            [query, document.page_content]
            for document in documents
        ]

        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            document
            for document, _ in ranked[:settings.final_k]
        ]
