"""
Module containing retrieval mechanisms: Dense (FAISS), Sparse (BM25), Web, and RRF Fusion.
"""

import faiss
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """FAISS-based Dense Retrieval module using SentenceTransformers."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def fit(self, documents: List[str]):
        self.documents = documents
        embeddings = self.model.encode(documents, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx != -1:
                results.append({
                    "doc_id": int(idx),
                    "text": self.documents[idx],
                    "score": float(score),
                    "rank": rank + 1
                })
        return results


class BM25Retriever:
    """BM25 Sparse Retrieval module."""
    def __init__(self):
        self.bm25 = None
        self.documents = []

    def fit(self, documents: List[str]):
        self.documents = documents
        tokenized_corpus = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "doc_id": int(idx),
                "text": self.documents[idx],
                "score": float(scores[idx]),
                "rank": rank + 1
            })
        return results


class WebRetrieverMock:
    """Retrieves live web coverage using DDGS or falls back gracefully."""
    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=top_k))
                results = []
                for idx, r in enumerate(ddg_results):
                    results.append({
                        "doc_id": f"web_{idx}",
                        "text": r.get("body", r.get("title", "")),
                        "score": 1.0,
                        "rank": idx + 1
                    })
                return results
        except Exception:
            return [{
                "doc_id": "web_0",
                "text": f"Recent coverage on: {query}. The administration outlined key focus areas in the newly announced policy proposal.",
                "score": 1.0,
                "rank": 1
            }]


def reciprocal_rank_fusion(
    rank_lists: List[List[Dict[str, Any]]], 
    rrf_k: int = 60, 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Combines multiple ranked retrieval results using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: Dict[Any, float] = {}
    doc_map: Dict[Any, Dict[str, Any]] = {}

    for r_list in rank_lists:
        for item in r_list:
            doc_id = item["doc_id"]
            rank = item["rank"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_map[doc_id] = item
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank)

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    fused_results = []
    for rank, (doc_id, score) in enumerate(sorted_docs):
        item = doc_map[doc_id].copy()
        item["score"] = score
        item["rank"] = rank + 1
        fused_results.append(item)
        
    return fused_results