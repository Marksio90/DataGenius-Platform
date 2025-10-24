"""
rag_explain.py
Docstring (PL): Lokalny retriever (gated). Jeśli brak FAISS/sqlite-vss, zwraca komunikat.
Na potrzeby Stage 6 – prosta implementacja in-memory z TF-IDF jako fallback.
"""
from __future__ import annotations
from typing import List, Dict, Any
import math

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_TFIDF = True
except Exception:
    HAS_TFIDF = False

def simple_retrieval(docs: List[str], query: str, topk: int = 3) -> Dict[str, Any]:
    if not HAS_TFIDF:
        return {"status":"TMIV-RAG-000","message":"Brak TF-IDF (scikit-learn). Zainstaluj, aby użyć prostego retrievera."}
    vec = TfidfVectorizer()
    X = vec.fit_transform(docs + [query])
    sims = cosine_similarity(X[-1], X[:-1]).ravel()
    idx = sims.argsort()[::-1][:topk]
    results = [{"doc": docs[i], "score": float(sims[i])} for i in idx]
    return {"status":"OK", "results": results}