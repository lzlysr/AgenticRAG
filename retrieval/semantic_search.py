"""FAISS 稠密检索 + BGE-reranker 重排序工具"""
import json
import os
import pickle
import sys

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIVE_INDEX_DIR as INDEX_DIR, SEMANTIC_TOP_K, RERANK_TOP_K

_index = None
_chunk_ids = None
_chunk_store = None


def _load():
    global _index, _chunk_ids, _chunk_store
    if _index is None:
        _index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
        with open(os.path.join(INDEX_DIR, "chunk_ids.json"), "r") as f:
            _chunk_ids = json.load(f)
        with open(os.path.join(INDEX_DIR, "chunk_store.pkl"), "rb") as f:
            _chunk_store = pickle.load(f)


def semantic_search(query: str, top_k: int = SEMANTIC_TOP_K, rerank_top_k: int = RERANK_TOP_K) -> list[dict]:
    """FAISS 稠密检索 + rerank，返回 [{"chunk_id", "text", "title", "score"}]"""
    _load()
    from retrieval.embedder import encode
    from retrieval.reranker import rerank

    q_vec = encode([query])
    scores, indices = _index.search(q_vec, top_k)

    candidates = []
    candidate_texts = [] # 只保存文本，交给 reranker。
    # 因为 query 只有一个，所以结果第一维长度是 1
    # indices[0] 才是这个 query 的搜索结果
    for i, idx in enumerate(indices[0]):
        # 当请求的 top_k 大于索引中的向量数量时，FAISS 可能用 -1 表示没有结果。
        if idx == -1:
            continue
        cid = _chunk_ids[idx]
        doc = _chunk_store[cid]
        candidates.append(doc)
        candidate_texts.append(doc["text"])

    if not candidates:
        return []

    # rerank
    reranked = rerank(query, candidate_texts, top_k=rerank_top_k)

    results = []
    for orig_idx, score in reranked:
        doc = candidates[orig_idx]
        results.append({
            "chunk_id": doc["chunk_id"],
            "text": doc["text"],
            "title": doc.get("title", ""),
            "score": float(score),
            "source": "semantic+rerank",
        })
    return results
