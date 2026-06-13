"""BM25 关键字搜索工具"""
import json
import os
import pickle
import re
import sys

import jieba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIVE_INDEX_DIR as INDEX_DIR, BM25_TOP_K

_bm25 = None # BM25 检索模型
_chunk_ids = None # 下标到 chunk_id 的映射
_chunk_store = None # chunk_id 到完整文本的映射


def tokenize(text: str) -> list[str]:
    """中英文混合分词：jieba 切中文，空格切英文/数字"""
    text = text.lower()
    # 初步切分：按中文和非中文边界拆分，把文本切成一些初步片段。
    segments = re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+(?:\.[0-9]+)*', text)
    tokens = []
    # 细切分：对每个片段，如果是中文就用 jieba 切词；如果不是中文就直接当成一个 token。
    for seg in segments:
        if re.match(r'[\u4e00-\u9fff]', seg):
            tokens.extend(jieba.lcut(seg))
        else:
            tokens.append(seg)
    return [t for t in tokens if len(t.strip()) > 0] # 把空字符串去掉。


def _load():
    global _bm25, _chunk_ids, _chunk_store
    if _bm25 is None:
        with open(os.path.join(INDEX_DIR, "bm25.pkl"), "rb") as f:
            _bm25 = pickle.load(f)
        with open(os.path.join(INDEX_DIR, "chunk_ids.json"), "r") as f:
            _chunk_ids = json.load(f)
        with open(os.path.join(INDEX_DIR, "chunk_store.pkl"), "rb") as f:
            _chunk_store = pickle.load(f)


def keyword_search(query: str, top_k: int = BM25_TOP_K) -> list[dict]:
    """BM25 关键字检索 + reranker 重排序，返回 [{"chunk_id", "text", "title", "score"}]"""
    _load()
    tokens = tokenize(query)
    scores = _bm25.get_scores(tokens)
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            break
        cid = _chunk_ids[idx]
        doc = _chunk_store[cid]
        results.append({
            "chunk_id": cid,
            "text": doc["text"],
            "title": doc.get("title", ""),
            "score": float(scores[idx]),
            "source": "bm25",
        })

    # Rerank BM25 results for better precision
    if len(results) > 5:
        from retrieval.reranker import rerank
        passages = [r["text"] for r in results]
        reranked = rerank(query, passages, top_k=5)
        results = [results[idx] for idx, _ in reranked]

    return results
