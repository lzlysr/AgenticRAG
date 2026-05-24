"""RRF (Reciprocal Rank Fusion) 多路召回融合"""
import sys, os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K
from retrieval.reranker import rerank

RRF_K = 60  # RRF 标准参数


def rrf_fuse(results_list: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """RRF 融合多路检索结果，按 chunk_id 去重合并排名。

    Args:
        results_list: 每路检索返回的 list[dict]，每个 dict 包含 chunk_id, text, title, score, source
        k: RRF 参数，默认 60

    Returns:
        按 RRF 分数降序排列的去重结果列表
    """
    chunk_scores = {}  # chunk_id -> rrf_score
    chunk_data = {}    # chunk_id -> best result dict

    for results in results_list:
        for rank, r in enumerate(results):
            cid = r.get("chunk_id", "")
            if not cid:
                continue
            rrf_score = 1.0 / (k + rank + 1)
            chunk_scores[cid] = chunk_scores.get(cid, 0) + rrf_score
            # 保留第一次出现的完整数据
            if cid not in chunk_data:
                chunk_data[cid] = r

    # 按 RRF 分数降序排列
    sorted_ids = sorted(chunk_scores, key=lambda x: chunk_scores[x], reverse=True)
    fused = []
    for cid in sorted_ids:
        entry = dict(chunk_data[cid])
        entry["score"] = chunk_scores[cid]
        entry["source"] = "hybrid_rrf"
        fused.append(entry)

    return fused


def hybrid_fuse_and_rerank(query: str, results_list: list[list[dict]],
                           top_k: int = RERANK_TOP_K) -> list[dict]:
    """RRF 融合 + CrossEncoder 重排。

    1. RRF 融合多路结果
    2. 取 top-N 候选（最多 15）
    3. CrossEncoder 重排 → top_k
    """
    fused = rrf_fuse(results_list)

    if len(fused) <= top_k:
        return fused[:top_k]

    # 取 top-15 候选做重排（避免 reranker 输入太大）
    candidates = fused[:15]
    passages = [c["text"] for c in candidates]
    reranked = rerank(query, passages, top_k=top_k)

    return [candidates[idx] for idx, _ in reranked]


def multi_tool_search(query: str, tool_names: list[str], tool_registry: dict,
                      top_k: int = RERANK_TOP_K) -> list[dict]:
    """并行调用多个检索工具，RRF 融合 + 重排。

    Args:
        query: 搜索查询
        tool_names: 工具名列表，如 ["semantic_search", "keyword_search"]
        tool_registry: 工具函数注册表
        top_k: 最终返回数量

    Returns:
        融合后的 top_k 结果
    """
    def _call_tool(name):
        fn = tool_registry.get(name)
        if fn is None:
            return []
        try:
            return fn(query)
        except Exception:
            return []

    # 并行调用
    with ThreadPoolExecutor(max_workers=len(tool_names)) as pool:
        futures = {pool.submit(_call_tool, name): name for name in tool_names}
        results_list = []
        for future in futures:
            results_list.append(future.result())

    return hybrid_fuse_and_rerank(query, results_list, top_k=top_k)
