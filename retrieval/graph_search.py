"""知识图谱检索工具：实体匹配 → BFS 图遍历 → chunk 收集排序"""
import json
import os
import pickle
import sys
from collections import deque

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIVE_INDEX_DIR as INDEX_DIR, RERANK_TOP_K

# ── 参数 ──
GRAPH_TOP_ENTITIES = 5      # 种子实体数
GRAPH_MAX_HOPS = 2          # BFS 遍历深度
GRAPH_RERANK_TOP_K = RERANK_TOP_K  # 最终返回 chunk 数
GRAPH_MAX_CANDIDATES = 20   # rerank 前最大候选 chunk 数

# ── 单例状态 ──
_graph = None
_entity_data = None
_chunk_store = None


def _load_graph():
    """懒加载知识图谱"""
    global _graph
    if _graph is None:
        graph_path = os.path.join(INDEX_DIR, "knowledge_graph.json")
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _graph = nx.node_link_graph(data)
        print(f"[graph_search] Loaded graph: {_graph.number_of_nodes()} nodes, {_graph.number_of_edges()} edges")
    return _graph


def _load_entity_embeddings():
    """懒加载实体 embedding"""
    global _entity_data
    if _entity_data is None:
        emb_path = os.path.join(INDEX_DIR, "entity_embeddings.pkl")
        with open(emb_path, "rb") as f:
            _entity_data = pickle.load(f)
        print(f"[graph_search] Loaded {len(_entity_data['entities'])} entity embeddings")
    return _entity_data


def _load_chunk_store():
    """懒加载 chunk store"""
    global _chunk_store
    if _chunk_store is None:
        with open(os.path.join(INDEX_DIR, "chunk_store.pkl"), "rb") as f:
            _chunk_store = pickle.load(f)
    return _chunk_store


def _match_entities(query: str, top_k: int = GRAPH_TOP_ENTITIES, device: str = None) -> list[tuple[str, float]]:
    """用 embedding 相似度匹配 query 到最相关的实体节点"""
    from retrieval.embedder import encode

    entity_data = _load_entity_embeddings()
    entities = entity_data["entities"]
    embeddings = entity_data["embeddings"]  # (N, D), normalized

    # 编码 query
    q_vec = encode([query], device=device)  # (1, D), normalized

    # 余弦相似度（向量已归一化，直接点积）
    scores = (embeddings @ q_vec.T).flatten()  # (N,)

    # top-k
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(entities[i], float(scores[i])) for i in top_indices]
    return results


def _bfs_collect_chunks(G: nx.MultiDiGraph, seed_entities: list[str],
                        max_hops: int = GRAPH_MAX_HOPS) -> dict[str, float]:
    """从种子实体出发 BFS 遍历，收集相关 chunk_id 及其分数

    返回 {chunk_id: score}，score 基于距离衰减
    """
    chunk_scores = {}  # chunk_id -> best_score
    visited = set()
    queue = deque()  # (entity, depth, base_score)

    for entity, score in seed_entities:
        if G.has_node(entity):
            queue.append((entity, 0, score))
            visited.add(entity)

    while queue:
        entity, depth, base_score = queue.popleft()

        # 收集当前实体关联的 chunk（从节点的 mentions 属性）
        node_data = G.nodes.get(entity, {})
        for cid in node_data.get("mentions", []):
            decay = 1.0 / (1 + depth)  # 距离衰减
            score = base_score * decay
            if cid not in chunk_scores or score > chunk_scores[cid]:
                chunk_scores[cid] = score

        # 收集边上的 chunk
        if G.has_node(entity):
            for _, neighbor, edge_data in G.edges(entity, data=True):
                cid = edge_data.get("chunk_id")
                if cid:
                    decay = 1.0 / (1 + depth)
                    score = base_score * decay
                    if cid not in chunk_scores or score > chunk_scores[cid]:
                        chunk_scores[cid] = score

                # BFS 扩展
                if depth < max_hops and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1, base_score))

            # 也检查入边（有向图）
            for predecessor, _, edge_data in G.in_edges(entity, data=True):
                cid = edge_data.get("chunk_id")
                if cid:
                    decay = 1.0 / (1 + depth)
                    score = base_score * decay
                    if cid not in chunk_scores or score > chunk_scores[cid]:
                        chunk_scores[cid] = score

                if depth < max_hops and predecessor not in visited:
                    visited.add(predecessor)
                    queue.append((predecessor, depth + 1, base_score))

    return chunk_scores


def graph_search(query: str, top_k: int = GRAPH_RERANK_TOP_K, device: str = None) -> list[dict]:
    """知识图谱检索：实体匹配 → BFS 图遍历 → chunk rerank

    返回 [{"chunk_id", "text", "title", "score", "source"}]
    """
    G = _load_graph()
    chunk_store = _load_chunk_store()

    # ① 实体匹配
    seed_entities = _match_entities(query, top_k=GRAPH_TOP_ENTITIES, device=device)

    if not seed_entities:
        return []

    # ② 图遍历收集 chunk
    chunk_scores = _bfs_collect_chunks(G, seed_entities, max_hops=GRAPH_MAX_HOPS)

    if not chunk_scores:
        return []

    # 按图分数排序，取 top 候选做 rerank
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    candidates = sorted_chunks[:GRAPH_MAX_CANDIDATES]

    # ③ Rerank
    candidate_texts = []
    candidate_docs = []
    for cid, _ in candidates:
        doc = chunk_store.get(cid)
        if doc:
            candidate_texts.append(doc["text"])
            candidate_docs.append(doc)

    if not candidate_docs:
        return []

    from retrieval.reranker import rerank
    reranked = rerank(query, candidate_texts, top_k=top_k)

    results = []
    for orig_idx, score in reranked:
        doc = candidate_docs[orig_idx]
        results.append({
            "chunk_id": doc["chunk_id"],
            "text": doc["text"],
            "title": doc.get("title", ""),
            "score": float(score),
            "source": "graph",
        })
    return results
