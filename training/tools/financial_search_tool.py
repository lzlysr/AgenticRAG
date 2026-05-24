"""金融文档检索工具集 — verl BaseTool 实现

支持 4 种检索工具，与 SFT 训练数据一致：
- keyword_search: BM25 关键词检索（纯 CPU）
- semantic_search: FAISS 稠密检索 + BGE reranker（需 GPU）
- graph_search: 知识图谱 BFS + 实体 embedding 匹配（需 GPU）
- hybrid_search: 多工具 RRF 融合 + reranker（需 GPU）

embedding/reranker 在指定 GPU 上运行，不占训练卡。
"""
import json
import os
import pickle
import re
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Optional

import numpy as np

from verl.tools.base_tool import BaseTool, ToolResponse

# ── 共享单例（跨工具实例复用）──────────────────────────────────────

_shared = {
    "chunk_store": None,
    "chunk_ids": None,
    "bm25": None,
    "faiss_index": None,
    "graph": None,
    "entity_data": None,
    "lock": Lock(),
    "retrieval_server_url": None,
}


def _call_embed(url: str, texts: list[str]) -> np.ndarray:
    """通过 HTTP 调用远程 embedding 服务"""
    import urllib.request
    data = json.dumps({"texts": texts}).encode()
    req = urllib.request.Request(f"{url}/embed", data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return np.array(result["embeddings"], dtype=np.float32)


def _call_rerank(url: str, query: str, passages: list[str]) -> list[float]:
    """通过 HTTP 调用远程 reranker 服务"""
    import urllib.request
    data = json.dumps({"query": query, "passages": passages}).encode()
    req = urllib.request.Request(f"{url}/rerank", data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["scores"]


def _load_chunk_store(index_dir: str):
    if _shared["chunk_store"] is None:
        with open(os.path.join(index_dir, "chunk_ids.json")) as f:
            _shared["chunk_ids"] = json.load(f)
        with open(os.path.join(index_dir, "chunk_store.pkl"), "rb") as f:
            _shared["chunk_store"] = pickle.load(f)
    return _shared["chunk_ids"], _shared["chunk_store"]


def _load_bm25(index_dir: str):
    if _shared["bm25"] is None:
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            _shared["bm25"] = pickle.load(f)
    return _shared["bm25"]


def _load_faiss(index_dir: str):
    if _shared["faiss_index"] is None:
        import faiss
        _shared["faiss_index"] = faiss.read_index(
            os.path.join(index_dir, "faiss.index")
        )
    return _shared["faiss_index"]


def _load_graph(index_dir: str):
    if _shared["graph"] is None:
        import networkx as nx
        with open(os.path.join(index_dir, "knowledge_graph.json"), "r") as f:
            data = json.load(f)
        _shared["graph"] = nx.node_link_graph(data)
    return _shared["graph"]


def _load_entity_embeddings(index_dir: str):
    if _shared["entity_data"] is None:
        with open(os.path.join(index_dir, "entity_embeddings.pkl"), "rb") as f:
            _shared["entity_data"] = pickle.load(f)
    return _shared["entity_data"]


# ── 分词 ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    import jieba
    tokens = []
    parts = re.split(r'([\u4e00-\u9fff]+)', text.lower())
    for part in parts:
        if re.match(r'^[\u4e00-\u9fff]+$', part):
            tokens.extend(jieba.lcut(part))
        else:
            tokens.extend(part.split())
    return [t for t in tokens if t.strip()]


# ── RRF 融合 ──────────────────────────────────────────────────────

def _rrf_fuse(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    chunk_scores = {}
    chunk_data = {}
    for results in results_list:
        for rank, r in enumerate(results):
            cid = r.get("chunk_id", "")
            if not cid:
                continue
            rrf_score = 1.0 / (k + rank + 1)
            chunk_scores[cid] = chunk_scores.get(cid, 0) + rrf_score
            if cid not in chunk_data:
                chunk_data[cid] = r
    sorted_ids = sorted(chunk_scores, key=lambda x: chunk_scores[x], reverse=True)
    return [dict(chunk_data[cid], score=chunk_scores[cid]) for cid in sorted_ids]


# ── 基类 ──────────────────────────────────────────────────────────

class _BaseFinancialTool(BaseTool):
    """所有金融检索工具的基类"""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self.index_dir = config.get("index_dir", "data/financial_all/indexes")
        self.top_k = config.get("top_k", 3)
        self.max_text_len = config.get("max_text_len", 300)
        self.retrieval_server_url = config.get("retrieval_server_url", "http://localhost:8790")
        self._instance_dict = {}
        # 首次使用时打印一次
        with _shared["lock"]:
            if _shared["retrieval_server_url"] is None:
                _shared["retrieval_server_url"] = self.retrieval_server_url
                print(f"[tool] Using retrieval server: {self.retrieval_server_url}")

    def _format_results(self, results: list[dict]) -> str:
        if results:
            parts = [f"[{r['chunk_id']}] {r['text'][:self.max_text_len]}" for r in results]
            return "\n".join(parts)
        return "(no results)"

    def _rerank(self, query: str, candidates: list[dict], top_k: int = None) -> list[dict]:
        top_k = top_k or self.top_k
        if len(candidates) <= top_k:
            return candidates[:top_k]
        passages = [c["text"] for c in candidates[:15]]
        scores = _call_rerank(self.retrieval_server_url, query, passages)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [candidates[i] for i, _ in indexed[:top_k]]

    def _encode(self, texts: list[str]) -> np.ndarray:
        return _call_embed(self.retrieval_server_url, texts)

    async def create(self, instance_id: Optional[str] = None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {"results": []}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        query = parameters.get("query", "")
        results = self._search(query, parameters)
        text = self._format_results(results)
        self._instance_dict[instance_id]["results"].append(text)
        return ToolResponse(text=text), 0.0, {"num_results": len(results), "query": query}

    def _search(self, query: str, parameters: dict) -> list[dict]:
        raise NotImplementedError

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs):
        self._instance_dict.pop(instance_id, None)


# ── 4 种工具实现 ──────────────────────────────────────────────────

class KeywordSearchTool(_BaseFinancialTool):
    """BM25 关键词检索（纯 CPU）"""

    def _search(self, query: str, parameters: dict) -> list[dict]:
        chunk_ids, chunk_store = _load_chunk_store(self.index_dir)
        bm25 = _load_bm25(self.index_dir)
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            cid = chunk_ids[idx]
            chunk = chunk_store.get(cid, {})
            results.append({
                "chunk_id": cid,
                "text": chunk.get("text", ""),
                "title": chunk.get("title", ""),
                "score": float(scores[idx]),
            })
        return results


class SemanticSearchTool(_BaseFinancialTool):
    """FAISS 稠密检索 + reranker"""

    def _search(self, query: str, parameters: dict) -> list[dict]:
        chunk_ids, chunk_store = _load_chunk_store(self.index_dir)
        faiss_index = _load_faiss(self.index_dir)
        q_vec = self._encode([query])
        scores, indices = faiss_index.search(q_vec, 20)
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            cid = chunk_ids[idx]
            doc = chunk_store.get(cid, {})
            candidates.append({
                "chunk_id": cid,
                "text": doc.get("text", ""),
                "title": doc.get("title", ""),
                "score": float(scores[0][i]),
            })
        return self._rerank(query, candidates)


class GraphSearchTool(_BaseFinancialTool):
    """知识图谱 BFS 检索"""

    def _search(self, query: str, parameters: dict) -> list[dict]:
        import networkx as nx
        chunk_ids, chunk_store = _load_chunk_store(self.index_dir)
        graph = _load_graph(self.index_dir)
        entity_data = _load_entity_embeddings(self.index_dir)

        # 实体匹配
        entities = entity_data["entities"]
        embeddings = entity_data["embeddings"]
        q_vec = self._encode([query])
        scores = (embeddings @ q_vec.T).flatten()
        top_ent_indices = np.argsort(scores)[::-1][:5]
        seed_entities = [entities[i] for i in top_ent_indices if scores[i] > 0.3]

        if not seed_entities:
            return []

        # BFS 收集 chunk
        chunk_scores = {}
        visited = set()
        queue = deque()
        for ent in seed_entities:
            if ent in graph:
                queue.append((ent, 0))
                visited.add(ent)

        while queue:
            node, depth = queue.popleft()
            if depth > 2:
                continue
            node_data = graph.nodes[node]
            for cid in node_data.get("chunk_ids", []):
                score = 1.0 / (1 + depth)
                chunk_scores[cid] = max(chunk_scores.get(cid, 0), score)
            if depth < 2:
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        # 按分数排序
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = []
        for cid, score in sorted_chunks[:20]:
            doc = chunk_store.get(cid, {})
            if doc:
                candidates.append({
                    "chunk_id": cid,
                    "text": doc.get("text", ""),
                    "title": doc.get("title", ""),
                    "score": score,
                })

        return self._rerank(query, candidates)


class HybridSearchTool(_BaseFinancialTool):
    """多工具 RRF 融合检索"""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        # 子工具实例（共享 config）
        self._keyword = KeywordSearchTool.__new__(KeywordSearchTool)
        self._keyword.__dict__.update(self.__dict__)
        self._semantic = SemanticSearchTool.__new__(SemanticSearchTool)
        self._semantic.__dict__.update(self.__dict__)
        self._graph = GraphSearchTool.__new__(GraphSearchTool)
        self._graph.__dict__.update(self.__dict__)
        self._sub_tools = {
            "keyword_search": self._keyword,
            "semantic_search": self._semantic,
            "graph_search": self._graph,
        }

    def _search(self, query: str, parameters: dict) -> list[dict]:
        # 从参数中获取要融合的工具列表
        tools = parameters.get("tools", ["keyword_search", "semantic_search"])
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except (json.JSONDecodeError, TypeError):
                tools = ["keyword_search", "semantic_search"]

        # 并行调用子工具
        results_list = []
        with ThreadPoolExecutor(max_workers=len(tools)) as pool:
            futures = []
            for t in tools:
                sub = self._sub_tools.get(t)
                if sub:
                    futures.append(pool.submit(sub._search, query, {}))
            for f in futures:
                try:
                    results_list.append(f.result())
                except Exception:
                    results_list.append([])

        if not results_list:
            return []

        # RRF 融合 + rerank
        fused = _rrf_fuse(results_list)
        return self._rerank(query, fused)
