"""金融文档检索工具集 — verl BaseTool 实现

支持 4 种检索工具，与 SFT 训练数据一致：
- keyword_search: BM25 关键词检索（纯 CPU）
- semantic_search: FAISS 稠密检索 + BGE reranker（需 GPU）
- graph_search: 知识图谱 BFS + 实体 embedding 匹配（需 GPU）
- hybrid_search: 多工具 RRF 融合 + reranker（需 GPU）

embedding/reranker 在指定 GPU 上运行，不占训练卡。

q：为什么不用 retrieval/ 下面已有函数？主要是运行边界不同：

retrieval/*.py
  给普通 Python pipeline / eval / run_pipeline 用。

training/tools/financial_search_tool.py
  给 verl multi-turn tool calling 用，必须继承 verl.tools.base_tool.BaseTool

GRPO 的 rollout 是 verl/Ray/vLLM 管理的多进程训练环境，不是你普通 Python 脚本里直接 import retrieval.semantic_search 的那条路径。所以这里单独写了 verl 兼容的工具类。

而且，retrieval/ 只是在单进程内避免重复加载，不是跨 worker 共享模型，不能代替retrieval_server。
"""
import asyncio
import json
import os
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Optional

import numpy as np

from retrieval.graph_search import _bfs_collect_chunks
from retrieval.hybrid_search import rrf_fuse
from retrieval.keyword_search import tokenize
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
    # urllib 是 Python 标准库自带的一组 URL/HTTP 工具，不需要额外安装。
    import urllib.request
    # .encode() 转成字节数据，供 HTTP 请求发送
    data = json.dumps({"texts": texts}).encode()
    req = urllib.request.Request(f"{url}/embed", data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    # 转成 float32 很重要，因为 FAISS 通常要求查询向量使用 float32
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
        # “返回顺序”是 Reranker 顺序，但字典中的 score 不是 Reranker score
        return [candidates[i] for i, _ in indexed[:top_k]]

    def _encode(self, texts: list[str]) -> np.ndarray:
        return _call_embed(self.retrieval_server_url, texts)

    async def create(self, instance_id: Optional[str] = None, **kwargs):
        '''
        verl要求工具对象能调用 create/execute/release，但子类不必全部重写；无状态工具通常只需重写 execute，有状态工具则应完整实现三者。

        每条 rollout 或工具 session 开始时创建实例

        **kwargs：prepare_agentic_grpo_data.py 中的 "create_kwargs" 字段，但是没用到。

        这不影响当前检索，因为总reward由外部 reward_agentic_rag.py 计算，工具自身的 calc_reward()固定返回0。但意味着parquet中为四个工具重复保存的这些字段目前没有实际作用。
        '''
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {"results": []}
        # 返回实例ID + 空的初始工具响应
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        '''
        verl要求工具对象能调用 create/execute/release，但子类不必全部重写；无状态工具通常只需重写 execute，有状态工具则应完整实现三者。
        
        **kwargs：verl调用时传入的{"agent_data": agent_data}，没有使用。
        '''
        query = parameters.get("query", "")
        # 注意：客户端工具调用已经 asyncio 并发，但 retrieval_server.py 的 embedding和reranker各自有锁，因此同类模型请求会在服务端排队；本地BM25、FAISS和图遍历仍可并发执行。
        results = await asyncio.to_thread(self._search, query, parameters)
        text = self._format_results(results)
        self._instance_dict[instance_id]["results"].append(text)
        # ToolResponse(text=text) 把检索结果封装成标准工具响应，给模型看。
        # 即时 reward 固定为 0.0，说明工具调用本身没有奖励。
        return ToolResponse(text=text), 0.0, {"num_results": len(results), "query": query}

    def _search(self, query: str, parameters: dict) -> list[dict]:
        '''占位接口，具体实现由四个子类分别覆盖。'''
        raise NotImplementedError

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        '''工具级最终奖励固定为 0。'''
        return 0.0

    async def release(self, instance_id: str, **kwargs):
        '''
        verl要求工具对象能调用 create/execute/release，但子类不必全部重写；无状态工具通常只需重写 execute，有状态工具则应完整实现三者。

        rollout 结束后删除实例状态
        '''
        self._instance_dict.pop(instance_id, None)


# ── 4 种工具实现 ──────────────────────────────────────────────────

class KeywordSearchTool(_BaseFinancialTool):
    """BM25 关键词检索（纯 CPU）"""

    def _search(self, query: str, parameters: dict) -> list[dict]:
        chunk_ids, chunk_store = _load_chunk_store(self.index_dir)
        bm25 = _load_bm25(self.index_dir)
        tokens = tokenize(query)
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
        _, chunk_store = _load_chunk_store(self.index_dir)
        graph = _load_graph(self.index_dir)
        entity_data = _load_entity_embeddings(self.index_dir)

        # 实体匹配
        entities = entity_data["entities"]
        embeddings = entity_data["embeddings"]
        q_vec = self._encode([query])
        scores = (embeddings @ q_vec.T).flatten()
        top_ent_indices = np.argsort(scores)[::-1][:5]
        seed_entities = [(entities[i], float(scores[i])) for i in top_ent_indices]

        if not seed_entities:
            return []

        # 复用在线图检索的 BFS：同时收集节点 mentions、入边和出边 chunk。
        chunk_scores = _bfs_collect_chunks(graph, seed_entities)

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
        # __new__()只创建对象，不调用__init__()
        # 这里没有正常调用：KeywordSearchTool(config, schema)
        # 而是使用 __new__() 创建一个未初始化对象，再复制当前 Hybrid 实例的属性。
        # 这样做的目的是：
        # 1. 不重复调用 BaseTool 初始化；
        # 2. 让子工具共享 Hybrid 的 index_dir、top_k、URL 等配置；
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
                    futures.append((t, pool.submit(sub._search, query, {})))
            for tool_name, future in futures:
                try:
                    results_list.append(future.result())
                except Exception as exc:
                    print(f"[hybrid_search] Tool {tool_name} failed: {exc}")
                    results_list.append([])

        if not results_list:
            return []

        # RRF 融合 + rerank
        fused = rrf_fuse(results_list)
        return self._rerank(query, fused)
