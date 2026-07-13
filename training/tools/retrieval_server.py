#!/usr/bin/env python3
"""BGE Embedding + Reranker FastAPI 服务

q：为什么只把 BGE embedding + reranker 做成 server？

因为这两个最重、最容易抢 GPU：
BGE-M3 embedder：semantic_search、graph_search 需要
BGE reranker：semantic_search、graph_search、hybrid_search 需要

q：非得做成server吗？
不是“非得”做成 server，但在 verl GRPO 这条链路里，做成 server 是更稳的工程选择。
区别：

直接固定在某个 GPU
= 每个调用检索工具的 Python 进程里自己加载 BGE-M3 / reranker，并指定 device=cuda:X。
而且，之前retrieval/下定义好的那些工具 的缓存跨进程不共享！它不能天然解决：多个进程同时抢同一张 GPU。

做成 server
= 只有一个独立进程加载 BGE-M3 / reranker，其他 verl worker 通过 HTTP 调它

server 是“固定 GPU”的一种更隔离、更可控的实现方式。

数据流转框图：

  GRPO / verl rollout worker
      |
      |  tool_call: keyword_search / semantic_search / graph_search / hybrid_search
      v
  training.tools.financial_search_tool
      |
      |  keyword_search: 本地读取 bm25.pkl + chunk_store.pkl
      |
      |  semantic_search:
      |    本地读取 faiss.index + chunk_store.pkl
      |    HTTP POST /embed  -> 本服务上的 BGE-M3
      |    HTTP POST /rerank -> 本服务上的 BGE reranker
      |
      |  graph_search:
      |    本地读取 knowledge_graph.json + entity_embeddings.pkl + chunk_store.pkl
      |    HTTP POST /embed  -> 本服务上的 BGE-M3
      |    HTTP POST /rerank -> 本服务上的 BGE reranker
      |
      |  hybrid_search:
      |    本地并行调用 keyword / semantic / graph
      |    RRF 融合后 HTTP POST /rerank -> 本服务上的 BGE reranker
      v
  training.tools.retrieval_server
      |
      |  /embed:  texts -> embedding vectors
      |  /rerank: query + passages -> relevance scores
      v
  返回 tool observation 文本给 verl，模型继续下一轮生成或最终回答

在其他机器上部署，GRPO 训练通过内网 HTTP 调用。

用法:
  # 在有 GPU 的机器上启动（占 ~2GB 显存）
  python training/tools/retrieval_server.py --port 8790 --device cuda:0

  # 测试
  curl -X POST http://<IP>:8790/embed -H 'Content-Type: application/json' \
    -d '{"texts": ["永辉超市注册资本"]}'
  curl -X POST http://<IP>:8790/rerank -H 'Content-Type: application/json' \
    -d '{"query": "永辉超市", "passages": ["永辉超市注册资本100亿", "沃尔玛全球门店"]}'
"""
import argparse
import time
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="AgenticRAG Retrieval Model Server")

# 模型在 main() 中加载一次，然后被所有请求复用。
# 注意：这是“server 进程内”的单例；GRPO worker 不会各自加载 BGE。
embedder = None
reranker = None

# SentenceTransformer/CrossEncoder 在同一实例上并发调用时不一定完全无风险。
# FastAPI 可能并发处理多个请求，所以这里用锁让同一个模型实例串行推理。
_embed_lock = Lock()
_rerank_lock = Lock()


class EmbedRequest(BaseModel):
    texts: list[str]


class RerankRequest(BaseModel):
    query: str
    passages: list[str]


@app.get("/health")
@app.post("/health")
def health():
    # GET 是标准健康检查方式；start_retrieval_server.sh 当前用的就是 GET。
    # POST 是兼容旧版 HTTPServer 实现和手工测试习惯，保留它不会影响调用方。
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest):
    # 输入: {"texts": ["文本1", "文本2", ...]}
    # 输出: {"embeddings": [[...], [...]], "elapsed": 秒数, "count": 文本数}
    # semantic_search 和 graph_search 会通过这个接口把查询转成向量。
    if embedder is None:
        raise HTTPException(status_code=503, detail="embedder not loaded")
    if not req.texts:
        raise HTTPException(status_code=400, detail="no texts")

    t0 = time.time()
    with _embed_lock:
        vecs = embedder.encode(req.texts, normalize_embeddings=True)
    elapsed = time.time() - t0
    return {
        # .encode() 返回的 vecs 是 numpy.ndarray，而 FastAPI 最终要把返回值转成 JSON。
        # 所以要转成 Python 原生 list：.tolist()
        "embeddings": vecs.tolist(),
        "elapsed": elapsed,
        "count": len(req.texts),
    }


@app.post("/rerank")
def rerank(req: RerankRequest):
    # 输入: {"query": "问题", "passages": ["候选chunk文本1", ...]}
    # 输出: {"scores": [分数1, ...], "elapsed": 秒数}
    # semantic_search、graph_search 和 hybrid_search 用它精排候选 chunk。
    if reranker is None:
        raise HTTPException(status_code=503, detail="reranker not loaded")
    if not req.query or not req.passages:
        raise HTTPException(status_code=400, detail="need query and passages")

    t0 = time.time()
    pairs = [[req.query, p] for p in req.passages]
    with _rerank_lock:
        scores = reranker.predict(pairs)
    elapsed = time.time() - t0
    return {
        # 因为 reranker.predict() 返回的 scores 很可能是 NumPy 类型，
        # 所以要转成 Python 原生 float，才能安全返回 JSON。
        "scores": [float(s) for s in scores],
        "elapsed": elapsed,
    }


def main():
    global embedder, reranker

    parser = argparse.ArgumentParser(description="BGE Embedding + Reranker Server")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embedding-model", type=str,
                        default="models/bge-m3")
    parser.add_argument("--reranker-model", type=str,
                        default="models/bge-reranker-v2-m3")
    args = parser.parse_args()

    print(f"Loading BGE-M3 embedder on {args.device}...")
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedder = SentenceTransformer(args.embedding_model, device=args.device)
    print(f"Loading BGE reranker on {args.device}...")
    reranker = CrossEncoder(args.reranker_model, max_length=512, device=args.device)

    print(f"\nRetrieval server ready at http://0.0.0.0:{args.port}")
    print(f"  POST /embed   — {{\"texts\": [\"...\", ...]}}")
    print(f"  POST /rerank  — {{\"query\": \"...\", \"passages\": [\"...\", ...]}}")
    print(f"  GET  /health  — health check")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
