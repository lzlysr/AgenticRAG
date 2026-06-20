#!/usr/bin/env python3
"""为已有 QA 数据的每个 hop 标注检索工具召回情况

对每个 hop 的 question，调用所有检索工具，检查哪些能召回目标 chunk_id，
将结果写入 search_tools 字段。

用法:
  python scripts/annotate_search_tools.py \
    --input data/financial_eval/train_qa_pairs.json \
    --output data/financial_eval/train_qa_pairs_annotated.json \
    --index-dir data/financial_all/indexes/ \
    --workers 10

为每一跳补充：search_tools（单独召回）、search_tools_hybrid（混合召回）、search_query
例如原始 hop：
{
  "hop_idx": 2,
  "question": "Where was John Smith born?",
  "answer": "London",
  "doc_chunk_id": "john_003"
}
标注后可能变成：
{
  "hop_idx": 2,
  "question": "Where was John Smith born?",
  "answer": "London",
  "doc_chunk_id": "john_003",
  "search_query": "Where was John Smith born?",
  "search_tools": [
    "keyword_search",
    "semantic_search"
  ],
  "search_tools_hybrid": [
    ["keyword_search", "semantic_search"],
    ["keyword_search", "semantic_search", "graph_search"]
  ]
}

策略：
对于当前 QA 的当前一跳，只要任何一种方式命中 -> 该跳可达。
如果所有非首跳hop都可达 -> 该 QA 被覆盖。
只要任何一跳没被任何工具命中，则认为该 QA 不被覆盖。

"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_write_lock = Lock()
_stats = {"total_hops": 0, "keyword_search": 0, "semantic_search": 0, "graph_search": 0, "hybrid_hit": 0}


def search_keyword(query: str, top_k: int = 10) -> list[dict]:
    """BM25 检索，返回结果列表"""
    from retrieval.keyword_search import keyword_search
    return keyword_search(query, top_k=top_k)


def search_semantic(query: str, top_k: int = 10) -> list[dict]:
    """FAISS 语义检索"""
    from retrieval.semantic_search import semantic_search
    return semantic_search(query, top_k=top_k)


def search_graph(query: str, top_k: int = 10) -> list[dict]:
    """知识图谱检索"""
    try:
        from retrieval.graph_search import graph_search
        return graph_search(query, top_k=top_k)
    except Exception:
        return []


# 3 个单工具
SINGLE_TOOLS = {
    "keyword_search": search_keyword,
    "semantic_search": search_semantic,
    "graph_search": search_graph,
}

# 4 种 hybrid 组合 (两两 + 三合一)
from itertools import combinations
HYBRID_COMBOS = list(combinations(["keyword_search", "semantic_search", "graph_search"], 2)) + \
                [("keyword_search", "semantic_search", "graph_search")]


def annotate_one(qa: dict, top_k: int = 10) -> dict:
    """为一条 QA 的所有 hop 标注检索工具"""
    from retrieval.hybrid_search import hybrid_fuse_and_rerank

    qa = dict(qa)
    new_hops = [] # 每个原始 hop 处理完后放入这个列表，最终：qa["hops"] = new_hops

    for hop in qa["hops"]:
        hop = dict(hop)
        # 使用原子问题，评测的是：用原子子问题能否召回当前 chunk。
        question = hop["question"]
        target_chunk = hop.get("doc_chunk_id", "")
        hop_idx = hop["hop_idx"]

        # 第一跳通常是 seed QA，不是由多跳检索过程找到的。因此不强求第一跳也被检索工具命中，直接跳过标注。
        if not target_chunk or hop_idx == 1:
            hop["search_tools"] = []
            hop["search_query"] = question
            new_hops.append(hop)
            continue

        # 1) 调用 3 个单工具，缓存原始结果，后面的 hybrid 不会重新执行这些基础检索。
        raw_results = {}
        for name, fn in SINGLE_TOOLS.items():
            try:
                raw_results[name] = fn(question, top_k=top_k)
            except Exception:
                raw_results[name] = []

        # 2) 检查单工具命中
        hit_tools = []
        for name, results in raw_results.items():
            if target_chunk in [r["chunk_id"] for r in results]:
                hit_tools.append(name)

        # 3) 检查 4 种 hybrid 组合命中
        # 1. 有一个重要的 Hybrid 标签问题：若两个工具组合中只有一个工具命中，另一个工具没有命中，那么仍然算作该组合命中。因为混合检索的目的是希望至少有一个工具能召回目标 chunk。
        # 2. Hybrid 可能比单工具命中更差。
        # 3. Hybrid 的候选池只来自各工具 top-10。
        hit_hybrids = []
        for combo in HYBRID_COMBOS:
            results_list = [raw_results[t] for t in combo if raw_results.get(t)]
            if not results_list:
                continue
            fused = hybrid_fuse_and_rerank(question, results_list, top_k=top_k)
            if target_chunk in [r["chunk_id"] for r in fused]:
                hit_hybrids.append(list(combo))

        hop["search_tools"] = hit_tools
        hop["search_tools_hybrid"] = hit_hybrids
        hop["search_query"] = question

        # 更新全局统计
        with _write_lock:
            _stats["total_hops"] += 1
            for t in hit_tools:
                _stats[t] = _stats.get(t, 0) + 1
            _stats["hybrid_hit"] = _stats.get("hybrid_hit", 0) + (1 if hit_hybrids else 0)

        new_hops.append(hop)

    qa["hops"] = new_hops
    return qa


def main():
    parser = argparse.ArgumentParser(description="Annotate search tools for existing QA data")
    parser.add_argument("--input", required=True, help="Input QA file (json or jsonl)")
    parser.add_argument("--output", required=True, help="Output annotated file")
    parser.add_argument("--index-dir", default="data/financial_all/indexes/",
                        help="Index directory")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for each search tool")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit QAs (0=all)")
    args = parser.parse_args()

    # 设置索引目录
    import config
    config.ACTIVE_INDEX_DIR = args.index_dir

    # 预热：加载检索模型到 GPU
    gpu_device = "cuda:0"
    print(f"Loading retrieval models on {gpu_device}...")
    from retrieval.embedder import _get_model as get_embedder
    from retrieval.reranker import _get_model as get_reranker
    get_embedder(gpu_device)
    get_reranker(gpu_device)

    # Monkeypatch: 让 encode/rerank 默认走 GPU
    import retrieval.embedder as _emb_mod
    import retrieval.reranker as _rnk_mod
    _orig_encode = _emb_mod.encode
    _orig_rerank = _rnk_mod.rerank
    _emb_mod.encode = lambda texts, batch_size=64, device=gpu_device: _orig_encode(texts, batch_size=batch_size, device=device)
    _rnk_mod.rerank = lambda query, passages, top_k=_rnk_mod.RERANK_TOP_K, device=gpu_device: _orig_rerank(query, passages, top_k=top_k, device=device)

    # 加载数据
    if args.input.endswith(".jsonl"):
        with open(args.input, "r", encoding="utf-8") as f:
            data = [json.loads(l) for l in f]
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

    if args.limit > 0:
        data = data[:args.limit]

    print(f"Loaded {len(data)} QAs from {args.input}")

    # 预热：加载检索模型
    print("Loading retrieval models...")
    search_keyword("test", top_k=1)
    search_semantic("test", top_k=1)
    search_graph("test", top_k=1)
    print("Models loaded.")

    # 并行标注
    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(annotate_one, qa, args.top_k): i
                   for i, qa in enumerate(data)}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            results.append((futures[fut], result))
            done += 1
            if done % 50 == 0 or done == len(data):
                elapsed = time.time() - t0
                print(f"  [{done}/{len(data)}] {elapsed:.0f}s elapsed")

    # 按原始顺序排列
    results.sort(key=lambda x: x[0])
    annotated = [r for _, r in results]

    # 保存
    if args.output.endswith(".jsonl"):
        with open(args.output, "w", encoding="utf-8") as f:
            for qa in annotated:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(annotated, f, ensure_ascii=False, indent=2)

    # 统计
    total_hops = _stats["total_hops"]
    print(f"\n=== 标注完成 ===")
    print(f"QAs: {len(annotated)}, 非首跳 Hops: {total_hops}")
    print(f"各工具召回率 (target chunk in top-{args.top_k}):")
    for tool in ["keyword_search", "semantic_search", "graph_search"]:
        cnt = _stats.get(tool, 0)
        rate = cnt / max(total_hops, 1) * 100
        print(f"  {tool}: {cnt}/{total_hops} ({rate:.1f}%)")
    print(f"  hybrid 组合命中: {_stats.get('hybrid_hit', 0)}/{total_hops} ({_stats.get('hybrid_hit', 0)/max(total_hops,1)*100:.1f}%)")

    # 统计每个 hop 的覆盖情况（单工具 or hybrid 至少命中一个）
    # 一个逻辑上的冗余现象：
    # 如果 hybrid 的输入只来自单工具 top-k 结果，那么理论上：
    # 若所有单工具 top-k 都没有目标 chunk，hybrid 不可能凭空召回目标 chunk。
    # 因为目标根本没有进入候选池。
    # Hybrid 不能提升“召回覆盖”，只能改变目标在融合结果中的保留与排序情况。
    # 甚至 hybrid 可能比单工具覆盖更低，因为目标可能被重排挤出去。
    # 若要让 hybrid 真正补充 top-k 覆盖，需要：
    # 各工具先召回更大的候选池
    # 然后 hybrid 截断到较小 top-k
    
    from collections import Counter
    coverage = Counter()
    for qa in annotated:
        for hop in qa["hops"]:
            if hop["hop_idx"] == 1:
                continue
            single = hop.get("search_tools", [])
            hybrid = hop.get("search_tools_hybrid", [])
            if single or hybrid:
                coverage["covered"] += 1
            else:
                coverage["uncovered"] += 1
            if single:
                coverage["single_hit"] += 1
            if hybrid:
                coverage["hybrid_hit"] += 1

    print(f"\n覆盖率:")
    print(f"  单工具命中: {coverage['single_hit']}/{total_hops} ({coverage['single_hit']/max(total_hops,1)*100:.1f}%)")
    print(f"  hybrid 命中: {coverage['hybrid_hit']}/{total_hops} ({coverage['hybrid_hit']/max(total_hops,1)*100:.1f}%)")
    print(f"  总覆盖(任一命中): {coverage['covered']}/{total_hops} ({coverage['covered']/max(total_hops,1)*100:.1f}%)")
    print(f"  未覆盖: {coverage['uncovered']}/{total_hops} ({coverage['uncovered']/max(total_hops,1)*100:.1f}%)")

    # QA 级别覆盖
    all_hit = sum(1 for qa in annotated if all(
        (h.get('search_tools') or h.get('search_tools_hybrid') or h['hop_idx'] == 1)
        for h in qa['hops']))
    print(f"\n全命中 QA: {all_hit}/{len(annotated)} ({all_hit/len(annotated)*100:.0f}%)")

    print(f"\n写出: {args.output}")


if __name__ == "__main__":
    main()
