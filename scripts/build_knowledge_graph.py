"""
从 corpus.json 构建知识图谱：LLM 抽取三元组 → NetworkX 图 → 持久化
可选：预计算实体 embedding 供后续检索使用，保存 entity_embeddings.pkl
把原本文本 chunk 里的事实，变成一个“实体—关系—实体”的图结构
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed # 多线程并行调用 LLM 抽三元组
from threading import Lock

import networkx as nx # 图结构处理库
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录
from config import DATA_DIR, INDEX_DIR, SYNTH_LLM_MODEL

# ── 全局计数 ──
_lock = Lock()
_stats = {"success": 0, "fail": 0, "total_triples": 0}

# ── 三元组抽取 prompt ──
EXTRACTION_PROMPT_EN = """You are a knowledge graph builder. Extract factual triples from the given text.

Each triple should be: (head_entity, relation, tail_entity)

Rules:
- Extract concrete, specific entities (people, places, organizations, works, concepts, numbers)
- Relations should be short verb phrases (e.g., "is_capital_of", "directed", "born_in", "has_population")
- Each triple must be factually stated in the text, not inferred
- Entity names should be canonical (full proper names, not pronouns)
- Extract 3-8 triples per text chunk
- If the text is too short or uninformative, return an empty list

Return ONLY a JSON array:
[{{"head": "entity1", "relation": "relation_type", "tail": "entity2"}}, ...]

Text:
{text}"""

EXTRACTION_PROMPT_ZH = """你是一个知识图谱构建器。从给定文本中抽取事实性三元组。

每个三元组格式：(头实体, 关系, 尾实体)

规则：
- 抽取具体、明确的实体（人名、地名、机构、作品、概念、数字）
- 关系用简短的动词短语（如"是...的子公司"、"营业收入为"、"位于"、"担任"、"同比增长"）
- 每个三元组必须是文本中明确陈述的事实，不能推断
- 实体名称应使用规范全称（不用代词）
- 每段文本抽取 3-8 个三元组
- 如果文本太短或无实质信息，返回空列表

仅返回 JSON 数组：
[{{"head": "实体1", "relation": "关系类型", "tail": "实体2"}}, ...]

文本：
{text}"""


def _get_extraction_prompt(lang=None):
    if lang is None:
        lang = getattr(__import__('config'), 'PROMPT_LANG', 'en')
    return EXTRACTION_PROMPT_ZH if lang == "zh" else EXTRACTION_PROMPT_EN


def extract_triples_from_chunk(chunk: dict, model: str) -> list[dict]:
    """对单个 chunk 调用 LLM 抽取三元组"""
    from llm.client import get_from_llm

    text = chunk["text"]
    if len(text.strip()) < 50:
        return []

    prompt = _get_extraction_prompt().format(text=text[:2000])

    for attempt in range(3):
        try:
            resp = get_from_llm(prompt, model=model)
            if not resp:
                continue

            # 解析 JSON
            import re
            # 先尝试提取 返回结果里包着 Markdown 代码块的 JSON，例如：
            # ```json
            # [{"head": "A", "relation": "是", "tail": "B"}]
            # ```
            m = re.search(r"```(?:json)?\s*\n?(.*?)```", resp, re.DOTALL)
            if m:
                resp = m.group(1).strip()

            # 找到 JSON 数组
            idx_start = resp.find("[")
            idx_end = resp.rfind("]")
            if idx_start != -1 and idx_end > idx_start:
                triples = json.loads(resp[idx_start:idx_end + 1])
            else:
                triples = json.loads(resp)

            if not isinstance(triples, list):
                continue

            # 验证格式
            valid = []
            for t in triples:
                if isinstance(t, dict) and all(k in t for k in ("head", "relation", "tail")):
                    head = str(t["head"]).strip()
                    rel = str(t["relation"]).strip()
                    tail = str(t["tail"]).strip()
                    if head and rel and tail:
                        valid.append({
                            "head": head,
                            "relation": rel,
                            "tail": tail,
                            "chunk_id": chunk["chunk_id"],
                        })
            return valid

        except (json.JSONDecodeError, Exception) as e:
            if attempt == 2:
                return []
            time.sleep(1)

    return []


def process_chunk(chunk: dict, model: str) -> list[dict]:
    """处理单个 chunk 并更新统计"""
    triples = extract_triples_from_chunk(chunk, model)
    with _lock:
        if triples:
            _stats["success"] += 1
            _stats["total_triples"] += len(triples)
        else:
            _stats["fail"] += 1
    return triples


def build_graph(all_triples: list[dict]) -> nx.MultiDiGraph:
    """从三元组列表构建 NetworkX 有向多重图"""
    # 为什么用多重图？因为两个实体之间可能有多种关系。MultiDiGraph 可以保存多条边。
    G = nx.MultiDiGraph()

    for t in all_triples:
        head = t["head"]
        tail = t["tail"]
        rel = t["relation"]
        chunk_id = t["chunk_id"]

        # 添加/更新节点
        if not G.has_node(head):
            # mentions 属性记录这个实体在哪些 chunk 中被提到，方便后续检索时定位原文
            G.add_node(head, mentions=[]) 
        # 把当前 chunk_id 加进去。如果一个实体在多个 chunk 中被提到，就会有多个 chunk_id。
        if chunk_id not in G.nodes[head]["mentions"]:
            G.nodes[head]["mentions"].append(chunk_id)

        if not G.has_node(tail):
            G.add_node(tail, mentions=[])
        if chunk_id not in G.nodes[tail]["mentions"]:
            G.nodes[tail]["mentions"].append(chunk_id)

        # 添加边
        G.add_edge(head, tail, relation=rel, chunk_id=chunk_id)

    return G


def save_graph(G: nx.MultiDiGraph, output_path: str):
    """将图保存为 JSON（node_link_data 格式）"""
    # NetworkX 图不能直接保存成 JSON，所以先转成：
    data = nx.node_link_data(G)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[save] Graph saved to {output_path}")


def compute_entity_embeddings(G: nx.MultiDiGraph, output_path: str):
    """预计算所有实体节点的 embedding"""
    from retrieval.embedder import encode

    entities = list(G.nodes())
    print(f"[embed] Computing embeddings for {len(entities)} entities...")

    # 分批编码
    batch_size = 256
    all_embeddings = []
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        emb = encode(batch, batch_size=batch_size)
        all_embeddings.append(emb)

    # vstack 合并成一个矩阵
    embeddings = np.vstack(all_embeddings)  # (N, D)

    # 保存为 {entity_name: index} 映射 + embedding 矩阵
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    data = {
        "entities": entities,
        "entity_to_idx": entity_to_idx,
        "embeddings": embeddings,
    }
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[embed] Saved entity embeddings to {output_path} (shape: {embeddings.shape})")


def print_graph_stats(G: nx.MultiDiGraph):
    """打印图的统计信息"""
    print("\n=== Knowledge Graph Statistics ===")
    print(f"Nodes (entities): {G.number_of_nodes()}")
    print(f"Edges (triples):  {G.number_of_edges()}")

    # 度分布 图里的度表示一个节点连了多少条边。如果最大度很高，说明某些实体是中心节点。
    degrees = [d for _, d in G.degree()]
    if degrees:
        print(f"Avg degree: {np.mean(degrees):.1f}")
        print(f"Max degree: {max(degrees)}")
        print(f"Median degree: {np.median(degrees):.0f}")

    # 连通性（转为无向图检查）
    UG = G.to_undirected()
    components = list(nx.connected_components(UG))
    print(f"Connected components: {len(components)}")
    if components:
        sizes = sorted([len(c) for c in components], reverse=True)
        print(f"Largest component: {sizes[0]} nodes ({sizes[0]/G.number_of_nodes()*100:.1f}%)")
        if len(sizes) > 1:
            print(f"Top 5 component sizes: {sizes[:5]}")

    # chunk 覆盖  表示有多少个 chunk 至少抽出了一个三元组
    all_chunks = set()
    for _, _, data in G.edges(data=True):
        all_chunks.add(data.get("chunk_id"))
    print(f"Chunks covered: {len(all_chunks)}")

    # 关系类型分布 如果关系类型特别多、每种只出现一两次，说明关系没有标准化。
    relations = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relation", "unknown")
        relations[rel] = relations.get(rel, 0) + 1
    sorted_rels = sorted(relations.items(), key=lambda x: x[1], reverse=True)
    print(f"Unique relation types: {len(relations)}")
    print("Top 10 relations:")
    for rel, cnt in sorted_rels[:10]:
        print(f"  {rel}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="Build knowledge graph from corpus")
    parser.add_argument("--corpus", default=os.path.join(DATA_DIR, "corpus.json"),
                        help="Path to corpus.json")
    parser.add_argument("--output-graph", default=os.path.join(INDEX_DIR, "knowledge_graph.json"),
                        help="Output graph JSON path")
    parser.add_argument("--output-embeddings", default=os.path.join(INDEX_DIR, "entity_embeddings.pkl"),
                        help="Output entity embeddings path")
    parser.add_argument("--model", default=SYNTH_LLM_MODEL, help="LLM model for extraction")
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit chunks to process (for testing)")
    parser.add_argument("--triples-cache", default=None,
                        help="Path to cache extracted triples (skip LLM if exists)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip entity embedding computation")
    parser.add_argument("--lang", choices=["en", "zh"], default=None,
                        help="Prompt language (default: config.PROMPT_LANG)")
    args = parser.parse_args()

    # 加载 corpus
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} chunks from {args.corpus}")

    if args.limit:
        corpus = corpus[:args.limit]
        print(f"Limited to {len(corpus)} chunks")

    # Step 1: 抽取三元组（支持缓存）
    triples_cache = args.triples_cache or os.path.join(INDEX_DIR, "triples_cache.jsonl")

    if os.path.exists(triples_cache):
        print(f"Loading cached triples from {triples_cache}")
        all_triples = []
        with open(triples_cache, "r", encoding="utf-8") as f:
            for line in f:
                all_triples.append(json.loads(line))
        print(f"Loaded {len(all_triples)} cached triples")
    else:
        print(f"Extracting triples with {args.workers} workers using {args.model}...")
        all_triples = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_chunk, chunk, args.model): chunk
                       for chunk in corpus}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
                triples = fut.result()
                all_triples.extend(triples)

        # 保存缓存
        os.makedirs(os.path.dirname(triples_cache), exist_ok=True)
        with open(triples_cache, "w", encoding="utf-8") as f:
            for t in all_triples:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"\nExtraction complete: {_stats}")
        print(f"Triples cached to {triples_cache}")

    print(f"Total triples: {len(all_triples)}")

    # Step 2: 构建图
    print("\nBuilding NetworkX graph...")
    G = build_graph(all_triples)
    print_graph_stats(G)

    # 保存图
    os.makedirs(os.path.dirname(args.output_graph), exist_ok=True)
    save_graph(G, args.output_graph)

    # Step 3: 计算实体 embedding
    if not args.skip_embeddings:
        compute_entity_embeddings(G, args.output_embeddings)
    else:
        print("[skip] Skipping entity embedding computation")

    print("\nDone!")


if __name__ == "__main__":
    main()

# 测试
# python scripts/build_knowledge_graph.py   --corpus data/datasets/corpus.json   --output-graph data/indexes/test_knowledge_graph.json   --output-embeddings data/indexes/entity_embeddings.pkl   --triples-cache data/indexes/test_triples_cache.jsonl   --model deepseek-v4-flash   --workers 20   --limit 5   --skip-embeddings