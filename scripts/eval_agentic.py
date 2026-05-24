#!/usr/bin/env python3
"""Agentic 评测：模型自主 tool_call → 环境执行检索 → 模型继续 → <answer>

与 GRPO rollout 完全一致的评测方式，使用 Qwen3 原生 tool calling 格式。

用法：
  # 需要先启动 vLLM
  python scripts/eval_agentic.py --model Qwen3-4B-GRPO-v4e --port 9099

  # 指定最大轮次和样本数
  python scripts/eval_agentic.py --model Qwen3-4B-GRPO-v4e --port 9099 \
    --max-turns 7 --max-samples 185
"""
import argparse
import json
import os
import pickle
import re
import string
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 与 SFT/GRPO 一致的配置 ──────────────────────────────────────

AGENTIC_SYSTEM_PROMPT = "你是一个金融文档问答 Agent。通过搜索相关文档来回答用户的问题。"

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "keyword_search", "description": "使用关键词匹配（BM25）搜索金融文档", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索查询关键词"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "semantic_search", "description": "使用语义向量检索金融文档", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "语义搜索查询"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "graph_search", "description": "使用知识图谱搜索金融文档中的实体关系", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "实体关系查询"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "hybrid_search", "description": "使用多个工具进行 RRF 融合检索并重排", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索查询"}, "tools": {"type": "array", "items": {"type": "string"}, "description": "要融合的工具列表"}}, "required": ["query", "tools"]}}},
]

# ── 检索工具 ──────────────────────────────────────────────────────

_retrieval = {"chunk_store": None, "chunk_ids": None, "bm25": None,
              "faiss_index": None, "graph": None, "entity_data": None,
              "embedder": None, "reranker": None}
_lock = Lock()


def _load_retrieval(index_dir: str, device: str):
    """加载检索索引（BM25 + FAISS + Graph + Embedder + Reranker）"""
    with _lock:
        if _retrieval["chunk_store"] is not None:
            return

        with open(os.path.join(index_dir, "chunk_ids.json")) as f:
            _retrieval["chunk_ids"] = json.load(f)
        with open(os.path.join(index_dir, "chunk_store.pkl"), "rb") as f:
            _retrieval["chunk_store"] = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            _retrieval["bm25"] = pickle.load(f)

        import faiss
        _retrieval["faiss_index"] = faiss.read_index(os.path.join(index_dir, "faiss.index"))

        import networkx as nx
        with open(os.path.join(index_dir, "knowledge_graph.json")) as f:
            _retrieval["graph"] = nx.node_link_graph(json.load(f))
        with open(os.path.join(index_dir, "entity_embeddings.pkl"), "rb") as f:
            _retrieval["entity_data"] = pickle.load(f)

        from sentence_transformers import SentenceTransformer, CrossEncoder
        _retrieval["embedder"] = SentenceTransformer(
            os.environ.get("BGE_M3_PATH", "models/bge-m3"), device=device)
        _retrieval["reranker"] = CrossEncoder(
            os.environ.get("BGE_RERANKER_PATH", "models/bge-reranker-v2-m3"),
            max_length=512, device=device)
        print(f"[eval] Retrieval loaded on {device}")


def _tokenize(text: str) -> list[str]:
    import jieba
    tokens = []
    for part in re.split(r'([\u4e00-\u9fff]+)', text.lower()):
        if re.match(r'^[\u4e00-\u9fff]+$', part):
            tokens.extend(jieba.lcut(part))
        else:
            tokens.extend(part.split())
    return [t for t in tokens if t.strip()]


def _keyword_search(query: str, top_k=3, max_len=300) -> str:
    scores = _retrieval["bm25"].get_scores(_tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    parts = []
    for idx in top_idx:
        if scores[idx] <= 0:
            break
        cid = _retrieval["chunk_ids"][idx]
        text = _retrieval["chunk_store"].get(cid, {}).get("text", "")
        parts.append(f"[{cid}] {text[:max_len]}")
    return "\n".join(parts) if parts else "(no results)"


def _semantic_search(query: str, top_k=3, max_len=300) -> str:
    q_vec = _retrieval["embedder"].encode([query], normalize_embeddings=True)
    scores, indices = _retrieval["faiss_index"].search(q_vec, 20)
    candidates = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        cid = _retrieval["chunk_ids"][idx]
        doc = _retrieval["chunk_store"].get(cid, {})
        candidates.append({"chunk_id": cid, "text": doc.get("text", ""), "score": float(scores[0][i])})
    # rerank
    if len(candidates) > top_k:
        passages = [c["text"] for c in candidates[:15]]
        rs = _retrieval["reranker"].predict([[query, p] for p in passages])
        ranked = sorted(enumerate(rs), key=lambda x: x[1], reverse=True)
        candidates = [candidates[i] for i, _ in ranked[:top_k]]
    else:
        candidates = candidates[:top_k]
    parts = [f"[{c['chunk_id']}] {c['text'][:max_len]}" for c in candidates]
    return "\n".join(parts) if parts else "(no results)"


def _graph_search(query: str, top_k=3, max_len=300) -> str:
    from collections import deque
    entities = _retrieval["entity_data"]["entities"]
    embeddings = _retrieval["entity_data"]["embeddings"]
    q_vec = _retrieval["embedder"].encode([query], normalize_embeddings=True)
    scores = (embeddings @ q_vec.T).flatten()
    top_ent = [entities[i] for i in np.argsort(scores)[::-1][:5] if scores[i] > 0.3]
    if not top_ent:
        return "(no results)"
    graph = _retrieval["graph"]
    chunk_scores, visited, queue = {}, set(), deque()
    for ent in top_ent:
        if ent in graph:
            queue.append((ent, 0)); visited.add(ent)
    while queue:
        node, depth = queue.popleft()
        if depth > 2:
            continue
        for cid in graph.nodes[node].get("chunk_ids", []):
            chunk_scores[cid] = max(chunk_scores.get(cid, 0), 1.0 / (1 + depth))
        if depth < 2:
            for nb in graph.neighbors(node):
                if nb not in visited:
                    visited.add(nb); queue.append((nb, depth + 1))
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    candidates = []
    for cid, sc in sorted_chunks:
        doc = _retrieval["chunk_store"].get(cid, {})
        if doc:
            candidates.append({"chunk_id": cid, "text": doc.get("text", ""), "score": sc})
    if len(candidates) > top_k:
        passages = [c["text"] for c in candidates[:15]]
        rs = _retrieval["reranker"].predict([[query, p] for p in passages])
        ranked = sorted(enumerate(rs), key=lambda x: x[1], reverse=True)
        candidates = [candidates[i] for i, _ in ranked[:top_k]]
    parts = [f"[{c['chunk_id']}] {c['text'][:max_len]}" for c in candidates]
    return "\n".join(parts) if parts else "(no results)"


def _hybrid_search(query: str, tools: list = None, top_k=3, max_len=300) -> str:
    tools = tools or ["keyword_search", "semantic_search"]
    all_results = []
    tool_fns = {"keyword_search": _keyword_search, "semantic_search": _semantic_search, "graph_search": _graph_search}
    for t in tools:
        fn = tool_fns.get(t)
        if fn:
            raw = fn(query, top_k=10, max_len=max_len)
            for line in raw.split("\n"):
                m = re.match(r'\[([^\]]+)\]\s*(.*)', line)
                if m:
                    all_results.append({"chunk_id": m.group(1), "text": m.group(2)})
    # RRF
    chunk_scores, chunk_data = {}, {}
    for rank, r in enumerate(all_results):
        cid = r["chunk_id"]
        chunk_scores[cid] = chunk_scores.get(cid, 0) + 1.0 / (60 + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = r
    sorted_ids = sorted(chunk_scores, key=lambda x: chunk_scores[x], reverse=True)
    candidates = [chunk_data[cid] for cid in sorted_ids[:15]]
    if len(candidates) > top_k:
        passages = [c["text"] for c in candidates]
        rs = _retrieval["reranker"].predict([[query, p] for p in passages])
        ranked = sorted(enumerate(rs), key=lambda x: x[1], reverse=True)
        candidates = [candidates[i] for i, _ in ranked[:top_k]]
    parts = [f"[{c['chunk_id']}] {c['text'][:max_len]}" for c in candidates]
    return "\n".join(parts) if parts else "(no results)"


TOOL_DISPATCH = {
    "keyword_search": lambda params: _keyword_search(params.get("query", "")),
    "semantic_search": lambda params: _semantic_search(params.get("query", "")),
    "graph_search": lambda params: _graph_search(params.get("query", "")),
    "hybrid_search": lambda params: _hybrid_search(params.get("query", ""), params.get("tools")),
}

# ── 答案提取与评分 ──────────────────────────────────────────────────

_CN_PUNCTUATION = '。，、；：？！""''【】《》（）｛｝〔〕·…—～'
_ALL_PUNCTUATION = set(string.punctuation) | set(_CN_PUNCTUATION)


def _normalize(text: str) -> str:
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif ch == '\u3000':
            result.append(' ')
        elif unicodedata.category(ch).startswith('Zs'):
            result.append(' ')
        else:
            result.append(ch)
    text = ''.join(result).lower()
    text = ''.join(ch for ch in text if ch not in _ALL_PUNCTUATION)
    text = re.sub(r'(\d)([\u4e00-\u9fff])', r'\1 \2', text)
    text = re.sub(r'([\u4e00-\u9fff])(\d)', r'\1 \2', text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _token_f1(pred: str, gold: str) -> float:
    pt, gt = _normalize(pred).split(), _normalize(gold).split()
    if not gt:
        return 1.0 if not pt else 0.0
    if not pt:
        return 0.0
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pt), common / len(gt)
    return 2 * p * r / (p + r)


def _extract_answer(text: str) -> str:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _best_f1(pred: str, gold: str, aliases: list) -> float:
    candidates = [gold] + [a for a in aliases if a]
    return max(_token_f1(pred, c) for c in candidates)


# ── 单样本 Agentic 推理 ──────────────────────────────────────────

def _build_system_prompt_with_tools() -> str:
    """构造和 GRPO rollout 一致的 system prompt（含 # Tools 段）

    Qwen3 chat_template 在 tools 参数存在时会注入这段。
    我们手动构造以避免依赖 vLLM 的 tool calling 功能。
    """
    tools_text = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    for schema in TOOL_SCHEMAS:
        tools_text += f"\n{json.dumps(schema, ensure_ascii=False)}"
    tools_text += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    return AGENTIC_SYSTEM_PROMPT + tools_text


def run_agentic_single(client: OpenAI, model: str, question: str,
                       max_turns: int = 7, temperature: float = 0.7) -> dict:
    """单个问题的 agentic 推理：多轮 tool_call → answer"""
    system_prompt = _build_system_prompt_with_tools()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    tool_calls_made = []
    evidence_pieces = []  # 收集所有检索到的 evidence
    full_trajectory = ""
    num_assistant_turns = 0

    for turn in range(max_turns):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
            )
        except Exception as e:
            full_trajectory += f"\n[ERROR] {e}"
            break

        choice = resp.choices[0]
        content = choice.message.content or ""
        num_assistant_turns += 1
        full_trajectory += content

        # 检查是否有 <tool_call> 标签（hermes 格式）
        tc_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
        if tc_match:
            try:
                tc_data = json.loads(tc_match.group(1))
                fn_name = tc_data.get("name", "")
                fn_args = tc_data.get("arguments", {})
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)
                tool_calls_made.append({"tool": fn_name, "args": fn_args})

                tool_fn = TOOL_DISPATCH.get(fn_name)
                result = tool_fn(fn_args) if tool_fn else f"(unknown tool: {fn_name})"
                evidence_pieces.append(result)

                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"})
                full_trajectory += f"\n[tool:{fn_name}] {result[:200]}...\n"
                continue
            except (json.JSONDecodeError, TypeError):
                pass

        # 没有 tool_call → 最终回答
        messages.append({"role": "assistant", "content": content})
        break

    return {
        "trajectory": full_trajectory,
        "tool_calls": tool_calls_made,
        "num_turns": num_assistant_turns,
        "answer": _extract_answer(full_trajectory),
        "evidence": "\n".join(evidence_pieces),
    }


# ── 主评测流程 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agentic evaluation")
    parser.add_argument("--model", required=True, help="Model name in vLLM or remote API")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--api-base", default=None,
                        help="Override API base URL (e.g. http://172.24.40.164:8086/v1 or https://kspmas.ksyun.com/v1/)")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key (default: EMPTY for vLLM)")
    parser.add_argument("--max-samples", type=int, default=185)
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--data-dir", default="data/financial_eval_zh")
    parser.add_argument("--index-dir", default="data/financial_all/indexes")
    parser.add_argument("--device", default=None, help="Retrieval device (default: env RETRIEVAL_DEVICE or cuda:1)")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = args.device or os.environ.get("RETRIEVAL_DEVICE", "cuda:1")

    # 加载检索索引
    print(f"[eval] Loading retrieval on {device}...")
    _load_retrieval(args.index_dir, device)

    # 加载 QA 数据
    qa_path = os.path.join(args.data_dir, "qa_pairs.json")
    with open(qa_path) as f:
        qa_data = json.load(f)
    qa_data = qa_data[:args.max_samples]
    print(f"[eval] {len(qa_data)} samples loaded")

    base_url = args.api_base or f"http://localhost:{args.port}/v1"
    client = OpenAI(api_key=args.api_key, base_url=base_url)
    print(f"[eval] API base: {base_url}, model: {args.model}")

    results = []
    total_em, total_f1 = 0, 0

    pbar = tqdm(total=len(qa_data), desc=f"Agentic {args.model}")

    def _eval_one(item):
        question = item.get("final_question", item.get("question", ""))
        gold = item.get("final_answer", item.get("answer", item.get("target", "")))
        aliases = item.get("answer_aliases", [])
        if isinstance(aliases, str):
            try:
                aliases = json.loads(aliases)
            except (json.JSONDecodeError, TypeError):
                aliases = [aliases]
        hops = item.get("hop_count", item.get("num_hops", 2))
        if isinstance(hops, str):
            hops = int(hops)

        t0 = time.time()
        out = run_agentic_single(client, args.model, question,
                                 max_turns=args.max_turns, temperature=args.temperature)
        elapsed = time.time() - t0

        pred = out["answer"]
        f1 = _best_f1(pred, gold, aliases)
        em = 1.0 if _normalize(pred) == _normalize(gold) else 0.0

        return {
            "question": question,
            "gold": gold,
            "pred": pred,
            "f1": f1,
            "em": em,
            "num_turns": out["num_turns"],
            "num_tool_calls": len(out["tool_calls"]),
            "tool_calls": out["tool_calls"],
            "evidence": out["evidence"],
            "latency": elapsed,
            "hops": hops,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_eval_one, item): i for i, item in enumerate(qa_data)}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            total_em += r["em"]
            total_f1 += r["f1"]
            n = len(results)
            pbar.update(1)
            pbar.set_postfix_str(f"EM={total_em/n:.3f} F1={total_f1/n:.3f}")

    pbar.close()

    # 汇总
    n = len(results)
    avg_em = total_em / n
    avg_f1 = total_f1 / n
    avg_turns = sum(r["num_turns"] for r in results) / n
    avg_tool_calls = sum(r["num_tool_calls"] for r in results) / n
    avg_latency = sum(r["latency"] for r in results) / n

    # 按 hop 分组
    by_hop = {}
    for r in results:
        h = r["hops"]
        by_hop.setdefault(h, []).append(r)

    summary = {
        "model": args.model,
        "num_samples": n,
        "avg_em": avg_em,
        "avg_f1": avg_f1,
        "avg_turns": avg_turns,
        "avg_tool_calls": avg_tool_calls,
        "avg_latency": avg_latency,
        "by_hop": {},
    }
    for h, items in sorted(by_hop.items()):
        summary["by_hop"][f"{h}hop"] = {
            "count": len(items),
            "em": sum(r["em"] for r in items) / len(items),
            "f1": sum(r["f1"] for r in items) / len(items),
            "avg_turns": sum(r["num_turns"] for r in items) / len(items),
            "avg_tool_calls": sum(r["num_tool_calls"] for r in items) / len(items),
        }

    print(f"\n{'='*60}")
    print(f"Agentic Eval: {args.model}")
    print(f"{'='*60}")
    print(f"  EM:         {avg_em:.3f}")
    print(f"  F1:         {avg_f1:.3f}")
    print(f"  Avg Turns:  {avg_turns:.1f}")
    print(f"  Avg Tools:  {avg_tool_calls:.1f}")
    print(f"  Avg Latency: {avg_latency:.1f}s")
    print(f"\n  By hop:")
    for h, s in summary["by_hop"].items():
        print(f"    {h}: EM={s['em']:.3f} F1={s['f1']:.3f} turns={s['avg_turns']:.1f} tools={s['avg_tool_calls']:.1f} (n={s['count']})")

    # 保存
    out_path = f"results/agentic_{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
