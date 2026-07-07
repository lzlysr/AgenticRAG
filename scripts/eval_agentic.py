#!/usr/bin/env python3
"""Agentic 评测：模型自主 tool_call → 环境执行检索 → 模型继续 → <answer>

与 GRPO rollout 完全一致的评测方式，使用 Qwen3 原生 tool calling 格式。评价tool调用能力

用法：
  # 需要先启动 vLLM
  python scripts/eval_agentic.py --model Qwen3-4B-GRPO-v4e

  # 指定最大轮次和样本数
  python scripts/eval_agentic.py --model Qwen3-4B-GRPO-v4e \
    --max-turns 7 --max-samples 185
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.hop_aware_eval import compute_hop_recall, aggregate_diagnostics
from evaluation.metrics import exact_match, f1_score
from evaluation.ablation import load_financial_qa_pairs
from llm.client import get_from_llm
from config import RESULTS_DIR, ACTIVE_DATA_DIR, ACTIVE_INDEX_DIR
from retrieval.keyword_search import tokenize

DEFAULT_FINANCIAL_QA_FILES = [
    os.path.join(ACTIVE_DATA_DIR, "train_qa_pairs_zh_clean.json"),
    os.path.join(ACTIVE_DATA_DIR, "qa_pairs_zh_clean.json"),
]


def _select_retrieval_device() -> str:
    """选择检索模型设备：优先 RETRIEVAL_DEVICE，否则选显存占用最少的 GPU。"""
    env_device = os.environ.get("RETRIEVAL_DEVICE")
    if env_device:
        return env_device
    import torch
    if not torch.cuda.is_available():
        return "cpu"

    mem_used = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        mem_used.append(total - free)
    gpu_idx = mem_used.index(min(mem_used))
    return f"cuda:{gpu_idx}"


def _resolve_qa_file(path: str) -> str:
    """裸文件名默认从 ACTIVE_DATA_DIR 下读取；显式路径按原样读取。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(ACTIVE_DATA_DIR, path)

# ── 与 SFT/GRPO 一致的配置 ──────────────────────────────────────

AGENTIC_SYSTEM_PROMPT = "你是一个金融文档问答 Agent。通过搜索相关文档来回答用户的问题。"

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "keyword_search", "description": "使用关键词匹配（BM25）搜索金融文档", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索查询关键词"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "semantic_search", "description": "使用语义向量检索金融文档", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "语义搜索查询"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "graph_search", "description": "使用知识图谱搜索金融文档中的实体关系", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "实体关系查询"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "hybrid_search", "description": "使用多个工具进行 RRF 融合检索并重排", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索查询"}, "tools": {"type": "array", "items": {"type": "string"}, "description": "要融合的工具列表"}}, "required": ["query", "tools"]}}},
]

# ── 检索工具 ──────────────────────────────────────────────────────

_retrieval = {"chunk_store": None, "chunk_ids": None, "bm25": None}
_lock = Lock()


def _load_retrieval(device: str):
    """从 config.ACTIVE_INDEX_DIR 加载 keyword_search 所需索引。"""
    with _lock:
        if _retrieval["chunk_store"] is not None:
            return

        index_dir = ACTIVE_INDEX_DIR
        os.environ.setdefault("RETRIEVAL_DEVICE", device)

        with open(os.path.join(index_dir, "chunk_ids.json")) as f:
            _retrieval["chunk_ids"] = json.load(f)
        with open(os.path.join(index_dir, "chunk_store.pkl"), "rb") as f:
            _retrieval["chunk_store"] = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            _retrieval["bm25"] = pickle.load(f)

        if device != "cpu":
            import retrieval.embedder as _emb
            import retrieval.reranker as _rnk

            _orig_emb_get = _emb._get_model
            _orig_rnk_get = _rnk._get_model

            def _emb_get_device(requested_device=None):
                return _orig_emb_get(requested_device or device)

            def _rnk_get_device(requested_device=None):
                return _orig_rnk_get(requested_device or device)

            _emb._get_model = _emb_get_device
            _rnk._get_model = _rnk_get_device
            _emb_get_device()
            _rnk_get_device()

        print(f"[eval] Retrieval loaded from {index_dir} on {device}")


def _format_tool_results(results: list[dict], max_len: int = 300) -> str:
    '''格式化检索结果为字符串，截断每条文本到 max_len 字符'''
    parts = []
    for item in results:
        cid = item.get("chunk_id", "")
        text = item.get("text", "")
        if cid and text:
            parts.append(f"[{cid}] {text[:max_len]}")
    return "\n".join(parts) if parts else "(no results)"


def _keyword_search_results(query: str, top_k=3) -> list[dict]:
    scores = _retrieval["bm25"].get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if scores[idx] <= 0:
            break
        cid = _retrieval["chunk_ids"][idx]
        doc = _retrieval["chunk_store"].get(cid, {})
        results.append({
            "chunk_id": cid,
            "text": doc.get("text", ""),
            "title": doc.get("title", ""),
            "score": float(scores[idx]),
            "source": "bm25",
        })
    return results


def _keyword_search(query: str, top_k=3, max_len=300) -> str:
    return _format_tool_results(_keyword_search_results(query, top_k=top_k), max_len=max_len)


def _semantic_search(query: str, top_k=3, max_len=300) -> str:
    from retrieval.semantic_search import semantic_search

    results = semantic_search(query, top_k=15, rerank_top_k=top_k)
    return _format_tool_results(results, max_len=max_len)


def _graph_search(query: str, top_k=3, max_len=300) -> str:
    from retrieval.graph_search import graph_search

    results = graph_search(query, top_k=top_k)
    return _format_tool_results(results, max_len=max_len)


def _hybrid_search(query: str, tools: list = None, top_k=3, max_len=300) -> str:
    tools = tools or ["keyword_search", "semantic_search"]
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",") if t.strip()]
    from retrieval.graph_search import graph_search
    from retrieval.hybrid_search import multi_tool_search
    from retrieval.semantic_search import semantic_search

    tool_registry = {
        "keyword_search": lambda q: _keyword_search_results(q, top_k=10),
        "semantic_search": lambda q: semantic_search(q, top_k=15, rerank_top_k=10),
        "graph_search": lambda q: graph_search(q, top_k=10),
    }
    results = multi_tool_search(query, tools, tool_registry, top_k=top_k)
    return _format_tool_results(results, max_len=max_len)


TOOL_DISPATCH = {
    "keyword_search": lambda params: _keyword_search(params.get("query", "")),
    "semantic_search": lambda params: _semantic_search(params.get("query", "")),
    "graph_search": lambda params: _graph_search(params.get("query", "")),
    "hybrid_search": lambda params: _hybrid_search(params.get("query", ""), params.get("tools")),
}

# ── 答案提取与评分 ──────────────────────────────────────────────────


def _extract_answer(text: str) -> str:
    '''从模型完整输出轨迹里提取最终答案字符串，用来算 EM/F1。'''
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _best_f1(pred: str, gold: str, aliases: list) -> float:
    return f1_score(pred, gold, aliases=[a for a in aliases if a])


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


def run_agentic_single(model: str, question: str,
                       max_turns: int = 7) -> dict:
    """单个问题的 agentic 推理：多轮 tool_call → answer"""
    system_prompt = _build_system_prompt_with_tools()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    tool_calls_made = []
    evidence_pieces = []  # 收集所有检索到的 evidence（字符串）
    structured_evidence = []  # 结构化 evidence，用于 tool_call_metrics
    full_trajectory = ""
    num_assistant_turns = 0

    for turn in range(max_turns):
        try:
            # 不能用 agent_chat_json()：这里需要保留原始 XML tool_call/answer 文本，
            # agent_chat_json() 会把回复当 JSON 解析，解析失败时还会丢掉原始工具调用轨迹。
            content = get_from_llm(
                messages,
                model_name=model,
                max_tokens=1024,
            ) or ""
        except Exception as e:
            full_trajectory += f"\n[ERROR] {e}"
            break

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

                # 从结果字符串中解析 chunk_id（格式: "[chunk_id] text..."）
                chunk_ids = re.findall(r'^\[([^\]]+)\]', result, re.MULTILINE)
                structured_evidence.append({
                    "tool": fn_name,
                    "query": fn_args.get("query", ""),
                    "results": [{"chunk_id": cid} for cid in chunk_ids],
                })

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
        "structured_evidence": structured_evidence,
        "num_turns": num_assistant_turns,
        "answer": _extract_answer(full_trajectory),
        "evidence": "\n".join(evidence_pieces),
    }


# ── 主评测流程 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agentic evaluation")
    parser.add_argument("--model", default='Qwen3-4B')
    parser.add_argument("--max-samples", type=int, default=982) # 训练 797 + 测试 185
    # 单个问题最多允许模型进行多少轮 assistant 回复
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument(
        "--qa-files-financial",
        default='qa_pairs_zh_clean.json',
        help=(
            "Comma-separated financial QA JSON files. Default combines "
            "train_qa_pairs_zh_clean.json and qa_pairs_zh_clean.json."
        ),
    )
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    device = _select_retrieval_device()

    # 加载检索索引
    print(f"[eval] Loading retrieval on {device}...")
    _load_retrieval(device)

    # 加载 QA 数据
    if args.qa_files_financial:
        qa_files = [_resolve_qa_file(p.strip()) for p in args.qa_files_financial.split(",") if p.strip()]
    else:
        qa_files = DEFAULT_FINANCIAL_QA_FILES
    qa_data = load_financial_qa_pairs(qa_files)
    qa_data = qa_data[:args.max_samples]
    print(f"[eval] {len(qa_data)} samples loaded")

    results = []
    total_em, total_f1 = 0, 0

    pbar = tqdm(total=len(qa_data), desc=f"Agentic {args.model}")

    def _eval_one(idx, item):
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
        out = run_agentic_single(args.model, question, max_turns=args.max_turns)
        elapsed = time.time() - t0

        pred = out["answer"]
        f1 = _best_f1(pred, gold, aliases)
        em = exact_match(pred, gold, aliases=aliases)

        state_like = {"evidence": out["structured_evidence"]}
        hop_recall = compute_hop_recall(state_like, item)
        diagnostics = {
            "hop_recall": round(hop_recall, 3),
            "premature_collapse": hop_recall < 0.5 and len(out["tool_calls"]) == 0,
            "over_extension": len(out["tool_calls"]) > hops * 3,
            "step_alignment": round(len(out["structured_evidence"]) / hops, 3) if hops > 0 else 0,
            "hop_count": hops,
            "total_tool_calls": len(out["tool_calls"]),
            "iteration_count": out["num_turns"],
            "evidence_count": len(out["structured_evidence"]),
        }

        return {
            "idx": idx,
            "question": question,
            "gold": gold,
            "pred": pred,
            "f1": f1,
            "em": em,
            "num_turns": out["num_turns"],
            "num_tool_calls": len(out["tool_calls"]),
            "tool_calls": out["tool_calls"],
            "evidence": out["evidence"],
            "diagnostics": diagnostics,
            "latency": elapsed,
            "hops": hops,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_eval_one, i, item): i for i, item in enumerate(qa_data)}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            total_em += r["em"]
            total_f1 += r["f1"]
            n = len(results)
            pbar.update(1)
            pbar.set_postfix_str(f"EM={total_em/n:.3f} F1={total_f1/n:.3f}")

    pbar.close()
    # 并发写入时，写入顺序可能乱，所以再次按 idx 排序
    results = sorted(results, key=lambda r: r["idx"])

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
        "diagnostics": aggregate_diagnostics([r["diagnostics"] for r in results]),
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
    out_dir = os.path.join(RESULTS_DIR, "eval_agentic", "financial")
    os.makedirs(out_dir, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    out_path = os.path.join(out_dir, f"agentic_{n}_{safe_model}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
