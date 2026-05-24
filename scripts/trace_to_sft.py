#!/usr/bin/env python3
"""将收集的 Agent trace 转换为 ReAct 风格 SFT 格式

ReAct 格式: think + tool_call + observation 循环 → <answer>

用法:
  python scripts/trace_to_sft.py \
    --input data/financial_eval/traces_oracle_zh.jsonl \
    --output-dir data/financial_eval/sft/ \
    --lang zh
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT_EN = (
    "You are a financial document QA agent. You have access to the following tools:\n"
    "- keyword_search(query): Search documents using keyword matching (BM25)\n"
    "- semantic_search(query): Search documents using semantic similarity\n"
    "- graph_search(query): Search related entities and documents via knowledge graph\n"
    "- hybrid_search(query, tools): Search using multiple tools with RRF fusion and reranking\n"
    "- read_chunk(chunk_id): Read a specific document chunk by ID\n\n"
    "Answer the user's question by searching relevant documents. "
    "Think step by step and use tools to gather evidence before answering."
)

SYSTEM_PROMPT_ZH = (
    "你是一个金融文档问答 Agent。你可以使用以下工具：\n"
    "- keyword_search(query): 使用关键词匹配（BM25）搜索文档\n"
    "- semantic_search(query): 使用语义相似度搜索文档\n"
    "- graph_search(query): 使用知识图谱搜索相关实体和文档\n"
    "- hybrid_search(query, tools): 使用多个工具进行 RRF 融合检索并重排\n"
    "- read_chunk(chunk_id): 通过 ID 读取特定文档片段\n\n"
    "通过搜索相关文档来回答用户的问题。逐步思考并使用工具收集证据后再作答。"
)


def _get_system_prompt(lang=None):
    if lang is None:
        import config
        lang = getattr(config, "PROMPT_LANG", "en")
    return SYSTEM_PROMPT_ZH if lang == "zh" else SYSTEM_PROMPT_EN


def _build_evidence_map(trace_data: dict) -> dict[int, dict]:
    """按 step_id 索引 evidence"""
    emap = {}
    for e in trace_data.get("evidence", []):
        sid = e.get("step_id")
        if sid is not None:
            emap[sid] = e
    return emap


def _format_observation(evidence_entry: dict, max_results: int = 3, max_text: int = 300) -> str:
    """格式化工具返回结果"""
    results = evidence_entry.get("results", [])[:max_results]
    if not results:
        return "(no results)"
    parts = []
    for r in results:
        chunk_id = r.get("chunk_id", "?")
        text = r.get("text", "")[:max_text]
        parts.append(f"[{chunk_id}] {text}")
    return "\n".join(parts)


def _normalize_tool(tool_field):
    """将 tool 字段标准化为 (tool_name, arguments_dict)。

    单工具: "keyword_search" → ("keyword_search", {"query": sub_query})
    多工具: ["keyword_search", "semantic_search"] → ("hybrid_search", {"query": sub_query, "tools": [...]})
    """
    if isinstance(tool_field, list) and len(tool_field) > 1:
        return "hybrid_search", tool_field
    elif isinstance(tool_field, list):
        return tool_field[0] if tool_field else "keyword_search", None
    else:
        return tool_field or "keyword_search", None


def _make_tool_args(tool_name: str, sub_query: str, tool_list: list | None) -> dict:
    """构建工具调用参数"""
    if tool_name == "read_chunk":
        return {"chunk_id": sub_query}
    elif tool_name == "hybrid_search" and tool_list:
        return {"query": sub_query, "tools": tool_list}
    else:
        return {"query": sub_query}


AGENTIC_SYSTEM_PROMPT = "你是一个金融文档问答 Agent。通过搜索相关文档来回答用户的问题。"

# Qwen3 原生 tool schema（与 GRPO tool_config 一致）
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "keyword_search", "description": "使用关键词匹配（BM25）搜索金融文档", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索查询关键词"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "semantic_search", "description": "使用语义向量检索金融文档", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "语义搜索查询"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "graph_search", "description": "使用知识图谱搜索金融文档中的实体关系", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "实体关系查询"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "hybrid_search", "description": "使用多个工具进行 RRF 融合检索并重排", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索查询"}, "tools": {"type": "array", "items": {"type": "string"}, "description": "要融合的工具列表"}}, "required": ["query", "tools"]}}},
]


def to_react(trace_data: dict, lang: str = None) -> dict | None:
    """格式 A: ReAct 风格 — Qwen3 原生 tool calling 格式

    用 tokenizer.apply_chat_template(tools=TOOL_SCHEMAS) 生成训练文本，
    确保和 GRPO rollout 时的 tokenization 100% 一致。
    """
    plan = trace_data.get("plan", [])
    evidence_map = _build_evidence_map(trace_data)

    if not plan:
        return None

    messages = [
        {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
        {"role": "user", "content": trace_data["question"]},
    ]

    for i, step in enumerate(plan):
        sub_query = step.get("sub_query", "")
        tool_name, tool_list = _normalize_tool(step.get("tool", "keyword_search"))
        step_id = step.get("id", i + 1)

        # Think + tool call
        if i == 0:
            think = f"I need to search for: {sub_query}"
        else:
            think = f"Now I need to find: {sub_query}"

        tool_args = _make_tool_args(tool_name, sub_query, tool_list)
        tool_call_json = json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False)

        messages.append({
            "role": "assistant",
            "content": f"<think>{think}</think>\n<tool_call>\n{tool_call_json}\n</tool_call>",
        })

        # Tool response
        ev = evidence_map.get(step_id, {})
        observation = _format_observation(ev)
        messages.append({
            "role": "tool",
            "content": observation,
        })

    # Final answer
    messages.append({
        "role": "assistant",
        "content": f"<answer>{trace_data['pred']}</answer>",
    })

    return {"messages": messages, "tools": TOOL_SCHEMAS, "format": "react"}


def main():
    parser = argparse.ArgumentParser(description="Trace → SFT ReAct 格式转换")
    parser.add_argument("--input", required=True, help="traces jsonl 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--max-iter", type=int, default=2, help="最大迭代数过滤 (默认 2)")
    parser.add_argument("--require-correct", action="store_true", default=True,
                        help="只保留 em=1 的 trace (默认 True)")
    parser.add_argument("--lang", choices=["en", "zh"], default=None,
                        help="Prompt 语言（默认读 config.PROMPT_LANG）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取 traces
    traces = []
    with open(args.input) as f:
        for line in f:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print(f"[sft] 读取 {len(traces)} 条 trace")

    # 质量筛选
    filtered = []
    for t in traces:
        if "error" in t:
            continue
        if args.require_correct and t.get("em", 0) != 1:
            continue
        if t.get("iteration_count", 99) > args.max_iter:
            continue
        if not t.get("evidence"):
            continue
        filtered.append(t)

    print(f"[sft] 筛选后: {len(filtered)} 条 (em=1 & iter≤{args.max_iter} & evidence非空)")

    # 转换 ReAct 格式
    out_path = os.path.join(args.output_dir, "sft_react.jsonl")
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for t in filtered:
            result = to_react(t, lang=args.lang)
            if result:
                result["question"] = t["question"]
                result["gold"] = t["gold"]
                result["subset"] = t["subset"]
                result["hop_count"] = t["hop_count"]
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
    print(f"  react: {count} 条 → {out_path}")

    # 统计
    print(f"\n[sft] 按 subset 统计 (筛选后):")
    from collections import Counter
    subset_counts = Counter(t["subset"] for t in filtered)
    for subset, n in sorted(subset_counts.items()):
        print(f"  {subset:<22} {n:>4}")

    print(f"\n[sft] 转换完成！")


if __name__ == "__main__":
    main()
