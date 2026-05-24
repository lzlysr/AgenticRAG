#!/usr/bin/env python3
"""从 ground truth hop 结构构建 oracle trajectory

每条 QA 的 hops 已包含完整信息：
- hop.question → 搜索 query
- hop.doc_chunk_id → 目标 chunk（从 corpus 读取 text）
- hop.answer → 每步答案
- final_answer → 最终答案

输出格式兼容 trace_to_sft.py。

用法：
  python scripts/build_oracle_traces.py
  python scripts/build_oracle_traces.py --qa data/financial_eval/train_qa_pairs_zh.json --use-zh
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_TEXT_LEN = 500


def build_oracle_trace(qa: dict, corpus_map: dict, use_zh: bool = False) -> dict:
    """从 ground truth 构建单条 oracle trajectory"""

    question = qa.get("final_question_zh", qa["final_question"]) if use_zh else qa["final_question"]
    answer = qa.get("final_answer_zh", qa["final_answer"]) if use_zh else qa["final_answer"]

    plan = []
    evidence = []
    tool_calls = []
    trace = []

    # Router
    trace.append({"node": "router", "query_type": "multi_hop"})

    # Planner
    for hop in qa["hops"]:
        step_id = hop["hop_idx"]
        sub_query = hop["question"]
        depends_on = [step_id - 1] if step_id > 1 else []
        tools = hop.get("search_tools", [])
        if not tools:
            tools = ["keyword_search"]  # 第一跳默认
        tool = tools if len(tools) > 1 else tools[0]

        plan.append({
            "id": step_id,
            "sub_query": sub_query,
            "tool": tool,
            "depends_on": depends_on,
            "status": "done",
        })

    trace.append({"node": "planner", "iteration": 1, "plan": plan})

    # Executor: 每个 hop 对应一次检索
    for hop in qa["hops"]:
        step_id = hop["hop_idx"]
        chunk_id = hop["doc_chunk_id"]
        sub_query = hop["question"]
        tools = hop.get("search_tools", [])
        if not tools:
            tools = ["keyword_search"]
        tool = tools if len(tools) > 1 else tools[0]

        chunk = corpus_map.get(chunk_id, {})
        chunk_text = chunk.get("text", "")[:MAX_TEXT_LEN]
        chunk_title = chunk.get("title", "")

        evidence.append({
            "step_id": step_id,
            "sub_query": sub_query,
            "tool": tool,
            "results": [{
                "chunk_id": chunk_id,
                "text": chunk_text,
                "title": chunk_title,
                "score": 1.0,  # oracle: 完美匹配
            }]
        })

        tool_calls.append({
            "step_id": step_id,
            "tool": tool,
            "query": sub_query,
            "num_results": 1,
        })

        trace.append({
            "node": "executor",
            "step_id": step_id,
            "tool": tool,
            "num_results": 1,
        })

    # Verifier
    trace.append({"node": "verifier", "verdict": "sufficient", "feedback": ""})

    # Synthesizer
    trace.append({"node": "synthesizer", "answer_length": len(answer)})

    return {
        "question": question,
        "gold": answer,
        "pred": answer,  # oracle: 答案正确
        "em": 1.0,
        "f1": 1.0,
        "subset": qa["subset"],
        "hop_count": qa["hop_count"],
        "qa_type": qa.get("qa_type", ""),
        "iteration_count": 1,
        "total_tool_calls": len(qa["hops"]),
        "latency": 0.0,
        "trace": trace,
        "evidence": evidence,
        "plan": plan,
        "tool_calls": tool_calls,
        "verification_result": "sufficient",
        "verification_feedback": "",
        "source": "oracle",  # 标记数据来源
    }


def main():
    parser = argparse.ArgumentParser(description="构建 Oracle Trajectory")
    parser.add_argument("--qa", default=None, help="QA 文件路径")
    parser.add_argument("--corpus", default=None, help="Corpus 文件路径")
    parser.add_argument("--output", default=None, help="输出路径")
    parser.add_argument("--use-zh", action="store_true", help="使用翻译后的中文字段")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "financial_eval")

    qa_path = args.qa or os.path.join(base_dir, "train_qa_pairs_zh.json" if args.use_zh else "train_qa_pairs.json")
    corpus_path = args.corpus or os.path.join(base_dir, "corpus.json")
    output_path = args.output or os.path.join(base_dir, "traces_oracle.jsonl")

    # 加载数据
    with open(qa_path) as f:
        qa_data = json.load(f)
    print(f"[oracle] QA: {len(qa_data)} 条 from {qa_path}")

    with open(corpus_path) as f:
        corpus = json.load(f)
    corpus_map = {c["chunk_id"]: c for c in corpus}
    print(f"[oracle] Corpus: {len(corpus)} chunks")

    # 验证覆盖率
    missing = 0
    for qa in qa_data:
        for hop in qa["hops"]:
            if hop["doc_chunk_id"] not in corpus_map:
                missing += 1

    if missing:
        print(f"[oracle] 警告: {missing} 个 hop chunk 未在 corpus 中找到")
    else:
        print(f"[oracle] Chunk 覆盖率: 100%")

    # 构建 oracle traces
    traces = []
    for qa in qa_data:
        t = build_oracle_trace(qa, corpus_map, use_zh=args.use_zh)
        traces.append(t)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"[oracle] 生成 {len(traces)} 条 oracle traces → {output_path}")

    # 统计
    from collections import Counter
    subset_counts = Counter(t["subset"] for t in traces)
    print(f"\n按 subset:")
    for subset, n in sorted(subset_counts.items()):
        avg_hops = sum(t["total_tool_calls"] for t in traces if t["subset"] == subset) / n
        print(f"  {subset:<22} {n:>4} 条, 平均 {avg_hops:.1f} hops")

    # 预览
    print(f"\n预览 (第 1 条):")
    t = traces[0]
    print(f"  Q: {t['question'][:80]}")
    print(f"  A: {t['gold'][:60]}")
    print(f"  Plan steps: {len(t['plan'])}")
    for p in t["plan"]:
        print(f"    {p['id']}. {p['tool']}: {p['sub_query'][:60]}")
    print(f"  Evidence chunks: {[e['results'][0]['chunk_id'] for e in t['evidence']]}")


if __name__ == "__main__":
    main()
