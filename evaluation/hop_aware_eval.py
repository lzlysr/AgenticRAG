"""AgenticRAGTracer Hop-Aware 诊断评测 ★

核心诊断指标：
- hop_recall: gold hop 中被成功检索到的比例
- premature_collapse: agent 过早终止
- over_extension: agent 过度检索
- step_alignment: agent 步骤数 vs 实际 hop 数
"""


def compute_hop_recall(state: dict, qa_item: dict) -> float:
    """计算 hop-level 召回率：gold docs 中被检索到的比例"""
    gold_chunk_ids = set()
    for hop in qa_item.get("hops", []):
        cid = hop.get("doc_chunk_id", "")
        if cid:
            gold_chunk_ids.add(cid)

    if not gold_chunk_ids:
        return 1.0

    # 从 agent evidence 中收集所有检索到的 chunk_ids
    retrieved_ids = set()
    for e in state.get("evidence", []):
        for r in e.get("results", []):
            rid = r.get("chunk_id", "")
            if rid:
                retrieved_ids.add(rid)

    found = gold_chunk_ids & retrieved_ids
    return len(found) / len(gold_chunk_ids)


def diagnose_failure(state: dict, qa_item: dict) -> dict:
    """诊断 agent 执行失败模式"""
    hop_count = qa_item.get("hop_count", 2)
    total_tool_calls = state.get("total_tool_calls", 0)
    iteration_count = state.get("iteration_count", 0)
    evidence_count = len(state.get("evidence", []))

    hop_recall = compute_hop_recall(state, qa_item)

    # premature_collapse: 证据不足就停了
    premature = hop_recall < 0.5 and iteration_count <= 1

    # over_extension: 工具调用远超 hop 数
    over_ext = total_tool_calls > hop_count * 3

    # step_alignment: 执行步骤数 vs hop 数的比值
    alignment = evidence_count / hop_count if hop_count > 0 else 0

    return {
        "hop_recall": round(hop_recall, 3),
        "premature_collapse": premature,
        "over_extension": over_ext,
        "step_alignment": round(alignment, 3),
        "hop_count": hop_count,
        "total_tool_calls": total_tool_calls,
        "iteration_count": iteration_count,
        "evidence_count": evidence_count,
    }


def aggregate_diagnostics(diagnostics: list[dict]) -> dict:
    """聚合多条诊断结果"""
    if not diagnostics:
        return {}

    n = len(diagnostics)
    avg_hop_recall = sum(d["hop_recall"] for d in diagnostics) / n
    premature_rate = sum(1 for d in diagnostics if d["premature_collapse"]) / n
    over_ext_rate = sum(1 for d in diagnostics if d["over_extension"]) / n
    avg_alignment = sum(d["step_alignment"] for d in diagnostics) / n

    # 按 hop_count 分组
    by_hop = {}
    for d in diagnostics:
        h = d["hop_count"]
        by_hop.setdefault(h, []).append(d)

    hop_breakdown = {}
    for h, items in sorted(by_hop.items()):
        hop_breakdown[f"{h}hop"] = {
            "count": len(items),
            "avg_hop_recall": round(sum(d["hop_recall"] for d in items) / len(items), 3),
            "premature_rate": round(sum(1 for d in items if d["premature_collapse"]) / len(items), 3),
            "over_ext_rate": round(sum(1 for d in items if d["over_extension"]) / len(items), 3),
        }

    return {
        "overall": {
            "avg_hop_recall": round(avg_hop_recall, 3),
            "premature_collapse_rate": round(premature_rate, 3),
            "over_extension_rate": round(over_ext_rate, 3),
            "avg_step_alignment": round(avg_alignment, 3),
            "total_samples": n,
        },
        "by_hop": hop_breakdown,
    }
