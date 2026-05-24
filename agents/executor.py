"""执行器：按计划执行检索工具调用"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.state import AgentState
from agents.prompts import get_profile

# 工具注册表
TOOL_REGISTRY = {}
_ALL_TOOLS = {}  # 完整工具备份（消融实验用）


def _ensure_tools():
    if _ALL_TOOLS:
        return
    from retrieval.keyword_search import keyword_search
    from retrieval.semantic_search import semantic_search
    from retrieval.read_chunk import read_chunk
    from retrieval.graph_search import graph_search
    _ALL_TOOLS["keyword_search"] = keyword_search
    _ALL_TOOLS["semantic_search"] = semantic_search
    _ALL_TOOLS["read_chunk"] = read_chunk
    _ALL_TOOLS["graph_search"] = graph_search
    TOOL_REGISTRY.update(_ALL_TOOLS)


def _normalize_tool(tool_field) -> tuple[list[str], bool]:
    """将 tool 字段统一为工具名列表。

    Returns:
        (tool_names, is_hybrid): 工具名列表 + 是否为 hybrid 融合
    """
    if isinstance(tool_field, list):
        return [t for t in tool_field if isinstance(t, str)], len(tool_field) > 1
    if isinstance(tool_field, str):
        if tool_field == "hybrid_search":
            # SFT 模型生成的 hybrid_search 标记，需要从 step 参数中取 tools 列表
            return [], True
        return [tool_field], False
    return ["semantic_search"], False


def execute_step(state: AgentState) -> AgentState:
    """LangGraph node: 执行当前步骤的检索"""
    _ensure_tools()

    plan = state["plan"]
    current = state.get("current_step", 0)
    total_calls = state.get("total_tool_calls", 0)

    new_evidence = []
    new_tool_calls = []
    new_trace = []

    max_calls = get_profile()["max_retrieval_calls"]
    while current < len(plan) and total_calls < max_calls:
        step = plan[current]

        # 检查依赖是否满足
        deps = step.get("depends_on", [])
        completed_ids = {e["step_id"] for e in state.get("evidence", []) + new_evidence}
        if deps and not all(d in completed_ids for d in deps):
            break

        tool_names, is_hybrid = _normalize_tool(step.get("tool", "semantic_search"))
        sub_query = step.get("sub_query", state["query"])

        # hybrid_search: 从 step 参数中取 tools 列表
        if is_hybrid and not tool_names:
            hybrid_tools = step.get("tools", ["keyword_search", "semantic_search"])
            if isinstance(hybrid_tools, str):
                hybrid_tools = [hybrid_tools]
            tool_names = hybrid_tools

        # 根据已有证据上下文增强子查询
        if current > 0 and new_evidence:
            prev_results = new_evidence[-1].get("results", [])
            prev_answer = prev_results[0].get("text", "")[:200] if prev_results else ""
            if prev_answer:
                sub_query = f"{sub_query} (context: {prev_answer})"

        # 执行工具：多工具 → 并行 + RRF 融合，单工具 → 直接调用
        if len(tool_names) > 1 or is_hybrid:
            # 过滤掉不存在的工具，至少保留一个
            valid_tools = [t for t in tool_names if t in TOOL_REGISTRY]
            if not valid_tools:
                valid_tools = ["semantic_search"]

            if len(valid_tools) == 1:
                # 只有一个有效工具，直接调用
                results = TOOL_REGISTRY[valid_tools[0]](sub_query)
                tool_label = valid_tools[0]
            else:
                from retrieval.hybrid_search import multi_tool_search
                results = multi_tool_search(sub_query, valid_tools, TOOL_REGISTRY)
                tool_label = "+".join(valid_tools)
        else:
            tool_name = tool_names[0]
            tool_fn = TOOL_REGISTRY.get(tool_name, TOOL_REGISTRY.get("semantic_search"))
            results = tool_fn(sub_query)
            tool_label = tool_name

        total_calls += 1
        step["status"] = "done"

        evidence_entry = {
            "step_id": step["id"],
            "sub_query": step["sub_query"],
            "tool": tool_label,
            "results": results[:5],  # 保留 top 5
        }
        new_evidence.append(evidence_entry)
        new_tool_calls.append({
            "step_id": step["id"],
            "tool": tool_label,
            "query": sub_query,
            "num_results": len(results),
        })
        new_trace.append({
            "node": "executor",
            "step_id": step["id"],
            "tool": tool_label,
            "num_results": len(results),
        })

        current += 1

    return {
        "current_step": current,
        "evidence": new_evidence,
        "tool_calls": new_tool_calls,
        "total_tool_calls": total_calls,
        "trace": new_trace,
    }


def should_continue_executing(state: AgentState) -> str:
    """条件边：所有步骤执行完 → verifier，否则继续执行"""
    current = state.get("current_step", 0)
    plan = state.get("plan", [])
    total_calls = state.get("total_tool_calls", 0)

    max_calls = get_profile()["max_retrieval_calls"]
    if current >= len(plan) or total_calls >= max_calls:
        return "verify"
    return "execute"
