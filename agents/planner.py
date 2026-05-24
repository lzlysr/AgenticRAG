"""规划器：将复杂查询分解为子任务 DAG"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.client import agent_chat_json
from agents.state import AgentState
from agents.prompts import get_profile

TOOL_DESCRIPTIONS = {
    "keyword_search": "BM25 keyword search, good for exact names/entities",
    "semantic_search": "Dense retrieval + reranking, good for semantic similarity",
    "read_chunk": "Read a specific document by chunk_id (use when you have a specific chunk_id from previous steps)",
    "graph_search": "Knowledge graph search: finds related entities and documents through entity relationship traversal. Best for multi-hop questions involving entity connections (e.g., 'Who directed the film starring X?')",
}


def plan(state: AgentState) -> AgentState:
    """LangGraph node: 生成或重新生成检索计划"""
    query = state["query"]
    iteration = state.get("iteration_count", 0)

    profile = get_profile()

    feedback_section = ""
    if iteration > 0 and state.get("verification_feedback"):
        evidence_summary = "\n".join(
            f"- Step {e['step_id']} [{e.get('tool', '?')}]: \"{e['sub_query']}\" -> {len(e.get('results', []))} results"
            for e in state.get("evidence", [])
        )
        feedback_section = profile["replan_feedback"].format(
            feedback=state["verification_feedback"],
            evidence_summary=evidence_summary or "No evidence yet",
        )

    # 动态生成可用工具列表（消融实验时 TOOL_REGISTRY 可能被过滤）
    from agents.executor import TOOL_REGISTRY, _ensure_tools
    _ensure_tools()
    tools_section = "\n".join(
        f"- {name}: {TOOL_DESCRIPTIONS.get(name, 'search tool')}"
        for name in TOOL_REGISTRY
    )
    if not tools_section:
        tools_section = "- semantic_search: Dense retrieval + reranking"

    prompt = profile["planner"].format(query=query, feedback_section=feedback_section, tools_section=tools_section)
    result = agent_chat_json(prompt)

    if not result or not isinstance(result, list):
        result = [{"id": 1, "sub_query": query, "tool": "semantic_search", "depends_on": []}]

    # 标记所有步骤为 pending
    for step in result:
        step["status"] = "pending"

    return {
        "plan": result,
        "current_step": 0,
        "iteration_count": iteration + 1,
        "trace": [{"node": "planner", "iteration": iteration + 1, "plan": result}],
    }
