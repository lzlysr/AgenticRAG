"""
规划器：将复杂查询分解为子任务 DAG。DAG 是 Directed Acyclic Graph，中文叫 有向无环图。

例如：
[
    {
        "id": 1,
        "sub_query": "Who founded Company A?",
        "tool": "semantic_search",
        "depends_on": [],
        "status": "pending",
    },
    {
        "id": 2,
        "sub_query": "Where was the founder born?",
        "tool": ["keyword_search", "semantic_search"],
        "depends_on": [1],
        "status": "pending",
    },
]
"""
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
    # 这里始终读取用户最初的问题，而不是当前子问题。因为规划器的职责是根据用户的原始查询和当前的验证反馈来生成整个检索计划，而不是针对某个子问题进行规划。
    query = state["query"]
    iteration = state.get("iteration_count", 0) # 当前迭代次数，第一次进入规划器时 iteration_count 不存在，默认为0。每次生成新计划时，iteration_count +1。

    profile = get_profile()

    feedback_section = ""
    # 已经至少执行过一次 Planner 且 Verifier 提供了非空反馈。才进入重规划。
    if iteration > 0 and state.get("verification_feedback"):
        # evidence_summary 中的results只有数量，没有具体内容？
        # 重规划后 Step ID 与旧 evidence 冲突：
        # 旧 evidence 不清空，新计划又从 1 开始，依赖可能被旧轮次错误满足？
        evidence_summary = "\n".join(
            f"- Iteration {e.get('iteration', '?')} Step {e['step_id']} "
            f"[{e.get('tool', '?')}]: \"{e['sub_query']}\" -> {len(e.get('results', []))} results"
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

    # result 的每一个子任务里的 depends_on 是 step_id 列表，而不是 sub_query 列表。因为依赖关系是基于步骤编号，而不是具体的查询内容。好像无法保证子问题真正的依赖关系？依靠模型本身的能力和 vertifier 兜底

    if not result or not isinstance(result, list):
        result = [{"id": 1, "sub_query": query, "tool": "semantic_search", "depends_on": []}]

    # 标记所有步骤为 pending （即将发生的），并记录本轮规划编号。
    for step in result:
        step["status"] = "pending"
        step["iteration"] = iteration + 1

    return {
        "plan": result,
        "current_step": 0, # 重置为 0，即从第一个子任务开始执行
        "iteration_count": iteration + 1,
        "trace": [{"node": "planner", "iteration": iteration + 1, "plan": result}],
    }
