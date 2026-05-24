"""LangGraph StateGraph 组装：PEV 多智能体编排 ★★★

流程图：
[router]
   ├── simple ──→ [simple_rag] ──→ END
   └── multi_hop ──→ [planner] ──→ [executor] ──→ [verifier]
                        ↑                           /        \\
                     (replan)                 sufficient   budget_exhausted
                        │                        │              │
                        └── insufficient ────    ↓              ↓
                                            [synthesizer] ──→ END
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.router import route_query, route_decision
from agents.planner import plan
from agents.executor import execute_step, should_continue_executing
from agents.verifier import verify, after_verification
from agents.synthesizer import synthesize, simple_rag


def build_graph(enable_verifier: bool = True, enabled_tools: list[str] | None = None):
    """构建 AgenticRAG LangGraph 图

    Args:
        enable_verifier: 是否启用验证器（消融实验用）
        enabled_tools: 允许使用的工具列表（消融实验用）
    """
    # 配置 executor 工具集（每次先恢复完整工具再过滤）
    from agents.executor import TOOL_REGISTRY, _ensure_tools, _ALL_TOOLS
    _ensure_tools()
    TOOL_REGISTRY.clear()
    if enabled_tools is not None:
        for name in enabled_tools:
            if name in _ALL_TOOLS:
                TOOL_REGISTRY[name] = _ALL_TOOLS[name]
    else:
        TOOL_REGISTRY.update(_ALL_TOOLS)

    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("router", route_query)
    graph.add_node("simple_rag", simple_rag)
    graph.add_node("planner", plan)
    graph.add_node("executor", execute_step)
    graph.add_node("synthesizer", synthesize)

    # 入口 → 路由
    graph.set_entry_point("router")

    # 路由条件边
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "simple": "simple_rag",
            "multi_hop": "planner",
        },
    )

    # simple_rag → END
    graph.add_edge("simple_rag", END)

    # planner → executor
    graph.add_edge("planner", "executor")

    if enable_verifier:
        graph.add_node("verifier", verify)

        # executor → verifier（当所有步骤完成）
        graph.add_conditional_edges(
            "executor",
            should_continue_executing,
            {
                "execute": "executor",
                "verify": "verifier",
            },
        )

        # verifier 条件边
        graph.add_conditional_edges(
            "verifier",
            after_verification,
            {
                "synthesize": "synthesizer",
                "replan": "planner",
            },
        )
    else:
        # 无验证器：executor 完成后直接合成
        graph.add_conditional_edges(
            "executor",
            should_continue_executing,
            {
                "execute": "executor",
                "verify": "synthesizer",  # 直接跳到合成
            },
        )

    # synthesizer → END
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_query(query: str, **kwargs) -> dict:
    """运行单个查询，返回完整 state"""
    app = build_graph(**kwargs)
    initial_state = {
        "query": query,
        "query_type": "multi_hop",
        "plan": [],
        "current_step": 0,
        "evidence": [],
        "tool_calls": [],
        "verification_result": "",
        "verification_feedback": "",
        "final_answer": "",
        "iteration_count": 0,
        "total_tool_calls": 0,
        "trace": [],
    }
    result = app.invoke(initial_state)
    return result


# 预编译默认图（单例）
_default_app = None


def get_default_app():
    global _default_app
    if _default_app is None:
        _default_app = build_graph()
    return _default_app
