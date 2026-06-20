"""查询复杂度分类器：simple / multi_hop"""
# 负责调用 llm 判断一个非常关键的问题：
# 当前用户问题是简单问题，还是需要多跳规划的问题？实际上项目只用到了multi_hop。
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.client import agent_chat_json
from agents.state import AgentState
from agents.prompts import get_profile


def route_query(state: AgentState) -> AgentState:
    """LangGraph node: 路由查询复杂度"""
    query = state["query"]
    profile = get_profile()
    result = agent_chat_json(profile["router"].format(query=query))

    query_type = "multi_hop"
    if result and result.get("query_type") == "simple":
        query_type = "simple"

    return {
        "query_type": query_type,
        "trace": [{"node": "router", "query_type": query_type, "detail": result}],
    }


def route_decision(state: AgentState) -> str:
    """条件边：根据路由结果决定走 simple 还是 multi_hop"""
    return state.get("query_type", "multi_hop")
