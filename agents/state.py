"""LangGraph AgentState 定义"""
from typing import Literal, TypedDict, Annotated
import operator


class AgentState(TypedDict):
    query: str                                                      # 原始查询
    query_type: Literal["simple", "multi_hop"]                      # 路由结果
    plan: list[dict]                                                # 子任务 [{"id", "sub_query", "depends_on", "status"}]
    current_step: int                                               # 当前子任务索引
    evidence: Annotated[list[dict], operator.add]                   # 累积证据
    tool_calls: Annotated[list[dict], operator.add]                 # 工具调用日志
    verification_result: str                                        # sufficient | insufficient | contradiction
    verification_feedback: str                                      # 重规划指导
    final_answer: str                                               # 最终答案
    iteration_count: int                                            # PEV 循环次数
    total_tool_calls: int                                           # 总工具调用次数
    trace: Annotated[list[dict], operator.add]                      # 执行轨迹
