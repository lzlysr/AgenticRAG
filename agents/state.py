"""LangGraph AgentState 定义"""
from typing import Literal, TypedDict, Annotated
import operator

# Annotated: 这个字段的基础类型是 list[dict]，多个节点同时或连续返回该字段时，使用 operator.add 合并。
# 可以实现 LangGraph 的 reducer功能：累加功能。

class AgentState(TypedDict):
    query: str                                                      # 原始查询
    query_type: Literal["simple", "multi_hop"]                      # 路由结果
    plan: list[dict]                              # 子任务 [{"id", "sub_query","depends_on", "status"}]
    current_step: int                                               # 当前子任务索引
    evidence: Annotated[list[dict], operator.add]                   # 累积证据
    tool_calls: Annotated[list[dict], operator.add]                 # 工具调用日志
    verification_result: str                                # sufficient | insufficient |contradiction
    verification_feedback: str                                      # 重规划指导
    final_answer: str                                               # 最终答案
    iteration_count: int                                            # PEV 循环次数
    total_tool_calls: int                                           # 总工具调用次数
    trace: Annotated[list[dict], operator.add]                      # 执行轨迹


# 各节点的读写关系：
#
# | 字段                  | 写入节点        | 读取节点                    |
# |-----------------------|----------------|-----------------------------|
# | query                 | 初始化          | Router, Planner, Synthesizer |
# | query_type            | Router         | 条件边(route_decision)       |
# | plan                  | Planner        | Executor                    |
# | current_step          | Executor       | Executor(循环)              |
# | evidence              | Executor(追加) | Verifier, Synthesizer        |
# | verification_result   | Verifier       | 条件边(after_verification)   |
# | verification_feedback | Verifier       | Planner(replan 时)           |
# | iteration_count       | Planner(+1)    | Verifier(预算检查)           |
# | total_tool_calls      | Executor(+N)   | Executor, Verifier(预算检查) |
# | final_answer          | Synthesizer    | 输出                        |
