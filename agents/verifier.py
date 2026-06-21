"""验证器：校验证据充分性，触发重规划"""

# 它主要负责两件事：

# 1. 调用 LLM 判断当前证据能否回答原始问题；
# 2. 根据验证结果和预算，决定进入答案合成还是重新规划。

# 另外还加了一段“证据重复检测”，用于避免重规划不断召回相同文档而陷入循环。

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.client import agent_chat_json
from agents.state import AgentState
from agents.prompts import get_profile

# 新旧 evidence chunk_id 重叠度超过此阈值时，强制 sufficient
# 含义：如果重规划后召回的文档几乎与之前一样，继续重规划大概率也不会获得新信息，因此强制结束循环。
# 它本质上是一种防停滞机制，不是严格的证据充分性判断。
EVIDENCE_OVERLAP_THRESHOLD = 0.9


def _get_chunk_ids(evidence_list: list[dict]) -> set[str]:
    """从 evidence 列表提取所有 chunk_id"""
    ids = set()
    for e in evidence_list:
        for r in e.get("results", []):
            cid = r.get("chunk_id", "")
            if cid:
                ids.add(cid)
    return ids


def verify(state: AgentState) -> AgentState:
    """LangGraph node: 验证证据充分性（当前全部 evidence 能否回答原始问题。）"""
    query = state["query"]
    evidence = state.get("evidence", [])
    iteration = state.get("iteration_count", 0)

    # 构造证据文本
    evidence_text = ""
    for e in evidence:
        evidence_text += (
            f"\n--- Iteration {e.get('iteration', '?')} / "
            f"Step {e['step_id']}: {e['sub_query']} (tool: {e['tool']}) ---\n"
        )
        for r in e.get("results", [])[:3]:
            evidence_text += f"[{r.get('chunk_id', '?')}] {r.get('text', '')[:500]}\n"

    # 构造 Verifier Prompt
    profile = get_profile()
    prompt = profile["verifier"].format(query=query, evidence_text=evidence_text or "No evidence collected.")
    result = agent_chat_json(prompt)

    verdict = "sufficient"
    feedback = ""
    if result:
        verdict = result.get("verdict", "sufficient")
        feedback = result.get("feedback", "")

    # Evidence 去重检测：如果是 replan 后的第 2+ 轮，检查新 evidence 是否和旧 evidence 高度重复
    # 但是根据 Planner 的计数逻辑，这个条件实际上第一轮验证就满足。不过第一轮通常没有“旧 evidence”，所以后续：if curr_chunks and prev_chunks:不会触发强制 sufficient。
    if verdict == "insufficient" and iteration >= 1:
        prev_evidence = [e for e in evidence if e.get("iteration") != iteration]
        curr_evidence = [e for e in evidence if e.get("iteration") == iteration]

        prev_chunks = _get_chunk_ids(prev_evidence)
        curr_chunks = _get_chunk_ids(curr_evidence)

        if curr_chunks and prev_chunks:
            overlap = len(curr_chunks & prev_chunks) / len(curr_chunks)
            if overlap >= EVIDENCE_OVERLAP_THRESHOLD:
                verdict = "sufficient"
                feedback = f"evidence_dedup: {overlap:.0%} overlap, forcing sufficient"

    return {
        "verification_result": verdict,
        "verification_feedback": feedback,
        "trace": [{"node": "verifier", "iteration": iteration, "verdict": verdict, "feedback": feedback}],
    }


def after_verification(state: AgentState) -> str:
    """条件边：sufficient/budget exhausted → synthesizer，insufficient → planner"""
    profile = get_profile()
    verdict = state.get("verification_result", "sufficient")
    iteration = state.get("iteration_count", 0)
    total_calls = state.get("total_tool_calls", 0)

    budget_exhausted = (iteration >= profile["max_iterations"]
                        or total_calls >= profile["max_retrieval_calls"])

    if verdict == "sufficient" or budget_exhausted:
        return "synthesize"
    return "replan" # 重规划后的新 evidence 会继续追加，旧 evidence 不会清除？
