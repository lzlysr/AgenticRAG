"""验证器：校验证据充分性，触发重规划"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.client import agent_chat_json
from agents.state import AgentState
from agents.prompts import get_profile

# 新旧 evidence chunk_id 重叠度超过此阈值时，强制 sufficient
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
    """LangGraph node: 验证证据充分性"""
    query = state["query"]
    evidence = state.get("evidence", [])
    iteration = state.get("iteration_count", 0)

    evidence_text = ""
    for e in evidence:
        evidence_text += f"\n--- Step {e['step_id']}: {e['sub_query']} (tool: {e['tool']}) ---\n"
        for r in e.get("results", [])[:3]:
            evidence_text += f"[{r.get('chunk_id', '?')}] {r.get('text', '')[:500]}\n"

    profile = get_profile()
    prompt = profile["verifier"].format(query=query, evidence_text=evidence_text or "No evidence collected.")
    result = agent_chat_json(prompt)

    verdict = "sufficient"
    feedback = ""
    if result:
        verdict = result.get("verdict", "sufficient")
        feedback = result.get("feedback", "")

    # Evidence 去重检测：如果是 replan 后的第 2+ 轮，检查新 evidence 是否和旧 evidence 高度重复
    if verdict == "insufficient" and iteration >= 1:
        plan = state.get("plan", [])
        # 当前轮的 plan step_id 集合
        current_step_ids = {step["id"] for step in plan}
        prev_evidence = [e for e in evidence if e["step_id"] not in current_step_ids]
        curr_evidence = [e for e in evidence if e["step_id"] in current_step_ids]

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
        "trace": [{"node": "verifier", "verdict": verdict, "feedback": feedback}],
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
    return "replan"
