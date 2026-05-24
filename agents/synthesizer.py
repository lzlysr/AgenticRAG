"""合成器：基于证据生成最终答案"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.client import agent_chat
from agents.state import AgentState
from agents.prompts import get_profile


def _extract_short_answer(text: str) -> str:
    """从 LLM 输出中提取简短答案（去掉解释性内容）"""
    import re
    text = text.strip()
    if not text:
        return text
    # 提取 <answer>...</answer> 标签内容
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # 去掉常见前缀
    for prefix in ["Answer:", "answer:", "The answer is", "the answer is",
                    "Based on the evidence,", "Based on the context,"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # 取第一行（如果有换行）
    first_line = text.split("\n")[0].strip()
    # 取第一句（如果答案过长）— 但保留合理长度的完整句
    if len(first_line) > 150:
        # 尝试取第一句
        for sep in [". ", "。", "; "]:
            if sep in first_line:
                first_line = first_line[:first_line.index(sep)].strip()
                break
    # 去掉尾部句号
    first_line = first_line.rstrip(".")
    return first_line


def synthesize(state: AgentState) -> AgentState:
    """LangGraph node: 合成最终答案"""
    query = state["query"]
    evidence = state.get("evidence", [])

    evidence_text = ""
    for e in evidence:
        for r in e.get("results", [])[:3]:
            evidence_text += f"[{r.get('chunk_id', '?')}] {r.get('text', '')[:500]}\n\n"

    profile = get_profile()
    prompt = profile["synthesizer"].format(query=query, evidence_text=evidence_text or "No evidence available.")
    answer = agent_chat(prompt)
    answer = _extract_short_answer(answer)

    return {
        "final_answer": answer,
        "trace": [{"node": "synthesizer", "answer_length": len(answer)}],
    }


def simple_rag(state: AgentState) -> AgentState:
    """LangGraph node: 简单 RAG 直通（单次检索 + 生成）"""
    from retrieval.semantic_search import semantic_search

    query = state["query"]
    results = semantic_search(query)

    profile = get_profile()
    context = "\n\n".join(r["text"][:500] for r in results[:3])
    prompt = profile["simple_rag"].format(context=context, query=query)
    answer = agent_chat(prompt)
    answer = _extract_short_answer(answer)

    evidence = [{
        "step_id": 1,
        "sub_query": query,
        "tool": "semantic_search",
        "results": results[:5],
    }]

    return {
        "final_answer": answer.strip(),
        "evidence": evidence,
        "total_tool_calls": 1,
        "trace": [{"node": "simple_rag", "num_results": len(results)}],
    }
