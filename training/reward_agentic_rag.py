"""GRPO 奖励函数 v9a：检索质量优先（Stage 3, 基于 v6a 改进）

Stage 3 目标：在 v14e（Judge_C=0.334, Faith=0.199）基础上提升 CtxP。
核心改动：hop_precision_recall 权重从 0.20→0.30，直接强化检索精度。

评分策略：
- hop_precision_recall × 0.30（检索质量：命中 gold chunks 且不引入过多噪声）
- Judge Faithfulness × 0.25（答案是否基于 evidence）
- Judge Correctness × 0.25（答案是否正确）
- grounded_answer × 0.10（答案关键词是否出现在 evidence 中）
- format × 0.10（<answer> + <tool_call>）
- 搜索不足惩罚：tool_calls < hop_count 时扣 0.05

verl 接口：compute_score(solution_str, ground_truth, **kwargs) -> float
"""
import json
import os
import random
import re
import string
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# ── LLM Judge API 配置（环境变量覆盖）──────────────────────────
_JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8086/v1")
_JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "EMPTY")
_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-oss-120b")
_JUDGE_CLIENT = None
_MAX_EVIDENCE_CHARS = 3000


def _get_judge_client():
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        from openai import OpenAI
        _JUDGE_CLIENT = OpenAI(api_key=_JUDGE_API_KEY, base_url=_JUDGE_BASE_URL)
    return _JUDGE_CLIENT


def _call_judge(prompt: str, retries: int = 1) -> dict:
    client = _get_judge_client()
    for i in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
            if m:
                text = m.group(1).strip()
            start = text.find('{')
            if start >= 0:
                end = text.rfind('}')
                if end > start:
                    return json.loads(text[start:end+1])
        except Exception:
            pass
    return {}


_CORRECTNESS_PROMPT = """Judge answer correctness.
Question: {question}
Predicted: {prediction}
Gold: {gold}
Score: 1.0=semantically equivalent, 0.7=mostly correct, 0.5=partially correct, 0.3=mostly wrong, 0.0=completely wrong.
Reply JSON: {{"score": <float>}}"""

_FAITHFULNESS_PROMPT = """Judge if the answer is supported by evidence.
Question: {question}
Answer: {answer}
Evidence: {evidence}
Score: 1.0=fully supported, 0.7=mostly supported, 0.5=half supported, 0.3=barely supported, 0.0=not supported.
Reply JSON: {{"score": <float>}}"""


def _judge_correctness(question: str, pred: str, gold: str) -> float:
    prompt = _CORRECTNESS_PROMPT.format(question=question, prediction=pred, gold=gold)
    result = _call_judge(prompt)
    return min(1.0, max(0.0, float(result.get("score", 0.0))))


def _judge_faithfulness(question: str, pred: str, evidence: str) -> float:
    evidence = evidence[:_MAX_EVIDENCE_CHARS]
    prompt = _FAITHFULNESS_PROMPT.format(question=question, answer=pred, evidence=evidence)
    result = _call_judge(prompt)
    return min(1.0, max(0.0, float(result.get("score", 0.0))))


# ── 中文 normalize + 工具函数 ──────────────────────────────────

_CN_PUNCTUATION = '。，、；：？！""''【】《》（）｛｝〔〕·…—～'
_ALL_PUNCTUATION = set(string.punctuation) | set(_CN_PUNCTUATION)


def _normalize(text: str) -> str:
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif ch == '\u3000':
            result.append(' ')
        elif unicodedata.category(ch).startswith('Zs'):
            result.append(' ')
        else:
            result.append(ch)
    text = ''.join(result).lower()
    text = ''.join(ch for ch in text if ch not in _ALL_PUNCTUATION)
    text = re.sub(r'(\d)([\u4e00-\u9fff])', r'\1 \2', text)
    text = re.sub(r'([\u4e00-\u9fff])(\d)', r'\1 \2', text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_f1(pred: str, gold: str, aliases: list[str]) -> float:
    candidates = [gold] + aliases
    return max(_token_f1(pred, c) for c in candidates if c)


def _extract_answer(text: str) -> str:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _extract_evidence(text: str) -> str:
    matches = re.findall(r'เพิ่มเติม(.*?)', text, re.DOTALL)
    return "\n".join(m.strip() for m in matches if m.strip())


def _extract_retrieved_chunks(text: str) -> set:
    return set(re.findall(r'\[([a-z]+_\d+)\]', text))


def _check_tool_call_format(text: str) -> bool:
    pattern = r'<tool_call>\s*\{[^}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:'
    return bool(re.search(pattern, text))


def _count_tool_calls(text: str) -> int:
    return len(re.findall(r'<tool_call>', text))


def _grounded_answer_score(pred: str, evidence: str) -> float:
    if not pred or not evidence:
        return 0.0
    pred_tokens = set(_normalize(pred).split())
    evidence_tokens = set(_normalize(evidence).split())
    pred_tokens = {t for t in pred_tokens if len(t) > 1}
    if not pred_tokens:
        return 0.0
    overlap = pred_tokens & evidence_tokens
    return len(overlap) / len(pred_tokens)


def _hop_precision_recall(retrieved_chunks: set, gold_chunks: list) -> float:
    """Precision-aware hop matching (F1)"""
    if not gold_chunks:
        return 0.0
    gold_set = set(str(c) for c in gold_chunks)
    if not retrieved_chunks:
        return 0.0
    hit = len(gold_set & retrieved_chunks)
    recall = hit / len(gold_set)
    precision = hit / len(retrieved_chunks) if retrieved_chunks else 0.0
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


# ── 主函数 ──────────────────────────────────────────────────────

_thread_pool = ThreadPoolExecutor(max_workers=2)


def compute_score(solution_str, ground_truth, **kwargs):
    """verl 奖励函数 v9a（Stage 3: 检索质量优先）

    score = hop_pr*0.30 + faith*0.25 + corr*0.25 + grounded*0.10 + format*0.10 - insufficient_search
    """
    if isinstance(ground_truth, dict):
        gold = ground_truth.get("target", ground_truth.get("answer", ""))
        question = ground_truth.get("question", "")
        aliases = ground_truth.get("answer_aliases", [])
        gold_chunks = ground_truth.get("gold_chunks", [])
        hop_count = int(ground_truth.get("hop_count", 2))
    else:
        gold = str(ground_truth)
        question = ""
        aliases = []
        gold_chunks = []
        hop_count = 2

    if hasattr(gold_chunks, 'tolist'):
        gold_chunks = gold_chunks.tolist()
    if hasattr(aliases, 'tolist'):
        aliases = aliases.tolist()

    pred = _extract_answer(solution_str)

    # </think> 重复惩罚
    if solution_str.count("</think>") > 5:
        return 0.0

    # ── 格式奖励（0.10）──
    has_answer_tag = "<answer>" in solution_str and "</answer>" in solution_str
    has_tool_call = _check_tool_call_format(solution_str)
    num_tool_calls = _count_tool_calls(solution_str)
    format_bonus = 0.0
    if has_answer_tag:
        format_bonus += 0.06
    if has_tool_call:
        format_bonus += 0.04

    # ── Hop precision-recall（0.30）──
    retrieved_chunks = _extract_retrieved_chunks(solution_str)
    hop_pr = _hop_precision_recall(retrieved_chunks, gold_chunks)
    # hop_pr 为 0 时给硬惩罚（而非 0.30×0=0 无梯度信号）
    if hop_pr > 0:
        hop_score = hop_pr * 0.30
    else:
        hop_score = -0.05

    # ── 搜索不足惩罚 ──
    insufficient_penalty = 0.0
    if hop_count > 0 and num_tool_calls < hop_count:
        insufficient_penalty = 0.05

    # ── Grounded answer（0.10）──
    evidence = _extract_evidence(solution_str)
    grounded = _grounded_answer_score(pred, evidence) if pred else 0.0
    grounded_score = grounded * 0.10

    # ── 无答案 → 只给格式分 + hop ──
    if not pred:
        score = format_bonus + hop_score - insufficient_penalty
        if random.randint(1, 16) == 1:
            print(f"[reward_v9a] NO_PRED hop_pr={hop_pr:.2f} fmt={format_bonus:.2f}")
        return max(0.0, score)

    # ── Correctness（0.25）+ Faithfulness（0.25）──
    f1 = _best_f1(pred, gold, aliases)
    em = 1.0 if _normalize(pred) == _normalize(gold) else 0.0

    if em == 1.0 or f1 >= 0.8:
        corr_score = 1.0
        if evidence:
            faith_score = _judge_faithfulness(question, pred, evidence)
        else:
            faith_score = 0.0
    else:
        corr_future = _thread_pool.submit(_judge_correctness, question, pred, gold)
        if evidence:
            faith_future = _thread_pool.submit(_judge_faithfulness, question, pred, evidence)
            corr_score = corr_future.result()
            faith_score = faith_future.result()
        else:
            corr_score = corr_future.result()
            faith_score = 0.0

    score = (hop_score
             + faith_score * 0.25
             + corr_score * 0.25
             + grounded_score
             + format_bonus
             - insufficient_penalty)
    score = min(1.0, max(0.0, score))

    if random.randint(1, 16) == 1:
        print(f"[reward_v9a] gold={gold[:30]} pred={pred[:30]} "
              f"hop_pr={hop_pr:.2f} faith={faith_score:.1f} corr={corr_score:.1f} "
              f"grnd={grounded:.2f} fmt={format_bonus:.2f} pen={insufficient_penalty:.2f} "
              f"score={score:.2f} tools={num_tool_calls}")

    return score
