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
from concurrent.futures import ThreadPoolExecutor

from evaluation.metrics import _normalize, exact_match, f1_score

# ── LLM Judge API 配置（环境变量覆盖）──────────────────────────
_JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8086/v1")
_JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "EMPTY")
_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-oss-120b")
_JUDGE_CLIENT = None
_MAX_EVIDENCE_CHARS = 3000


def _get_judge_client():
    '''
    llm.client 默认使用 JUDGE_LLM_MODEL，奖励脚本使用 JUDGE_MODEL，这是两个环境变量。
    不能复用？
    '''
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        from openai import OpenAI
        _JUDGE_CLIENT = OpenAI(api_key=_JUDGE_API_KEY, base_url=_JUDGE_BASE_URL)
    return _JUDGE_CLIENT


def _call_judge(prompt: str, retries: int = 1) -> dict:
    '''
    当前奖励固定 temperature=0、max_tokens=512。
    而 llm.client 的 judge_chat_json() 使用模型注册表中的默认温度、max_len、top_p、thinking 配置。
    不能复用。
    '''
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


_CORRECTNESS_PROMPT = """你是一个严格的评测法官。评估预测答案与标准答案相比的正确性。

## 输入
**问题：** {question}
**预测答案：** {prediction}
**标准答案：** {gold}

## 评分标准
- 1.0：预测答案与标准答案语义等价（措辞或格式可以不同）。
- 0.7：预测答案抓住了要点，但有轻微不准确或多余细节。
- 0.5：预测答案部分正确，包含一些正确信息，但也有重大错误或遗漏。
- 0.3：预测答案与标准答案有少量重叠，但大部分是错误的。
- 0.0：预测答案完全错误或与问题无关。

只返回 JSON，不要输出解释：{{"score": <float>}}"""

_FAITHFULNESS_PROMPT = """你是一个严格的评测法官。判断答案是否被证据支持。

## 任务
1. 识别答案中的关键论断。
2. 逐项检查证据是否直接支持这些论断。
3. 根据被支持的论断比例给出分数。

## 输入
**问题：** {question}
**答案：** {answer}
**证据（检索到的文档）：**
{evidence}

## 评分标准
- 1.0：答案中的所有论断都被证据直接支持。
- 0.7：大部分论断被支持，少量细节缺乏证据。
- 0.5：约一半的论断被证据支持。
- 0.3：仅少部分答案被证据支持。
- 0.0：答案没有任何证据支持，或与证据直接矛盾。

重要：只判断证据是否支持答案，不要判断答案本身是否正确。错误答案也可能忠实于证据。

只返回 JSON，不要输出解释：{{"score": <float>}}"""


def _judge_correctness(question: str, pred: str, gold: str) -> float:
    '''与 llm_judge.py 的 judge_answer_correctness() 类似'''
    prompt = _CORRECTNESS_PROMPT.format(question=question, prediction=pred, gold=gold)
    result = _call_judge(prompt)
    return min(1.0, max(0.0, float(result.get("score", 0.0))))


def _judge_faithfulness(question: str, pred: str, evidence: str) -> float:
    '''与 llm_judge.py 的 judge_faithfulness() 类似'''
    evidence = evidence[:_MAX_EVIDENCE_CHARS]
    prompt = _FAITHFULNESS_PROMPT.format(question=question, answer=pred, evidence=evidence)
    result = _call_judge(prompt)
    return min(1.0, max(0.0, float(result.get("score", 0.0))))


# ── 工具函数 ───────────────────────────────────────────────────


def _extract_answer(text: str) -> str:
    '''提取 <answer>，即最终答案'''
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _extract_evidence(text: str) -> str:
    '''提取 <tool_response>，即检索到的工具返回的内容。'''
    matches = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", text, re.DOTALL)
    return "\n".join(m.strip() for m in matches if m.strip())


def _extract_retrieved_chunks(text: str) -> set:
    '''提取金融数据的 chunk_id'''
    return set(re.findall(r'\[([a-z]+_\d+)\]', text))


def _check_tool_call_format(text: str) -> bool:
    '''检查是否存在 <tool_call> 格式的工具调用。'''
    pattern = r'<tool_call>\s*\{[^}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:'
    return bool(re.search(pattern, text))


def _count_tool_calls(text: str) -> int:
    '''根据 <tool_call> 的次数统计工具调用次数'''
    return len(re.findall(r'<tool_call>', text))


def _grounded_answer_score(pred: str, evidence: str) -> float:
    '''衡量答案是否被证据覆盖（关键词级别）。'''
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
    """
    Precision-aware hop matching (F1)。比较检索到的 chunk ID 与标准答案中的 chunk ID。
    使用 _hop_precision_recall 可作为 llm_judge.py 中的 Context Precision 的确定性替代指标。
    """
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

    这是 verl 调用的统一入口。

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

    # 数据从 Parquet/Pandas 进入 verl 后，可能不是普通列表。因此需要转换为 list。
    if hasattr(gold_chunks, 'tolist'):
        gold_chunks = gold_chunks.tolist()
    if hasattr(aliases, 'tolist'):
        aliases = aliases.tolist()

    pred = _extract_answer(solution_str)

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

    # ── 搜索不足惩罚 ── (没有过度搜索的惩罚？）
    insufficient_penalty = 0.0
    if hop_count > 0 and num_tool_calls < hop_count:
        insufficient_penalty = 0.05

    # ── Grounded answer（0.10）──
    evidence = _extract_evidence(solution_str)
    grounded = _grounded_answer_score(pred, evidence) if pred else 0.0
    grounded_score = grounded * 0.10

    # ── 无答案 → 只给格式分 + hop ──
    # 属于过程奖励，能鼓励模型先学会检索。
    if not pred:
        score = format_bonus + hop_score - insufficient_penalty
        # 随机打印无答案日志。既能观察训练情况，又不会刷屏。
        if random.randint(1, 16) == 1:
            print(f"[reward_v9a] NO_PRED hop_pr={hop_score:.2f} fmt={format_bonus:.2f}")
        return max(0.0, score)

    # ── Correctness（0.25）+ Faithfulness（0.25）──
    aliases = [alias for alias in aliases if alias]
    f1 = f1_score(pred, gold, aliases=aliases)
    em = exact_match(pred, gold, aliases=aliases)

    # 如果 em 和 f1 较高，直接认为 correctness 满分，不再调用 Correctness Judge。
    if em == 1.0 or f1 >= 0.8:
        corr_score = 1.0
        # 即使答案规则匹配正确，仍要检查是否有 evidence 支持。
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
