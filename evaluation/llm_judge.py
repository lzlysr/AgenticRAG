"""LLM-as-a-Judge：faithfulness, context precision, answer correctness"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.client import judge_chat_json

# ── 最大输入长度（字符），防止 evidence 过长导致 judge 不稳定 ──────────
MAX_EVIDENCE_CHARS = 6000
MAX_GOLD_DOCS_CHARS = 3000


def _get_lang(lang=None):
    if lang:
        return lang
    import config
    return getattr(config, "PROMPT_LANG", "en")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"


# ── Faithfulness ──────────────────────────────────────────────────────
FAITHFULNESS_PROMPT_EN = """You are a strict evaluation judge. Determine whether the **answer** is supported by the **evidence**.

## Task
1. First, identify the key claim(s) in the answer.
2. For each claim, check whether the evidence contains information that directly supports it.
3. Assign a score based on the proportion of claims supported.

## Inputs
**Question:** {question}
**Answer:** {answer}
**Evidence (retrieved documents):**
{evidence}

## Scoring rubric
- 1.0: ALL claims in the answer are directly supported by the evidence.
- 0.7: Most claims are supported; minor details lack evidence.
- 0.5: About half of the claims are supported by evidence.
- 0.3: Only a small part of the answer is supported.
- 0.0: The answer has NO support from the evidence, or directly contradicts it.

**Important:** Judge ONLY whether the evidence supports the answer's claims. Do NOT judge whether the answer is correct — a wrong answer can still be faithful to (supported by) the evidence.

Respond in JSON: {{"claims": ["<claim1>", ...], "supported": [true/false, ...], "score": <float>, "reasoning": "<brief explanation>"}}"""

FAITHFULNESS_PROMPT_ZH = """你是一个严格的评测法官。判断**答案**是否被**证据**支持。

## 任务
1. 首先，识别答案中的关键论断。
2. 对每个论断，检查证据中是否包含直接支持它的信息。
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
- 0.3：仅少部分答案被支持。
- 0.0：答案没有任何证据支持，或与证据直接矛盾。

**重要：** 仅判断证据是否支持答案的论断。不要判断答案是否正确——一个错误的答案仍然可以是忠实于证据的。

以 JSON 格式回复：{{"claims": ["<论断1>", ...], "supported": [true/false, ...], "score": <float>, "reasoning": "<简要说明>"}}"""


# ── Context Precision ─────────────────────────────────────────────────
CONTEXT_PRECISION_PROMPT_EN = """You are an evaluation judge. Assess whether the retrieved documents contain the information needed to answer the question, by comparing them against the gold reference documents.

## Task
1. Read the gold reference documents to understand what information is needed.
2. Check whether the retrieved documents contain equivalent or overlapping information for EACH gold document.
3. Score based on how many gold documents' information is covered by the retrieved documents.

## Inputs
**Question:** {question}

**Gold Reference Documents (what SHOULD be retrieved):**
{gold_docs}

**Retrieved Documents (what WAS actually retrieved):**
{evidence}

## Scoring rubric
- 1.0: Retrieved docs cover the key information from ALL gold reference documents.
- 0.7: Retrieved docs cover MOST gold reference documents (one minor gap).
- 0.5: Retrieved docs cover about HALF of the gold reference documents.
- 0.3: Retrieved docs cover only ONE of the gold reference documents.
- 0.0: Retrieved docs contain NONE of the information from gold reference documents.

Respond in JSON: {{"gold_doc_count": <int>, "covered_count": <int>, "score": <float>, "reasoning": "<brief explanation>"}}"""

CONTEXT_PRECISION_PROMPT_ZH = """你是一个评测法官。通过与标准参考文档对比，评估检索到的文档是否包含回答问题所需的信息。

## 任务
1. 阅读标准参考文档，了解需要哪些信息。
2. 检查检索到的文档是否包含与每个标准文档等价或重叠的信息。
3. 根据标准文档信息的覆盖程度给出分数。

## 输入
**问题：** {question}

**标准参考文档（应该被检索到的）：**
{gold_docs}

**检索到的文档（实际检索到的）：**
{evidence}

## 评分标准
- 1.0：检索文档覆盖了所有标准参考文档的关键信息。
- 0.7：检索文档覆盖了大部分标准参考文档（有一处小缺口）。
- 0.5：检索文档覆盖了约一半的标准参考文档。
- 0.3：检索文档仅覆盖了一个标准参考文档。
- 0.0：检索文档不包含任何标准参考文档的信息。

以 JSON 格式回复：{{"gold_doc_count": <int>, "covered_count": <int>, "score": <float>, "reasoning": "<简要说明>"}}"""


# ── Answer Correctness ────────────────────────────────────────────────
CORRECTNESS_PROMPT_EN = """You are an evaluation judge. Rate the correctness of the predicted answer compared to the gold answer.

**Question:** {question}
**Predicted Answer:** {prediction}
**Gold Answer:** {gold}

## Scoring rubric
- 1.0: Prediction is semantically equivalent to the gold answer (may differ in wording/format).
- 0.7: Prediction captures the main point but includes minor inaccuracies or extra details.
- 0.5: Prediction is partially correct — contains some correct information but also significant errors or omissions.
- 0.3: Prediction has a small overlap with the gold answer but is mostly wrong.
- 0.0: Prediction is completely wrong or irrelevant.

Respond in JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

CORRECTNESS_PROMPT_ZH = """你是一个评测法官。评估预测答案与标准答案相比的正确性。

**问题：** {question}
**预测答案：** {prediction}
**标准答案：** {gold}

## 评分标准
- 1.0：预测答案与标准答案语义等价（措辞/格式可以不同）。
- 0.7：预测答案抓住了要点，但有轻微不准确或多余细节。
- 0.5：预测答案部分正确——包含一些正确信息，但也有重大错误或遗漏。
- 0.3：预测答案与标准答案有少量重叠，但大部分是错的。
- 0.0：预测答案完全错误或无关。

以 JSON 格式回复：{{"score": <float>, "reasoning": "<简要说明>"}}"""


# ── Prompt 选择 ──────────────────────────────────────────────────────
_PROMPTS = {
    "faithfulness": {"en": FAITHFULNESS_PROMPT_EN, "zh": FAITHFULNESS_PROMPT_ZH},
    "context_precision": {"en": CONTEXT_PRECISION_PROMPT_EN, "zh": CONTEXT_PRECISION_PROMPT_ZH},
    "correctness": {"en": CORRECTNESS_PROMPT_EN, "zh": CORRECTNESS_PROMPT_ZH},
}


def get_judge_prompt(name: str, lang=None) -> str:
    lang = _get_lang(lang)
    return _PROMPTS[name][lang]


def judge_faithfulness(question: str, answer: str, evidence: str, lang=None) -> float:
    evidence = _truncate(evidence, MAX_EVIDENCE_CHARS)
    prompt = get_judge_prompt("faithfulness", lang).format(question=question, answer=answer, evidence=evidence)
    result = judge_chat_json(prompt)
    return result.get("score", 0.0) if result else 0.0


def judge_context_precision(question: str, evidence: str, gold_docs: str, lang=None) -> float:
    evidence = _truncate(evidence, MAX_EVIDENCE_CHARS)
    gold_docs = _truncate(gold_docs, MAX_GOLD_DOCS_CHARS)
    prompt = get_judge_prompt("context_precision", lang).format(question=question, evidence=evidence, gold_docs=gold_docs)
    result = judge_chat_json(prompt)
    return result.get("score", 0.0) if result else 0.0


def judge_answer_correctness(question: str, prediction: str, gold: str, lang=None) -> float:
    prompt = get_judge_prompt("correctness", lang).format(question=question, prediction=prediction, gold=gold)
    result = judge_chat_json(prompt)
    return result.get("score", 0.0) if result else 0.0
