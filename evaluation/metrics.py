"""基础评测指标：EM、F1、CostTracker"""
import re
import string
import time
from collections import Counter


def _normalize(text: str) -> str:
    """标准化文本用于评测比较"""
    text = text.lower()
    # 移除标点
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # 移除冠词
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # 合并空白
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold: str, aliases: list[str] | None = None) -> float:
    """EM 精确匹配，支持 answer_aliases"""
    candidates = [gold] + (aliases or [])
    return max(1.0 if _normalize(prediction) == _normalize(c) else 0.0 for c in candidates)


def f1_score(prediction: str, gold: str, aliases: list[str] | None = None) -> float:
    """Token-level F1，支持 answer_aliases（取最大值）"""
    candidates = [gold] + (aliases or [])
    return max(_f1_single(prediction, c) for c in candidates)


def _f1_single(prediction: str, gold: str) -> float:
    """单个 gold 的 Token-level F1"""
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


class CostTracker:
    """统计 LLM 调用次数、工具调用次数、延迟"""

    def __init__(self):
        self.records = []

    def record(self, state: dict, latency: float):
        self.records.append({
            "total_tool_calls": state.get("total_tool_calls", 0),
            "iteration_count": state.get("iteration_count", 0),
            "latency": latency,
        })

    def summary(self) -> dict:
        if not self.records:
            return {}
        n = len(self.records)
        return {
            "num_queries": n,
            "avg_tool_calls": sum(r["total_tool_calls"] for r in self.records) / n,
            "avg_iterations": sum(r["iteration_count"] for r in self.records) / n,
            "avg_latency": sum(r["latency"] for r in self.records) / n,
            "total_latency": sum(r["latency"] for r in self.records),
        }
