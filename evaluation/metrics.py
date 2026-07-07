"""基础评测指标：EM（Exact Match）、F1（token overlap）、CostTracker（系统性能统计）"""
import re
import string
import time
import unicodedata
from collections import Counter

_CN_PUNCTUATION = '。，、；：？！""''【】《》（）｛｝〔〕·…—～'
_ALL_PUNCTUATION = set(string.punctuation) | set(_CN_PUNCTUATION)


def _normalize(text: str) -> str:
    """标准化文本用于评测比较，兼容中英文、全角字符、中文标点和数字。"""
    result = []
    for ch in str(text):
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif ch == "\u3000":
            result.append(" ")
        elif unicodedata.category(ch).startswith("Zs"):
            result.append(" ")
        else:
            result.append(ch)

    text = "".join(result).lower()
    text = "".join(ch for ch in text if ch not in _ALL_PUNCTUATION)
    text = re.sub(r"(\d)([\u4e00-\u9fff])", r"\1 \2", text)
    text = re.sub(r"([\u4e00-\u9fff])(\d)", r"\1 \2", text)
    # 移除冠词
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # 合并空白
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold: str, aliases: list[str] | None = None) -> float:
    """EM 精确匹配，支持 answer_aliases"""
    candidates = [gold] + (aliases or [])
    # 只要匹配任意 alias → EM = 1，否则 EM = 0
    return max(1.0 if _normalize(prediction) == _normalize(c) else 0.0 for c in candidates)


def f1_score(prediction: str, gold: str, aliases: list[str] | None = None) -> float:
    """Token-level F1，支持 answer_aliases（取最大值）"""
    candidates = [gold] + (aliases or [])
    # 取最大F1
    return max(_f1_single(prediction, c) for c in candidates)


def _f1_single(prediction: str, gold: str) -> float:
    """单个 gold 的 Token-level F1"""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    # 交集计算
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    # 衡量“答案重合程度”，不是完全匹配
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
