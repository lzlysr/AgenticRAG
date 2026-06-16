"""
LLM 调用封装：
服务离线数据合成的大规模并发、JSON 解析、重试和统计。用于多跳 QA 合成 pipeline，基于 mog-1 (GPT-5) via KS API
"""
import json
import re
import time
import threading
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

logger = logging.getLogger("synthesis")

# ---------- 全局统计 ----------
_stats_lock = threading.Lock()
_stats = {"calls": 0, "errors": 0, "total_latency": 0.0}

# ---------- 并发控制 ----------
# 并发信号量。用于限制同一时刻真正调用 LLM API 的请求数量。
_semaphore = None


def init_concurrency(max_concurrent: int = 20):
    """初始化并发信号量。最多允许 20 个线程同时进入真正的 LLM 调用。"""
    global _semaphore
    _semaphore = threading.Semaphore(max_concurrent)


def get_stats() -> dict:
    with _stats_lock:
        # 返回浅拷贝，外部修改不会影响内部统计
        return dict(_stats)


def reset_stats():
    with _stats_lock:
        _stats["calls"] = 0
        _stats["errors"] = 0
        _stats["total_latency"] = 0.0


def _record_call(latency: float, error: bool = False):
    with _stats_lock:
        _stats["calls"] += 1
        _stats["total_latency"] += latency
        if error:
            _stats["errors"] += 1


# ---------- JSON 解析 ----------
def _extract_json(text: str):
    """从 LLM 回复中提取 JSON"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        # 如果文本本身已经是合法 JSON，直接解析并返回
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 如果文本不是纯 JSON，尝试从中提取第一个 JSON 对象或数组进行解析
    for start_c, end_c in [('[', ']'), ('{', '}')]:
        idx_s = text.find(start_c)
        idx_e = text.rfind(end_c)
        if idx_s != -1 and idx_e > idx_s:
            try:
                return json.loads(text[idx_s:idx_e + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---------- 核心调用 ----------
def llm_call(prompt: str, model: str = "deepseek-v4-flash", temperature: float = 0.7,
             system_prompt: str = "You are a helpful assistant.",
             timeout: int = 200) -> str:
    """单次 LLM 调用（带并发控制，统一走 llm.client）"""
    # 把全局信号量保存到局部变量。
    sem = _semaphore
    # 如果有信号量，先申请一个名额。
    if sem:
        sem.acquire()
    try:
        t0 = time.time()
        from llm.client import get_from_llm
        resp = get_from_llm(prompt, model_name=model, temperature=temperature)
        _record_call(time.time() - t0)
        return resp or ""
    except Exception as e:
        _record_call(time.time() - t0 if 't0' in dir() else 0, error=True)
        raise
    finally:
        if sem:
            # 无论如何释放信号量
            sem.release()


def llm_call_with_retry(prompt: str, max_retries: int = 3,
                        model: str = "deepseek-v4-flash", temperature: float = 0.7,
                        return_json: bool = False,
                        timeout: int = 200) -> str | dict | list | None:
    """带重试的 LLM 调用，可选 JSON 解析"""
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = llm_call(prompt, model=model, temperature=temperature, timeout=timeout)
            if return_json:
                parsed = _extract_json(resp)
                if parsed is not None:
                    return parsed
                # JSON 解析失败，重试
                logger.warning(f"JSON parse failed (attempt {attempt+1}), raw: {resp[:200]}")
                last_error = ValueError(f"JSON parse failed: {resp[:200]}")
                time.sleep(1)
                continue
            return resp
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2 * (attempt + 1))

    if return_json:
        logger.error(f"All {max_retries} retries failed for JSON call: {last_error}")
        return None
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


def llm_judge(question: str, golden_answer: str, other_answer: str,
              judge_prompt: str, model: str = "deepseek-v4-flash") -> dict:
    """
    给定问题、标准答案和另一个答案，让 LLM 判断两个答案语义是否等价。
    EssEq 评分：判断 other_answer 是否等价于 golden_answer
    """
    prompt = f"Input:\nQuestion: {question}\nGolden answer: {golden_answer}\nOther answer: {other_answer}"
    result = llm_call_with_retry(
        prompt=f"{judge_prompt}\n\n{prompt}",
        model=model,
        return_json=True,
        max_retries=2,
    )
    if result is None:
        return {"avg_score": 0, "reasons": [], "raw_scores": []}
    if isinstance(result, list):
        result = result[0] if result else {}
    # 为什么分数和理由使用列表？因为有些评测可能会给出多个维度的评分和理由，比如准确性、完整性、流畅性等。或者是兼容多评委模式。
    return {
        "avg_score": result.get("answer_score", 0),
        "reasons": [result.get("answer_reason", "")],
        "raw_scores": [result.get("answer_score", 0)],
    }
