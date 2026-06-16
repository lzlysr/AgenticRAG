"""统一 LLM 调用接口

所有模型（本地 vLLM + 云端 API）统一通过 OpenAI 兼容协议调用。
通过环境变量配置模型地址和 API Key，无供应商耦合。

环境变量：
  VLLM_BASE_URL     — 本地 Agent 模型地址 (默认 http://localhost:9097/v1)
  VLLM_32B_URL      — 32B 大模型地址 (默认 http://localhost:9094/v1)
  JUDGE_BASE_URL    — Judge 模型地址 (默认 http://localhost:8086/v1)
  DEEPSEEK_API_KEY  — DeepSeek API Key
  AGENT_LLM_MODEL   — Agent 默认模型名
  JUDGE_LLM_MODEL   — Judge 默认模型名

云端 API 使用方式：在 MODEL_CONFIGS 中注册新条目，指定 url/model_name/api_key。
"""
import json
import logging
import os
import re
import sys
import time
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI, OpenAIError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AGENT_LLM_MODEL, JUDGE_LLM_MODEL

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# 环境变量
# ══════════════════════════════════════════════════════════════════
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:9097/v1")
VLLM_32B_URL = os.environ.get("VLLM_32B_URL", "http://localhost:9094/v1")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8086/v1")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


# ══════════════════════════════════════════════════════════════════
# 模型配置
# ══════════════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    """模型配置（支持本地 vLLM 和任意 OpenAI 兼容 API）"""
    url: str                           # API base URL
    model_name: str                    # 实际传给 API 的模型名
    api_key: str = "EMPTY"             # API Key（本地 vLLM 用 "EMPTY"）
    max_len: int = 32768
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: int = 0
    timeout: int = 200
    retry_attempts: int = 20
    think_bool: bool = False
    _client: Optional[Any] = field(default=None, repr=False)

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI client，首次访问时才创建连接"""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.url)
        return self._client


# ── 模型注册表 ─────────────────────────────────────────────────
# 添加新模型（本地或云端）：在此 dict 中新增 ModelConfig 条目即可
# 云端 API 示例：
#   MODEL_CONFIGS["gpt-4o-judge"] = ModelConfig(
#       url="https://api.openai.com/v1",
#       model_name="gpt-4o",
#       api_key=os.environ.get("OPENAI_API_KEY", ""),
#   )
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # ── 本地 vLLM 模型 ──
    "Qwen3-4B": ModelConfig(
        url=VLLM_BASE_URL,
        model_name="Qwen3-4B",
        max_len=30000,
    ),
    "Qwen3-32B": ModelConfig(
        url=VLLM_32B_URL,
        model_name="Qwen/Qwen3-32B",
        max_len=131072,
    ),
    "gpt-oss-120b": ModelConfig(
        url=JUDGE_BASE_URL,
        model_name="gpt-oss-120b",
        max_len=131072,
    ),
    "deepseek-v4-flash": ModelConfig(
        url="https://api.deepseek.com",
        model_name="deepseek-v4-flash",
        api_key=DEEPSEEK_API_KEY,
        max_len=131072,
    ),
}


# ══════════════════════════════════════════════════════════════════
# 统计
# ══════════════════════════════════════════════════════════════════
class LLMStats:
    """线程安全的 LLM 调用统计"""
    def __init__(self):
        self._lock = threading.Lock()
        self.agent_calls = 0
        self.judge_calls = 0
        self.total_latency = 0.0

    def record(self, kind: str, latency: float):
        with self._lock:
            if kind == "agent":
                self.agent_calls += 1
            else:
                self.judge_calls += 1
            self.total_latency += latency

    def reset(self):
        with self._lock:
            self.agent_calls = self.judge_calls = 0
            self.total_latency = 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "agent_calls": self.agent_calls,
                "judge_calls": self.judge_calls,
                "total_latency": round(self.total_latency, 2),
            }


stats = LLMStats()


# ══════════════════════════════════════════════════════════════════
# 核心调用函数
# ══════════════════════════════════════════════════════════════════
def _strip_think(text: str) -> str:
    """去掉 <think>...</think> 标签"""
    if text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def _format_messages(
    messages: Union[str, List[Dict[str, str]]], model_name: str = ""
) -> List[Dict[str, str]]:
    """将 str 或 messages list 统一为 OpenAI messages 格式"""
    if isinstance(messages, str):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": messages},
        ]
    return messages


def get_from_llm(
    messages: Union[str, List[Dict[str, str]]],
    model_name: str = "Qwen3-4B",
    **kwargs,
) -> Optional[str]:
    """统一 LLM 调用（本地 vLLM 或云端 API，全部走 OpenAI SDK）

    Args:
        messages: 文本 prompt 或 OpenAI messages list
        model_name: MODEL_CONFIGS 中的模型名
        **kwargs: temperature, top_p, max_len 等覆盖参数

    Returns:
        模型回复文本，失败返回 None
    """
    # 兼容旧脚本：历史接口用 model=... 指定模型。
    model_name = kwargs.pop("model", model_name)

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_CONFIGS.keys())}. "
            f"Add new models to MODEL_CONFIGS in llm/client.py."
        )

    config = MODEL_CONFIGS[model_name]
    formatted = _format_messages(messages, model_name)

    temperature = kwargs.get("temperature", config.temperature)
    top_p = kwargs.get("top_p", config.top_p)
    max_tokens = kwargs.get("max_len", kwargs.get("max_tokens", config.max_len))

    logger.info(f"Requesting {model_name} at {config.url}")

    for attempt in range(config.retry_attempts):
        try:
            resp = config.client.chat.completions.create(
                model=config.model_name,
                messages=formatted,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": config.think_bool},
                    "top_k": config.top_k,
                    "min_p": config.min_p,
                },
            )
            text = resp.choices[0].message.content
            if text:
                return _strip_think(text)
            logger.error(f"Empty response from {model_name}")
            time.sleep(5)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {model_name}: {e}")
            if attempt == config.retry_attempts - 1:
                logger.error(f"All {config.retry_attempts} attempts failed for {model_name}")
                logger.error(traceback.format_exc())
            else:
                time.sleep(min(5, 1 + attempt))
    return None


# ══════════════════════════════════════════════════════════════════
# 高层接口（Agent / Judge）
# ══════════════════════════════════════════════════════════════════
def _extract_json(text: str):
    """从 LLM 回复中提取 JSON（支持 markdown code block）"""
    if text is None:
        return None
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            idx_start = text.find(start_char)
            idx_end = text.rfind(end_char)
            if idx_start != -1 and idx_end > idx_start:
                try:
                    return json.loads(text[idx_start:idx_end + 1])
                except json.JSONDecodeError:
                    continue
        return None


def agent_chat(prompt: str, model: str = None, messages: list = None) -> str:
    """Agent 推理调用"""
    model = model or AGENT_LLM_MODEL
    t0 = time.time()

    if messages:
        resp = get_from_llm(messages, model_name=model)
    else:
        resp = get_from_llm(prompt, model_name=model)

    stats.record("agent", time.time() - t0)
    return resp or ""


def agent_chat_json(prompt: str, model: str = None, retries: int = 2) -> dict | None:
    """Agent 推理调用，返回解析后的 JSON"""
    for i in range(retries + 1):
        resp = agent_chat(prompt, model=model)
        result = _extract_json(resp)
        if result is not None:
            return result
    return None


def judge_chat(prompt: str, model: str = None) -> str:
    """Judge/合成调用"""
    model = model or JUDGE_LLM_MODEL
    t0 = time.time()
    resp = get_from_llm(prompt, model_name=model)
    stats.record("judge", time.time() - t0)
    return resp or ""


def judge_chat_json(prompt: str, model: str = None, retries: int = 2) -> dict | None:
    """Judge 调用，返回解析后的 JSON"""
    for i in range(retries + 1):
        resp = judge_chat(prompt, model=model)
        result = _extract_json(resp)
        if result is not None:
            return result
    return None


# ── 兼容别名（供旧脚本使用）──
def get_from_ks_openai(prompt: str, model: str = "gpt-oss-120b", **kwargs) -> str:
    """兼容旧接口：等价于 get_from_llm(prompt, model_name=model)"""
    return get_from_llm(prompt, model_name=model, **kwargs) or ""
