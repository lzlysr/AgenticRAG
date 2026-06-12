"""全局配置 — 所有路径支持环境变量覆盖"""
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型路径（环境变量覆盖）
MODEL_HUB = os.environ.get("MODEL_HUB", os.path.join(PROJECT_ROOT, "models"))
BGE_M3_PATH = os.environ.get("BGE_M3_PATH", os.path.join(MODEL_HUB, "bge-m3"))
BGE_RERANKER_PATH = os.environ.get("BGE_RERANKER_PATH", os.path.join(MODEL_HUB, "bge-reranker-v2-m3"))

# 数据和索引路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")
INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "indexes")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TMP_DIR = os.path.join(PROJECT_ROOT, "tmp")

# Agent 推理模型（环境变量覆盖）
AGENT_LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "Qwen3-4B")
# Prompt 语言：en / zh
PROMPT_LANG = os.environ.get("PROMPT_LANG", "zh")
# 评测 Judge 用：填写 llm/client.py 中 MODEL_CONFIGS 的 key
JUDGE_LLM_MODEL = os.environ.get("JUDGE_LLM_MODEL", "deepseek-v4-flash")
# 数据合成用：填写 llm/client.py 中 MODEL_CONFIGS 的 key
SYNTH_LLM_MODEL = os.environ.get("SYNTH_LLM_MODEL", "deepseek-v4-flash")

# 检索参数
RERANK_TOP_K = 5
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20

# 金融语料配置（环境变量覆盖）
NEWS_CORPUS_DIR = os.environ.get("NEWS_CORPUS_DIR", "")
NEWS_INDEX_DIR = os.environ.get("NEWS_INDEX_DIR", "")

# 实际使用的路径（优先金融语料）
ACTIVE_DATA_DIR = NEWS_CORPUS_DIR or DATA_DIR
ACTIVE_INDEX_DIR = NEWS_INDEX_DIR or INDEX_DIR

# 确保目录存在
for d in [DATA_DIR, INDEX_DIR, RESULTS_DIR, TMP_DIR, ACTIVE_DATA_DIR, ACTIVE_INDEX_DIR]:
    if d:
        os.makedirs(d, exist_ok=True)
