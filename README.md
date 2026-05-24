# AgenticRAG: 金融多跳 QA 场景下的 Agentic RAG 系统

> 从基准适配到 SFT+GRPO 端到端训练的完整复现包

## 目录结构

```
core/
├── config.py              # 全局配置（路径、模型、检索参数）
├── requirements.txt       # agenticrag 环境依赖（推理/评测/数据合成/SFT）
├── requirements-verl.txt  # verl 环境依赖（GRPO 训练）
│
├── agents/                # Agent 编排（LangGraph）
│   ├── graph.py           # Planner→Executor→Verifier→Synthesizer 循环图
│   ├── planner.py         # 子查询分解 + 工具选择
│   ├── executor.py        # 工具执行（检索/读取）
│   ├── verifier.py        # Evidence 充分性判断
│   ├── synthesizer.py     # 基于 evidence 生成最终答案
│   └── prompts.py         # 中英文 Prompt 模板
│
├── retrieval/             # 检索工具
│   ├── semantic_search.py # FAISS 向量检索（BGE-M3）
│   ├── keyword_search.py  # BM25 关键词检索（jieba 分词）
│   ├── graph_search.py    # 知识图谱检索（NetworkX）
│   ├── hybrid_search.py   # RRF 多路融合 + CrossEncoder 重排
│   ├── read_chunk.py      # 按 chunk_id 读取原文
│   ├── embedder.py        # BGE-M3 Embedding 封装
│   └── reranker.py        # BGE-Reranker-v2-m3 封装
│
├── llm/                   # LLM 统一接口
│   └── client.py          # 本地 vLLM + 云端 API 统一调用（OpenAI 兼容协议）
│
├── evaluation/            # 评测框架
│   ├── metrics.py         # EM/F1/hop_recall 等指标
│   ├── llm_judge.py       # 三维度 LLM Judge（Correctness/Faithfulness/CtxPrecision）
│   ├── hop_aware_eval.py  # hop-aware 诊断评测
│   └── run_eval.py        # 全量评测入口
│
├── training/              # 训练代码
│   ├── reward_agentic_rag.py         # GRPO 奖励函数（最终 v9a）
│   ├── start_grpo.sh                # GRPO 训练启动脚本
│   ├── sft_pipeline.sh              # SFT 一站式流水线
│   ├── sft_zh_react.yaml            # SFT 配置（ReAct 格式）
│   ├── tools/financial_search_tool.py  # verl BaseTool 实现
│   └── tools/retrieval_server.py     # Embedding+Reranker HTTP 服务
│
├── scripts/               # 数据合成 + 评测 + 工具脚本
│   ├── domain_multihop_synthesis.py  # 多跳 QA 合成核心
│   ├── eval_agentic.py               # Agentic 评测入口
│   ├── trace_to_sft.py               # Trace → SFT ReAct 格式转换
│   └── ...                           # 其余 20+ 脚本
│
├── data/
│   ├── datasets/          # AgenticRAGTracer 原始数据（1305 条）
│   ├── financial_eval/    # 金融评测+训练数据
│   │   ├── qa_pairs_zh_clean.json    # 中文评测集
│   │   ├── sft/                      # SFT 训练数据（ReAct 格式）
│   │   ├── sft_zh_llamafactory/      # LLaMA-Factory 格式
│   │   ├── grpo_agentic_*.parquet    # GRPO 训练数据
│   │   └── traces_oracle*.jsonl      # Oracle 轨迹
│   └── financial_all/     # 金融全语料 + 索引
│
├── docs/                  # 技术报告
├── LLaMA-Factory/         # SFT 训练框架（LoRA 微调）
└── verl/                  # GRPO 训练框架（多轮 Agentic RL）
```

## 快速开始

### 1. 环境准备

```bash
# 环境 1: agenticrag（推理/评测/数据合成/SFT）
conda create -n agenticrag python=3.11
conda activate agenticrag
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# SFT 训练额外安装：cd LLaMA-Factory && pip install -e .

# 环境 2: verl（GRPO 训练，与 agenticrag 严格隔离）
conda create -n verl python=3.12
conda activate verl
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
cd verl && pip install -e . && cd ..
pip install -r requirements-verl.txt
```

### 2. 模型准备

下载 BGE-M3 和 BGE-Reranker-v2-m3 到 `models/` 目录：
```bash
mkdir -p models
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
```

### 3. 配置模型 API

所有 LLM 调用统一通过 OpenAI 兼容协议（`llm/client.py`），支持本地 vLLM 和任意云端 API。

**本地 vLLM 部署**（推荐）：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --port 9097 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 32768
```

**环境变量配置**：
```bash
# 本地 vLLM 地址
export VLLM_BASE_URL="http://localhost:9097/v1"     # Agent 推理模型

# 模型和检索配置
export MODEL_HUB="./models"
export AGENT_LLM_MODEL="Qwen3-4B"
export PROMPT_LANG="zh"

# Judge 模型（评测 + GRPO reward 中的 LLM 评分）
# 方案 A: 本地 vLLM 部署
export JUDGE_BASE_URL="http://localhost:8086/v1"
export JUDGE_LLM_MODEL="gpt-oss-120b"
# 方案 B: 云端 API（推荐）— 在 llm/client.py 中注册模型后使用
# export JUDGE_LLM_MODEL="gpt-4o-judge"
```

**添加自定义模型**（本地或云端均可）：在 `llm/client.py` 的 `MODEL_CONFIGS` 中新增条目：
```python
# 云端 API 示例（如 OpenAI / DeepSeek / 通义千问等）
MODEL_CONFIGS["gpt-4o-judge"] = ModelConfig(
    url="https://api.openai.com/v1",
    model_name="gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)
# 本地 vLLM 示例
MODEL_CONFIGS["my-local-model"] = ModelConfig(
    url="http://localhost:8000/v1",
    model_name="my-model-name",
)
```

### 4. 运行评测

```bash
export NEWS_CORPUS_DIR=data/financial_eval
export NEWS_INDEX_DIR=data/financial_all/indexes

# Pipeline 模式评测
python scripts/run_cloud_eval.py --model Qwen3-4B --workers 1

# Agentic 模式评测
python scripts/eval_agentic.py --model Qwen3-4B --max-samples 50
```

## 全流程复现

### Step 1: 数据合成

详见 `docs/project_report_v2.md` 第二章。

```bash
# 1. 生成种子 QA
python scripts/gen_seed_qa.py

# 2. 多跳合成
python scripts/domain_multihop_synthesis.py --lang zh

# 3. 质量过滤
python scripts/judge_synthesis.py
python scripts/clean_synthesis.py
```

### Step 2: SFT 训练

```bash
# 1. 构建 Oracle Trace → SFT 数据
python scripts/build_oracle_traces.py
python scripts/trace_to_sft.py --lang zh
python scripts/convert_sft_to_llamafactory.py

# 2. 一站式 SFT（需要 LLaMA-Factory）
bash training/sft_pipeline.sh
```

### Step 3: GRPO 训练

```bash
# 1. 准备 GRPO 数据
python scripts/prepare_agentic_grpo_data.py

# 2. 启动 Retrieval Server（GRPO 训练时需要）
bash training/start_retrieval_server.sh

# 3. 启动 GRPO 训练（需要 verl）
export VERL_DIR=$(pwd)/verl
bash training/start_grpo.sh
```

## 关键配置说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `VLLM_BASE_URL` | `http://localhost:9097/v1` | Agent 推理模型 URL |
| `VLLM_32B_URL` | `http://localhost:9094/v1` | 32B 大模型 URL |
| `JUDGE_BASE_URL` | `http://localhost:8086/v1` | Judge 模型 URL（本地 vLLM） |
| `JUDGE_LLM_MODEL` | `gpt-oss-120b` | Judge 模型名（可改为 MODEL_CONFIGS 中注册的云端模型） |
| `MODEL_HUB` | `./models` | BGE Embedding/Reranker 模型目录 |
| `AGENT_LLM_MODEL` | `Qwen3-32B` | Agent 默认推理模型 |
| `PROMPT_LANG` | `zh` | Prompt 语言（zh/en） |
| `NEWS_CORPUS_DIR` | 空 | 金融语料路径（覆盖默认） |
| `NEWS_INDEX_DIR` | 空 | 金融索引路径（覆盖默认） |
