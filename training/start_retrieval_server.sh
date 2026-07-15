#!/bin/bash
# 启动 BGE Embedding + Reranker HTTP 服务（GPU 0）
# 用法: bash training/start_retrieval_server.sh

# 这样做的核心目的，是把检索模型从 GRPO 训练进程中独立出来，避免每个 rollout worker 都加载一份 BGE 模型，造成 GPU 显存浪费。

set -x # 让 Bash 在执行每条命令前，把命令打印到终端。

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd) # 获取项目根目录
PYTHON=${PYTHON:-/data/lzl/anaconda3/envs/agenticrag/bin/python} # 默认 agenticrag 环境启动
LOG=${SCRIPT_DIR}/logs/retrieval_server.log

mkdir -p ${SCRIPT_DIR}/logs

# 杀掉旧的 retrieval server
pkill -f "retrieval_server.py" 2>/dev/null
sleep 1

# 使用 nohup 后，即使终端关闭，服务仍能继续运行。最后的 & 表示后台运行。
echo "Starting retrieval server on GPU 0..."
CUDA_VISIBLE_DEVICES=1 nohup ${PYTHON} \
  ${SCRIPT_DIR}/training/tools/retrieval_server.py \
  --port 8790 --device cuda:0 \
  > ${LOG} 2>&1 &

echo "PID: $!"
echo "Waiting for server to load models (~30s)..."

for i in $(seq 1 60); do
    sleep 5
    if curl --noproxy '*' --max-time 5 -s http://localhost:8790/health | grep -q "ok"; then
        echo "Retrieval server ready!"
        # 快速测试 不走代理
        curl --noproxy '*' --max-time 30 -s -X POST http://localhost:8790/embed \
          -H 'Content-Type: application/json' \
          -d '{"texts": ["测试查询"]}' | ${PYTHON} -c "import sys,json; d=json.load(sys.stdin); print(f'Embed OK: dim={len(d[\"embeddings\"][0])}, elapsed={d[\"elapsed\"]:.3f}s')"
        curl --noproxy '*' --max-time 30 -s -X POST http://localhost:8790/rerank \
          -H 'Content-Type: application/json' \
          -d '{"query": "永辉超市", "passages": ["永辉超市注册资本100亿", "沃尔玛全球门店"]}' | ${PYTHON} -c "import sys,json; d=json.load(sys.stdin); print(f'Rerank OK: scores={len(d[\"scores\"])}, elapsed={d[\"elapsed\"]:.3f}s')"
        exit 0
    fi
    echo "  waiting... (${i}/60)"
done

echo "ERROR: Server failed to start. Check ${LOG}"
exit 1
