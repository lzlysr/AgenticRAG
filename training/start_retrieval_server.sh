#!/bin/bash
# 启动 BGE Embedding + Reranker HTTP 服务（GPU 7）
# 用法: bash scripts/start_retrieval_server.sh

set -x

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
LOG=${SCRIPT_DIR}/logs/retrieval_server.log

mkdir -p ${SCRIPT_DIR}/logs

# 杀掉旧的 retrieval server
pkill -f "retrieval_server.py" 2>/dev/null
sleep 1

echo "Starting retrieval server on GPU 7..."
CUDA_VISIBLE_DEVICES=7 nohup ${PYTHON} \
  ${SCRIPT_DIR}/training/tools/retrieval_server.py \
  --port 8790 --device cuda:0 \
  > ${LOG} 2>&1 &

echo "PID: $!"
echo "Waiting for server to load models (~30s)..."

for i in $(seq 1 60); do
    sleep 5
    if curl -s http://localhost:8790/health | grep -q "ok"; then
        echo "Retrieval server ready!"
        # 快速测试
        curl -s -X POST http://localhost:8790/embed \
          -H 'Content-Type: application/json' \
          -d '{"texts": ["测试查询"]}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Embed OK: dim={len(d[\"embeddings\"][0])}, elapsed={d[\"elapsed\"]:.3f}s')"
        exit 0
    fi
    echo "  waiting... (${i}/60)"
done

echo "ERROR: Server failed to start. Check ${LOG}"
exit 1
