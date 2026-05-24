#!/bin/bash
# SFT 训练 + Merge + 评测 一站式脚本
# 用法: bash scripts/sft_pipeline.sh [TRAIN_GPU] [VLLM_GPU] [RETRIEVAL_GPU]
# 示例: bash scripts/sft_pipeline.sh 3 0 1
#   训练用 GPU 3，评测时 vLLM 用 GPU 0，embedder/reranker 用 GPU 1
set -e

cd "$(dirname "$0")/.."

# ============================================================
# 参数
# ============================================================
TRAIN_GPU=${1:-3}
VLLM_GPU=${2:-0}
RETRIEVAL_GPU=${3:-1}

YAML_PATH="training/qwen3_4b_sft_zh_func_call.yaml"
BASE_MODEL="/share/project/common/models/models/Qwen/Qwen3-4B"
OUTPUT_DIR="training/outputs/qwen3-4b-sft-zh-func_call"
MERGED_DIR="${OUTPUT_DIR}-merged"
VLLM_PORT=9097
SERVED_MODEL="Qwen3-4B-SFT-ZH"
EVAL_DATA_DIR="data/financial_eval_zh"
EVAL_INDEX_DIR="data/financial_all/indexes"
EVAL_LANG="zh"
EVAL_WORKERS=10
EVAL_MAX_SAMPLES=185
JUDGE_WORKERS=20

ANYRAG_PYTHON="${ANYRAG_PYTHON:-python}"
VERL_PYTHON="${VERL_PYTHON:-python}"
LLAMA_FACTORY="${LLAMA_FACTORY:?"请设置 LLAMA_FACTORY 路径"}"

echo "============================================"
echo "  SFT Pipeline"
echo "  Train GPU: $TRAIN_GPU"
echo "  vLLM GPU:  $VLLM_GPU"
echo "  Retrieval GPU: $RETRIEVAL_GPU (embedder+reranker)"
echo "============================================"

# ============================================================
# 通用函数
# ============================================================
cleanup_gpu() {
    echo "  清理 GPU 残留..."
    # 1. nvidia-smi 可见的进程
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
    for pid in $pids; do kill -9 "$pid" 2>/dev/null || true; done
    # 2. vLLM EngineCore 子进程（nvidia-smi 经常看不到）
    pids=$(ps aux | grep -i "vllm.*EngineCore\|EngineCore.*vllm" | grep -v grep | awk '{print $2}')
    for pid in $pids; do kill -9 "$pid" 2>/dev/null || true; done
    # 3. 其他 vLLM 残留
    pids=$(ps aux | grep -i "vllm" | grep -v grep | grep -v defunct | awk '{print $2}')
    for pid in $pids; do kill -9 "$pid" 2>/dev/null || true; done
    sleep 3
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

check_gpu_free() {
    local gpu_id=$1
    local gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    if [ "$gpu_mem" -gt 1000 ]; then
        echo "  ERROR: GPU $gpu_id 仍有 ${gpu_mem}MiB 占用"
        echo "  提示: top -c 查找残留进程"
        return 1
    fi
    echo "  GPU $gpu_id 空闲 (${gpu_mem}MiB)"
    return 0
}

# ============================================================
# Step 0: 清理 GPU 残留
# ============================================================
echo ""
echo "=== Step 0: 清理 GPU ==="
cleanup_gpu
check_gpu_free $TRAIN_GPU || exit 1
rm -rf /root/.cache/huggingface/datasets/json* 2>/dev/null

# ============================================================
# Step 1: SFT 训练
# ============================================================
echo ""
echo "=== Step 1: SFT 训练 (GPU $TRAIN_GPU) ==="
echo "  $(date)"

cd "$LLAMA_FACTORY"
CUDA_VISIBLE_DEVICES=$TRAIN_GPU $ANYRAG_PYTHON -m llamafactory.cli train \
    "$(cd - > /dev/null && pwd)/$YAML_PATH"
cd - > /dev/null

echo "[DONE] 训练完成 $(date)"
ls "$OUTPUT_DIR/" | grep checkpoint

# ============================================================
# Step 2: 清理训练后 GPU 残留
# ============================================================
echo ""
echo "=== Step 2: 清理训练后残留 ==="
sleep 3
cleanup_gpu

# ============================================================
# Step 3: Merge 所有 checkpoints
# ============================================================
echo ""
echo "=== Step 3: Merge LoRA ==="

for ckpt_dir in $(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n) "$OUTPUT_DIR"; do
    ckpt_name=$(basename "$ckpt_dir")
    if [ "$ckpt_name" = "$(basename $OUTPUT_DIR)" ]; then
        merge_dir="${MERGED_DIR}"
        label="final"
    else
        merge_dir="${OUTPUT_DIR}-${ckpt_name}-merged"
        label="$ckpt_name"
    fi

    if [ -d "$merge_dir" ] && [ -f "$merge_dir/config.json" ]; then
        echo "  [SKIP] $label"
        continue
    fi

    echo "  Merging $label → $merge_dir"
    cat > /tmp/export_ckpt.yaml << YAML
model_name_or_path: $BASE_MODEL
adapter_name_or_path: $ckpt_dir
template: qwen3_nothink
finetuning_type: lora
export_dir: $merge_dir
export_size: 2
export_device: cpu
export_legacy_format: false
YAML
    cd "$LLAMA_FACTORY"
    $ANYRAG_PYTHON -m llamafactory.cli export /tmp/export_ckpt.yaml 2>&1 | tail -1
    cd - > /dev/null
done

echo "  Merged:"
ls -d ${OUTPUT_DIR}*merged/ 2>/dev/null

# ============================================================
# Step 4: 串行评测所有 checkpoints
# ============================================================
echo ""
echo "=== Step 4: 串行评测 (vLLM GPU $VLLM_GPU, retrieval GPU $RETRIEVAL_GPU) ==="

for mdir in $(ls -d ${OUTPUT_DIR}*merged/ 2>/dev/null | sort); do
    dir_name=$(basename "$mdir")
    if echo "$dir_name" | grep -q "checkpoint"; then
        label=$(echo "$dir_name" | grep -oP 'checkpoint-\d+' | sed 's/checkpoint-/ckpt/')
    else
        label="final"
    fi
    result_file="results/eval_all_${EVAL_MAX_SAMPLES}_${SERVED_MODEL}-${label}_financial.json"

    if [ -f "$result_file" ]; then
        echo "  [SKIP] $label already evaluated"
        continue
    fi

    echo ""
    echo "  --- $label ---"

    # 4a. 清理并启动 vLLM
    cleanup_gpu
    check_gpu_free $VLLM_GPU || { echo "  跳过 $label"; continue; }

    CUDA_VISIBLE_DEVICES=$VLLM_GPU $VERL_PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$mdir" \
        --served-model-name "$SERVED_MODEL" \
        --port $VLLM_PORT \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        > tmp/vllm_eval_${label}.log 2>&1 &
    VLLM_PID=$!

    echo "  等待 vLLM (GPU $VLLM_GPU)..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
            echo "  vLLM 就绪 (${i}x3s)"
            break
        fi
        sleep 3
    done

    if ! curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
        echo "  ERROR: vLLM 超时，跳过 $label"
        kill -9 $VLLM_PID 2>/dev/null || true
        cleanup_gpu
        continue
    fi

    # 4b. 评测（embedder/reranker 指定 RETRIEVAL_GPU）
    RETRIEVAL_DEVICE="cuda:${RETRIEVAL_GPU}" python -u evaluation/run_eval.py \
        --model "$SERVED_MODEL" \
        --workers $EVAL_WORKERS \
        --max-samples $EVAL_MAX_SAMPLES \
        --data-dir "$EVAL_DATA_DIR" \
        --index-dir "$EVAL_INDEX_DIR" \
        --lang "$EVAL_LANG"

    # 重命名结果
    default_result="results/eval_all_${EVAL_MAX_SAMPLES}_${SERVED_MODEL}_financial.json"
    if [ -f "$default_result" ]; then
        mv "$default_result" "$result_file"
        echo "  → $result_file"
    fi

    # 4c. 关 vLLM + 清理
    kill -9 $VLLM_PID 2>/dev/null || true
    sleep 2
    cleanup_gpu
done

# ============================================================
# Step 5: 批量 Judge
# ============================================================
echo ""
echo "=== Step 5: LLM Judge ==="

RESULT_FILES=$(ls results/eval_all_*_${SERVED_MODEL}-*_financial.json 2>/dev/null)
if [ -n "$RESULT_FILES" ]; then
    python -u scripts/run_judge_batch.py \
        $RESULT_FILES \
        --workers $JUDGE_WORKERS \
        --lang "$EVAL_LANG" \
        --data-dir "$EVAL_DATA_DIR"
fi

# ============================================================
# Step 6: 汇总
# ============================================================
echo ""
echo "=== Step 6: 汇总结果 ==="

python3 << 'PYEOF'
import json, glob

files = sorted(glob.glob("results/eval_all_*_Qwen3-4B-SFT-ZH-*_financial_judged.json"))
if not files:
    files = sorted(glob.glob("results/eval_all_*_Qwen3-4B-SFT-ZH-*_financial.json"))

if not files:
    print("  没有找到结果文件")
else:
    print(f"{'Checkpoint':<15} {'EM':>6} {'F1':>6} {'Judge_C':>8} {'Judge_F':>8} {'Ctx_P':>8}")
    print('-' * 55)
    for f in files:
        d = json.load(open(f))
        s = d['summary']
        name = f.split('_Qwen3-4B-SFT-ZH-')[1].split('_financial')[0]
        print(f"{name:<15} {s['avg_em']:>6.3f} {s['avg_f1']:>6.3f} "
              f"{s.get('avg_judge_correctness',0):>8.3f} "
              f"{s.get('avg_judge_faithfulness',0):>8.3f} "
              f"{s.get('avg_judge_ctx_precision',0):>8.3f}")
PYEOF

echo ""
echo "=== 全部完成 $(date) ==="
