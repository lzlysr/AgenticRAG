#!/bin/bash
# 合并 verl GRPO FSDP checkpoint 到 HuggingFace 格式
# 用法: bash merge_grpo.sh <grpo_output_dir> <step> <target_dir>
# 示例: bash merge_grpo.sh training/outputs/grpo_v4 70 training/outputs/qwen3-4b-grpo-v4-merged
set -e

GRPO_DIR=${1:?用法: bash merge_grpo.sh <grpo_output_dir> <step> <target_dir>}
STEP=${2:?请指定 step，如 70}
TARGET_DIR=${3:?请指定输出目录}

ACTOR_DIR="${GRPO_DIR}/global_step_${STEP}/actor"

if [ ! -f "${ACTOR_DIR}/fsdp_config.json" ]; then
    echo "错误: ${ACTOR_DIR}/fsdp_config.json 不存在"
    exit 1
fi

if [ ! -f "${ACTOR_DIR}/huggingface/config.json" ]; then
    echo "警告: ${ACTOR_DIR}/huggingface/config.json 不存在，尝试从 SFT base 复制..."
    BASE_DIR="$(dirname $(dirname ${GRPO_DIR}))/training/outputs/qwen3-4b-sft-react-merged"
    if [ -f "${BASE_DIR}/config.json" ]; then
        cp "${BASE_DIR}/config.json" "${ACTOR_DIR}/huggingface/"
        echo "已从 ${BASE_DIR} 复制 config.json"
    else
        echo "错误: 找不到 base model config.json"
        exit 1
    fi
fi

export PATH=${VERL_PYTHON_DIR:-$(dirname $(which python))}:$PATH

# 转为绝对路径
ACTOR_DIR=$(cd "${ACTOR_DIR}" && pwd)
mkdir -p "${TARGET_DIR}"
TARGET_DIR=$(cd "${TARGET_DIR}" && pwd)

echo "合并 ${ACTOR_DIR} -> ${TARGET_DIR}"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${ACTOR_DIR}" \
    --target_dir "${TARGET_DIR}"

echo "完成: ${TARGET_DIR}"
