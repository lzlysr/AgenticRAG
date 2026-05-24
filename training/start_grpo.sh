#!/bin/bash
# Agentic GRPO v15e：Stage 3（检索质量优先）
#
# 三阶段策略：
#   Stage 1 (v11e): grounding-first (Faith×0.30) → 学会 grounding
#   Stage 2 (v14e): correctness-heavy (Corr×0.40) → 强化 correctness
#   Stage 3 (v15e): 检索质量优先 (hop_pr×0.30) → 提升 CtxP
#
# Base model: v14e step30 merged（Judge_C=0.334, Faith=0.199, CtxP=0.251）
# Reward: v9a（hop_pr×0.30 + Faith×0.25 + Corr×0.25 + grounded×0.10 + format×0.10）
# 1 epoch（小幅微调，不破坏已有的 Judge_C 和 Faith）
#
# 用法: bash scripts/start_grpo_v15e.sh
set -x

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
VERL_DIR=${VERL_DIR:?"请设置 VERL_DIR 指向 verl 安装目录"}

# ── 检查 retrieval server ──
if ! curl -s http://localhost:8790/health | grep -q "ok"; then
    echo "ERROR: Retrieval server not running! Run scripts/start_retrieval_server.sh first."
    exit 1
fi
echo "Retrieval server OK"

# ── GPU 清理 ──
cleanup_gpu() {
    ray stop --force 2>/dev/null
    pkill -9 -f 'verl\.trainer|python3 -m verl|VLLM::|vllm' 2>/dev/null
    pkill -9 -f 'ray::|raylet|gcs_server|ray-dashboard|dashboardagent|default_worker|monitor\.py|runtime_env' 2>/dev/null
    sleep 3
}
cleanup_gpu

# Stage 3 base = v14e step30 merged
MODEL_PATH=${PROJECT_DIR}/training/outputs/qwen3-4b-grpo-v14e-step30-merged
DATA_DIR=${PROJECT_DIR}/data/financial_eval
REWARD_FN=${PROJECT_DIR}/training/reward_agentic_rag_v9a.py
TOOL_CONFIG=${PROJECT_DIR}/training/config/financial_tool_config.yaml
OUTPUT_DIR=${PROJECT_DIR}/training/outputs/grpo_agentic_v15e
LOG=${PROJECT_DIR}/logs/grpo_agentic_v15e.log

mkdir -p ${PROJECT_DIR}/logs

# 清理旧输出
rm -rf ${OUTPUT_DIR}

export PATH=${VERL_PYTHON_DIR:-$(dirname $(which python))}:$PATH
export ATTN_BACKEND=flash_attn
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $VERL_DIR

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/grpo_agentic_train.parquet \
    data.val_files=${DATA_DIR}/grpo_agentic_val.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=7 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path=${REWARD_FN} \
    reward.custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=cv_agenticrag \
    trainer.experiment_name=qwen3_4b_grpo_agentic_v15e \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=3 \
    trainer.total_epochs=2 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    2>&1 | tee ${LOG}
