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
# 2 epochs（小幅微调，不破坏已有的 Judge_C 和 Faith）
#
# 用法: bash training/start_grpo.sh
set -euo pipefail
# DEBUG_GRPO 默认是 0，表示不打印每条命令；如果设置为 1，则打印每条命令。
if [[ "${DEBUG_GRPO:-0}" == "1" ]]; then
    set -x
fi

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)} # 自动计算根目录
VERL_DIR=${PROJECT_DIR}/verl # 指向verl安装目录

# ── 检查 retrieval server ──
if ! curl -s http://localhost:8790/health | grep -q "ok"; then
    echo "ERROR: Retrieval server not running! Run scripts/start_retrieval_server.sh first."
    exit 1
fi
echo "Retrieval server OK"

# ── GPU 清理（只清理lzl用户） ──
cleanup_gpu() {
    local cleanup_user="lzl"
    if [[ "$(id -un)" != "${cleanup_user}" ]]; then
        echo "ERROR: GPU cleanup must be run as ${cleanup_user}; current user is $(id -un)."
        return 1
    fi

    ray stop --force 2>/dev/null || true
    pkill -9 -u "${cleanup_user}" -f 'verl\.trainer|python3 -m verl' 2>/dev/null || true
    pkill -9 -u "${cleanup_user}" -f 'ray::|raylet|gcs_server|ray-dashboard|dashboardagent|default_worker|monitor\.py|runtime_env' 2>/dev/null || true
    sleep 3
}
cleanup_gpu || exit 1

# Stage 3 base = v14e step30 merged
MODEL_NAME=${MODEL_NAME:-Qwen3-4B-grpo-zh}
MODEL_PATH=${PROJECT_DIR}/models/Qwen3-4B-sft-zh
DATA_DIR=${PROJECT_DIR}/data/financial_eval
REWARD_FN=${PROJECT_DIR}/training/reward_agentic_rag.py
TOOL_CONFIG=${PROJECT_DIR}/training/config/financial_tool_config.yaml
OUTPUT_DIR=${PROJECT_DIR}/training/outputs/${MODEL_NAME}
LOG=${PROJECT_DIR}/logs/${MODEL_NAME}.log
RUN_LOG=${OUTPUT_DIR}/main_ppo.log

mkdir -p ${PROJECT_DIR}/logs

# 清理旧输出
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

export PATH=${VERL_PYTHON_DIR:-$(dirname $(which python))}:$PATH # python环境设置
export ATTN_BACKEND=flash_attn # 没用，因为 verl 的 FSDP 模型加载默认值已经是"flash_attention_2"
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH:-} # 项目根目录加入 Python 模块搜索路径
export CUDA_VISIBLE_DEVICES=0,3

cd $VERL_DIR

set +e
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/grpo_agentic_train.parquet \
    data.val_files=${DATA_DIR}/grpo_agentic_val.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
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
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.enforce_eager=True \
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
    trainer.project_name=AGENTICRAG \
    trainer.experiment_name=${MODEL_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=3 \
    trainer.total_epochs=2 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    2>&1 | tee ${LOG} ${RUN_LOG} | python ${PROJECT_DIR}/scripts/filter_grpo_console.py
train_status=${PIPESTATUS[0]}
set -e

python ${PROJECT_DIR}/scripts/plot_grpo_log.py \
    --log ${LOG} \
    --out-dir ${OUTPUT_DIR} || true

exit ${train_status}

# ── main_ppo 参数说明 ───────────────────────────────────────────
# python3 -m verl.trainer.main_ppo
#   启动 verl 的 PPO/GRPO 统一训练入口；具体算法由下面的 Hydra 参数覆盖决定。
#
# 算法：
# algorithm.adv_estimator=grpo
#   使用 GRPO 优势估计。同一问题采样多条回答，按组内相对奖励计算 advantage，不训练 Critic。
# algorithm.use_kl_in_reward=False
#   不把 KL 惩罚直接减到环境 reward 中；当前改由 actor.use_kl_loss=True 在 Actor loss 中约束。
#
# 数据：
# data.train_files=.../grpo_agentic_train.parquet
#   GRPO 训练集路径，每条记录包含聊天 prompt、标准答案、gold chunks 和工具配置等字段。
# data.val_files=.../grpo_agentic_val.parquet
#   验证集路径，用于训练期间定期 rollout 和评估。
# data.train_batch_size=32
#   每个训练 batch 取 32 个问题；rollout.n=4 时最多生成 32x4=128 条候选轨迹。
# data.max_prompt_length=1024
#   输入 prompt 的最大 token 数。
# data.max_response_length=4096
#   一条 rollout 的总响应预算；assistant 输出和工具 observation 都会占用该预算。
# data.dataloader_num_workers=0
#   不额外启动 PyTorch DataLoader 子进程。金融 GRPO 数据很小，瓶颈在 rollout/update；
#   设为 0 可减少训练结束时 DataLoader worker 被 Ray 清理杀掉的尾部噪声。
# data.filter_overlong_prompts=True
#   训练前过滤 token 数超过 max_prompt_length 的 prompt。
# data.truncation=error
#   遇到仍需截断的输入时直接报错，避免静默截断问题内容。
# data.return_raw_chat=True
#   保留原始消息列表，而不是只返回模板化文本，供多轮 Tool Agent 循环继续追加消息。
#
# 模型和混合引擎：
# actor_rollout_ref.model.path=${MODEL_PATH}
#   Actor、rollout 和 reference model 使用的初始模型目录。
# actor_rollout_ref.hybrid_engine=True
#   Actor 训练与 vLLM rollout 在同一组 Ray worker/GPU 上分时复用资源；当前 PPO trainer 要求开启。
# actor_rollout_ref.model.use_remove_padding=True
#   计算时移除 padding token，减少无效显存和算力开销。
# actor_rollout_ref.model.enable_gradient_checkpointing=True
#   用反向传播时重算部分前向结果换取更低的激活显存占用。
#
# Actor 更新：
# actor_rollout_ref.actor.optim.lr=5e-6
#   Actor 优化器学习率。
# actor_rollout_ref.actor.ppo_mini_batch_size=16
#   一次 Actor 参数更新使用多少个 prompt 所对应的 rollout 数据。实际是16 × 4 = 64条轨迹
#   所以 128 条轨迹会拆成：128 ÷ 64 = 2个PPO mini-batch。
#   使用两张 GPU，FSDP 再将每个全局 mini-batch 分给两张卡：64 ÷ 2 = 每张GPU 32条轨迹
#   又因为： ppo_micro_batch_size_per_gpu=1，每张 GPU 每次只前后向一条轨迹，累计 32 次梯度后执行一次 #   optimizer.step()。因此，每个 trainer step 内部实际是：
#   2个PPO mini-batch 每个mini-batch执行1次optimizer.step() = 2次Actor参数更新

# 外层训练 batch：32 prompts / 128 trajectories

# PPO mini-batch 1：
# 16 prompts × 4 = 64 trajectories
# → 做一次 optimizer step

# PPO mini-batch 2：
# 16 prompts × 4 = 64 trajectories
# → 再做一次 optimizer step

# actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
#   每张 GPU 单次前后向处理 1 条样本，通过梯度累积组成 mini-batch。
# actor_rollout_ref.actor.use_kl_loss=True
#   在 Actor loss 中加入相对于 reference model 的 KL 约束。
# actor_rollout_ref.actor.kl_loss_coef=0.05
#   Actor loss 中 KL 项的权重；越大越不允许新策略偏离 reference model。
# actor_rollout_ref.actor.kl_loss_type=low_var_kl
#   使用低方差的 k3 KL 估计器计算 token 级 KL loss。
# actor_rollout_ref.actor.entropy_coeff=0
#   不额外加入熵奖励；探索主要来自 rollout 采样。
# actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
#   Actor FSDP 按 bf16 加载模型权重，避免默认 fp32 让 4B 模型常驻显存翻倍。
# actor_rollout_ref.actor.fsdp_config.param_offload=True
#   Actor 参数允许卸载到 CPU，降低与 vLLM rollout 共卡时的显存峰值。
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
#   优化器状态允许卸载到 CPU，进一步降低训练阶段显存压力。
#
# vLLM Rollout 和工具调用：
# actor_rollout_ref.rollout.name=vllm
#   使用 vLLM 作为 rollout 推理后端。
# actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
#   每条样本默认进入 verl 的 Tool Agent 多轮循环，而不是普通单轮生成。必须显示设置，
#   否则模型不会触发多轮 tool calling
# actor_rollout_ref.rollout.tensor_model_parallel_size=1
#   一个 rollout 模型实例不做跨 GPU 张量并行，每个实例使用 1 张 GPU。
# actor_rollout_ref.rollout.load_format=safetensors
#   让 vLLM 直接从模型目录预加载 base model，避免 hybrid engine 首轮再同步完整 base 权重。
# actor_rollout_ref.rollout.layered_summon=True
#   LoRA 同步时分层收集 adapter 参数，降低 FSDP state_dict 的瞬时显存峰值。
# actor_rollout_ref.rollout.gpu_memory_utilization=0.5
#   vLLM 引擎用于模型执行和 KV cache 的目标显存比例为 50%。该值需要给 FSDP 权重同步留出余量。
# actor_rollout_ref.rollout.max_model_len=5120
#   vLLM 单条序列的最大上下文长度，等于 max_prompt_length + max_response_length。
# actor_rollout_ref.rollout.max_num_seqs=16
#   vLLM 单个引擎允许的最大并发序列数；默认 1024 会在 sampler warmup 阶段制造过大的 dummy batch。
# actor_rollout_ref.rollout.enforce_eager=True
#   禁用 vLLM CUDA graph capture，降低和 FSDP 权重同步共卡时的显存峰值，代价是 rollout 速度变慢。
# actor_rollout_ref.rollout.multi_turn.enable=True
#   开启 assistant -> tool -> assistant 的多轮交互。
# actor_rollout_ref.rollout.multi_turn.max_assistant_turns=7
#   每条轨迹最多生成 7 个 assistant turn，达到上限后停止继续调用模型。
# actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG}
#   指定四种金融检索工具的 schema、实现类、索引路径和 retrieval server 地址。
# actor_rollout_ref.rollout.multi_turn.format=hermes
#   按 Hermes 工具调用协议解析模型生成的 tool call，并把工具结果追加回消息历史。
# actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024
#   单次工具响应最多保留 1024 个字符；超出后按 verl 的截断策略裁剪。
# actor_rollout_ref.rollout.n=4
#   每个问题采样 4 条候选轨迹，作为 GRPO 组内奖励比较的样本组。
# actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
#   rollout 策略重新计算 token log probability 时，每张 GPU 的 micro-batch 大小。
#
# Reference model：
# actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
#   Reference model 计算 token log probability/KL 时，每张 GPU 的 micro-batch 大小。
# actor_rollout_ref.ref.fsdp_config.param_offload=True
#   Reference model 参数允许卸载到 CPU，降低其常驻 GPU 显存占用。
#
# 奖励函数：
# reward.custom_reward_function.path=${REWARD_FN}
#   自定义奖励 Python 文件路径，verl 会动态导入该模块。
# reward.custom_reward_function.name=compute_score
#   指定模块中真正被调用的奖励入口函数名。
#
# Trainer：
# trainer.critic_warmup=0
#   不等待 Critic 预热；GRPO 不依赖单独的价值 Critic，可以立即更新 Actor。
# trainer.logger='["console"]'
#   只把训练指标输出到终端，不上传 WandB 等外部平台。
# trainer.project_name=AGENTICRAG
#   日志系统中的项目名。
# trainer.experiment_name=Qwen3-4B-grpo-zh
#   本次实验名，用于区分不同训练配置。
# trainer.n_gpus_per_node=2
#   每个节点使用 2 张 GPU，应与 CUDA_VISIBLE_DEVICES 暴露的 GPU 数量一致。
# trainer.nnodes=1
#   使用 1 个训练节点，即单机训练。
# trainer.save_freq=5
#   每 5 个训练 step 保存一次 checkpoint。
# trainer.test_freq=3
#   每 3 个训练 step 在验证集上执行一次评估。
# trainer.total_epochs=2
#   完整遍历训练集 2 轮。
# trainer.default_local_dir=${OUTPUT_DIR}
#   checkpoint、训练状态和其他产物的本地输出目录。
# 2>&1 | tee ${LOG}
#   合并标准错误与标准输出，同时显示在终端并写入日志文件。
