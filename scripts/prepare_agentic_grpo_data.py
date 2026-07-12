#!/usr/bin/env python3
"""生成 Agentic GRPO 训练数据（Search-R1 风格）

将 oracle traces 转为 verl multi-turn agentic 格式：
- prompt: [system, user] 聊天消息（不含 evidence）
- 模型在 rollout 中自主生成 tool_call，环境执行检索返回结果
- reward 基于最终答案质量

用法:
  python scripts/prepare_agentic_grpo_data.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT = "你是一个金融文档问答 Agent。通过搜索相关文档来回答用户的问题。"

# 数据源路径
QA_PATH = "data/financial_eval/train_qa_pairs_zh_clean.json"  # 含 answer_aliases
OUTPUT_TRAIN = "data/financial_eval/grpo_agentic_train.parquet"
OUTPUT_VAL = "data/financial_eval/grpo_agentic_val.parquet"
VAL_RATIO = 0.1


def build_dataset():
    """从 QA 数据提取 question/gold、answer_aliases 和 gold_chunks"""
    with open(QA_PATH) as f:
        qa_data = json.load(f)

    records = []
    matched = 0
    for i, item in enumerate(qa_data):
        question = item["final_question"]
        gold = item["final_answer"]
        subset = item.get("subset", "unknown")
        hop_count = int(item.get("hop_count", 1))

        aliases = item.get("answer_aliases", [])
        # 提取每个 hop 的 doc_chunk_id
        gold_chunks = [hop.get("doc_chunk_id", "") for hop in item.get("hops", []) if hop.get("doc_chunk_id")]
        if not aliases:
            aliases = [gold]
        else:
            matched += 1
            if gold not in aliases:
                aliases = [gold] + aliases

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        record = {
            "data_source": "financial_agentic_rag",
            "prompt": prompt,
            "ability": "multi_hop_qa",
            "reward_model": {
                "ground_truth": {
                    "target": gold,
                    "answer": gold,
                    "question": question,
                    "answer_aliases": aliases,
                    "gold_chunks": gold_chunks,
                    "hop_count": hop_count,
                },
            },
            "extra_info": {
                "index": i,
                "need_tools_kwargs": True,
                "question": question,
                "split": "train",
                "subset": subset,
                "hop_count": hop_count,
                "tools_kwargs": {
                    tool_name: {
                        "create_kwargs": {
                            "ground_truth": gold,
                            "question": question,
                            "data_source": "financial_agentic_rag",
                        }
                    }
                    for tool_name in [
                        "keyword_search",
                        "semantic_search",
                        "graph_search",
                        "hybrid_search",
                    ]
                },
            },
            "metadata": {
                "subset": subset,
                "hop_count": hop_count,
            },
        }
        records.append(record)

    print(f"[agentic-grpo] {len(records)} records, {matched} matched aliases")
    return records


def main():
    print("[agentic-grpo] 生成训练数据（从 QA 797 条数据）...")

    records = build_dataset()

    # train/val split
    import random
    random.seed(42)
    indices = list(range(len(records)))
    random.shuffle(indices)
    val_size = int(len(records) * VAL_RATIO)
    val_indices = set(indices[:val_size])

    train_records = [records[i] for i in range(len(records)) if i not in val_indices]
    val_records = [records[i] for i in range(len(records)) if i in val_indices]
    for record in val_records:
        record["extra_info"]["split"] = "val"

    # 验证格式
    sample = records[0]
    assert isinstance(sample["prompt"], list), "prompt 应为 list"
    assert sample["prompt"][0]["role"] == "system", "第一条应为 system"
    assert sample["extra_info"]["need_tools_kwargs"] is True, "need_tools_kwargs 应为 True"

    df_train = pd.DataFrame(train_records)
    df_val = pd.DataFrame(val_records)
    df_train.to_parquet(OUTPUT_TRAIN)
    df_val.to_parquet(OUTPUT_VAL)
    print(f"[agentic-grpo] train: {len(df_train)} 条 → {OUTPUT_TRAIN}")
    print(f"[agentic-grpo] val:   {len(df_val)} 条 → {OUTPUT_VAL}")


if __name__ == "__main__":
    main()
