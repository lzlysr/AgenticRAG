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
SFT_PATH = "data/financial_eval/sft/sft_react.jsonl"
QA_PATH = "data/financial_eval/train_qa_pairs_zh.json"  # 含 answer_aliases
OUTPUT_TRAIN = "data/financial_eval/grpo_agentic_train.parquet"
OUTPUT_VAL = "data/financial_eval/grpo_agentic_val.parquet"
VAL_RATIO = 0.1


def _build_aliases_index(qa_path):
    """从 QA 数据构建 gold_answer → (aliases, gold_chunks) 索引"""
    with open(qa_path) as f:
        qa_data = json.load(f)
    index = {}
    for item in qa_data:
        gold = item.get("final_answer", "")
        aliases = item.get("answer_aliases", [])
        # 提取每个 hop 的 doc_chunk_id
        chunks = [hop.get("doc_chunk_id", "") for hop in item.get("hops", []) if hop.get("doc_chunk_id")]
        if gold:
            index[gold] = {"aliases": aliases, "gold_chunks": chunks}
    print(f"[agentic-grpo] Loaded {len(index)} QA items with aliases + chunks")
    return index


def build_dataset():
    """从 SFT 数据提取 question/gold，关联 QA 数据的 answer_aliases 和 gold_chunks"""
    qa_index = _build_aliases_index(QA_PATH)

    records = []
    matched = 0
    with open(SFT_PATH) as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            question = entry["question"]
            gold = entry["gold"]
            subset = entry.get("subset", "unknown")
            hop_count = int(entry.get("hop_count", 1))

            # 从 QA 数据关联 aliases 和 gold_chunks（用 gold answer 匹配）
            qa_info = qa_index.get(gold, {})
            aliases = qa_info.get("aliases", [])
            gold_chunks = qa_info.get("gold_chunks", [])
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
    print("[agentic-grpo] 生成训练数据（从 SFT 797 条）...")

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
