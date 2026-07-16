#!/usr/bin/env python3
"""将 ReAct SFT 数据转换为 LLaMA-Factory sharegpt 格式

- sft_react_zh.jsonl → 用 Qwen3 chat template 转成 lf_react_zh.jsonl

用法：
  python scripts/convert_sft_to_llamafactory.py
"""
import json
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_HUB

SFT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "financial_eval", "sft")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "financial_eval", "sft_zh_llamafactory")


def convert_react(input_path: str, output_path: str):
    """ReAct messages + tools → 用 Qwen3 tokenizer apply_chat_template 生成训练文本

    确保 SFT 训练文本和 GRPO rollout tokenization 100% 一致。
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_HUB, "Qwen3-4B"), trust_remote_code=True
    )

    count = 0
    with open(input_path) as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            msgs = d.get("messages", [])
            tools = d.get("tools", [])
            if not msgs:
                continue
            
            # 把 messages + tools schema 渲染成 Qwen3 官方聊天格式 得到的text是str
            # 把工具 schema（原始的 tools 字段） 注入到 system 里，变成 Qwen3 真正吃到的文本
            # 用 tokenizer 生成完整文本（含 # Tools 和 <tool_response> 格式）
            # 保证：训练阶段 SFT tokenizer 和 推理阶段 Qwen3 chat template 一致
            text = tok.apply_chat_template(
                msgs, tools=tools, tokenize=False,
            )

            # 拆回 messages 给 LlamaFactory sharegpt 格式
            # 把 Qwen3 官方 tool-call 模板先固化进文本里，再伪装回 LLaMA-Factory 能吃的 sharegpt messages。
            # 按 <|im_start|> 和 <|im_end|> 分割
            # 这样做的真实目的有两个：
            # 1. 保留 Qwen3 官方 tools 注入格式:
            # 让 system 里带上 Qwen3 tokenizer 生成的 # Tools / tool schema 文本。
            # 2. 仍然兼容 LLaMA-Factory sharegpt 数据入口
            # 让 LLaMA-Factory 继续按 messages 训练，并且只对 assistant 部分算 loss。
            parts = re.split(r'<\|im_start\|>(system|user|assistant)\n', text)
            messages = []
            i = 1
            while i < len(parts) - 1:
                role = parts[i].strip()
                content = parts[i + 1]
                # 去掉尾部的 <|im_end|> 和后续空白
                content = re.sub(r'<\|im_end\|>\s*$', '', content).strip()
                if role and content:
                    messages.append({"role": role, "content": content})
                i += 2

            if messages:
                fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1

    print(f"  react: {count} 条 → {output_path}")
    return count


def register_datasets(out_dir: str):
    """注册到 LLaMA-Factory dataset_info.json"""
    llama_factory_dir = os.environ.get("LLAMA_FACTORY", "./LLaMA-Factory")
    info_path = os.path.join(llama_factory_dir, "data", "dataset_info.json")

    with open(info_path) as f:
        info = json.load(f)

    datasets = {"financial_agent_zh_react": "lf_react_zh.jsonl"}

    for name, filename in datasets.items():
        info[name] = {
            "file_name": os.path.join(out_dir, filename),
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "observation_tag": "tool",
                "system_tag": "system",
            }
        }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n注册 {len(datasets)} 个数据集到 {info_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[convert] SFT → LLaMA-Factory sharegpt 格式")

    convert_react(
        os.path.join(SFT_DIR, "sft_react_zh.jsonl"),
        os.path.join(OUT_DIR, "lf_react_zh.jsonl"),
    )

    # 注册到 LLaMA-Factory
    register_datasets(OUT_DIR)

    # 预览
    print("\n[preview] ReAct 格式第 1 条:")
    with open(os.path.join(OUT_DIR, "lf_react_zh.jsonl")) as f:
        sample = json.loads(f.readline())
    for m in sample["messages"][:4]:
        print(f"  [{m['role']}] {m['content'][:80]}...")


if __name__ == "__main__":
    main()
