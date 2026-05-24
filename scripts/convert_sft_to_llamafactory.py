#!/usr/bin/env python3
"""将 SFT 数据转换为 LLaMA-Factory sharegpt 格式

三种格式都转换：
- sft_react.jsonl → 解析 text 字段为 messages
- sft_plan_execute.jsonl → 已有 messages，直接适配
- sft_function_call.jsonl → 已有 messages，适配 function_call → content

用法：
  python scripts/convert_sft_to_llamafactory.py
"""
import json
import os
import re
import sys

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
        "/share/project/common/models/models/Qwen/Qwen3-4B", trust_remote_code=True
    )

    count = 0
    with open(input_path) as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            msgs = d.get("messages", [])
            tools = d.get("tools", [])
            if not msgs:
                continue

            # 用 tokenizer 生成完整文本（含 # Tools 和 <tool_response> 格式）
            text = tok.apply_chat_template(
                msgs, tools=tools, tokenize=False,
            )

            # 拆回 messages 给 LlamaFactory sharegpt 格式
            # 按 <|im_start|> 和 <|im_end|> 分割
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


def convert_plan_execute(input_path: str, output_path: str):
    """Plan-Execute messages → sharegpt（已有 messages，直接写）"""
    count = 0
    with open(input_path) as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            msgs = d.get("messages", [])
            if msgs:
                fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                count += 1

    print(f"  plan_execute: {count} 条 → {output_path}")
    return count


def convert_function_call(input_path: str, output_path: str):
    """Function Calling → sharegpt（将 function_call 和 function role 转为文本）"""
    count = 0
    with open(input_path) as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            msgs = d.get("messages", [])
            converted = []

            for m in msgs:
                role = m["role"]

                if role == "assistant" and m.get("function_call"):
                    # 将 function_call 转为文本
                    fc = m["function_call"]
                    content = f'<tool_call>{{"name": "{fc["name"]}", "arguments": {fc["arguments"]}}}</tool_call>'
                    converted.append({"role": "assistant", "content": content})
                elif role == "function":
                    # function → tool
                    converted.append({"role": "tool", "content": m.get("content", "")})
                else:
                    content = m.get("content", "")
                    if content:
                        converted.append({"role": role, "content": content})

            if converted:
                fout.write(json.dumps({"messages": converted}, ensure_ascii=False) + "\n")
                count += 1

    print(f"  function_call: {count} 条 → {output_path}")
    return count


def register_datasets(out_dir: str):
    """注册到 LLaMA-Factory dataset_info.json"""
    llama_factory_dir = os.environ.get("LLAMA_FACTORY", "./LLaMA-Factory")
    info_path = os.path.join(llama_factory_dir, "data", "dataset_info.json")

    with open(info_path) as f:
        info = json.load(f)

    datasets = {
        "financial_agent_zh_react": "lf_react.jsonl",
        "financial_agent_zh_plan_exec": "lf_plan_execute.jsonl",
        "financial_agent_zh_func_call": "lf_function_call.jsonl",
    }

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
        os.path.join(SFT_DIR, "sft_react.jsonl"),
        os.path.join(OUT_DIR, "lf_react.jsonl"),
    )
    convert_plan_execute(
        os.path.join(SFT_DIR, "sft_plan_execute.jsonl"),
        os.path.join(OUT_DIR, "lf_plan_execute.jsonl"),
    )
    convert_function_call(
        os.path.join(SFT_DIR, "sft_function_call.jsonl"),
        os.path.join(OUT_DIR, "lf_function_call.jsonl"),
    )

    # 注册到 LLaMA-Factory
    register_datasets(OUT_DIR)

    # 预览
    print("\n[preview] ReAct 格式第 1 条:")
    with open(os.path.join(OUT_DIR, "lf_react.jsonl")) as f:
        sample = json.loads(f.readline())
    for m in sample["messages"][:4]:
        print(f"  [{m['role']}] {m['content'][:80]}...")


if __name__ == "__main__":
    main()
