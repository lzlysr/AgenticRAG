#!/usr/bin/env python3
"""将金融 QA 数据的英文/混杂部分翻译为中文

用法：
  python -u scripts/translate_qa.py
  python -u scripts/translate_qa.py --input data/financial_eval/train_qa_pairs.json --output data/financial_eval/train_qa_pairs_zh.json
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.client import judge_chat_json

WORKERS = 15

TRANSLATE_PROMPT = """你是一个专业的金融文档翻译器。将以下金融问答的英文部分翻译为中文。

## 翻译规则
1. 保留所有中文公司名、专有名词不变（如"步步高商业连锁股份有限公司"、"永辉超市"）
2. 保留所有数字、金额、百分比不变（如"3,851,242,298元"、"24.45%"）
3. 只翻译英文部分为自然的中文
4. 比较类问题保留"哪个更高/更大/更多"的中文句式
5. 如果问题已经是纯中文，原样返回
6. 如果答案是纯数字/金额/中文实体名，原样返回
7. 翻译后的问题必须保留所有原始信息，不能丢失任何细节

## 输入
**问题（Question）：** {question}
**答案（Answer）：** {answer}

## 输出格式
返回 JSON：{{"question_zh": "翻译后的中文问题", "answer_zh": "翻译后的中文答案"}}"""


def _is_chinese(text: str) -> bool:
    """判断文本是否已经是纯中文（或纯数字/符号）"""
    # 去掉数字、标点、空格后，检查剩余字符是否都是中文
    cleaned = re.sub(r'[\d\s\.,;:!?\-\+\(\)\[\]{}%¥￥$€£/\\=<>"\'\u2018\u2019\u201c\u201d]', '', text)
    if not cleaned:
        return True  # 纯数字/符号
    chinese_chars = sum(1 for c in cleaned if '\u4e00' <= c <= '\u9fff')
    return chinese_chars / len(cleaned) > 0.8


def translate_one(qa: dict) -> dict:
    """翻译单条 QA"""
    question = qa["final_question"]
    answer = qa["final_answer"]

    # 如果已经是中文，跳过
    if _is_chinese(question) and _is_chinese(answer):
        return {"question_zh": question, "answer_zh": answer, "skipped": True}

    prompt = TRANSLATE_PROMPT.format(question=question, answer=answer)
    result = judge_chat_json(prompt)

    if result and "question_zh" in result:
        return {
            "question_zh": result["question_zh"],
            "answer_zh": result.get("answer_zh", answer),
            "skipped": False,
        }
    # fallback: 返回原文
    return {"question_zh": question, "answer_zh": answer, "skipped": True}


def process_file(input_path: str, output_path: str):
    """处理单个文件"""
    with open(input_path) as f:
        data = json.load(f)

    print(f"[translate] 处理 {input_path}: {len(data)} 条")

    done = [0]
    skipped = [0]
    t0 = time.time()
    lock = Lock()

    def _process(qa):
        result = translate_one(qa)

        # 写回
        qa["final_question_zh"] = result["question_zh"]
        qa["final_answer_zh"] = result["answer_zh"]

        with lock:
            done[0] += 1
            if result["skipped"]:
                skipped[0] += 1
            d = done[0]

        if d % 50 == 0 or d == len(data):
            elapsed = time.time() - t0
            rate = d / elapsed * 60
            print(f"  [{d}/{len(data)}] {rate:.0f}/min, skipped={skipped[0]}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_process, qa) for qa in data]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"  Error: {e}")

    elapsed = time.time() - t0
    print(f"[translate] 完成: {done[0]} 条, 跳过 {skipped[0]} 条, 耗时 {elapsed:.0f}s")

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[translate] 保存到 {output_path}")

    # 抽样展示
    print(f"\n[translate] 翻译示例 (前 5 条非跳过):")
    shown = 0
    for qa in data:
        if qa["final_question"] == qa.get("final_question_zh"):
            continue
        print(f"  EN: {qa['final_question'][:80]}")
        print(f"  ZH: {qa['final_question_zh'][:80]}")
        print(f"  A_EN: {qa['final_answer'][:40]}")
        print(f"  A_ZH: {qa['final_answer_zh'][:40]}")
        print()
        shown += 1
        if shown >= 5:
            break


def main():
    parser = argparse.ArgumentParser(description="翻译 QA 数据为中文")
    parser.add_argument("--input", default=None, help="输入文件 (默认跑测试集+训练集)")
    parser.add_argument("--output", default=None, help="输出文件")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "financial_eval")

    if args.input:
        output = args.output or args.input.replace(".json", "_zh.json")
        process_file(args.input, output)
    else:
        # 默认跑测试集 + 训练集
        for name in ["qa_pairs.json", "train_qa_pairs.json"]:
            inp = os.path.join(base_dir, name)
            out = os.path.join(base_dir, name.replace(".json", "_zh.json"))
            if os.path.exists(inp):
                process_file(inp, out)


if __name__ == "__main__":
    main()
