#!/usr/bin/env python3
"""用 LLM 增强 answer_aliases：补充短形式、中文翻译、无单位数值等"""
import json
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.client import judge_chat_json

INPUT_FILE = "data/financial_eval/qa_pairs.json"
OUTPUT_FILE = "data/financial_eval/qa_pairs_v2.json"
WORKERS = 15

ALIAS_PROMPT_EN = """You are helping improve answer matching for a QA evaluation system.

Given a question and its gold answer, generate additional answer aliases that a correct model might produce. Focus on forms that are semantically equivalent but differ in format.

**Question:** {question}
**Gold Answer:** {gold}
**Existing Aliases (do NOT repeat these):** {existing}

Generate 5-10 NEW aliases following these rules:
1. If the answer is in English, add Chinese translations (and vice versa)
2. If the answer contains a number with units (e.g., "31,957,697.48元"), add the bare number without units
3. If the answer is a comparison result like "X更高" or "X is larger", add just the entity name "X" without the comparison suffix
4. If the answer is a full sentence, add the key phrase/entity only
5. If the answer contains a company's full name, add common abbreviations (e.g., "永辉超市股份有限公司" → "永辉超市")
6. Add variations with/without parentheses, with/without "的" particles

Respond in JSON: {{"aliases": ["alias1", "alias2", ...]}}"""

ALIAS_PROMPT_ZH = """你正在帮助改进 QA 评测系统的答案匹配。

给定一个问题及其标准答案，生成模型可能产出的等价答案别名。重点关注语义等价但格式不同的形式。

**问题：** {question}
**标准答案：** {gold}
**已有别名（不要重复这些）：** {existing}

生成 5-10 个新的别名，遵循以下规则：
1. 如果答案是英文，添加中文翻译（反之亦然）
2. 如果答案包含带单位的数字（如"31,957,697.48元"），添加不带单位的纯数字
3. 如果答案是比较结果（如"X更高"），只添加实体名"X"
4. 如果答案是完整句子，只添加关键短语/实体
5. 如果答案包含公司全称，添加常用简称（如"永辉超市股份有限公司"→"永辉超市"）
6. 添加有/无括号、有/无"的"等变体

以 JSON 格式回复：{{"aliases": ["别名1", "别名2", ...]}}"""


def _get_alias_prompt(lang=None):
    if lang is None:
        import config
        lang = getattr(config, "PROMPT_LANG", "en")
    return ALIAS_PROMPT_ZH if lang == "zh" else ALIAS_PROMPT_EN


def gen_aliases(qa):
    """为单条 QA 生成增强 aliases"""
    existing = qa.get("answer_aliases", [])
    prompt = _get_alias_prompt().format(
        question=qa["final_question"][:500],
        gold=qa["final_answer"],
        existing=json.dumps(existing[:5], ensure_ascii=False),
    )
    result = judge_chat_json(prompt)
    if result and "aliases" in result:
        return result["aliases"]
    return []


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"[alias] 总样本: {len(data)}, workers: {WORKERS}")
    done = [0]
    t0 = time.time()

    def _process(qa):
        new_aliases = gen_aliases(qa)
        existing = set(qa.get("answer_aliases", []))
        added = [a for a in new_aliases if a not in existing]
        qa["answer_aliases"] = list(existing) + added
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(data):
            elapsed = time.time() - t0
            rate = done[0] / elapsed * 60
            print(f"  [{done[0]}/{len(data)}] {rate:.0f}/min, 新增 {len(added)} aliases")
        return len(added)

    total_added = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_process, qa) for qa in data]
        for f in as_completed(futures):
            try:
                total_added += f.result()
            except Exception as e:
                print(f"  Error: {e}")

    # 统计
    alias_counts = [len(qa["answer_aliases"]) for qa in data]
    print(f"\n[alias] 完成！新增 {total_added} 条 aliases")
    print(f"[alias] Aliases 数量: min={min(alias_counts)}, max={max(alias_counts)}, avg={sum(alias_counts)/len(alias_counts):.1f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[alias] 已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
