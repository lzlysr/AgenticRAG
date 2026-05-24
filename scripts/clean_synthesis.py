#!/usr/bin/env python3
"""清洗合成 QA：过滤 trivial、去重、统计质量

用法:
  # 默认清洗
  python scripts/clean_synthesis.py \
    --input data/financial_all/synthesis_v2/multihop_results.jsonl \
    --output data/financial_all/synthesis_v2/multihop_clean.jsonl

  # 只统计不写出
  python scripts/clean_synthesis.py \
    --input data/financial_all/synthesis_v2/multihop_results.jsonl \
    --dry-run

过滤规则:
  1. 3hop+ 最后一跳是简单查表（trivial prefix + 纯数字答案）
  2. 所有跳都是简单查表
  3. question 前缀去重（前80字符相同视为重复）
  4. chunk overlap 去重（≥80% chunk_id 重叠）
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict


# ============================================================
# Trivial detection
# ============================================================

TRIVIAL_PREFIXES_EN = [
    "what is the total", "what was the total",
    "what is the ending", "what was the ending",
    "what is the beginning", "what was the beginning",
    "what is the amount", "what was the amount",
    "what is the net", "what was the net",
    "what is the subtotal", "what was the subtotal",
    "what is the balance", "what was the balance",
    "what was the beginning balance", "what is the beginning balance",
    "what is the ending balance", "what was the ending balance",
]

TRIVIAL_PREFIXES_ZH = [
    "总共是多少", "总计是多少", "合计是多少",
    "期末余额是多少", "期初余额是多少",
    "金额是多少", "净额是多少",
    "小计是多少", "余额是多少",
]

TRIVIAL_KEYWORDS_EN = [
    "page reference", "page number", "title of the report",
    "name of the report", "what page",
]

TRIVIAL_KEYWORDS_ZH = [
    "页码", "第几页", "报告标题", "报告名称",
]

# 合并中英文（兼容两种语言的数据）
TRIVIAL_PREFIXES = TRIVIAL_PREFIXES_EN + TRIVIAL_PREFIXES_ZH
TRIVIAL_KEYWORDS = TRIVIAL_KEYWORDS_EN + TRIVIAL_KEYWORDS_ZH


def _is_numeric_answer(answer: str) -> bool:
    """答案是否为纯数字（允许逗号、小数点、货币符号）"""
    cleaned = answer.lower().strip()
    for rm in [",", ".", " ", "元", "yuan", "rmb", "hk$", "hkd", "$", "¥",
               "万", "亿", "千", "百", "million", "billion", "thousand"]:
        cleaned = cleaned.replace(rm, "")
    return cleaned.lstrip("-").isdigit() and len(cleaned) > 0


def is_trivial_hop(hop: dict) -> bool:
    """判断单跳是否为 trivial 查表"""
    q = hop["question"].lower().strip()
    a = hop.get("answer", "")

    # 前缀匹配 + 数字答案
    for pfx in TRIVIAL_PREFIXES:
        if q.startswith(pfx) and (_is_numeric_answer(a) or len(a.strip()) < 5):
            return True

    # 关键词匹配（页码、报告标题等）
    for kw in TRIVIAL_KEYWORDS:
        if kw in q:
            return True

    return False


# ============================================================
# Filters
# ============================================================

def filter_trivial_last_hop(results: list) -> tuple[list, list]:
    """过滤 3hop+ 最后一跳 trivial 的 QA"""
    keep, removed = [], []
    for r in results:
        if r["hop_count"] >= 3 and is_trivial_hop(r["hops"][-1]):
            removed.append(r)
        else:
            keep.append(r)
    return keep, removed


def filter_all_trivial(results: list) -> tuple[list, list]:
    """过滤所有跳都是 trivial 的 QA"""
    keep, removed = [], []
    for r in results:
        if all(is_trivial_hop(h) for h in r["hops"]):
            removed.append(r)
        else:
            keep.append(r)
    return keep, removed


def dedup_by_question(results: list, prefix_len: int = 80) -> tuple[list, list]:
    """按 question 前缀去重"""
    keep, removed = [], []
    seen = set()
    for r in results:
        key = r["question"][:prefix_len].lower().strip()
        if key in seen:
            removed.append(r)
        else:
            seen.add(key)
            keep.append(r)
    return keep, removed


def dedup_by_chunk_overlap(results: list, threshold: float = 0.8) -> tuple[list, list]:
    """按 chunk_id 集合重叠去重"""
    keep, removed = [], []
    seen_sets = []
    for r in results:
        chunks = {h["chunk_id"] for h in r["hops"] if h.get("chunk_id")}
        is_dup = False
        for prev in seen_sets:
            if not chunks or not prev:
                continue
            overlap = len(chunks & prev) / max(len(chunks), len(prev))
            if overlap >= threshold:
                is_dup = True
                break
        if is_dup:
            removed.append(r)
        else:
            keep.append(r)
            seen_sets.append(chunks)
    return keep, removed


# ============================================================
# Stats
# ============================================================

def print_stats(results: list, label: str = ""):
    n = len(results)
    if n == 0:
        print(f"  [{label}] 空")
        return

    types = Counter(f'{r["hop_count"]}hop_{r["qa_type"]}' for r in results)
    inf = sum(c for t, c in types.items() if "inference" in t)
    comp = sum(c for t, c in types.items() if "comparison" in t)
    alias_counts = [len(r.get("answer_aliases", [])) for r in results]

    hop3plus = [r for r in results if r["hop_count"] >= 3]
    trivial_last = sum(1 for r in hop3plus if is_trivial_hop(r["hops"][-1]))

    print(f"\n  [{label}] {n} 条")
    print(f"  类型分布:")
    for t, c in sorted(types.items()):
        print(f"    {t}: {c} ({c/n*100:.1f}%)")
    print(f"  inference: {inf} ({inf/n*100:.1f}%) | comparison: {comp} ({comp/n*100:.1f}%)")
    print(f"  alias: avg={sum(alias_counts)/n:.1f}, max={max(alias_counts)}")
    if hop3plus:
        print(f"  3hop+ trivial last hop: {trivial_last}/{len(hop3plus)} ({trivial_last/len(hop3plus)*100:.1f}%)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="清洗合成 QA")
    parser.add_argument("--input", required=True, help="输入 JSONL")
    parser.add_argument("--output", default=None, help="输出 JSONL（不指定则为 input 同目录 multihop_clean.jsonl）")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写出")
    parser.add_argument("--removed-output", default=None, help="保存被过滤的 QA（可选）")
    args = parser.parse_args()

    if args.output is None and not args.dry_run:
        import os
        d = os.path.dirname(args.input)
        args.output = os.path.join(d, "multihop_clean.jsonl")

    # Load
    results = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    print(f"加载: {len(results)} 条")
    print_stats(results, "原始")

    all_removed = []

    # Filter 1: 3hop+ trivial last hop
    results, removed = filter_trivial_last_hop(results)
    all_removed.extend(removed)
    print(f"\n过滤 3hop+ trivial last hop: -{len(removed)} 条")

    # Filter 2: all hops trivial
    results, removed = filter_all_trivial(results)
    all_removed.extend(removed)
    print(f"过滤全跳 trivial: -{len(removed)} 条")

    # Filter 3: question dedup
    results, removed = dedup_by_question(results)
    all_removed.extend(removed)
    print(f"question 前缀去重: -{len(removed)} 条")

    # Filter 4: chunk overlap dedup
    results, removed = dedup_by_chunk_overlap(results)
    all_removed.extend(removed)
    print(f"chunk overlap 去重: -{len(removed)} 条")

    # Re-index IDs
    for i, r in enumerate(results):
        r["id"] = f"mhop_{i:06d}"

    print_stats(results, "清洗后")
    print(f"\n总计: {len(results)} 条可用 (过滤 {len(all_removed)} 条, {len(all_removed)/(len(results)+len(all_removed))*100:.1f}%)")

    if not args.dry_run:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写出: {args.output}")

        if args.removed_output:
            with open(args.removed_output, "w", encoding="utf-8") as f:
                for r in all_removed:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"过滤样本: {args.removed_output}")


if __name__ == "__main__":
    main()
