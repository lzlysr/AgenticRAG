#!/usr/bin/env python3
"""训练/测试集划分：分层采样保证训测分布一致

domain_multihop_synthesis.py
  -> multihop_merged.jsonl / multihop_results.jsonl  原始合成结果

clean_synthesis.py
  -> multihop_clean.jsonl  规则清洗后

judge_synthesis.py 评分模式
  -> multihop_judged.jsonl  加 judge 字段

judge_synthesis.py --filter-only
  -> multihop_final.jsonl  按 judge 分数过滤后

split_train_test.py
  -> train/test

用法:
  # 默认 80/20 划分
  python scripts/split_train_test.py \
    --input data/financial_all/synthesis_v2/multihop_merged.jsonl \
    --output-dir data/financial_all/synthesis_v2/ \
    --test-ratio 0.2

  # 指定测试集大小
  python scripts/split_train_test.py \
    --input data/financial_all/synthesis_v2/multihop_merged.jsonl \
    --output-dir data/financial_all/synthesis_v2/ \
    --test-size 200

划分策略:
  1. 按 (hop_count, qa_type) 分层采样，保证各类别训测比例一致
  2. 同一 seed chunk 产生的所有 QA 不能同时出现在训练和测试集（防止数据泄露）
  3. 输出统计对比训测分布
"""
import argparse
import json
import random
from collections import Counter, defaultdict


def group_by_seed_chunk(items: list) -> dict[str, list]:
    """按 seed chunk (hop1 的 chunk_id) 分组，同源 QA 归为一组"""
    groups = defaultdict(list)
    for r in items:
        # 用第一跳的 chunk_id 作为 seed 标识
        seed_key = r["hops"][0]["doc_chunk_id"] if r["hops"] else r["id"]
        groups[seed_key].append(r)
    return groups


def stratified_split(items: list, test_ratio: float = 0.2,
                     test_size: int = 0, seed: int = 42) -> tuple[list, list]:
    """分层采样划分训练/测试集

    策略：
    1. 按 seed chunk 分组（防泄露）
    2. 每组标记 stratum = (hop_count, qa_type) 的主类别
    3. 在每个 stratum 内按组随机划分
    """
    random.seed(seed)

    # 按 seed chunk 分组
    groups = group_by_seed_chunk(items)
    print(f"Seed chunk 分组: {len(groups)} 组, 平均 {len(items)/len(groups):.1f} 条/组")

    # 每组标记主 stratum（组内可能有多种 hop/type，取众数）
    # 这里的 stratum 是一个字符串，如 "2hop_inference"，表示该组的主要类型
    # 要做到更精确，需要使用组级优化算法，而不是简单给每组一个主标签。
    group_strata = {}
    for gkey, gitems in groups.items():
        strata = Counter(f"{r['hop_count']}hop_{r['qa_type']}" for r in gitems)
        # most_common(1) 会返回计数相同情况下首先遇到的类别。group_strata的值是一个字符串，如 "2hop_inference"，表示该组的主要类型。
        group_strata[gkey] = strata.most_common(1)[0][0]

    # 按 stratum 分桶
    stratum_groups = defaultdict(list)  # stratum → [group_key, ...]
    for gkey, stratum in group_strata.items():
        stratum_groups[stratum].append(gkey)
    # 得到类似：
    # {
    #     "2hop_inference": ["bbg_0016", "yh_0003", ...],
    #     "3hop_inference": ["hq_0098", "jjy_0021", ...],
    #     "2hop_comparison": ["zb_0042", ...],
    # }

    # 计算目标测试集大小
    if test_size > 0:
        target_test = test_size
    else:
        target_test = int(len(items) * test_ratio)

    # 在每个 stratum 内按比例抽组到测试集
    test_groups = set()
    train_groups = set()

    for stratum, gkeys in sorted(stratum_groups.items()):
        # 随机打乱组
        random.shuffle(gkeys)
        # 该 stratum 的总条目数（当前这个 stratum 类别里，一共有多少条 QA 样本。）
        stratum_total = sum(len(groups[gk]) for gk in gkeys)
        # 按全局比例分配该 stratum 的测试条目数
        stratum_test_target = int(stratum_total * target_test / len(items))
        stratum_test_target = max(stratum_test_target, 1)  # 每类至少 1 条

        test_count = 0
        # 为了保持 group 完整性不可避免的偏差。如果 stratum_test_target 是 10 条，但前面抽了 9 条后剩下的组都很大（每组 5 条），就可能抽到 15 条，超过目标。但这是为了保证同一 seed chunk 不被拆分。
        for gk in gkeys:
            if test_count >= stratum_test_target:
                break
            test_groups.add(gk)
            test_count += len(groups[gk])

    # 剩余归训练集
    train_groups = set(groups.keys()) - test_groups

    # 构建最终列表
    # 确保：一个 seed group 中的所有 QA 都在同一个集合中（训练或测试），不会拆分。
    train_items = []
    test_items = []
    for gk in groups:
        if gk in test_groups:
            test_items.extend(groups[gk])
        else:
            train_items.extend(groups[gk])

    # 随机打乱
    random.shuffle(train_items)
    random.shuffle(test_items)

    return train_items, test_items


def print_distribution(items: list, label: str):
    """打印数据分布"""
    n = len(items)
    if n == 0:
        print(f"  [{label}] 空")
        return

    print(f"\n  [{label}] {n} 条")

    # hop 分布
    hop_dist = Counter(r["hop_count"] for r in items)
    print(f"  hop 分布:")
    for h, c in sorted(hop_dist.items()):
        print(f"    {h}hop: {c} ({c/n*100:.1f}%)")

    # 类型分布
    # Counter({
    #   2: 100,
    #   3: 300,
    #   4: 100
    # })
    type_dist = Counter(f'{r["hop_count"]}hop_{r["qa_type"]}' for r in items)
    print(f"  细分类型:")
    for t, c in sorted(type_dist.items()):
        print(f"    {t}: {c} ({c/n*100:.1f}%)")

    # inference/comparison
    inf = sum(1 for r in items if r["qa_type"] == "inference")
    comp = n - inf
    print(f"  inference: {inf} ({inf/n*100:.1f}%) | comparison: {comp} ({comp/n*100:.1f}%)")

    # judge 分数
    judged = [r for r in items if "judge" in r and r["judge"].get("total", 0) > 0]
    if judged:
        totals = [r["judge"]["total"] for r in judged]
        print(f"  judge: avg={sum(totals)/len(totals):.1f}, min={min(totals)}, max={max(totals)}")

    # seed chunk 数
    seeds = set(r["hops"][0]["doc_chunk_id"] for r in items if r["hops"])
    print(f"  独立 seed chunks: {len(seeds)}")


def check_leakage(train_items: list, test_items: list):
    """检查训测集之间是否有 seed chunk 泄露"""
    train_seeds = set(r["hops"][0]["doc_chunk_id"] for r in train_items if r["hops"])
    test_seeds = set(r["hops"][0]["doc_chunk_id"] for r in test_items if r["hops"])
    overlap = train_seeds & test_seeds
    if overlap:
        print(f"\n  ⚠ 泄露检查: {len(overlap)} 个 seed chunk 同时出现在训练和测试集!")
        for s in list(overlap)[:5]:
            print(f"    {s}")
    else:
        print(f"\n  ✓ 泄露检查通过: 训测 seed chunk 无重叠")


def main():
    parser = argparse.ArgumentParser(description="训练/测试集划分")
    parser.add_argument("--input", required=True, help="输入 JSONL")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="测试集比例 (默认 0.2)")
    parser.add_argument("--test-size", type=int, default=0, help="测试集大小 (优先于 ratio)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--train-name", default="train.jsonl", help="训练集文件名")
    parser.add_argument("--test-name", default="test.jsonl", help="测试集文件名")
    args = parser.parse_args()

    # Load
    items = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    print(f"加载: {len(items)} 条")
    print_distribution(items, "全量")

    # Split
    train_items, test_items = stratified_split(
        items,
        test_ratio=args.test_ratio,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Re-index
    for i, r in enumerate(train_items):
        r["id"] = f"mhop_{i:06d}"
        r["split"] = "train"
    for i, r in enumerate(test_items):
        r["id"] = f"mhop_test_{i:06d}"
        r["split"] = "test"

    # Print comparison
    print(f"\n{'='*60}")
    print(f"划分结果: train {len(train_items)} / test {len(test_items)} "
          f"(实际比例 {len(test_items)/len(items)*100:.1f}%)")
    print_distribution(train_items, "训练集")
    print_distribution(test_items, "测试集")
    check_leakage(train_items, test_items)

    # Write
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, args.train_name)
    test_path = os.path.join(args.output_dir, args.test_name)

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for r in test_items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n写出:")
    print(f"  训练集: {train_path} ({len(train_items)} 条)")
    print(f"  测试集: {test_path} ({len(test_items)} 条)")


if __name__ == "__main__":
    main()
