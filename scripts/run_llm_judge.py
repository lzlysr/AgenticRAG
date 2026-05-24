"""离线 LLM Judge 评测脚本：对已有全量结果运行 judge 评分

支持三个维度：
- judge_correctness: 答案正确性（只需 question + pred + gold，始终可离线评估）
- judge_faithfulness: 答案忠实度（需要 evidence_text，仅当结果中包含时可评估）
- judge_ctx_precision: 检索精度（需要 evidence_text + gold_docs，仅当结果中包含时可评估）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_question_map(qa_pairs_path: str, corpus_path: str) -> tuple[dict[str, str], dict[str, list], dict[str, dict]]:
    """建立 question[:100] → full_question、question → hops 和 chunk_id → doc 的映射"""
    with open(qa_pairs_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_map = {c["chunk_id"]: c for c in json.load(f)}

    qmap = {}      # prefix → full_question
    hops_map = {}   # full_question → hops
    for qa in qa_pairs:
        full_q = qa["final_question"]
        prefix = full_q[:100]
        qmap[prefix] = full_q
        hops_map[full_q] = qa.get("hops", [])
    return qmap, hops_map, corpus_map


def restore_full_question(truncated: str, qmap: dict[str, str]) -> str:
    """从截断问题恢复完整问题，找不到则返回原文"""
    if truncated in qmap:
        return qmap[truncated]
    for prefix, full_q in qmap.items():
        if full_q.startswith(truncated) or truncated.startswith(prefix):
            return full_q
    return truncated


def run_judge_on_results(results_path: str, configs: list[str] | None = None,
                         max_samples: int = 0) -> dict:
    """对已有结果文件运行 LLM Judge 评分"""
    from evaluation.llm_judge import (
        judge_answer_correctness, judge_faithfulness, judge_context_precision,
    )

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "datasets"
    )
    qa_pairs_path = os.path.join(data_dir, "qa_pairs.json")
    corpus_path = os.path.join(data_dir, "corpus.json")
    qmap, hops_map, corpus_map = build_question_map(qa_pairs_path, corpus_path)
    print(f"[judge] Loaded {len(qmap)} questions, {len(corpus_map)} corpus chunks")

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    target_configs = configs or list(results.keys())

    for config_name in target_configs:
        if config_name not in results:
            print(f"[judge] Config '{config_name}' not found, skipping")
            continue

        config_data = results[config_name]
        per_sample = config_data.get("per_sample", [])
        samples = per_sample[:max_samples] if max_samples > 0 else per_sample

        print(f"\n[judge] Processing '{config_name}': {len(samples)} samples")
        correctness_scores = []
        faithfulness_scores = []
        ctx_precision_scores = []

        for i, sample in enumerate(samples):
            full_question = restore_full_question(sample["question"], qmap)
            pred = sample.get("pred", "")
            gold = sample.get("gold", "")
            evidence_text = sample.get("evidence_text", "")

            # 1) judge_correctness
            if "judge_correctness" not in sample:
                try:
                    score = judge_answer_correctness(full_question, pred, gold)
                except Exception as e:
                    print(f"  [{i+1}] Judge correctness error: {e}")
                    score = 0.0
                sample["judge_correctness"] = round(score, 3)
                sample["question_full"] = full_question
            correctness_scores.append(sample["judge_correctness"])

            # 2) judge_faithfulness（需要 evidence_text）
            if evidence_text.strip() and "judge_faithfulness" not in sample:
                try:
                    jf = judge_faithfulness(full_question, pred, evidence_text)
                except Exception as e:
                    print(f"  [{i+1}] Judge faithfulness error: {e}")
                    jf = 0.0
                sample["judge_faithfulness"] = round(jf, 3)
            if "judge_faithfulness" in sample:
                faithfulness_scores.append(sample["judge_faithfulness"])

            # 3) judge_ctx_precision（需要 evidence_text + gold_docs）
            if evidence_text.strip() and "judge_ctx_precision" not in sample:
                hops = hops_map.get(full_question, [])
                gold_docs_parts = []
                for h in hops:
                    cid = h.get("doc_chunk_id", "")
                    doc = corpus_map.get(cid, {})
                    doc_text = doc.get("text", h.get("answer", ""))
                    doc_title = doc.get("title", "")
                    gold_docs_parts.append(
                        f"[Gold doc {h['hop_idx']}] {doc_title}\n{doc_text}"
                    )
                gold_docs_text = "\n\n".join(gold_docs_parts)
                if gold_docs_text.strip():
                    try:
                        jcp = judge_context_precision(full_question, evidence_text, gold_docs_text)
                    except Exception as e:
                        print(f"  [{i+1}] Judge ctx_precision error: {e}")
                        jcp = 0.0
                    sample["judge_ctx_precision"] = round(jcp, 3)
            if "judge_ctx_precision" in sample:
                ctx_precision_scores.append(sample["judge_ctx_precision"])

            if (i + 1) % 20 == 0 or i == 0:
                avg_c = sum(correctness_scores) / len(correctness_scores)
                print(f"  [{i+1}/{len(samples)}] correctness={sample['judge_correctness']:.3f} avg={avg_c:.3f}")

        # 聚合
        if correctness_scores:
            config_data["avg_judge_correctness"] = round(
                sum(correctness_scores) / len(correctness_scores), 3
            )
        if faithfulness_scores:
            config_data["avg_judge_faithfulness"] = round(
                sum(faithfulness_scores) / len(faithfulness_scores), 3
            )
        if ctx_precision_scores:
            config_data["avg_judge_ctx_precision"] = round(
                sum(ctx_precision_scores) / len(ctx_precision_scores), 3
            )

        print(f"  => correctness={config_data.get('avg_judge_correctness', 'N/A')}"
              f"  faithfulness={config_data.get('avg_judge_faithfulness', 'N/A')}"
              f"  ctx_precision={config_data.get('avg_judge_ctx_precision', 'N/A')}")

    # 保存结果
    base_name = os.path.basename(results_path)
    name_without_ext = base_name.rsplit(".json", 1)[0]
    out_path = os.path.join(os.path.dirname(results_path), f"{name_without_ext}_judged.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[judge] Results saved to {out_path}")

    print_summary_table(results, target_configs)
    return results


def print_summary_table(results: dict, configs: list[str]):
    """打印对比汇总表"""
    has_faith = any("avg_judge_faithfulness" in results.get(c, {}) for c in configs)
    has_ctx = any("avg_judge_ctx_precision" in results.get(c, {}) for c in configs)

    header = f"{'Config':<35} {'EM':>6} {'F1':>6} {'Judge_C':>8}"
    if has_faith:
        header += f" {'Judge_F':>8}"
    if has_ctx:
        header += f" {'Ctx_P':>8}"
    header += f" {'N':>5}"

    width = len(header) + 2
    print(f"\n{'='*width}")
    print(header)
    print(f"{'-'*width}")
    for name in configs:
        if name not in results:
            continue
        r = results[name]
        line = f"{name:<35} {r.get('avg_em', 0):>6.3f} {r.get('avg_f1', 0):>6.3f} {r.get('avg_judge_correctness', 0):>8.3f}"
        if has_faith:
            line += f" {r.get('avg_judge_faithfulness', 0):>8.3f}"
        if has_ctx:
            line += f" {r.get('avg_judge_ctx_precision', 0):>8.3f}"
        line += f" {r.get('num_samples', 0):>5}"
        print(line)
    print(f"{'='*width}")


def main():
    parser = argparse.ArgumentParser(description="对已有结果运行 LLM Judge 评分")
    parser.add_argument("results_path", help="结果 JSON 文件路径")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="指定要评测的 config（默认全部）")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="每个 config 最大评测样本数（0=全部）")
    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        print(f"[judge] File not found: {args.results_path}")
        sys.exit(1)

    run_judge_on_results(
        args.results_path,
        configs=args.configs,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
