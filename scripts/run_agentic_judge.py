      
#!/usr/bin/env python3
"""对 Agentic(eval_agentic) 评测结果补跑 LLM Judge（三维度：faithfulness + context_precision + answer_correctness）

用法：
  python scripts/run_agentic_judge.py results/agentic_Qwen3-4B-GRPO-v5e.json

输出：在原文件同目录生成 *_judged.json
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.llm_judge import (
    judge_answer_correctness,
    judge_faithfulness,
    judge_context_precision,
)

WORKERS = 10


def main():
    result_file = sys.argv[1] if len(sys.argv) > 1 else "results/eval_agentic/financial/agentic_185_Qwen3-4B.json"

    with open(result_file) as f:
        data = json.load(f)

    results = data["results"]
    has_evidence = any(r.get("evidence") for r in results)
    print(f"[judge] 总样本: {len(results)}, workers: {WORKERS}, has_evidence: {has_evidence}")

    # 断点续跑：检查三个维度是否都已完成
    to_judge = [r for r in results if "judge_correctness" not in r]
    already_done = len(results) - len(to_judge)
    if already_done:
        print(f"[judge] answer_correctness 已有 {already_done} 条")
    print(f"[judge] 待 judge: {len(to_judge)}")

    done = [0]
    t0 = time.time()

    def _judge_one(sample):
        pred = sample.get("pred", "")
        gold = sample.get("gold", "")
        question = sample.get("question", "")
        evidence = sample.get("evidence", "")

        # answer_correctness（必跑）
        if "judge_correctness" not in sample:
            sample["judge_correctness"] = judge_answer_correctness(
                question=question, prediction=pred, gold=gold,
            )

        # faithfulness（需要 evidence）
        if evidence and "judge_faithfulness" not in sample:
            sample["judge_faithfulness"] = judge_faithfulness(
                question=question, answer=pred, evidence=evidence,
            )

        # context_precision（需要 evidence + gold 作为参考）
        if evidence and "judge_context_precision" not in sample:
            sample["judge_context_precision"] = judge_context_precision(
                question=question, evidence=evidence, gold_docs=gold,
            )

        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(to_judge):
            elapsed = time.time() - t0
            rate = done[0] / elapsed * 60
            print(f"  [{done[0]}/{len(to_judge)}] {rate:.0f}/min")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_judge_one, s) for s in to_judge]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"  Judge error: {e}")

    # 统计
    def _avg(key):
        vals = [r[key] for r in results if key in r]
        return sum(vals) / len(vals) if vals else 0

    avg_jc = _avg("judge_correctness")
    avg_jf = _avg("judge_faithfulness")
    avg_jcp = _avg("judge_context_precision")

    # 按 hop 分组
    by_hop = {}
    for r in results:
        h = r.get("hops", 2)
        if isinstance(h, str):
            h = int(h)
        by_hop.setdefault(h, []).append(r)

    print(f"\n{'='*60}")
    print(f"Judge Results: {os.path.basename(result_file)}")
    print(f"{'='*60}")
    print(f"  Judge Correctness:      {avg_jc:.3f}")
    if has_evidence:
        print(f"  Judge Faithfulness:     {avg_jf:.3f}")
        print(f"  Judge Context Precision: {avg_jcp:.3f}")
    print(f"  Token F1:               {data['summary']['avg_f1']:.3f}")
    print(f"  EM:                     {data['summary']['avg_em']:.3f}")

    print(f"\n  By hop:")
    for h in sorted(by_hop.keys()):
        items = by_hop[h]
        n = len(items)
        jc = sum(r.get("judge_correctness", 0) for r in items) / n
        f1 = sum(float(r["f1"]) for r in items) / n
        line = f"    {h}hop: Judge_C={jc:.3f} F1={f1:.3f}"
        if has_evidence:
            jf = sum(r.get("judge_faithfulness", 0) for r in items) / n
            jcp = sum(r.get("judge_context_precision", 0) for r in items) / n
            line += f" Faith={jf:.3f} CtxP={jcp:.3f}"
        line += f" (n={n})"
        print(line)

    # 保存
    data["summary"]["judge_correctness"] = avg_jc
    if has_evidence:
        data["summary"]["judge_faithfulness"] = avg_jf
        data["summary"]["judge_context_precision"] = avg_jcp
    out_path = result_file.replace(".json", "_judged.json").replace("eval_agentic", "run_agentic_judge")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()