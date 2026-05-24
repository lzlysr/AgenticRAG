#!/usr/bin/env python3
"""云端模型评测脚本：扁平并行跑所有 sample，共享一个 graph

特性：
- 单 graph 实例，检索模型只加载一次
- sample 级线程池并行（适合云端 API 模型）
- 断点续跑（--resume）
- 自动按 subset 聚合指标

用法：
  # 跑 mcs-1，10 并发
  NEWS_CORPUS_DIR=data/financial_eval NEWS_INDEX_DIR=data/financial_all/indexes \
  python -u scripts/run_cloud_eval.py --model mcs-1 --workers 10

  # 断点续跑
  python -u scripts/run_cloud_eval.py --model mcs-1 --workers 10 --resume
"""
import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKPOINT_EVERY = 20

_save_lock = Lock()
_results = []  # [{question, gold, pred, em, f1, subset, ...}]
_done_count = 0


def run_one(app, qa, idx, total):
    """跑单条 QA"""
    from evaluation.metrics import exact_match, f1_score

    query = qa["final_question"]
    gold = qa["final_answer"]
    subset = qa["subset"]
    aliases = qa.get("answer_aliases", [])

    initial_state = {
        "query": query, "query_type": "multi_hop",
        "plan": [], "current_step": 0, "evidence": [],
        "tool_calls": [], "verification_result": "",
        "verification_feedback": "", "final_answer": "",
        "iteration_count": 0, "total_tool_calls": 0, "trace": [],
    }

    t0 = time.time()
    try:
        state = app.invoke(initial_state)
    except Exception as e:
        state = initial_state
        state["final_answer"] = ""
        state["_error"] = str(e)
    latency = time.time() - t0

    pred = state.get("final_answer", "")
    em = exact_match(pred, gold, aliases=aliases)
    f1 = f1_score(pred, gold, aliases=aliases)

    result = {
        "question": query,
        "gold": gold,
        "pred": pred,
        "em": em,
        "f1": round(f1, 3),
        "subset": subset,
        "hop_count": qa["hop_count"],
        "qa_type": qa["qa_type"],
        "tool_calls": state.get("total_tool_calls", 0),
        "iterations": state.get("iteration_count", 0),
        "latency": round(latency, 2),
    }
    # 保存 evidence_text 用于 LLM Judge（faithfulness / context_precision）
    evidence = state.get("evidence", [])
    if evidence:
        parts = []
        for e in evidence:
            for r in e.get("results", [])[:3]:
                parts.append(f"[{r.get('chunk_id', '?')}] {r.get('text', '')[:500]}")
        result["evidence_text"] = "\n\n".join(parts)
    if state.get("_error"):
        result["error"] = state["_error"][:200]

    return result


def save_results(results, out_path):
    """保存结果"""
    # 按 subset 聚合
    subset_results = defaultdict(list)
    for r in results:
        subset_results[r["subset"]].append(r)

    output = {
        "total": len(results),
        "num_errors": sum(1 for r in results if "error" in r),
    }

    valid = [r for r in results if "error" not in r]
    if valid:
        output["avg_em"] = round(sum(r["em"] for r in valid) / len(valid), 3)
        output["avg_f1"] = round(sum(r["f1"] for r in valid) / len(valid), 3)
        output["avg_latency"] = round(sum(r["latency"] for r in valid) / len(valid), 2)
        output["avg_iterations"] = round(sum(r["iterations"] for r in valid) / len(valid), 2)
        output["avg_tool_calls"] = round(sum(r["tool_calls"] for r in valid) / len(valid), 2)

    # 按 subset 汇总
    output["subsets"] = {}
    for subset, items in sorted(subset_results.items()):
        sv = [r for r in items if "error" not in r]
        if sv:
            output["subsets"][subset] = {
                "n": len(items),
                "n_valid": len(sv),
                "avg_em": round(sum(r["em"] for r in sv) / len(sv), 3),
                "avg_f1": round(sum(r["f1"] for r in sv) / len(sv), 3),
                "avg_latency": round(sum(r["latency"] for r in sv) / len(sv), 2),
            }

    output["per_sample"] = results

    with _save_lock:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="云端模型评测")
    parser.add_argument("--model", required=True, help="模型名 (e.g. mcs-1)")
    parser.add_argument("--workers", type=int, default=10, help="并行 sample 数")
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    parser.add_argument("--output", default=None, help="输出路径 (默认 results/<model>_financial.json)")
    args = parser.parse_args()

    # 切换模型
    import config as cfg
    import llm.client as llm_client
    print(f"[eval] Model: {cfg.AGENT_LLM_MODEL} -> {args.model}")
    cfg.AGENT_LLM_MODEL = args.model
    llm_client.AGENT_LLM_MODEL = args.model

    # 加载数据
    from evaluation.run_eval import load_qa_pairs
    qa_pairs = load_qa_pairs()
    print(f"[eval] Loaded {len(qa_pairs)} QA pairs")
    print(f"[eval] Subsets: {dict(Counter(q['subset'] for q in qa_pairs))}")

    # 输出路径
    model_tag = args.model.replace("/", "_").replace("-", "").lower()
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", f"{model_tag}_financial.json"
    )

    # 断点续跑
    existing = {}
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        for r in data.get("per_sample", []):
            if "error" not in r:
                existing[r["question"]] = r
        print(f"[eval] Resumed: {len(existing)} completed samples")

    # 构建 graph（只一次）
    from agents.graph import build_graph
    print("[eval] Building graph...")
    app = build_graph(
        enable_verifier=True,
        enabled_tools=['keyword_search', 'semantic_search', 'read_chunk']
    )
    print("[eval] Graph ready")

    # 分离已跑和待跑
    to_run = []
    results = []
    for qa in qa_pairs:
        prev = existing.get(qa["final_question"])
        if prev:
            results.append(prev)
        else:
            to_run.append(qa)

    print(f"[eval] To run: {len(to_run)}, Already done: {len(results)}")
    if not to_run:
        print("[eval] All done!")
        save_results(results, out_path)
        return

    # 并行执行
    print(f"[eval] Starting with {args.workers} workers...")
    t_start = time.time()
    done = [0]
    done_lock = Lock()

    def _run_and_track(qa):
        r = run_one(app, qa, 0, len(to_run))
        with done_lock:
            results.append(r)
            done[0] += 1
            d = done[0]

        status = "OK" if "error" not in r else f"ERR: {r.get('error','')[:60]}"
        if d % 10 == 0 or d == 1 or d == len(to_run) or "error" in r:
            elapsed = time.time() - t_start
            rate = d / elapsed * 60
            eta = (len(to_run) - d) / (d / elapsed) if d > 0 else 0
            print(f"  [{d}/{len(to_run)}] {r['subset']} EM={r['em']:.0f} F1={r['f1']:.3f} "
                  f"lat={r['latency']:.1f}s {status} | {rate:.0f}/min ETA {eta/60:.0f}min")

        if d % CHECKPOINT_EVERY == 0:
            save_results(results, out_path)

        return r

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_and_track, qa) for qa in to_run]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"  Unexpected error: {e}")

    elapsed = time.time() - t_start
    print(f"\n[eval] Done in {elapsed/60:.1f} min")

    # 保存最终结果
    save_results(results, out_path)
    print(f"[eval] Saved to {out_path}")

    # 打印汇总
    valid = [r for r in results if "error" not in r]
    errors = len(results) - len(valid)
    print(f"\n{'='*60}")
    print(f"Model: {args.model} | Total: {len(results)} | Errors: {errors}")
    if valid:
        print(f"Overall: EM={sum(r['em'] for r in valid)/len(valid):.3f} "
              f"F1={sum(r['f1'] for r in valid)/len(valid):.3f}")
    print(f"{'='*60}")

    subset_results = defaultdict(list)
    for r in valid:
        subset_results[r["subset"]].append(r)
    print(f"{'Subset':<25} {'N':>4} {'EM':>6} {'F1':>6} {'Lat':>6}")
    print("-" * 50)
    for subset in sorted(subset_results.keys()):
        items = subset_results[subset]
        n = len(items)
        em = sum(r["em"] for r in items) / n
        f1 = sum(r["f1"] for r in items) / n
        lat = sum(r["latency"] for r in items) / n
        print(f"{subset:<25} {n:>4} {em:>6.3f} {f1:>6.3f} {lat:>5.1f}s")


if __name__ == "__main__":
    main()
