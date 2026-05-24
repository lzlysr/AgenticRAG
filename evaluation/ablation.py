"""消融实验配置与运行器 ★"""
import json
import os
import sys
import time
from threading import Lock

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

ABLATION_CONFIGS = {
    "full_system": {
        "tools": ["keyword_search", "semantic_search", "read_chunk"],
        "verifier": True,
        "description": "完整系统（BM25 + FAISS + Reranker + Verifier）",
    },
    "no_keyword": {
        "tools": ["semantic_search", "read_chunk"],
        "verifier": True,
        "description": "去掉 BM25 关键字搜索",
    },
    "no_read_chunk": {
        "tools": ["keyword_search", "semantic_search"],
        "verifier": True,
        "description": "去掉 chunk 阅读工具",
    },
    "no_verifier": {
        "tools": ["keyword_search", "semantic_search", "read_chunk"],
        "verifier": False,
        "description": "去掉验证器（无自纠错循环）",
    },
    "semantic_only": {
        "tools": ["semantic_search"],
        "verifier": False,
        "description": "Naive RAG 基线（仅语义检索，无验证）",
    },
}


def _get_checkpoint_path(config_name: str, model: str | None = None) -> str:
    model_tag = f"_{model}" if model else ""
    return os.path.join(RESULTS_DIR, f"ablation_{config_name}{model_tag}.checkpoint.jsonl")


def _load_checkpoint(ckpt_path: str) -> dict[int, dict]:
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["idx"]] = r
    return done


def run_ablation(config_name: str, qa_pairs: list, max_samples: int = 50,
                  use_llm_judge: bool = False, model: str | None = None,
                  workers: int = 1, resume: bool = False) -> dict:
    """运行单个消融配置（支持多线程 + checkpoint + tqdm）"""
    from agents.graph import build_graph
    from evaluation.metrics import exact_match, f1_score, CostTracker
    from evaluation.hop_aware_eval import diagnose_failure, aggregate_diagnostics

    config = ABLATION_CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f"[ablation] Running: {config_name} - {config['description']}")
    print(f"  Tools: {config['tools']}, Verifier: {config['verifier']}")
    print(f"  Model: {model or 'default'}, Workers: {workers}")
    print(f"  Samples: {min(max_samples, len(qa_pairs))}")
    print(f"{'='*60}")

    app = build_graph(
        enable_verifier=config["verifier"],
        enabled_tools=config["tools"],
    )

    cost = CostTracker()
    samples = qa_pairs[:max_samples]

    # Resume
    ckpt_path = _get_checkpoint_path(config_name, model)
    done_map = {}
    if resume:
        done_map = _load_checkpoint(ckpt_path)
        if done_map:
            print(f"[ablation] Resume: loaded {len(done_map)} completed from checkpoint")

    todo = [(i, qa) for i, qa in enumerate(samples) if i not in done_map]
    print(f"[ablation] To run: {len(todo)}/{len(samples)}")

    if not todo and done_map:
        results_list = [done_map[i] for i in sorted(done_map.keys()) if i < len(samples)]
    else:
        ckpt_lock = Lock()
        ckpt_f = open(ckpt_path, "a", encoding="utf-8")

        def _run_one(i, qa):
            query = qa["final_question"]
            gold = qa["final_answer"]

            initial_state = {
                "query": query, "query_type": "multi_hop",
                "plan": [], "current_step": 0,
                "evidence": [], "tool_calls": [],
                "verification_result": "", "verification_feedback": "",
                "final_answer": "", "iteration_count": 0,
                "total_tool_calls": 0, "trace": [],
            }

            t0 = time.time()
            try:
                state = app.invoke(initial_state)
            except Exception as e:
                state = initial_state
                state["final_answer"] = ""
            latency = time.time() - t0

            pred = state.get("final_answer", "")
            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)
            diag = diagnose_failure(state, qa)
            cost.record(state, latency)

            record = {
                "idx": i,
                "subset": qa["subset"],
                "hop_count": qa["hop_count"],
                "question": query, "gold": gold, "pred": pred,
                "em": em, "f1": round(f1, 3),
                "diagnostics": diag,
                "tool_calls": state.get("total_tool_calls", 0),
                "iterations": state.get("iteration_count", 0),
                "latency": latency,
            }

            with ckpt_lock:
                ckpt_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                ckpt_f.flush()

            return record

        new_results = []
        running_em, running_f1 = [], []
        pbar = tqdm(total=len(todo), desc=f"Ablation {config_name}",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] EM={postfix[0]:.3f} F1={postfix[1]:.3f}",
                    postfix=[0.0, 0.0])

        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_one, i, qa): i for i, qa in todo}
                for fut in as_completed(futures):
                    record = fut.result()
                    new_results.append(record)
                    running_em.append(record["em"])
                    running_f1.append(record["f1"])
                    pbar.postfix[0] = sum(running_em) / len(running_em)
                    pbar.postfix[1] = sum(running_f1) / len(running_f1)
                    pbar.update(1)
        else:
            for i, qa in todo:
                record = _run_one(i, qa)
                new_results.append(record)
                running_em.append(record["em"])
                running_f1.append(record["f1"])
                pbar.postfix[0] = sum(running_em) / len(running_em)
                pbar.postfix[1] = sum(running_f1) / len(running_f1)
                pbar.update(1)

        pbar.close()
        ckpt_f.close()

        for r in new_results:
            done_map[r["idx"]] = r
        results_list = [done_map[i] for i in sorted(done_map.keys()) if i < len(samples)]

    # 聚合
    n = len(results_list)
    avg_em = sum(r["em"] for r in results_list) / n
    avg_f1 = sum(r["f1"] for r in results_list) / n
    diag_agg = aggregate_diagnostics([r["diagnostics"] for r in results_list])

    result = {
        "config_name": config_name,
        "description": config["description"],
        "model": model,
        "num_samples": n,
        "avg_em": round(avg_em, 3),
        "avg_f1": round(avg_f1, 3),
        "cost": cost.summary(),
        "diagnostics": diag_agg,
        "per_sample": results_list,
    }

    # 保存单配置结果
    model_tag = f"_{model}" if model else ""
    out_path = os.path.join(RESULTS_DIR, f"ablation_{config_name}_{n}{model_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[ablation] {config_name} saved to {out_path}")

    # 清理 checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


def run_all_ablations(qa_pairs: list, max_samples: int = 50, configs: list[str] | None = None,
                      use_llm_judge: bool = False, model: str | None = None,
                      workers: int = 1, resume: bool = False) -> dict:
    """运行所有消融实验"""
    configs = configs or list(ABLATION_CONFIGS.keys())
    results = {}

    for name in configs:
        if name not in ABLATION_CONFIGS:
            print(f"[ablation] Unknown config: {name}, skipping")
            continue
        results[name] = run_ablation(name, qa_pairs, max_samples,
                                     use_llm_judge=use_llm_judge,
                                     model=model, workers=workers,
                                     resume=resume)

    # 保存汇总
    model_tag = f"_{model}" if model else ""
    out_path = os.path.join(RESULTS_DIR, f"ablation_results{model_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[ablation] All results saved to {out_path}")

    print_comparison_table(results)
    return results


def print_comparison_table(results: dict):
    """打印消融实验对比表"""
    has_judge = any(
        "avg_judge_correctness" in r or "avg_judge_faithfulness" in r
        for r in results.values()
    )

    if has_judge:
        print(f"\n{'='*106}")
        print(f"{'Config':<20} {'EM':>6} {'F1':>6} {'Hop↑':>6} {'Judge_C':>8} {'Judge_F':>8} {'Ctx_P':>8} {'Calls':>6} {'Lat(s)':>8}")
        print(f"{'-'*106}")
    else:
        print(f"\n{'='*80}")
        print(f"{'Config':<20} {'EM':>6} {'F1':>6} {'Hop↑':>6} {'Calls':>6} {'Lat(s)':>8}")
        print(f"{'-'*80}")

    for name, r in results.items():
        em = r.get("avg_em", 0)
        f1 = r.get("avg_f1", 0)
        hop = r.get("diagnostics", {}).get("overall", {}).get("avg_hop_recall", 0)
        calls = r.get("cost", {}).get("avg_tool_calls", 0)
        lat = r.get("cost", {}).get("avg_latency", 0)
        if has_judge:
            jc = r.get("avg_judge_correctness", 0)
            jf = r.get("avg_judge_faithfulness", 0)
            jcp = r.get("avg_judge_ctx_precision", 0)
            print(f"{name:<20} {em:>6.3f} {f1:>6.3f} {hop:>6.3f} {jc:>8.3f} {jf:>8.3f} {jcp:>8.3f} {calls:>6.1f} {lat:>8.1f}")
        else:
            print(f"{name:<20} {em:>6.3f} {f1:>6.3f} {hop:>6.3f} {calls:>6.1f} {lat:>8.1f}")

    print(f"{'='*106}" if has_judge else f"{'='*80}")
