"""
评测主入口
QA数据 → LangGraph Agent → 推理 → EM/F1 → diagnostics → 可选LLM judge → checkpoint → 汇总
"""
import pickle
import json
import os
import sys
import time
from threading import Lock

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIVE_DATA_DIR, RESULTS_DIR


def load_qa_pairs(subset: str | None = None, data_dir: str | None = None) -> list:
    """
    加载 QA 数据
    
    subset： 可选的子集名称，用于只加载特定子集的数据。比如 3hop_inference、2hop_comparison。默认 None，评全部。
    """
    qa_path = os.path.join(data_dir or ACTIVE_DATA_DIR, "qa_pairs.json")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    if subset:
        qa_pairs = [q for q in qa_pairs if q["subset"] == subset]
    return qa_pairs


def _get_checkpoint_path(model: str | None, subset: str | None) -> str:
    tag = subset or "all"
    model_tag = f"_{model}" if model else ""
    return os.path.join(RESULTS_DIR, f"eval_{tag}{model_tag}.checkpoint.jsonl")


def _load_checkpoint(ckpt_path: str) -> dict[int, dict]:
    """加载已完成的结果，返回 {idx: record}"""
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    # idx = QA在数据集中的顺序编号
                    done[r["idx"]] = r
    return done


def run_full_eval(
    max_samples: int = 200,
    subset: str | None = None,
    use_llm_judge: bool = False,
    model: str | None = None,
    workers: int = 1,
    resume: bool = False,
    data_dir: str | None = None,
    index_dir: str | None = None,
    lang: str | None = None,
):
    """运行完整评测"""
    import config as cfg
    import llm.client as llm_client
    from agents.graph import build_graph
    from evaluation.metrics import exact_match, f1_score, CostTracker
    from evaluation.hop_aware_eval import diagnose_failure, aggregate_diagnostics

    if model:
        print(f"[eval] Model: {cfg.AGENT_LLM_MODEL} -> {model}")
        cfg.AGENT_LLM_MODEL = model
        llm_client.AGENT_LLM_MODEL = model

    if lang:
        cfg.PROMPT_LANG = lang
        print(f"[eval] Prompt lang: {lang}")

    if index_dir:
        cfg.ACTIVE_INDEX_DIR = index_dir
        cfg.INDEX_DIR = index_dir
        print(f"[eval] Index dir: {index_dir}")

    qa_pairs = load_qa_pairs(subset, data_dir=data_dir)
    print(f"[eval] Loaded {len(qa_pairs)} QA pairs" + (f" (subset: {subset})" if subset else ""))

    # 使用 LLM judge 时，从当前索引目录加载 chunk_store.pkl，以便 judge context precision
    corpus_lookup = {}
    if use_llm_judge:
        chunk_store_path = os.path.join(index_dir or cfg.ACTIVE_INDEX_DIR, "chunk_store.pkl")
        if os.path.exists(chunk_store_path):
            with open(chunk_store_path, "rb") as f:
                corpus_lookup = pickle.load(f)
            print(f"[eval] Loaded {len(corpus_lookup)} chunks from chunk_store.pkl for LLM judge")
        else:
            print(f"[eval] WARNING: chunk_store.pkl not found at {chunk_store_path}; context precision will use hop answers only")

    # GPU 预热：embedder/reranker 指定 GPU（环境变量 RETRIEVAL_DEVICE 或自动选空闲卡）
    import torch
    if torch.cuda.is_available():
        gpu_device = os.environ.get("RETRIEVAL_DEVICE")
        if not gpu_device:
            # 自动选显存占用最少的 GPU
            mem_used = []
            for i in range(torch.cuda.device_count()):
                mem_used.append(torch.cuda.mem_get_info(i)[1] - torch.cuda.mem_get_info(i)[0])
            gpu_idx = mem_used.index(min(mem_used))
            gpu_device = f"cuda:{gpu_idx}"
        import retrieval.embedder as _emb
        import retrieval.reranker as _rnk
        # 强制 embedder 和 reranker 加载到指定 GPU 上
        _orig_emb_get = _emb._get_model
        _orig_rnk_get = _rnk._get_model
        def _emb_get_gpu(device=None):
            return _orig_emb_get(device or gpu_device)
        def _rnk_get_gpu(device=None):
            return _orig_rnk_get(device or gpu_device)
        # 用新包装的函数替换原来的两个_get_model函数：
        # 这之后，项目里任何地方再调用 retrieval.embedder._get_model() 或 retrieval.reranker._get_model() 都会默认使用上面确定的 gpu_device 。
        _emb._get_model = _emb_get_gpu
        _rnk._get_model = _rnk_get_gpu
        # 预加载模型
        _emb_get_gpu()
        _rnk_get_gpu()
        print(f"[eval] Embedder/Reranker pre-loaded on {gpu_device}")

    app = build_graph()
    cost = CostTracker()

    samples = qa_pairs[:max_samples]

    # Resume 支持：加载 checkpoint
    ckpt_path = _get_checkpoint_path(model, subset)
    done_map = {}
    if resume:
        done_map = _load_checkpoint(ckpt_path)
        if done_map:
            print(f"[eval] Resume: loaded {len(done_map)} completed results from checkpoint")

    # 过滤待跑的样本
    todo = [(i, qa) for i, qa in enumerate(samples) if i not in done_map]
    print(f"[eval] To run: {len(todo)}/{len(samples)}")

    if not todo and done_map:
        print("[eval] All samples already completed, aggregating results...")
        # done_map 是“并发写入”的，写入顺序可能乱，所以再次按 idx 排序
        results = [done_map[i] for i in sorted(done_map.keys()) if i < len(samples)]
    else:
        # checkpoint 写锁
        ckpt_lock = Lock()
        ckpt_f = open(ckpt_path, "a", encoding="utf-8")

        def _run_one(i, qa):
            query = qa["final_question"]
            gold = qa["final_answer"]

            initial_state = {
                "query": query,
                "query_type": "multi_hop",
                "plan": [],
                "current_step": 0,
                "evidence": [],
                "tool_calls": [],
                "verification_result": "",
                "verification_feedback": "",
                "final_answer": "",
                "iteration_count": 0,
                "total_tool_calls": 0,
                "trace": [],
            }

            t0 = time.time()
            try:
                state = app.invoke(initial_state)
            except Exception as e:
                print(f"  [{i+1}] ERROR: {e}")
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
                "question": query,
                "gold": gold,
                "prediction": pred,
                "em": em,
                "f1": f1,
                "diagnostics": diag,
                "latency": latency,
            }

            # 保存 evidence_text（供离线 judge faithfulness / context_precision）
            evidence_list = state.get("evidence", [])
            seen_chunks = set()
            evidence_parts = []
            for e in evidence_list:
                if not isinstance(e, dict):
                    continue
                for r in e.get("results", []):
                    cid = r.get("chunk_id", "")
                    if cid in seen_chunks:
                        continue
                    seen_chunks.add(cid)
                    title = r.get("title", "")
                    # 原来的500够吗？
                    text = r.get("text", "")[:1000]
                    evidence_parts.append(f"[{cid}] {title}\n{text}")
            record["evidence_text"] = "\n\n".join(evidence_parts)

            # 在线的 llm judge
            if use_llm_judge:
                from evaluation.llm_judge import (
                    judge_answer_correctness,
                    judge_faithfulness,
                    judge_context_precision,
                )
                # 原来的200够吗？
                evidence_text = "\n".join(
                    r.get("text", "")[:1000]
                    for e in state.get("evidence", [])
                    for r in e.get("results", [])[:2]
                )
                record["judge_correctness"] = judge_answer_correctness(query, pred, gold)
                record["judge_faithfulness"] = judge_faithfulness(query, pred, record["evidence_text"])
                gold_docs_parts = []
                for h in qa.get("hops", []):
                    cid = h.get("doc_chunk_id", "")
                    doc = corpus_lookup.get(cid, {})
                    doc_text = doc.get("text", h.get("answer", ""))
                    doc_title = doc.get("title", "")
                    gold_docs_parts.append(
                        f"[Gold doc {h.get('hop_idx', '?')}] {doc_title}\n{doc_text}"
                    )
                gold_docs_text = "\n\n".join(gold_docs_parts)
                # evidence_text 是在线 judge 的短上下文；record["evidence_text"] 是保存到结果里的完整检索上下文。context precision 更适合用后者。
                record["judge_context_precision"] = judge_context_precision(
                    query,
                    record["evidence_text"],
                    gold_docs_text,
                ) if record["evidence_text"].strip() and gold_docs_text.strip() else 0.0

            # 即时写入 checkpoint
            with ckpt_lock:
                ckpt_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                ckpt_f.flush()

            return record

        new_results = []
        running_em = []
        running_f1 = []
        pbar = tqdm(total=len(todo), desc=f"Eval {model or 'default'}",
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

        # 合并 checkpoint + 新结果
        for r in new_results:
            done_map[r["idx"]] = r
        # done_map 是“并发写入”的，写入顺序可能乱，所以再次按 idx 排序
        results = [done_map[i] for i in sorted(done_map.keys()) if i < len(samples)]

    # 聚合
    n = len(results)
    avg_em = sum(r["em"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n
    diag_agg = aggregate_diagnostics([r["diagnostics"] for r in results])

    summary = {
        "num_samples": n,
        "avg_em": round(avg_em, 3),
        "avg_f1": round(avg_f1, 3),
        "cost": cost.summary(),
        "diagnostics": diag_agg,
    }

    if use_llm_judge:
        summary["avg_judge_correctness"] = round(
            sum(r.get("judge_correctness", 0) for r in results) / n, 3
        )
        summary["avg_judge_faithfulness"] = round(
            sum(r.get("judge_faithfulness", 0) for r in results) / n, 3
        )
        summary["avg_judge_context_precision"] = round(
            sum(r.get("judge_context_precision", 0) for r in results) / n, 3
        )

    # 保存最终结果
    out = {"summary": summary, "results": results}
    tag = subset or "all"
    model_tag = f"_{model}" if model else ""
    data_tag = "_financial" if data_dir and "financial" in data_dir else ""
    out_path = os.path.join(RESULTS_DIR, f"eval_{tag}_{n}{model_tag}{data_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 清理 checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"[eval] Checkpoint cleaned: {ckpt_path}")

    print(f"\n[eval] Summary:")
    print(f"  EM: {avg_em:.3f}, F1: {avg_f1:.3f}")
    print(f"  Diagnostics: {json.dumps(diag_agg.get('overall', {}), indent=2)}")
    print(f"  Results saved to {out_path}")

    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # --subset：只评某个 subset，比如 3hop_inference、2hop_comparison。默认 None，评全部。
    parser.add_argument("--subset", default=None)
    # --max-samples：最多评多少条，默认 5。是在加载并按 subset 过滤后取前 N 条。
    parser.add_argument("--max-samples", type=int, default=1305)
    # --model：覆盖 AGENT_LLM_MODEL。默认 None，用 .env/config.py 里的 AGENT_LLM_MODEL。
    parser.add_argument("--model", default='Qwen3-4B', help="Override AGENT_LLM_MODEL")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    # --llm-judge：是否额外启用 LLM-as-a-Judge。默认不开。开启后每条样本会额外调用 correctness 和 faithfulness judge。
    parser.add_argument("--llm-judge", default=False, action="store_true")
    parser.add_argument("--resume", default=True, action="store_true", help="Resume from checkpoint")
    # --data-dir：覆盖数据目录，要求里面有 qa_pairs.json。默认用 config.ACTIVE_DATA_DIR。
    parser.add_argument("--data-dir", default=None, help="Override data directory (qa_pairs.json location)")
    # --index-dir：覆盖索引目录。会改 config.ACTIVE_INDEX_DIR 和 config.INDEX_DIR。
    parser.add_argument("--index-dir", default=None, help="Override index directory")
    parser.add_argument("--lang", default=None, choices=["en", "zh"], help="Prompt language")
    args = parser.parse_args()
    run_full_eval(max_samples=args.max_samples, subset=args.subset,
                  use_llm_judge=args.llm_judge, model=args.model,
                  workers=args.workers, resume=args.resume,
                  data_dir=args.data_dir, index_dir=args.index_dir,
                  lang=args.lang)
