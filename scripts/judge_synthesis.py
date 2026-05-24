#!/usr/bin/env python3
"""LLM Judge 清洗：对合成 QA 做质量评分，过滤低质量数据

用法:
  # 评分（20 并发）
  python scripts/judge_synthesis.py \
    --input data/financial_all/synthesis_v2/multihop_clean.jsonl \
    --corpus data/financial_all/corpus_all.json \
    --output data/financial_all/synthesis_v2/multihop_judged.jsonl \
    --workers 20

  # 只跑前 50 条测试
  python scripts/judge_synthesis.py \
    --input data/financial_all/synthesis_v2/multihop_clean.jsonl \
    --corpus data/financial_all/corpus_all.json \
    --output data/financial_all/synthesis_v2/multihop_judged.jsonl \
    --workers 20 --limit 50

  # 按分数过滤（总分 ≤ 8 过滤）
  python scripts/judge_synthesis.py \
    --input data/financial_all/synthesis_v2/multihop_judged.jsonl \
    --filter-only --min-score 9 \
    --output data/financial_all/synthesis_v2/multihop_final.jsonl

评分维度（每项 1-5 分）：
  1. answer_correctness: 答案是否可从给定 chunk 内容中验证
  2. multihop_necessity: 是否真正需要所有 hop 才能回答
  3. question_clarity: 问题是否清晰无歧义
"""
import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.synthesis_llm import init_concurrency, llm_call, _extract_json, get_stats, reset_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("judge_synthesis")


# ============================================================
# Judge prompt
# ============================================================

def _get_lang(lang=None):
    if lang:
        return lang
    import config
    return getattr(config, "PROMPT_LANG", "en")


JUDGE_PROMPT_TEMPLATE_EN = """You are a strict quality evaluator for multi-hop question-answering datasets.

Given a multi-hop QA item with its supporting evidence chunks, evaluate it on three dimensions.
Score each dimension from 1 (worst) to 5 (best).

## Scoring Criteria

### 1. Answer Correctness (answer_correctness)
Can the answer be verified from the provided chunk contents?
- 5: Answer is directly and clearly supported by the chunks
- 4: Answer is supported but requires minor inference
- 3: Answer is partially supported, some information is ambiguous
- 2: Answer is weakly supported, key evidence is missing or unclear
- 1: Answer appears incorrect or contradicts the chunks

### 2. Multi-hop Necessity (multihop_necessity)
Does answering the question truly require information from ALL hops?
- 5: Every hop provides essential, non-redundant information; removing any hop makes the question unanswerable
- 4: Most hops are necessary, one hop adds minor but useful context
- 3: The question could potentially be answered with fewer hops
- 2: One or more hops are clearly unnecessary or redundant
- 1: The question can be answered from a single hop; other hops are decorative

### 3. Question Clarity (question_clarity)
Is the question clear, unambiguous, and well-formed?
- 5: Crystal clear, no ambiguity, a human could answer it given the right documents
- 4: Mostly clear with minor verbosity
- 3: Somewhat unclear or overly complex phrasing
- 2: Ambiguous, could be interpreted multiple ways
- 1: Incomprehensible or severely malformed

## Input

**Question**: {question}

**Answer**: {answer}

**Hop chain**:
{hop_chain}

## Output

Return ONLY a JSON object:
```json
{{"answer_correctness": <1-5>, "multihop_necessity": <1-5>, "question_clarity": <1-5>, "total": <sum>, "reason": "<one sentence explanation>"}}
```"""

JUDGE_PROMPT_TEMPLATE_ZH = """你是一个严格的多跳问答数据集质量评估专家。

给定一条多跳 QA 及其支撑证据 chunk，从三个维度进行评估。
每个维度从 1（最差）到 5（最好）评分。

## 评分标准

### 1. 答案正确性 (answer_correctness)
答案是否可从给定 chunk 内容中验证？
- 5：答案被 chunk 直接且清晰地支持
- 4：答案被支持，但需要少量推理
- 3：答案部分被支持，部分信息模糊
- 2：答案支持较弱，关键证据缺失或不清晰
- 1：答案看起来不正确或与 chunk 矛盾

### 2. 多跳必要性 (multihop_necessity)
回答该问题是否真正需要所有跳的信息？
- 5：每一跳都提供必要且不冗余的信息；去掉任何一跳都无法回答
- 4：大部分跳是必要的，有一跳提供了次要但有用的上下文
- 3：该问题可能用更少的跳就能回答
- 2：一个或多个跳明显不必要或冗余
- 1：该问题可以从单个跳回答；其他跳是装饰性的

### 3. 问题清晰度 (question_clarity)
问题是否清晰、无歧义、表述良好？
- 5：非常清晰，无歧义，给定正确文档人类就能回答
- 4：大部分清晰，有少量冗余
- 3：有些不清晰或措辞过于复杂
- 2：有歧义，可以有多种解读
- 1：无法理解或严重格式问题

## 输入

**问题**: {question}

**答案**: {answer}

**推理链**:
{hop_chain}

## 输出

仅返回一个 JSON 对象：
```json
{{"answer_correctness": <1-5>, "multihop_necessity": <1-5>, "question_clarity": <1-5>, "total": <sum>, "reason": "<一句话说明>"}}
```"""


def get_judge_prompt(lang=None):
    return JUDGE_PROMPT_TEMPLATE_ZH if _get_lang(lang) == "zh" else JUDGE_PROMPT_TEMPLATE_EN


def build_hop_chain(qa: dict, corpus_lookup: dict) -> str:
    """构建 hop chain 文本，包含 chunk 内容摘要"""
    lines = []
    for hop in qa["hops"]:
        chunk_id = hop.get("chunk_id", "?")
        chunk = corpus_lookup.get(chunk_id, {})
        # 截取 chunk 前 500 字符作为 evidence
        text_preview = chunk.get("text", "")[:500].replace("\n", " ")
        title = hop.get("title", chunk.get("title", "?"))

        lines.append(f"**Hop {hop['hop_idx']}** (chunk: {chunk_id}, doc: {title})")
        lines.append(f"  Sub-question: {hop['question']}")
        lines.append(f"  Sub-answer: {hop.get('answer', '?')}")
        lines.append(f"  Evidence: {text_preview}")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# Judge execution
# ============================================================

def judge_one(qa: dict, corpus_lookup: dict, model: str = "gpt-oss-120b", lang: str = None) -> dict:
    """对单条 QA 做 judge 评分"""
    hop_chain = build_hop_chain(qa, corpus_lookup)
    prompt = get_judge_prompt(lang).format(
        question=qa["question"],
        answer=qa["answer"],
        hop_chain=hop_chain,
    )

    for attempt in range(3):
        try:
            resp = llm_call(prompt, model=model, temperature=0.0, timeout=120)
            parsed = _extract_json(resp)
            if parsed and isinstance(parsed, dict) and "answer_correctness" in parsed:
                # 确保分数是整数
                for key in ["answer_correctness", "multihop_necessity", "question_clarity"]:
                    parsed[key] = int(parsed.get(key, 1))
                parsed["total"] = parsed["answer_correctness"] + parsed["multihop_necessity"] + parsed["question_clarity"]
                return parsed
            logger.warning(f"Judge parse failed (attempt {attempt+1}): {resp[:200]}")
        except Exception as e:
            logger.warning(f"Judge call failed (attempt {attempt+1}): {e}")
            time.sleep(2 * (attempt + 1))

    # 默认返回最低分
    return {"answer_correctness": 0, "multihop_necessity": 0, "question_clarity": 0,
            "total": 0, "reason": "judge_failed"}


# ============================================================
# Filter mode
# ============================================================

def filter_by_score(input_path: str, output_path: str, min_score: int = 9):
    """按已有 judge 分数过滤"""
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    total = len(results)
    has_judge = [r for r in results if "judge" in r]
    if not has_judge:
        print(f"错误: 输入文件中没有 judge 评分，请先运行评分模式")
        return

    keep = [r for r in results if r.get("judge", {}).get("total", 0) >= min_score]
    removed = total - len(keep)

    # 统计分数分布
    scores = Counter(r.get("judge", {}).get("total", 0) for r in results)
    print(f"分数分布:")
    for s in sorted(scores.keys()):
        marker = " ← cutoff" if s == min_score else ""
        print(f"  total={s}: {scores[s]} 条{marker}")

    # 各维度低分统计
    for dim in ["answer_correctness", "multihop_necessity", "question_clarity"]:
        low = sum(1 for r in results if r.get("judge", {}).get(dim, 0) <= 2)
        print(f"  {dim} ≤ 2: {low} 条")

    # Re-index
    for i, r in enumerate(keep):
        r["id"] = f"mhop_{i:06d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n过滤: {removed}/{total} ({removed/total*100:.1f}%)")
    print(f"保留: {len(keep)} 条 (min_score={min_score})")
    print(f"写出: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Judge 清洗合成 QA")
    parser.add_argument("--input", required=True)
    parser.add_argument("--corpus", default="data/financial_all/corpus_all.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="只评前 N 条")
    parser.add_argument("--filter-only", action="store_true", help="只按分数过滤（不调 LLM）")
    parser.add_argument("--min-score", type=int, default=9, help="过滤阈值（总分 < min_score 过滤）")
    parser.add_argument("--resume", action="store_true", help="跳过已有 judge 的条目")
    parser.add_argument("--lang", choices=["en", "zh"], default=None,
                        help="Prompt 语言（默认读 config.PROMPT_LANG）")
    args = parser.parse_args()

    # Filter-only mode
    if args.filter_only:
        filter_by_score(args.input, args.output, args.min_score)
        return

    # Load corpus
    logger.info(f"Loading corpus from {args.corpus}")
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus_lookup = {c["chunk_id"]: c for c in json.load(f)}
    logger.info(f"Corpus: {len(corpus_lookup)} chunks")

    # Load QA
    results = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} QAs")

    if args.limit > 0:
        results = results[:args.limit]
        logger.info(f"Limited to {len(results)} QAs")

    # Init
    init_concurrency(args.workers)
    reset_stats()

    # Judge
    write_lock = Lock()
    judged_count = [0]
    skipped = [0]

    def _judge_item(idx, qa):
        if args.resume and "judge" in qa:
            skipped[0] += 1
            return qa

        scores = judge_one(qa, corpus_lookup, model=args.model, lang=args.lang)
        qa["judge"] = scores

        with write_lock:
            judged_count[0] += 1
            if judged_count[0] % 50 == 0:
                stats = get_stats()
                logger.info(f"Progress: {judged_count[0]}/{len(results)}, "
                           f"LLM calls: {stats['calls']}, errors: {stats['errors']}")
        return qa

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_judge_item, i, r): i for i, r in enumerate(results)}
        judged_results = [None] * len(results)
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                judged_results[idx] = fut.result()
            except Exception as e:
                logger.error(f"Item {idx} failed: {e}")
                judged_results[idx] = results[idx]

    elapsed = time.time() - t0
    stats = get_stats()

    # Write results with judge scores
    with open(args.output, "w", encoding="utf-8") as f:
        for r in judged_results:
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    scored = [r for r in judged_results if r and "judge" in r and r["judge"].get("total", 0) > 0]
    if scored:
        totals = [r["judge"]["total"] for r in scored]
        below_threshold = sum(1 for t in totals if t < args.min_score)
        print(f"\n{'='*50}")
        print(f"Judge 完成: {len(scored)} 条, 用时 {elapsed/60:.1f}min")
        print(f"LLM calls: {stats['calls']}, errors: {stats['errors']}")
        print(f"分数: avg={sum(totals)/len(totals):.1f}, min={min(totals)}, max={max(totals)}")
        print(f"低于 {args.min_score} 分: {below_threshold} ({below_threshold/len(scored)*100:.1f}%)")
        print(f"写出: {args.output}")
        print(f"\n下一步过滤:")
        print(f"  python scripts/judge_synthesis.py \\")
        print(f"    --input {args.output} --filter-only --min-score {args.min_score} \\")
        print(f"    --output {os.path.dirname(args.output)}/multihop_final.jsonl")
        print(f"{'='*50}")

        # 分数分布
        score_dist = Counter(totals)
        print(f"\n分数分布:")
        for s in sorted(score_dist.keys()):
            bar = "█" * score_dist[s]
            print(f"  {s:2d}: {score_dist[s]:4d} {bar}")


if __name__ == "__main__":
    main()
