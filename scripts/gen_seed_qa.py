#!/usr/bin/env python3
"""种子 QA 生成：从 chunk 生成原子 QA 对

用法:
  python scripts/gen_seed_qa.py \
    --corpus data/news_corpus/en/corpus.json \
    --output data/news_synthesis/seeds.jsonl \
    --model mog-1 --workers 20 --limit 2000
"""
import argparse
import json
import logging
import os
import random
import re
import string
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.synthesis_llm import (
    llm_call_with_retry,
    init_concurrency,
    get_stats,
    reset_stats,
)

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gen_seed_qa")

# ---------- 文本工具 ----------
def _tokens(text: str) -> list[str]:
    return re.findall(r'\w+', text, flags=re.UNICODE)


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


# ---------- 过滤逻辑 ----------
def filter_qa(question: str, answer: str, chunk_text: str) -> bool:
    """返回 True 表示保留"""
    # 答案过长
    if len(_tokens(answer)) >= 10:
        return False
    # 答案在问题中出现
    if normalize_answer(answer) in normalize_answer(question):
        return False
    # 答案 token 与问题重叠过多
    atokens = _tokens(answer)
    qtokens = _tokens(question)
    intokens = [t for t in atokens if t in qtokens]
    if len(intokens) > 0.5 * len(atokens):
        return False
    # 含 and/or（复合答案）
    if "and" in atokens or "or" in atokens or "&" in answer:
        return False
    # 问题引用文档本身
    qtokens_lower = [t.lower() for t in _tokens(question)]
    if "document" in qtokens_lower or "article" in qtokens_lower or "according" in qtokens_lower:
        return False
    # 要求全名的问题
    q_lower = question.lower()
    for pattern in ["full name", "original name", "alternate name", "alternative name", "name one"]:
        if pattern in q_lower:
            return False
    return True


# ---------- 单 chunk 处理 ----------
def process_chunk(chunk: dict, prompts: dict, model: str, gen_qa_num: int = 3) -> list[dict]:
    """对单个 chunk 生成 seed QA"""
    chunk_id = chunk["chunk_id"]
    text = chunk["text"]
    title = chunk.get("title", "")

    # 跳过过短的 chunk（兼容中文：用字符数兜底）
    if len(text.split()) < 50 and len(text) < 200:
        return []

    # 生成 QA
    gen_prompt = prompts["gen_qa_prompt"].format(
        gen_qa_num=gen_qa_num,
        input_doc=f'"{title}"\n{text}' if title else text,
    )
    raw_qas = llm_call_with_retry(gen_prompt, model=model, return_json=True, max_retries=3)
    if not raw_qas or not isinstance(raw_qas, list):
        return []

    # 过滤
    filtered = []
    seen_answers = set()
    seen_questions = set()
    for item in raw_qas:
        if not isinstance(item, dict):
            continue
        q = item.get("question", "")
        a = item.get("answer", "")
        if not q or not a:
            continue
        norm_a = normalize_answer(a)
        norm_q = normalize_answer(q)
        if norm_a in seen_answers or norm_q in seen_questions:
            continue
        if not filter_qa(q, a, text):
            continue
        seen_answers.add(norm_a)
        seen_questions.add(norm_q)
        filtered.append({"question": q, "answer": a})

    filtered = filtered[:gen_qa_num]

    # 精炼答案
    results = []
    for qa in filtered:
        refine_prompt = prompts["refine_prompt"].format(
            question=qa["question"],
            original_answer=qa["answer"],
        )
        refined = llm_call_with_retry(refine_prompt, model=model, return_json=True, max_retries=2)
        if refined and isinstance(refined, dict):
            refined_answer = refined.get("refined_answer", qa["answer"])
        else:
            refined_answer = qa["answer"]

        # 再次检查精炼后答案长度
        if len(_tokens(refined_answer)) >= 10:
            continue

        results.append({
            "chunk_id": chunk_id,
            "title": title,
            "question": qa["question"],
            "answer": qa["answer"],
            "refined_answer": refined_answer,
        })

    return results


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="Generate seed QA from news chunks")
    parser.add_argument("--corpus", default="data/news_corpus/en/corpus.json")
    parser.add_argument("--output", default="data/news_synthesis/seeds.jsonl")
    parser.add_argument("--prompts", default="scripts/synthesis_prompts.yaml")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=2000, help="Max chunks to sample")
    parser.add_argument("--gen-qa-num", type=int, default=3, help="Max QA per chunk")
    parser.add_argument("--resume", action="store_true", help="Skip already processed chunks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 加载 prompts
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # 加载语料
    logger.info(f"Loading corpus from {args.corpus}")
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    logger.info(f"Total chunks: {len(corpus)}")

    # 采样
    random.seed(args.seed)
    if args.limit and args.limit < len(corpus):
        corpus = random.sample(corpus, args.limit)
    logger.info(f"Sampled {len(corpus)} chunks")

    # 断点续跑
    processed_chunks = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_chunks.add(data["chunk_id"])
                except Exception:
                    continue
        logger.info(f"Resume: skipping {len(processed_chunks)} already processed chunks")
        corpus = [c for c in corpus if c["chunk_id"] not in processed_chunks]

    if not corpus:
        logger.info("No chunks to process")
        return

    # 初始化
    init_concurrency(args.workers)
    reset_stats()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 并行处理
    write_lock = Lock()
    total_seeds = 0

    def _process_and_write(chunk):
        nonlocal total_seeds
        try:
            seeds = process_chunk(chunk, prompts, args.model, args.gen_qa_num)
            if seeds:
                with write_lock:
                    with open(args.output, "a", encoding="utf-8") as f:
                        for s in seeds:
                            f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    total_seeds += len(seeds)
            return len(seeds)
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.get('chunk_id', '?')}: {e}")
            return 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_process_and_write, c) for c in corpus]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating seeds"):
            fut.result()

    # 统计
    stats = get_stats()
    logger.info(f"Done! Generated {total_seeds} seed QAs from {len(corpus)} chunks")
    logger.info(f"LLM calls: {stats['calls']}, errors: {stats['errors']}, "
                f"total latency: {stats['total_latency']:.1f}s")


if __name__ == "__main__":
    main()
