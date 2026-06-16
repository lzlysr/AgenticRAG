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
    """使用正则提取连续的“单词字符”"""
    return re.findall(r'\w+', text, flags=re.UNICODE)


def normalize_answer(s: str) -> str:
    """标准化。通过小写化、去除标点、去除冠词、合并多余空格等方式，降低表面差异的影响"""
    def remove_articles(text):
        # 删除英文冠词
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        # 合并多余空格
        return " ".join(text.split())
    def remove_punc(text):
        # 删除标点并转小写
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


# ---------- 过滤逻辑 ----------
def filter_qa(question: str, answer: str, chunk_text: str) -> bool:
    """判断一条 QA 是否符合“原子 QA”的要求。返回 True 表示保留"""
    # 答案过长
    if len(_tokens(answer)) >= 10:
        return False
    # 答案在问题中出现 这种 QA 没有训练价值，因此删除。
    if normalize_answer(answer) in normalize_answer(question):
        return False
    # 答案 token 与问题重叠过多
    atokens = _tokens(answer)
    qtokens = _tokens(question)
    intokens = [t for t in atokens if t in qtokens]
    # 统计答案 token 中有多少也出现在问题里，如果超过 50%，说明问题和答案高度重叠，可能是一些无意义的 QA，应该过滤掉。
    if len(intokens) > 0.5 * len(atokens):
        return False
    # 含 and/or（复合答案）
    # 这有利于生成原子 QA，但也可能误删合法实体，例如某些公司或作品名称中本来就含有 and。
    # 此外，因为 atokens 没有转小写，答案中的：And 可能漏过检查。
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
    """
    对单个 chunk 生成 seed QA
    每个 chunk 单独配一整段 gen_qa_prompt，然后单独调用一次 LLM。会不会有点浪费？是的，但好处是每个 chunk 都能得到针对性的 QA，且更容易控制生成质量和数量。
    """
    chunk_id = chunk["chunk_id"]
    text = chunk["text"]
    title = chunk.get("title", "")

    # 跳过过短的 chunk（兼容中文：用字符数兜底）
    if len(text.split()) < 50 and len(text) < 200:
        return []

    # 生成 QA
    gen_prompt = prompts["gen_qa_prompt"].format(
        gen_qa_num=gen_qa_num,
        input_doc=f'"{title}"\n{text}' if title else text, # 没有标题则直接用文本
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
        # 同一个 chunk 中，如果两个 QA 的答案高度相似（标准化后完全一样），或者问题高度相似，我们只保留第一个，后面出现的类似 QA 就丢弃掉。这是为了增加种子 QA 的多样性，避免过多重复的 QA。
        if norm_a in seen_answers or norm_q in seen_questions:
            continue
        # filter_qa内部会标准化问题和答案，因此这里直接传原始文本就行，不需要再标准化一次。
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

    # 如果语料库的chunk数量超过限制，就随机采样
    random.seed(args.seed)
    if args.limit and args.limit < len(corpus):
        corpus = random.sample(corpus, args.limit)
    logger.info(f"Sampled {len(corpus)} chunks")

    # 断点续跑
    processed_chunks = set()
    if args.resume and os.path.exists(args.output):
        # 只有同时满足：使用了 --resume，且输出文件已经存在，才会启用断点续跑功能。它会读取已经生成的 QA 文件，提取出其中的 chunk_id，记录在 processed_chunks 集合里。然后在后续处理时，如果一个 chunk 的 chunk_id 已经在 processed_chunks 里，就跳过这个 chunk，不再处理。这可以避免重复生成已经处理过的 chunk 的 QA，从而节省时间和计算资源。
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

    # 并行处理  多个线程会同时生成结果，但不能同时写同一个文件，所以使用 write_lock。
    write_lock = Lock()
    total_seeds = 0

    def _process_and_write(chunk):
        # nonlocal 表示修改的是外层 main() 中的 total_seeds。
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
