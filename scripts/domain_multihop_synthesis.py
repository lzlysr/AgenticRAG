#!/usr/bin/env python3
"""核心多跳 QA 合成 pipeline：从 seed QA 自底向上生成多跳问题

Seed QA （seeds.jsonl，目前没有从gen_seed_qa.py真正生成）
  ↓
逐 hop 扩展（DomainMultiHopPipeline）
  ↓
检索（FAISS + BM25 + Graph）
  ↓
LLM生成新QA
  ↓
过滤（规则 + overlap + depth）
  ↓
LLM merge（生成多跳问题）
  ↓
4层验证（semantic / reasoning / single-doc / full-doc）
  ↓
输出 multi-hop QA

用法:
  python scripts/domain_multihop_synthesis.py \
    --seeds data/news_synthesis/seeds.jsonl \
    --corpus data/news_corpus/en/corpus.json \
    --index-dir data/news_indexes/en/ \
    --output data/news_synthesis/multihop_raw.jsonl \
    --num-hop 4 --workers 10 --model mog-1
"""
import argparse
import json
import logging
import os
import pickle
import random
import re
import string
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from threading import Lock

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.synthesis_llm import (
    llm_call_with_retry,
    llm_judge,
    init_concurrency,
    get_stats,
    reset_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("multihop_synthesis")

# ============================================================
# 文本工具（复刻 AgenticRAGTracer）
# ============================================================

def _tokens(text: str) -> list[str]:
    return re.findall(r'\w+', text, flags=re.UNICODE)


def normalize_answer(s: str) -> str:
    """标准化。做语义对齐时用，降低表面差异的影响"""
    def remove_articles(text):
        # 去冠词 + 停用词
        return re.sub(r"\b(a|an|the|do|does|is|are|was|were|of|under|in|at|on|with|by|for|from|about)\b", " ", text)
    def white_space_fix(text):
        # 合并多余空格
        return " ".join(text.split())
    def remove_punc(text):
        # 删除标点并转小写
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _years(text):
    """年份检测器,用于 multi-hop QA 的时间一致性约束"""
    return re.findall(r'\b\d{4}s?\b', text, flags=re.UNICODE | re.IGNORECASE)


def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def simple_partial_presence(phrase: str, sentence: str) -> bool:
    """检查 phrase（上一跳答案） 的关键 token 是否部分出现在 sentence（当前新生成的问题） 中"""
    # 过滤无意义词
    prepositions = {"Of", "Under", "In", "At", "On", "With", "By", "For", "From", "About",
                    "An", "The", "Do", "Does", "Is", "Were", "Was", "Are"}
    def filtertokens(tokens):
        # 只保留以下 token：首字母大写、全部大写、数字，然后排除无意义词
        return [t for t in tokens if (t[0].isupper() or t.isupper() or is_numeric(t)) and t not in prepositions]

    p_tokens = filtertokens(_tokens(phrase))
    s_tokens = filtertokens(_tokens(sentence))
    if not p_tokens:
        return False
    # 完全匹配 → 不算 partial
    plen = len(p_tokens)
    for i in range(len(s_tokens) - plen + 1):
        # 如果 phrase 完整出现在 sentence 中：返回 False
        # 为什么“完整匹配”反而返回 False？
        # 因为函数名叫 partial_presence，它专门检测部分泄露。完整泄露在前面已经有其他过滤逻辑处理了
        if s_tokens[i:i+plen] == p_tokens:
            return False
    # 如果上一跳答案和当前问题的关键 token 有交集（&），即
    # 如果 phrase 的部分关键 token 出现在 sentence 中：返回 True
    return bool(set(p_tokens) & set(s_tokens))


def f1_score(prediction: str, ground_truths) -> float:
    if prediction is None or ground_truths is None:
        return 0.0
    if prediction.startswith("I cannot answer this question"):
        return 0.0
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    # 模型答案只要和任意一个标准答案别名匹配得好，就使用最高分。
    max_f1 = 0.0
    for gt in ground_truths:
        if gt is None:
            continue
        pred_norm = normalize_answer(prediction)
        gt_norm = normalize_answer(gt)
        # 对于 yes/no 类答案，不允许通过部分 token 重合获得分数。eg：下面情况不算分
        # pred_norm = "yes"
        # gt_norm = "yes company profitable"
        # 只要预测或标准答案属于 yes/no/noanswer，就要求两者标准化后必须完全一致；
        # 不一致则该标准答案直接记为 0。
        if pred_norm in ["yes", "no", "noanswer"] or gt_norm in ["yes", "no", "noanswer"]:
            if pred_norm != gt_norm:
                continue
        ptoks = pred_norm.split()
        gtoks = gt_norm.split()
        common = Counter(ptoks) & Counter(gtoks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(ptoks)
        rec = num_same / len(gtoks)
        f1 = (2 * prec * rec) / (prec + rec)
        max_f1 = max(max_f1, f1)
    return max_f1


# ============================================================
# DomainRetriever：直接加载项目索引做检索。
# ============================================================

# 作用：用来给当前 seed / 当前 hop 找下一跳可连接的文档 chunk。
# 同时保证多跳路径上文档的多样性：
# 1.排除已经使用过的 chunk（ exclude_ids ）。
# 2.跨文档偏好（ seed_title ）。

# 不依赖前面封装好的 semantic_search()、keyword_search()。在脚本内部完成：
# Query
#   ↓
# BGE-M3 编码
#   ↓
# FAISS 稠密召回 ─┐
#                  ├→ RRF 融合 → CrossEncoder 重排
# BM25 关键词召回 ─┘
#   ↓
# 跨文档结果调整
#   ↓
# 返回 top_k 个 chunk


class _GPUModelPool:
    """
    多 GPU 模型池：在多个 GPU 上加载 embedder + reranker，用 Queue 分发
    模型池的主要目标是： 
    在每张 GPU 上只加载一份 embedding 模型和一份 reranker 模型，然后让多个检索任务复用。
    """

    def __init__(self, gpu_ids: list[int] = None, max_per_gpu: int = 5):
        # max_per_gpu：每张 GPU 允许同时分配多少个检索名额，不表示每张 GPU 加载 5 份模型
        import torch
        from queue import Queue

        if gpu_ids is None:
            n_gpus = torch.cuda.device_count()
            gpu_ids = list(range(n_gpus)) if n_gpus > 0 else []

        self._queue = Queue() # 保存当前可用的设备 slot。放的不是模型对象，而是 device 字符串
        self._models = {}  # gpu_id → (embedder, reranker)

        if not gpu_ids:
            # CPU fallback
            logger.info("GPU pool: no GPU, using CPU")
            self._load_on_device("cpu")
            self._queue.put("cpu")
        else:
            for gid in gpu_ids:
                device = f"cuda:{gid}"
                self._load_on_device(device)
                # 每个 GPU 加载好模型后，放入 max_per_gpu 个 slot 供检索任务使用
                for _ in range(max_per_gpu):
                    self._queue.put(device)
            logger.info(f"GPU pool: {len(gpu_ids)} GPUs {gpu_ids}, "
                        f"{max_per_gpu}/GPU, total {len(gpu_ids)*max_per_gpu} slots")

    def _load_on_device(self, device: str):
        """在指定设备上加载 embedder 和 reranker 模型，并保存到 self._models 中"""
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from config import BGE_M3_PATH, BGE_RERANKER_PATH

        logger.info(f"  Loading embedder + reranker on {device}")
        embedder = SentenceTransformer(BGE_M3_PATH, device=device)
        reranker = CrossEncoder(BGE_RERANKER_PATH, max_length=512, device=device)
        self._models[device] = (embedder, reranker)

    def acquire(self):
        """获取一个 GPU slot，返回 (device, embedder, reranker)"""
        device = self._queue.get()
        emb, rnk = self._models[device]
        return device, emb, rnk

    def release(self, device: str):
        """归还 GPU slot"""
        self._queue.put(device)

# 为什么单独实现 DomainRetriever，不直接用hybrid_search.py？主要原因是它有一些特殊需求：
# 一个 seed 会不断扩展多跳，需要高频检索；
# 希望优先检索不同文档中的内容；
# 需要记录候选来自 FAISS 还是 BM25；
# 多线程合成时，需要为不同线程分配 GPU；
# 所以这个类是专门服务于多跳 QA 合成流程的。

class DomainRetriever:
    """直接调用 FAISS + BM25 + Reranker 进行检索（多 GPU 模型池）"""

    def __init__(self, index_dir: str, gpu_ids: list[int] = None, max_per_gpu: int = 5):
        import faiss
        import numpy as np

        logger.info(f"Loading indexes from {index_dir}")
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "chunk_ids.json"), "r") as f:
            self.chunk_ids = json.load(f)
        with open(os.path.join(index_dir, "chunk_store.pkl"), "rb") as f:
            self.chunk_store = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        logger.info(f"Loaded {len(self.chunk_ids)} chunks, FAISS dim={self.faiss_index.d}")

        # 多 GPU 模型池
        self._pool = _GPUModelPool(gpu_ids=gpu_ids, max_per_gpu=max_per_gpu)

    def search(self, query: str, top_k: int = 10, exclude_ids: set = None,
               seed_title: str = None) -> list[dict]:
        """混合检索：FAISS + BM25 → RRF 融合 → Reranker 重排

        Args:
            exclude_ids：检索时排除的 chunk_id 集合（通常是当前 multi-hop 路径上已经使用过的 chunk_id，避免重复）
            seed_title: 是原始 seed，也就是 hop_1 的标题，不是当前这一跳的标题。如果提供，会确保结果中包含来自不同文档的 chunk（跨文档偏好）
        """
        import numpy as np
        exclude_ids = exclude_ids or set()

        # 从 GPU 池获取模型
        device, embedder, reranker = self._pool.acquire()
        try:
            return self._search_impl(query, top_k, exclude_ids, seed_title,
                                     embedder, reranker)
        finally:
            self._pool.release(device)

    def _search_impl(self, query, top_k, exclude_ids, seed_title,
                     embedder, reranker):
        """实际的检索实现，接受 embedder 和 reranker 作为参数，供 search() 调用"""
        import numpy as np

        # FAISS 检索
        q_vec = embedder.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype=np.float32)
        # 召回 top_k 的 4 倍候选，后续通过 RRF + reranker 精排到 top_k
        faiss_k = min(top_k * 4, len(self.chunk_ids))
        scores_f, indices_f = self.faiss_index.search(q_vec, faiss_k)
        faiss_results = []
        # 因为只有一个 query，所以取[0]
        for i, idx in enumerate(indices_f[0]):
            if idx == -1:
                continue
            cid = self.chunk_ids[idx]
            if cid in exclude_ids:
                continue
            doc = self.chunk_store[cid]
            faiss_results.append({
                "chunk_id": cid,
                "text": doc["text"],
                "title": doc.get("title", ""),
                "score": float(scores_f[0][i]),
            })

        # BM25 检索
        from retrieval.keyword_search import tokenize
        tokens = tokenize(query)
        # get_scores 返回每个 chunk 对这个 query 的 BM25 分数
        # 是一个 NumPy 数组，长度等于语料库里的 chunk 数量。
        # 这个数组的下标和 chunk_ids.json 的下标一一对应
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = bm25_scores.argsort()[-faiss_k:][::-1]
        bm25_results = []
        for idx in bm25_top:
            if bm25_scores[idx] <= 0:
                break
            cid = self.chunk_ids[idx]
            if cid in exclude_ids:
                continue
            doc = self.chunk_store[cid]
            bm25_results.append({
                "chunk_id": cid,
                "text": doc["text"],
                "title": doc.get("title", ""),
                "score": float(bm25_scores[idx]),
            })

        # RRF 融合
        rrf_k = 60
        chunk_scores = {}
        chunk_data = {}
        chunk_sources = {}  # chunk_id → set of tool names 保存每个 chunk 是被哪些检索方式找到的
        for rank, r in enumerate(faiss_results):
            cid = r["chunk_id"]
            chunk_scores[cid] = chunk_scores.get(cid, 0) + 1.0 / (rrf_k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = r
            # 记录来源。setdefault(cid, set()) 的意思是：
            # 如果 cid 不在 chunk_sources 中，就创建一个空集合；set可以防止来源重复。
            chunk_sources.setdefault(cid, set()).add("semantic_search")
        for rank, r in enumerate(bm25_results):
            cid = r["chunk_id"]
            chunk_scores[cid] = chunk_scores.get(cid, 0) + 1.0 / (rrf_k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = r
            chunk_sources.setdefault(cid, set()).add("keyword_search")

        sorted_ids = sorted(chunk_scores, key=lambda x: chunk_scores[x], reverse=True)
        candidates = [chunk_data[cid] for cid in sorted_ids[:top_k * 3]]

        if not candidates:
            return []

        # Reranker 重排
        passages = [c["text"] for c in candidates]
        pairs = [[query, p] for p in passages]
        scores = reranker.predict(pairs)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        reranked = indexed[:min(top_k * 2, len(candidates))]
        results = []
        for orig_idx, score in reranked:
            doc = candidates[orig_idx]
            results.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "title": doc.get("title", ""),
                "score": float(score),
                "_sources": sorted(chunk_sources.get(doc["chunk_id"], {"semantic_search"})),
            })

        # 多跳 QA 的一个常见问题是：后续检索一直从同一篇文章、同一份报告中找相邻内容。
        # 这样虽然用了多个 chunk，但并不是真正有挑战性的跨文档推理。
        # 所以增加跨文档偏好：如果 seed_title 提供，确保结果中有不同来源的文档。
        # 增加生成真正跨文档多跳问题的机会
        if seed_title and results:
            same_doc = [r for r in results if r["title"] == seed_title]
            cross_doc = [r for r in results if r["title"] != seed_title]
            if cross_doc:
                # 交替排列：优先放跨文档的，保证被选到
                interleaved = []
                ci, si = 0, 0 # cross_doc 当前下标，same_doc 当前下标
                for _ in range(top_k):
                    # 先加入跨文档
                    if ci < len(cross_doc):
                        interleaved.append(cross_doc[ci])
                        ci += 1
                    # 再加入同文档
                    if si < len(same_doc) and len(interleaved) < top_k:
                        interleaved.append(same_doc[si])
                        si += 1
                results = interleaved[:top_k]
                logger.info(f"    Cross-doc boost: {len(cross_doc)} cross / {len(same_doc)} same → top-{top_k} has {sum(1 for r in results if r['title'] != seed_title)} cross")
            else:
                results = results[:top_k]
        else:
            results = results[:top_k]

        return results


# ============================================================
# DomainMultiHopPipeline
# ============================================================

# 作用：把证据逐步组织成多跳问题，并通过一系列过滤和验证，判断它是否真的是一个合格的多跳 QA。

# 流程：

# Seed QA
#   ↓
# 整理已有问题、答案、chunk_id
#   ↓
# 排除已经使用过的 chunk 和相邻页 chunk
#   ↓
# 检索新 chunk （使用 DomainRetriever）
#   ↓
# 从新 chunk 生成原子 QA
#   ↓
# 过滤原子 QA
#   ↓
# 与已有链合并成多跳问题
#   ↓
# 前置规则过滤
#   ↓
# 四重验证
#   ↓
# 保存为新的 hop 链

# current_data 的结构：
# {
#     "hop_1": {
#         "question": "Who founded Company A?", # 当前新增文档中生成的原子问题。
#         "answer": "John Smith",
#         "doc": "...",
#         "final_question": "Who founded Company A?", # 把当前问题与前面所有 hop 合成后的多跳问题。
#         "final_answer": "John Smith",
#         "refined_answer": "John Smith",
#         "qa_type": "initial_qa",
#         "chunk_id": "docA_001",
#         "title": "Company A"
#     },

#     "hop_2": {
#         "question": "Where was John Smith born?",
#         "answer": "London",
#         "doc": "...",

#         "final_question":
#             "Where was the founder of Company A born?",

#         "final_answer": "London",
#         "refined_answer": "London",

#         "qa_type": "inference",
#         "chunk_id": "docB_003",
#         "title": "John Smith",

#         "optional_answers": ["London", "London, England"],
#         "verify_result": [...],
#         "search_tools": ["semantic_search"],
#         "search_query": "Who founded Company A?"
#     }
# }

class DomainMultiHopPipeline:
    """适配自 AgenticRAGTracer 的 MultiHopPipeline"""

    def __init__(self, model_id: str, retriever: DomainRetriever, prompts: dict,
                 merge_model_id: str = None, corpus_lookup: dict = None):
        self.model_id = model_id # 普通步骤使用的模型
        self.merge_model_id = merge_model_id or model_id # 专门负责多跳问题合成的模型
        self.retriever = retriever
        self.prompts = prompts
        self._corpus_lookup = corpus_lookup or {}

    def compare_verify(self, prompt: str, final_question: str,
                       option_answers: list[str], std: str, desc: str) -> tuple:
        """
        验证：让 LLM 回答问题，与标准答案比对。

        std 和 desc 是用来判断这一轮验证是否失败，以及失败原因叫什么的。

        Args:
            std： 表示这次验证的“期望标准”。
            desc： 表示失败原因的描述。
        """
        llm_answer = llm_call_with_retry(prompt, model=self.model_id, max_retries=2) or ""
        # f1 主要用于记录和日志，不直接决定验证是否通过。
        f1 = f1_score(llm_answer, option_answers)
        # 让另一个 LLM 判断：llm_answer 是否与标准答案语义等价
        esseq = llm_judge(
            final_question,
            golden_answer=option_answers[0], # 确保第一个答案是最标准的答案
            other_answer=llm_answer,
            judge_prompt=self.prompts["EssEq_prompt"],
            model=self.model_id,
        )
        verification = ''
        # avg_score >= 1 代表 LLM 基本答对了。
        # std = "mid" 表示这一步 LLM 理论上不应该答对。
        # 比如不给文档或者给部分文档，只给最终问题，LLM 不应该能回答。
        if std == 'mid' and esseq["avg_score"] >= 1:
            verification = desc
        # avg_score < 1 代表 LLM 没答对
        # std = "final" 表示这一步 LLM 理论上应该答对。
        elif std == 'final' and esseq["avg_score"] < 1:
            verification = desc
        if not verification:
            verification = 'pass'
        return verification, llm_answer, f1, esseq

    def process_seed(self, seed: dict, corpus_lookup: dict,
                     num_hop: int, topk: int = 10,
                     gen_qa_num: int = 5, every_hop_qa_num: int = 15,
                     max_valid_per_hop: int = 3,
                     max_qa_per_seed: int = 5,
                     **kwargs) -> list[dict]:
        """
        从一个 seed 出发，逐跳扩展到 num_hop
        
        Args:
            seed：  初始单跳 QA。（来自 gen_seed_qa.py 生成的 seeds.jsonl）
            corpus_lookup:	chunk_id 到原文映射。
            num_hop:	最多扩展到几跳。
            topk：	每轮检索多少 chunk。
            gen_qa_num：	每个新 chunk 最多生成多少原子 QA。
            every_hop_qa_num：	每层最多保留多少条链继续扩展。
            max_valid_per_hop：	每个父链单次扩展最多生成多少有效子链。
            max_qa_per_seed：   一个 seed 最多输出多少条 QA。
        """
        chunk_id = seed["chunk_id"]
        doc = corpus_lookup.get(chunk_id, {})
        original_doc = f'"{seed.get("title", "")}"\n{doc.get("text", "")}'

        current_results = [{
            "hop_1": {
                "question": seed["question"],
                "answer": seed.get("refined_answer", seed["answer"]),
                "doc": original_doc,
                "final_question": seed["question"],
                "final_answer": seed.get("refined_answer", seed["answer"]),
                "refined_answer": seed.get("refined_answer", seed["answer"]),
                "qa_type": "initial_qa",
                "chunk_id": chunk_id,
                "title": seed.get("title", ""),
            }
        }]

        valid_results = []
        seed_title = seed.get("title", "")
        # per-hop 产出配额（可通过 kwargs 覆盖）
        # 表示一个 seed 最多输出 2 条 2-hop QA，2 条 3-hop QA，1 条 4-hop QA。
        # 但这只是各层输出配额，还会受到： max_qa_per_seed 的限制。
        hop_quotas = kwargs.get("hop_quotas", {2: 2, 3: 2, 4: 1})

        # 第 1 轮:
        # current_results = [
        #   {hop_1}
        # ]
        # 扩展后：
        # temp_results = [
        #   {hop_1, hop_2a},
        #   {hop_1, hop_2b}
        # ]
        # 赋值：
        # current_results = [
        #   {hop_1, hop_2a},
        #   {hop_1, hop_2b}
        # ]
        # 第 2 轮循环：创建一个新的 temp_results = []。for current_data in current_results:
        # 第 1 次处理整条链 {hop_1, hop_2a}
        # 第 2 次处理整条链 {hop_1, hop_2b}
        # 然后分别扩展出：
        # {hop_1, hop_2a, hop_3x}
        # {hop_1, hop_2a, hop_3y}
        # {hop_1, hop_2b, hop_3z}
        
        for hop in range(1, num_hop):
            hop_level = hop + 1  # 当前正在生成的 hop 层级
            # 1. 扩展所有父链
            # 例如当前有三条 2-hop 链，那么每条都会尝试扩展成若干条 3-hop 链。
            # temp_results 示当前 hop 层级所有父链扩展出来的有效子链集合。
            # temp_results 是 process_seed 每一层的“中间候选池”：
            # 它保存当前 hop 扩展出来的有效多跳链；其中一部分会进入最终输出 valid_results，另一部分还会作为下一轮 current_results 继续扩展更高跳数。
            temp_results = [] # 每轮创建一个新的[]
            for current_data in current_results:
                try:
                    # 每条父链最多生成：max_valid_per_hop 条有效子链。
                    new_items = self._extend_one_hop(
                        current_data, hop, topk, gen_qa_num,
                        max_valid=max_valid_per_hop,
                        seed_title=seed_title,
                    )
                    temp_results.extend(new_items)
                # 某条链扩展失败，只跳过这条链，不终止整个 seed。
                except Exception as e:
                    logger.warning(f"Error extending hop {hop+1}: {e}")

            # 2. 基于 chunk 集合去重。避免不同问题实际上使用了几乎相同的证据组合。
            # chunk overlap dedup：chunk_id 集合 >= 80% 相同视为重复
            temp_results = self._dedup_by_chunk_overlap(temp_results)

            logger.info(f"  Hop {hop_level}: {len(temp_results)} valid from {len(current_results)} parents")

            # 3. 按 hop 层级限制输出到最终结果的数量（quota=0 表示不输出但仍保留作为下一步输入）
            quota = hop_quotas.get(hop_level, 2)
            if quota > 0:
                output_items = temp_results[:quota]
                # 只把前 quota 条加入最终输出。
                # 但是没有被加入 valid_results 的其他 temp_results，仍继续参与下一层扩展。
                valid_results.extend(output_items)
                if len(temp_results) > quota:
                    logger.info(f"  Hop {hop_level} quota applied: output {quota}, pipeline {len(temp_results)}")
            # quota = 0。当前层完全不输出，但仍保留为下一跳输入。
            else:
                logger.info(f"  Hop {hop_level} quota=0: skip output, keep {len(temp_results)} for next hop")

            # 一个 seed 的总输出上限
            if len(valid_results) >= max_qa_per_seed:
                logger.info(f"  Seed cap reached: {len(valid_results)} >= {max_qa_per_seed}")
                valid_results = valid_results[:max_qa_per_seed]
                break

            if hop + 1 < num_hop:
                random.seed(42)
                # 多跳扩展会出现分支爆炸，所以所以每层最多随机选 every_hop_qa_num 条继续扩展。
                temp_results = random.sample(temp_results, min(len(temp_results), every_hop_qa_num))
                # 当前轮生成出来的新链，成为下一轮要继续扩展的父链。
                current_results = temp_results

        return valid_results

    @staticmethod
    def _dedup_by_chunk_overlap(results: list) -> list:
        """去重：chunk_id 集合 >= 80% 相同的视为重复，只保留第一个"""
        # 假设同一个 seed 扩展出了三条 2-hop 候选链：输入
        # results = [
        #     {
        #         "hop_1": {"chunk_id": "A"},
        #         "hop_2": {"chunk_id": "B"},
        #     },
        #     {
        #         "hop_1": {"chunk_id": "A"},
        #         "hop_2": {"chunk_id": "B"},
        #     },
        #     {
        #         "hop_1": {"chunk_id": "A"},
        #         "hop_2": {"chunk_id": "C"},
        #     },
        # ]
        if len(results) <= 1:
            return results
        deduped = [] # 最终保留下来的结果列表。
        seen_chunk_sets = [] # 之前已经保留下来的结果所使用的 chunk 集合。
        for r in results:
            # 1. 提取最大 hop
            max_hop = max(int(k.split("_")[1]) for k in r if k.startswith("hop_"))
            # 2. 提取 chunk 集合
            chunks = set()
            for h in range(1, max_hop + 1):
                cid = r.get(f"hop_{h}", {}).get("chunk_id")
                if cid:
                    chunks.add(cid)
            # 3. 检查与已有结果的重叠
            # 当前结果暂时认为不重复
            is_dup = False
            # 当前结果必须和所有之前保留的结果逐一比较。假设
            # chunks = {"A", "B", "C"}
            # seen_chunk_sets = [
            #     {"A", "B", "D"},
            #     {"A", "B", "C"},
            #     {"E", "F", "G"},
            # ]
            for prev_chunks in seen_chunk_sets:
                if not chunks or not prev_chunks:
                    continue
                overlap = len(chunks & prev_chunks) / max(len(chunks), len(prev_chunks))
                # 只要和任意一条已保留结果高度重合，就不再保留当前结果。
                if overlap >= 0.8:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(r)
                seen_chunk_sets.append(chunks)
        return deduped

    def _extend_one_hop(self, current_data: dict, hop: int,
                        topk: int, gen_qa_num: int,
                        max_valid: int = 3,
                        seed_title: str = None) -> list[dict]:
        """扩展一跳，找到 max_valid 个有效结果后提前退出"""
        temp_results = []
        # 当 hop=2 时，当前已有 hop_1 和 hop_2，这里取 hop_2。
        # 后续检索主要围绕最后一跳进行。
        last_hop_data = current_data[f"hop_{hop}"]
        # 收集全部已有文档和 chunk_id
        full_docs = [current_data[f"hop_{h}"]["doc"] for h in range(1, hop+1)]
        full_chunk_ids = set()
        for h in range(1, hop+1):
            cid = current_data[f"hop_{h}"].get("chunk_id")
            if cid:
                full_chunk_ids.add(cid)

        # P3: 排除同公司页码相邻的 chunk（同一张表格区域）
        if hasattr(self, '_corpus_lookup') and self._corpus_lookup:
            nearby_ids = set()
            for cid in list(full_chunk_ids):
                chunk_info = self._corpus_lookup.get(cid, {})
                pages = set(chunk_info.get("pages", []))
                if not pages:
                    continue
                # 从 chunk_id 推断公司前缀 (yh_0000 → yh)
                company_prefix = cid.split("_")[0]
                # 扩展页码范围 ±1
                expanded_pages = set()
                for p in pages:
                    expanded_pages.update([p - 1, p, p + 1])
                # 找同公司页码相邻的 chunk
                for other_cid, other_info in self._corpus_lookup.items():
                    if other_cid in full_chunk_ids:
                        continue
                    if not other_cid.startswith(company_prefix + "_"):
                        continue
                    other_pages = set(other_info.get("pages", []))
                    # 页码与 expanded_pages 有交集, 则认为是相邻 chunk
                    if other_pages & expanded_pages:
                        nearby_ids.add(other_cid)
            if nearby_ids:
                logger.info(f"    P3: excluding {len(nearby_ids)} nearby chunks (same section)")
                # |：排除
                full_chunk_ids = full_chunk_ids | nearby_ids

        # 保存的是每一阶段的累计问题和答案，不是各 hop 自己的原子问题。
        full_questions = [current_data[f"hop_{h}"]["final_question"] for h in range(1, hop+1)]
        full_answers = [current_data[f"hop_{h}"]["refined_answer"] for h in range(1, hop+1)]
        full_types = [current_data[f"hop_{h}"]["qa_type"] for h in range(1, hop+1)]

        # 双路检索：主 query 用 final_question（保持主题关联），
        # 辅 query 用 refined_answer（实体检索）
        # 主查询：上一跳累计问题
        primary_query = last_hop_data.get("final_question", last_hop_data["question"])
        retrieved_primary = self.retriever.search(
            query=primary_query, top_k=topk,
            exclude_ids=full_chunk_ids, seed_title=seed_title,
        )
        # 辅查询：上一跳答案，但前提是它看起来不像纯数字（纯数字通常是金额、日期等，检索价值不大）
        secondary_query = last_hop_data.get("refined_answer", "")
        retrieved_secondary = []
        if secondary_query and not secondary_query.replace(",", "").replace(".", "").replace(" ", "").isdigit():
            retrieved_secondary = self.retriever.search(
                query=secondary_query, top_k=topk // 2,
                exclude_ids=full_chunk_ids, seed_title=seed_title,
            )
        # 合并去重（primary 优先）
        seen_ids = {r["chunk_id"] for r in retrieved_primary}
        # 复制一份新的列表，避免后面对 retrieved 做 append() 时，顺手把 retrieved_primary 原列表也改了。
        retrieved = list(retrieved_primary)
        for r in retrieved_secondary:
            if r["chunk_id"] not in seen_ids:
                seen_ids.add(r["chunk_id"])
                retrieved.append(r)

        # graph_search 一路（如果 KG 可用）。图谱检索只使用主问题，不使用答案查询。
        retrieved_graph = []
        try:
            from retrieval.graph_search import graph_search
            retrieved_graph = graph_search(primary_query, top_k=topk)
            # 过滤已排除的 chunk
            retrieved_graph = [r for r in retrieved_graph if r["chunk_id"] not in full_chunk_ids]
        except Exception as e:
            logger.debug(f"graph_search failed: {e}")

        # 构建 chunk → 工具来源映射
        chunk_tool_sources = {}
        for r in retrieved:
            cid = r["chunk_id"]
            # 从 _sources 字段获取（由 _search_impl 设置）
            sources = r.get("_sources", ["semantic_search"])
            chunk_tool_sources.setdefault(cid, set()).update(sources)
        for r in retrieved_graph:
            cid = r["chunk_id"]
            chunk_tool_sources.setdefault(cid, set()).add("graph_search")
            if cid not in seen_ids:
                seen_ids.add(cid)
                retrieved.append(r)

        logger.info(f"    Retrieved {len(retrieved)} docs (primary={len(retrieved_primary)}, secondary={len(retrieved_secondary)}, graph={len(retrieved_graph)}) query: {primary_query[:60]}")

        # 扩展统计信息
        # docs	实际处理了多少候选文档
        # qas_gen	LLM 共生成多少原子 QA
        # qas_filtered	规则过滤后剩余多少
        # merged	合成出多少多跳候选
        # v_semantic	语义检查淘汰数
        # v_reasoning	无文档检查淘汰数
        # v_singledoc	缺失文档检查淘汰数
        # v_fulldoc	全文档仍无法回答的淘汰数
        # valid	最终有效数
        stats = {"docs": 0, "qas_gen": 0, "qas_filtered": 0, "merged": 0,
                 "v_semantic": 0, "v_reasoning": 0, "v_singledoc": 0, "v_fulldoc": 0, "valid": 0}
        # 类型配额：comparison 不超过 40%
        type_counts = {"inference": 0, "comparison": 0}
        max_comparison = max(1, int(max_valid * 0.4))

        for doc_idx, new_item in enumerate(retrieved):
            if len(temp_results) >= max_valid:
                # 如果已经找到足够有效结果就提前停止
                logger.info(f"    Early exit: found {max_valid} valid results after {doc_idx} docs")
                break

            new_doc_text = f'"{new_item["title"]}"\n{new_item["text"]}'
            new_chunk_id = new_item["chunk_id"]
            stats["docs"] += 1

            # 生成新 QA
            gen_prompt = self.prompts["gen_qa_prompt"].format(
                gen_qa_num=gen_qa_num,
                input_doc=new_doc_text,
            )
            new_qas = llm_call_with_retry(gen_prompt, model=self.model_id, return_json=True)
            if not new_qas or not isinstance(new_qas, list):
                logger.info(f"    Doc {doc_idx}: no QA generated")
                continue
            stats["qas_gen"] += len(new_qas)

            # 过滤 QA
            filter_qas = self._filter_generated_qas(new_qas, full_answers, full_questions, gen_qa_num)
            stats["qas_filtered"] += len(filter_qas)
            if not filter_qas:
                logger.info(f"    Doc {doc_idx}: all QAs filtered out")
                continue

            for nq in filter_qas:
                if len(temp_results) >= max_valid:
                    break

                mid_question, mid_answer = nq["question"], nq["answer"]

                # 精炼答案
                refine_prompt = self.prompts["refine_prompt"].format(
                    question=mid_question, original_answer=mid_answer
                )
                refined_result = llm_call_with_retry(refine_prompt, model=self.model_id, return_json=True)
                if refined_result and isinstance(refined_result, dict):
                    mid_answer = refined_result.get("refined_answer", mid_answer)

                # 跳过重复答案
                if any(normalize_answer(mid_answer) == normalize_answer(fa) for fa in full_answers):
                    continue

                # 合成多跳问题
                merged_results = self._merge_multihop(
                    current_data, hop, mid_question, mid_answer, new_doc_text, full_types
                )
                if not merged_results:
                    continue
                stats["merged"] += len(merged_results)

                for merged in merged_results:
                    if len(temp_results) >= max_valid:
                        break
                    # 类型配额：comparison 达上限后只接受 inference
                    merged_type = merged.get("type", "inference")
                    if merged_type == "comparison" and type_counts["comparison"] >= max_comparison:
                        logger.info(f"    Type quota: skipping comparison (already {type_counts['comparison']})")
                        continue
                    result = self._verify_and_build(
                        current_data, hop, last_hop_data, merged,
                        mid_question, mid_answer, new_doc_text, new_chunk_id,
                        new_item["title"], full_docs, full_questions, full_answers,
                        stats=stats,
                        search_tools=sorted(chunk_tool_sources.get(new_chunk_id, {"semantic_search"})),
                        search_query=primary_query,
                    )
                    if result is not None:
                        temp_results.append(result)
                        stats["valid"] += 1
                        r_type = merged.get("type", "inference")
                        type_counts[r_type] = type_counts.get(r_type, 0) + 1

        logger.info(f"    Hop extend stats: {stats}")
        return temp_results

    @staticmethod
    def _qa_depth_score(question: str, answer: str) -> int:
        """
        评估 QA 的推理深度（越高越好），用于排序优先保留非 trivial 的 QA
        这个评分器几乎完全按英文关键词设计。如果使用中文 prompt，需要增加中文关键词与简单问题模板。
        """
        q_lower = question.lower()
        score = 0
        # 正向：推理/因果/关系类问题
        reasoning_keywords = ["which", "who", "why", "how", "compared", "led to",
                              "resulted in", "contributed", "caused", "responsible",
                              "relationship", "between", "difference", "impact"]
        for kw in reasoning_keywords:
            if kw in q_lower:
                score += 2
        # 负向：简单查表型
        trivial_prefixes = ["what is the total", "what was the total",
                            "what is the ending", "what was the ending",
                            "what is the beginning", "what was the beginning",
                            "what is the amount", "what was the amount",
                            "what is the net", "what was the net",
                            "what is the subtotal", "what is the balance",
                            "what was the balance"]
        for pfx in trivial_prefixes:
            if q_lower.startswith(pfx):
                score -= 3
                break
        # 纯数字答案降分
        # 先删除：逗号、小数点、空格、元、yuan、rmb
        ans_clean = answer.replace(",", "").replace(".", "").replace(" ", "").replace("元", "").replace("yuan", "").replace("rmb", "")
        # 如果剩下的部分是纯数字（可能带一个负号），则认为是纯数字答案，降分。
        if ans_clean.lstrip("-").isdigit():
            score -= 1
        return score

    def _filter_generated_qas(self, raw_qas: list, full_answers: list,
                              full_questions: list, max_num: int) -> list:
        """过滤生成的原子 QA，并按推理深度排序。过滤重复、答案泄露、复合答案、引用文档等"""
        filtered = []
        pre_answers = []
        pre_questions = []

        for nq in raw_qas:
            if not isinstance(nq, dict):
                continue
            question = nq.get("question", "")
            answer = nq.get("answer", "")
            if not question or not answer:
                continue
            if len(_tokens(answer)) >= 10:
                continue
            if normalize_answer(answer) in normalize_answer(question):
                continue
            atokens = _tokens(answer)
            qtokens = _tokens(question)
            if len([t for t in atokens if t in qtokens]) > 0.5 * len(atokens):
                continue
            # 去重
            skip = False
            for pa in pre_answers:
                if normalize_answer(answer) == normalize_answer(pa):
                    skip = True; break
            for pq in pre_questions:
                if normalize_answer(question) == normalize_answer(pq):
                    skip = True; break
            if skip:
                continue
            # 当前顺序存在一个细节：pre_answers.append(answer)和pre_questions.append(question)
            # 发生在 and/or 等后续过滤之前。也就是说，一条后来因为包含 and 被淘汰的 QA，仍然会占据去重集合，可能导致后面一条本来合格的同答案 QA 被误删。
            # 更合理的是所有规则通过后，再加入 pre_answers 和 pre_questions。
            pre_answers.append(answer)
            pre_questions.append(question)
            # 排除含 and/or 等的复合问题和复合答案
            if "and" in atokens or "or" in atokens or "&" in answer:
                continue
            if "and" in qtokens:
                continue
            # 上一跳答案不应出现在新问题中
            if full_answers and simple_partial_presence(full_answers[-1], question):
                continue
            # 排除要求全名/引用文档的问题
            q_lower = question.lower()
            if any(p in q_lower for p in ["full name", "original name", "alternate name",
                                           "alternative name", "name one"]):
                continue
            qtokens_lower = [t.lower() for t in _tokens(question)]
            if any(w in qtokens_lower for w in ["document", "article", "according"]):
                continue
            filtered.append(nq)

        # 按推理深度排序，优先保留非 trivial（琐碎的） 的 QA。推理深度越高，score 越大。
        filtered.sort(key=lambda x: self._qa_depth_score(x["question"], x["answer"]), reverse=True)
        return filtered[:max_num]

    def _merge_multihop(self, current_data: dict, hop: int,
                        mid_question: str, mid_answer: str,
                        new_doc: str, full_types: list) -> list[dict]:
        """调用 LLM 合成多跳问题"""
        Data = []
        for h in range(1, hop + 1):
            info = current_data[f"hop_{h}"]
            Data.append(
                f"Hop_{h}:\n"
                f"Question: {info['final_question']}\n"
                f"Answer: {info['refined_answer']}\n"
                f"Document: {info['doc']}"
            )

        # 只要已有链中出现过 comparison，后续就使用 comparison 专用模板。
        if "comparison" not in full_types:
            template = self.prompts["merge_qa_prompt_morehop"]
        else:
            template = self.prompts["merge_qa_prompt_morehop_comparison"]

        merge_prompt = template.format(
            max_num=3,
            Data="\n".join(Data),
            New_question=mid_question,
            New_answer=mid_answer,
            New_document=new_doc,
        )

        merged_qas = llm_call_with_retry(merge_prompt, model=self.merge_model_id, return_json=True)
        if not merged_qas:
            logger.info(f"    Merge returned empty/None (type={type(merged_qas).__name__})")
            return []
        logger.info(f"    Merge raw result (type={type(merged_qas).__name__}): {json.dumps(merged_qas, ensure_ascii=False, default=str)[:300]}")
        if isinstance(merged_qas, dict):
            merged_qas = [merged_qas]
        if not isinstance(merged_qas, list):
            logger.info(f"    Merge returned non-list: {type(merged_qas)}")
            return []
        valid = [m for m in merged_qas if isinstance(m, dict) and m.get("final_question")]
        if not valid:
            logger.info(f"    Merge returned {len(merged_qas)} items but 0 have 'final_question'. Keys: {[list(m.keys()) if isinstance(m, dict) else type(m) for m in merged_qas[:3]]}")
        return valid

    def _verify_and_build(self, current_data, hop, last_hop_data, merged,
                          mid_question, mid_answer, new_doc, new_chunk_id,
                          new_title, full_docs, full_questions, full_answers,
                          stats: dict = None,
                          search_tools: list = None,
                          search_query: str = ""):
        """四重验证 + 构建最终结果"""
        qa_type = merged.get("type", "inference")
        final_question = merged["final_question"]
        final_answer = merged["final_answer"]
        _s = stats or {}

        # --- 过滤泛化答案（comparison 应有明确差异） ---
        generic_patterns = ["both", "neither", "same", "they are the same", "equal"]
        if qa_type == "comparison" and any(p in final_answer.lower() for p in generic_patterns):
            logger.info(f"    Pre-filter: generic comparison answer '{final_answer[:40]}'")
            return None

        # --- 前置过滤 ---
        if qa_type == "inference":
            # 年份过滤：只过滤真正的 4 位年份，不误伤普通数字
            # 因为纯年份答案容易生成简单的查表型问题，推理深度通常较低。
            if _years(mid_answer) and len(mid_answer.strip()) == 4:
                logger.info(f"    Pre-filter: year answer '{mid_answer}'")
                return None
            # inference 最终答案必须等于新原子答案
            if normalize_answer(final_answer) != normalize_answer(mid_answer):
                logger.info(f"    Pre-filter: answer mismatch")
                return None
            # P2: 用字符数而非 token 数比较，对中文更公平
            # 要求 final_question 至少比最长的前序问题长 5 个字符
            max_prev_len = max((len(pq) for pq in full_questions), default=0)
            if len(final_question) < max_prev_len + 5:
                logger.info(f"    Pre-filter: final_question not longer ({len(final_question)} < {max_prev_len}+5)")
                return None
            # 新原子问题不能几乎原样成为最终问题
            # P2: mid_question substring 检查放宽——只在 normalized 长度占比 > 80% 时过滤
            norm_mid = normalize_answer(mid_question[:-1]) if mid_question.endswith("?") else normalize_answer(mid_question)
            norm_final = normalize_answer(final_question)
            if norm_mid in norm_final and len(norm_mid) > 0.8 * len(norm_final):
                logger.info(f"    Pre-filter: mid_question substring (>{80}% overlap)")
                return None

        # 年份检查
        # 目的：防止 merge 时丢失时间限定，导致问题语义改变。
        pre_years = []
        for pq in full_questions:
            pre_years += _years(pq)
        # 收集已有累计问题里的年份，再加上新问题中的年份：
        qyear = _years(mid_question) + pre_years
        fqyear = _years(final_question)
        if qyear:
            # 如果已有问题中出现的任何年份没有出现在最终问题：
            missing = [yr for yr in qyear if yr not in fqyear]
            if missing:
                return None

        # 中间答案泄露检查
        # 收集：第一跳答案 和 所有 inference hop 的最终答案。
        # omparison hop 的答案不加入泄露检查。
        pre_inf_answers = []
        for h in range(1, hop + 1):
            info = current_data[f"hop_{h}"]
            if h == 1 or info["qa_type"] == "inference":
                pre_inf_answers.append(info["final_answer"])
        for pa in pre_inf_answers:
            # 最终问题不能直接暴露桥接答案。
            if normalize_answer(pa) in normalize_answer(final_question):
                return None

        # 精炼最终答案
        refine_prompt = self.prompts["refine_prompt"].format(
            question=final_question, original_answer=final_answer
        )
        refined_result = llm_call_with_retry(refine_prompt, model=self.model_id, return_json=True)
        if refined_result and isinstance(refined_result, dict):
            refined_answer = refined_result.get("refined_answer", final_answer)
        else:
            refined_answer = final_answer
        if len(_tokens(refined_answer)) >= 10:
            return None

        # 生成答案别名
        opt_prompt = self.prompts["more_optional_answer_prompt"].format(
            refined_answer=refined_answer
        )
        option_answers = llm_call_with_retry(opt_prompt, model=self.model_id, return_json=True)
        if not option_answers or not isinstance(option_answers, list):
            option_answers = [refined_answer]
        # 截断到最多 20 个别名
        option_answers = option_answers[:20]

        # === 四重验证 ===
        verification_steps = []

        # 1. 语义检查
        if qa_type == "inference":
            check_prompt = self.prompts["inference_check_prompt"].format(
                Question1=last_hop_data["final_question"],
                Answer1=last_hop_data["refined_answer"],
                Document1=last_hop_data["doc"],
                Question2=mid_question,
                Answer2=mid_answer,
                Document2=new_doc,
                Final_question=final_question,
                Final_answer=final_answer,
                qa_type=qa_type,
            )
        else:
            check_prompt = self.prompts["comparison_check_prompt"].format(
                Question1=last_hop_data["final_question"],
                Answer1=last_hop_data["refined_answer"],
                Document1=last_hop_data["doc"],
                Question2=mid_question,
                Answer2=mid_answer,
                Document2=new_doc,
                Final_question=final_question,
                Final_answer=final_answer,
                qa_type=qa_type,
            )
        check_result = llm_call_with_retry(check_prompt, model=self.model_id, return_json=True)
        if not check_result or not isinstance(check_result, dict):
            logger.info(f"    Semantic check: no valid JSON response")
            return None
        verification_steps.append({"step": "semantic_check", "result": check_result})
        if str(check_result.get("valid", "false")).lower() != "true":
            _s["v_semantic"] = _s.get("v_semantic", 0) + 1
            logger.info(f"    FAIL semantic check: [{check_result.get('error_type', '')}] {check_result.get('justification', '')[:80]}")
            return None

        # 2. 推理检查（不给文档，LLM 应该无法回答）过滤仅靠常识、记忆或问题表面就能回答的问题。
        if qa_type == "inference":
            reasoning_prompt = self.prompts["reasoning_prompt"].format(problem=final_question)
        else:
            reasoning_prompt = self.prompts["comparison_reasoning_prompt"].format(problem=final_question)

        # std="mid" 表示：
        # 如果 LLM 答对 → 失败
        # 如果 LLM 答错 → 通过
        v, llm_ans, f1, esseq = self.compare_verify(
            prompt=reasoning_prompt,
            final_question=final_question,
            option_answers=option_answers,
            std="mid", desc="reasoning",
        )
        verification_steps.append({
            "step": "reasoning_check", "verification": v,
            "llm_answer": llm_ans, "f1": f1,
        })
        if v != "pass":
            _s["v_reasoning"] = _s.get("v_reasoning", 0) + 1
            logger.info(f"    FAIL reasoning check: LLM answered '{llm_ans[:60]}' (f1={f1:.2f})")
            return None

        # 3. 去单文档检查（给部分文档，LLM 应该无法回答）
        # 保证：每一份文档都对最终答案不可缺少。
        current_full_docs = full_docs + [new_doc]
        # 只检查最后 N-1 个文档的子集（与 AgenticRAGTracer 一致）
        r = len(current_full_docs) - 1
        # 枚举所有“缺少一份文档”的组合：
        # combinations() 返回的是迭代器，里面是所有长度为 r 的组合，每个组合是一个元组。
        for combo in combinations(current_full_docs, r):
            combo_docs = "\n\n".join(combo) if len(combo) > 1 else combo[0]
            singlehop_prompt = self.prompts["singlehop_prompt"].format(
                Document=combo_docs, Question=final_question,
            )
            v, llm_ans, f1, esseq = self.compare_verify(
                prompt=singlehop_prompt,
                final_question=final_question,
                option_answers=option_answers,
                std="mid", desc=f"only_{len(combo)}_docs",
            )
            verification_steps.append({
                "step": "single_doc_check", "doc_count": len(combo),
                "verification": v, "llm_answer": llm_ans, "f1": f1,
            })
            if v != "pass":
                _s["v_singledoc"] = _s.get("v_singledoc", 0) + 1
                logger.info(f"    FAIL single_doc check ({len(combo)} docs): LLM answered '{llm_ans[:60]}'")
                return None

        # 4. 全文档检查（给全部文档，LLM 应该能回答）
        Data = []
        # 构造已有 hop 数据
        for h in range(1, hop + 1):
            info = current_data[f"hop_{h}"]
            Data.append(
                # 每一跳的原子问题，不是累计 final_question。
                f"Question{h}: {info['question']}\n" # 
                f"Answer{h}: {info['refined_answer']}\n"
                f"Supporting Document{h}: {info['doc']}"
            )
        if qa_type == "inference":
            # 对新一跳的 inference，故意不提供新原子答案
            # 因为新原子答案通常就是最终答案，如果直接提供，相当于泄露答案。
            Data.append(
                f"Question{hop+1}: {mid_question}\n"
                f"Supporting Document{hop+1}: {new_doc}"
            )
            full_prompt = self.prompts["multihop_inference_prompt_morehop"].format(
                Data="\n".join(Data), FinalQuestion=final_question,
            )
        else:
            Data.append(
                # comparison 会提供新原子答案：
                # 因为比较问题通常需要已知两个局部答案，再比较它们。
                f"Question{hop+1}: {mid_question}\n"
                f"Answer{hop+1}: {mid_answer}\n"
                f"Supporting Document{hop+1}: {new_doc}"
            )
            full_prompt = self.prompts["multihop_comparison_prompt_morehop"].format(
                Data="\n".join(Data), FinalQuestion=final_question,
            )
        # std="final" 表示：
        # 如果 LLM 答对 → 通过
        # 如果 LLM 答错 → 失败
        v, llm_ans, f1, esseq = self.compare_verify(
            prompt=full_prompt,
            final_question=final_question,
            option_answers=option_answers,
            std="final", desc="cannot_answer",
        )
        verification_steps.append({
            "step": "full_doc_check", "verification": v,
            "llm_answer": llm_ans, "f1": f1,
        })
        if v != "pass":
            _s["v_fulldoc"] = _s.get("v_fulldoc", 0) + 1
            logger.info(f"    FAIL full_doc check: LLM answered '{llm_ans[:60]}'")
            return None
        logger.info(f"    PASS all 4 checks! Q: {final_question[:80]}...")

        # === 构建结果 ===
        new_hop_data = dict(current_data)
        new_hop_data[f"hop_{hop+1}"] = {
            "question": mid_question,
            "answer": mid_answer,
            "doc": new_doc,
            "final_question": final_question,
            "final_answer": final_answer,
            "refined_answer": refined_answer,
            "optional_answers": option_answers,
            "qa_type": qa_type,
            "chunk_id": new_chunk_id,
            "title": new_title,
            "verify_result": verification_steps,
            "search_tools": search_tools or ["semantic_search"],
            "search_query": search_query,
        }
        return new_hop_data


# ============================================================
# 主流程
# ============================================================

def _flatten_result(data: dict, id_prefix: str, idx: int) -> dict:
    """将内部 hop 结构转为输出 JSONL 格式"""
    # Pipeline 内部生成的数据类似：
    # {
    #     "hop_1": {...},
    #     "hop_2": {...},
    #     "hop_3": {...}
    # }
    # _flatten_result() 会把它转换成：
    # {
    #     "id": "...",
    #     "final_question": "...",
    #     "final_answer": "...",
    #     "hop_count": 3,
    #     "qa_type": "inference",
    #     "subset": "3hop_inference",
    #     "hops": [...],
    #     "verification": {...}
    # }
    # 找出最大 hop 数
    max_hop = 0
    for k in data:
        if k.startswith("hop_"):
            h = int(k.split("_")[1])
            max_hop = max(max_hop, h)
    # 获取最后一跳
    last_hop = data[f"hop_{max_hop}"]
    hops = []
    for h in range(1, max_hop + 1):
        info = data[f"hop_{h}"]
        # 所有跳都用原子答案 (answer 字段)
        # refined_answer 存的是 final_answer 精炼值（对 comparison 是比较结论），不是原子答案
        # final answer 已单独存在顶层的 "answer" 字段
        hops.append({
            "hop_idx": h,
            "question": info.get("question", info.get("final_question", "")),
            "answer": info.get("answer", ""),
            "doc_chunk_id": info.get("chunk_id", ""),
            "title": info.get("title", ""),
            "search_tools": info.get("search_tools", []),
            "search_query": info.get("search_query", ""),
        })

    verify = last_hop.get("verify_result", [])
    verification = {
        "semantic": any(s.get("step") == "semantic_check" for s in verify),
        "reasoning": any(s.get("step") == "reasoning_check" and s.get("verification") == "pass" for s in verify),
        "single_doc": any(s.get("step") == "single_doc_check" and s.get("verification") == "pass" for s in verify),
        "full_doc": any(s.get("step") == "full_doc_check" and s.get("verification") == "pass" for s in verify),
    }

    # 生成 subset 标签：{hop_count}hop_{qa_type}
    qa_type = last_hop["qa_type"]
    subset = f"{max_hop}hop_{qa_type}"

    return {
        "id": f"{id_prefix}_{idx:06d}",
        "final_question": last_hop["final_question"],
        "final_answer": last_hop["refined_answer"],
        "answer_aliases": last_hop.get("optional_answers", [last_hop["refined_answer"]]),
        "hop_count": max_hop,
        "qa_type": qa_type,
        "subset": subset,
        "hops": hops,
        "verification": verification,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-hop QA synthesis from seed QAs")
    parser.add_argument("--seeds", default="data/news_synthesis/seeds.jsonl")
    parser.add_argument("--corpus", default="data/news_corpus/en/corpus.json")
    parser.add_argument("--index-dir", default="data/news_indexes/en/")
    parser.add_argument("--output", default="data/news_synthesis/multihop_raw.jsonl")
    parser.add_argument("--prompts", default="scripts/synthesis_prompts.yaml")
    parser.add_argument("--lang", choices=["en", "zh"], default="en",
                        help="语言：en=英文prompt, zh=中文prompt（自动切换 prompts 文件）")
    parser.add_argument("--model", default="gpt-oss-120b", help="Default model for most steps")
    parser.add_argument("--merge-model", default="mog-2", help="Model for multi-hop merge (hardest step)")
    parser.add_argument("--num-hop", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--gen-qa-num", type=int, default=3)
    parser.add_argument("--every-hop-qa-num", type=int, default=5)
    parser.add_argument("--max-valid-per-hop", type=int, default=3)
    parser.add_argument("--max-qa-per-seed", type=int, default=5, help="Max QA output per seed")
    parser.add_argument("--hop-quotas", type=str, default="2:1,3:3,4:3",
                        help="Per-hop output quotas, e.g. '2:1,3:3,4:3'")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--gpu-ids", type=str, default=None, help="GPU IDs, e.g. '0,1,2,3'")
    parser.add_argument("--max-per-gpu", type=int, default=5, help="Max concurrent retrievals per GPU")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of seeds (0=all)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 加载 prompts（根据 --lang 自动选择文件）
    prompts_path = args.prompts
    if args.lang == "zh" and args.prompts == "scripts/synthesis_prompts.yaml":
        prompts_path = "scripts/synthesis_prompts_zh.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    logger.info(f"Loaded prompts from {prompts_path} (lang={args.lang})")

    # 加载语料
    logger.info(f"Loading corpus from {args.corpus}")
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_lookup = {c["chunk_id"]: c for c in corpus}
    logger.info(f"Corpus: {len(corpus)} chunks")

    # 加载 seeds
    logger.info(f"Loading seeds from {args.seeds}")
    seeds = []
    with open(args.seeds, "r", encoding="utf-8") as f:
        for line in f:
            try:
                seeds.append(json.loads(line.strip()))
            except Exception:
                continue
    logger.info(f"Loaded {len(seeds)} seed QAs")

    # 随机抽样
    if args.limit > 0:
        random.seed(args.seed)
        seeds = random.sample(seeds, min(args.limit, len(seeds)))
        logger.info(f"Sampled {len(seeds)} seeds")

    # 断点续跑
    processed_seeds = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 用第一跳的 doc_chunk_id + question 作为去重键
                    hops = data.get("hops", [])
                    if hops:
                        key = f"{hops[0].get('doc_chunk_id', '')}|{hops[0].get('question', '')}"
                        processed_seeds.add(key)
                except Exception:
                    continue
        logger.info(f"Resume: {len(processed_seeds)} seeds already processed")
        seeds = [s for s in seeds if f"{s['chunk_id']}|{s['question']}" not in processed_seeds]

    if not seeds:
        logger.info("No seeds to process")
        return

    # 初始化
    init_concurrency(args.workers)
    reset_stats()
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else None
    retriever = DomainRetriever(args.index_dir, gpu_ids=gpu_ids, max_per_gpu=args.max_per_gpu)
    pipeline = DomainMultiHopPipeline(args.model, retriever, prompts,
                                    merge_model_id=args.merge_model,
                                    corpus_lookup=corpus_lookup)
    logger.info(f"Models: default={args.model}, merge={args.merge_model}")
    # 解析 hop_quotas
    hop_quotas = {}
    for part in args.hop_quotas.split(","):
        k, v = part.split(":")
        hop_quotas[int(k)] = int(v)
    logger.info(f"Hop quotas: {hop_quotas}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 并行处理
    write_lock = Lock()
    global_idx = [0]  # mutable counter
    total_valid = [0]

    def _process_seed(seed_item):
        try:
            results = pipeline.process_seed(
                seed_item, corpus_lookup,
                num_hop=args.num_hop,
                topk=args.topk,
                gen_qa_num=args.gen_qa_num,
                every_hop_qa_num=args.every_hop_qa_num,
                max_valid_per_hop=args.max_valid_per_hop,
                max_qa_per_seed=args.max_qa_per_seed,
                hop_quotas=hop_quotas,
            )
            if results:
                # P2: 去重问题前缀 —— 同一 seed 下前缀相同的只保留第一条
                # 因为LLM 可能生成多个几乎相同的问题，
                seen_prefixes = set()
                deduped = []
                for r in results:
                    max_hop = max(int(k.split("_")[1]) for k in r if k.startswith("hop_"))
                    # 使用最终问题前 50 个字符作为去重键。
                    fq = r[f"hop_{max_hop}"]["final_question"][:50]
                    if fq not in seen_prefixes:
                        seen_prefixes.add(fq)
                        deduped.append(r)
                if len(deduped) < len(results):
                    logger.info(f"  Dedup: {len(results)} → {len(deduped)} (removed {len(results)-len(deduped)} duplicate prefixes)")
                results = deduped

                with write_lock:
                    with open(args.output, "a", encoding="utf-8") as f:
                        for r in results:
                            flat = _flatten_result(r, "mhop", global_idx[0])
                            f.write(json.dumps(flat, ensure_ascii=False) + "\n")
                            global_idx[0] += 1
                    total_valid[0] += len(results)
                logger.info(f"Seed {seed_item['chunk_id'][:8]}... → {len(results)} valid multi-hop QAs")
            return len(results) if results else 0
        except Exception as e:
            logger.error(f"Error processing seed {seed_item.get('chunk_id', '?')[:8]}: {e}")
            return 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_process_seed, s) for s in seeds]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Synthesizing"):
            fut.result()

    stats = get_stats()
    logger.info(f"Done! Generated {total_valid[0]} multi-hop QAs from {len(seeds)} seeds")
    logger.info(f"LLM calls: {stats['calls']}, errors: {stats['errors']}, "
                f"total latency: {stats['total_latency']:.1f}s")


if __name__ == "__main__":
    main()
