"""构建 FAISS + BM25 索引 + chunk store"""
import json
import os
import pickle
import sys

import numpy as np

# 去两次上一层目录，导入项目根路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, INDEX_DIR


def build_all(corpus_path: str = None, index_dir: str = None):
    """从 corpus.json 构建所有索引

    Args:
        corpus_path: corpus.json 路径，默认 DATA_DIR/corpus.json
        index_dir: 索引输出目录，默认 INDEX_DIR
    """
    if corpus_path is None:
        corpus_path = os.path.join(DATA_DIR, "corpus.json")
    if index_dir is None:
        index_dir = INDEX_DIR

    os.makedirs(index_dir, exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"[build_index] Building indexes for {len(corpus)} docs...")

    texts = [doc["text"] for doc in corpus]
    chunk_ids = [doc["chunk_id"] for doc in corpus]
    chunk_store = {doc["chunk_id"]: doc for doc in corpus}

    # 1. FAISS IndexFlatIP 。FAISS 向量检索擅长找语义相近的内容。
    # 给 semantic_search.py 用，走 BGE-M3 embedding。
    print("[build_index] Encoding with BGE-M3...")
    from retrieval.embedder import encode
    import faiss

    # 把所有文本编码成向量。
    embeddings = encode(texts, batch_size=64)
    dim = embeddings.shape[1]
    # 创建一个 FAISS 索引。
    # FAISS IndexFlatIP 是最简单的向量索引，使用内积（dot product）作为相似度。因为我们在 embedder.py 里对向量做了 L2 归一化，所以内积相当于余弦相似度。
    index = faiss.IndexFlatIP(dim)
    # index 只保存“向量”和“索引”。FAISS 会按照加入顺序给向量编号。后面检索返回的编号需要用 chunk_ids 转换。
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    print(f"[build_index] FAISS index: {index.ntotal} vectors, dim={dim}")

    # 2. BM25 。BM25 擅长找字面匹配的内容，是传统搜索引擎里的关键词排序算法。
    # 给 keyword_search.py 用，走 jieba/正则分词 + BM25。
    print("[build_index] Building BM25 (jieba + whitespace tokenizer)...")
    from rank_bm25 import BM25Okapi
    from retrieval.keyword_search import tokenize
    tokenized = [tokenize(t) for t in texts]
    # 根据所有分词后的文档构建 BM25 索引。
    bm25 = BM25Okapi(tokenized)
    # BM25 对象不能直接保存成 JSON，所以用 pickle.dump() 序列化保存。
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    # 3. chunk_ids（与 FAISS 对齐）因为 FAISS 返回的是向量编号，而不是 chunk_id。
    # FAISS index 位置 -> chunk_ids -> chunk_store -> 原始 chunk
    with open(os.path.join(index_dir, "chunk_ids.json"), "w") as f:
        json.dump(chunk_ids, f)

    # 4. chunk_store
    # chunk_store.pkl 是 Python 内部运行时缓存，用 pickle 图省事和加载方便。但如果想让索引目录更透明，chunk_store 改成 JSON 也完全可行。
    with open(os.path.join(index_dir, "chunk_store.pkl"), "wb") as f:
        pickle.dump(chunk_store, f)

    print(f"[build_index] All indexes saved to {index_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes")
    parser.add_argument("--corpus", default=None, help="Path to corpus.json")
    parser.add_argument("--index-dir", default=None, help="Output index directory")
    args = parser.parse_args()
    build_all(corpus_path=args.corpus, index_dir=args.index_dir)
