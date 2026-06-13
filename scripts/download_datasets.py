"""下载 AgenticRAGTracer 数据集并构建语料库"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

DATASET_NAME = "YqjMartin/AgenticRAGTracer"
# 多跳推理（Multi-hop Reasoning）是一种复杂的推理技术，要求模型在回答问题或解决任务时，跨越多个信息片段或知识点，逐步推导出最终答案，而非直接从单一信息源中获取结果。每次跨越称为一个“跳跃”（hop）。
# hop 数量表示问题需要几跳推理，comparison/inference 表示问题类型（对比推理 vs 直接推理）。共 6 个子集。
SUBSETS = [
    "2hop_comparison", "2hop_inference",
    "3hop_comparison", "3hop_inference",
    "4hop_comparison", "4hop_inference",
]


def _doc_hash(text: str) -> str:
    """根据文档内容生成一个短 ID，作为 chunk_id。"""
    # text.encode() 把字符串转成字节。.hexdigest() 把哈希值转成十六进制字符串。
    # 为什么用文档内容做 hash？因为同一文档无论在哪个问题中出现，内容相同就应该对应同一个 chunk_id，这样索引和检索才有效。如果用随机 ID，每次出现都不一样，就无法正确匹配了。
    return hashlib.md5(text.encode()).hexdigest()[:12]


def download_and_process():
    """下载所有子集，提取语料库和 QA 对"""
    # load_dataset 用来从 HuggingFace 下载数据集
    from datasets import load_dataset

    corpus = {}  # chunk_id -> {"chunk_id", "text", "title"}
    qa_pairs = []

    for subset in SUBSETS:
        print(f"[download] Loading {subset}...")
        ds = load_dataset(DATASET_NAME, subset, split="test") # test: 直接拿到测试集

        for row in ds:
            hop_count = int(subset[0])  # 2, 3, or 4
            qa_type = subset.split("_")[1]  # comparison or inference
            hops = []

            for i in range(1, hop_count + 1):
                hop_key = f"hop_{i}"
                if hop_key not in row:
                    break
                hop_data = row[hop_key]
                if isinstance(hop_data, str):
                    hop_data = json.loads(hop_data)

                doc_text = hop_data.get("doc", "")
                if doc_text:
                    cid = _doc_hash(doc_text)
                    # 如果 corpus 中还没有这个 chunk_id，就添加进去。这样同一文档无论在哪个问题中出现，内容相同就对应同一个 chunk_id。
                    if cid not in corpus:
                        corpus[cid] = {
                            "chunk_id": cid,
                            "text": doc_text,
                            "title": hop_data.get("title", ""),
                        }
                hops.append({
                    "hop_idx": i,
                    "question": hop_data.get("question", ""),
                    "answer": hop_data.get("answer", ""),
                    "doc_chunk_id": _doc_hash(doc_text) if doc_text else "",
                    "qa_type": hop_data.get("qa_type", ""),
                })

            qa_pairs.append({
                "final_question": row.get("final_question", ""),
                "final_answer": row.get("final_answer", ""),
                "hop_count": hop_count,
                "qa_type": qa_type,
                "subset": subset,
                "hops": hops,
            })

    # 保存
    corpus_list = list(corpus.values())
    corpus_path = os.path.join(DATA_DIR, "corpus.json")
    qa_path = os.path.join(DATA_DIR, "qa_pairs.json")

    # 把 corpus_list 这个 Python 列表保存成 JSON 文件。
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_list, f, ensure_ascii=False, indent=2)
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"[download] Corpus: {len(corpus_list)} unique docs")
    print(f"[download] QA pairs: {len(qa_pairs)} total")
    print(f"[download] Saved to {DATA_DIR}")
    return corpus_list, qa_pairs


if __name__ == "__main__":
    download_and_process()
