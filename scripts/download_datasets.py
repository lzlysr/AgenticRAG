"""下载 AgenticRAGTracer 数据集并构建语料库"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

DATASET_NAME = "YqjMartin/AgenticRAGTracer"
SUBSETS = [
    "2hop_comparison", "2hop_inference",
    "3hop_comparison", "3hop_inference",
    "4hop_comparison", "4hop_inference",
]


def _doc_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def download_and_process():
    """下载所有子集，提取语料库和 QA 对"""
    from datasets import load_dataset

    corpus = {}  # chunk_id -> {"chunk_id", "text", "title"}
    qa_pairs = []

    for subset in SUBSETS:
        print(f"[download] Loading {subset}...")
        ds = load_dataset(DATASET_NAME, subset, split="test")

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
