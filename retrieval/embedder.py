"""BGE-M3 Embedding 服务（多设备支持，线程安全），用 BGE-M3 模型生成文本向量。"""
import numpy as np
import sys, os
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BGE_M3_PATH

# 用来缓存已经加载过的模型。因为加载 BGE-M3 很慢，而且模型很大。你不希望每次调用 encode() 都重新加载模型。
_models = {}   # device: SentenceTransformer
# 因为多个线程同时使用同一个模型推理，可能会出现资源竞争问题。为了避免同一个模型被多个线程同时调用，就给每个设备配一把锁。
_locks = {}    # device: Lock
_global_lock = Lock()  # 这是全局锁，保护 _models/_locks 字典本身


def _get_model(device: str = None):
    """根据 device 获取对应的模型和锁；如果模型还没加载，就先加载。"""
    if device is None:
        device = "cpu"
    with _global_lock: # 进入全局锁。只要一个线程进入这里，其他线程就要等它出来。
        if device not in _models:
            # sentence_transformers 把文本变成 embedding 向量。
            from sentence_transformers import SentenceTransformer
            _models[device] = SentenceTransformer(BGE_M3_PATH, device=device)
            _locks[device] = Lock()
            print(f"[embedder] Loaded BGE-M3 on {device}")
        return _models[device], _locks[device]


def encode(texts: list[str], batch_size: int = 64, device: str = None) -> np.ndarray:
    """编码文本列表，返回归一化向量 (N, D)"""
    model, lock = _get_model(device)
    with lock:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True, 
            # 把输出向量做 L2 归一化。如果没有归一化，build_index.py 的 IndexFlatIP 会偏向向量模长更大的文本，不一定是语义最接近的文本。
        )
    return np.array(embeddings, dtype=np.float32)
