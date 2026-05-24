"""BGE-M3 Embedding 服务（多设备支持，线程安全）"""
import numpy as np
import sys, os
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BGE_M3_PATH

_models = {}   # device → SentenceTransformer
_locks = {}    # device → Lock
_global_lock = Lock()  # 保护 _models/_locks 字典本身


def _get_model(device: str = None):
    if device is None:
        device = "cpu"
    with _global_lock:
        if device not in _models:
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
        )
    return np.array(embeddings, dtype=np.float32)
