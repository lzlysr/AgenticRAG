"""BGE Reranker-v2-m3 重排序服务（多设备支持，线程安全）"""
import sys, os
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BGE_RERANKER_PATH, RERANK_TOP_K

_models = {}   # device → CrossEncoder
_locks = {}    # device → Lock
_global_lock = Lock()


def _get_model(device: str = None):
    if device is None:
        device = "cpu"
    with _global_lock:
        if device not in _models:
            from sentence_transformers import CrossEncoder
            _models[device] = CrossEncoder(BGE_RERANKER_PATH, max_length=512, device=device)
            _locks[device] = Lock()
            print(f"[reranker] Loaded BGE-reranker on {device}")
        return _models[device], _locks[device]


def rerank(query: str, passages: list[str], top_k: int = RERANK_TOP_K, device: str = None) -> list[tuple[int, float]]:
    """重排序，返回 [(原始index, score)] 按 score 降序，取 top_k"""
    if not passages:
        return []
    model, lock = _get_model(device)
    with lock:
        scores = model.predict([[query, p] for p in passages])
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:top_k]
