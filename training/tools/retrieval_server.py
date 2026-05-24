#!/usr/bin/env python3
"""BGE Embedding + Reranker HTTP 服务

在其他机器上部署，GRPO 训练通过内网 HTTP 调用。

用法:
  # 在有 GPU 的机器上启动（占 ~2GB 显存）
  python training/tools/retrieval_server.py --port 8790 --device cuda:0

  # 测试
  curl -X POST http://<IP>:8790/embed -H 'Content-Type: application/json' \
    -d '{"texts": ["永辉超市注册资本"]}'
  curl -X POST http://<IP>:8790/rerank -H 'Content-Type: application/json' \
    -d '{"query": "永辉超市", "passages": ["永辉超市注册资本100亿", "沃尔玛全球门店"]}'
"""
import argparse
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np


class RetrievalHandler(BaseHTTPRequestHandler):
    embedder = None
    reranker = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/embed":
            texts = body.get("texts", [])
            if not texts:
                self._json_response({"error": "no texts"}, 400)
                return
            t0 = time.time()
            vecs = self.embedder.encode(texts, normalize_embeddings=True)
            elapsed = time.time() - t0
            self._json_response({
                "embeddings": vecs.tolist(),
                "elapsed": elapsed,
                "count": len(texts),
            })

        elif self.path == "/rerank":
            query = body.get("query", "")
            passages = body.get("passages", [])
            if not query or not passages:
                self._json_response({"error": "need query and passages"}, 400)
                return
            t0 = time.time()
            pairs = [[query, p] for p in passages]
            scores = self.reranker.predict(pairs)
            elapsed = time.time() - t0
            self._json_response({
                "scores": [float(s) for s in scores],
                "elapsed": elapsed,
            })

        elif self.path == "/health":
            self._json_response({"status": "ok"})

        else:
            self._json_response({"error": f"unknown path: {self.path}"}, 404)

    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "ok"})
        else:
            self._json_response({"error": "use POST"}, 405)

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, format, *args):
        # 只打印非 health 的请求
        if "/health" not in str(args):
            BaseHTTPRequestHandler.log_message(self, format, *args)


def main():
    parser = argparse.ArgumentParser(description="BGE Embedding + Reranker Server")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embedding-model", type=str,
                        default="models/bge-m3")
    parser.add_argument("--reranker-model", type=str,
                        default="models/bge-reranker-v2-m3")
    args = parser.parse_args()

    print(f"Loading BGE-M3 embedder on {args.device}...")
    from sentence_transformers import SentenceTransformer, CrossEncoder
    RetrievalHandler.embedder = SentenceTransformer(args.embedding_model, device=args.device)
    print(f"Loading BGE reranker on {args.device}...")
    RetrievalHandler.reranker = CrossEncoder(args.reranker_model, max_length=512, device=args.device)

    server = HTTPServer(("0.0.0.0", args.port), RetrievalHandler)
    print(f"\nRetrieval server ready at http://0.0.0.0:{args.port}")
    print(f"  POST /embed   — {{\"texts\": [\"...\", ...]}}")
    print(f"  POST /rerank  — {{\"query\": \"...\", \"passages\": [\"...\", ...]}}")
    print(f"  GET  /health  — health check")
    server.serve_forever()


if __name__ == "__main__":
    main()
