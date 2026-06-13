#!/usr/bin/env python3
"""PDF 财报解析 → corpus.json

用法:
  python scripts/parse_pdf_corpus.py \
    --pdf 11323531.PDF \
    --output data/financial/corpus.json \
    --chunk-size 500
"""
import argparse
import json
import os
import re
import sys

import fitz  # PyMuPDF


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """逐页提取 PDF 文本"""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    doc.close()
    return pages


def detect_section_title(text: str) -> str | None:
    """检测章节标题"""
    patterns = [
        r"^第[一二三四五六七八九十]+节\s+.+",
        r"^[一二三四五六七八九十]+、\s*.+",
        r"^\([一二三四五六七八九十]+\)\s*.+",
        r"^\d+、\s*.+",
    ]
    # 取当前页第一行
    first_line = text.split("\n")[0].strip()
    for p in patterns:
        if re.match(p, first_line):
            return first_line
    return None


def chunk_pages(pages: list[dict], chunk_size: int = 500,
                overlap: int = 50, doc_title: str = "") -> list[dict]:
    """将页面文本切成 chunk

    策略：按段落分割，尽量不切断段落；超长段落强制切割
    """
    chunks = [] # 保存最终所有 chunk
    chunk_id = 0 # 当前 chunk 编号
    current_text = "" # 当前正在拼接的文本 buffer
    current_section = "" # 当前章节标题
    current_pages = [] # 当前 chunk 涉及哪些页

    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]

        # 检测章节
        section = detect_section_title(text)
        if section:
            current_section = section

        # 按段落分割（双换行或单换行+缩进）
        paragraphs = re.split(r'\n(?=\s{2,}|\S)', text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前 buffer + 新段落不超限，追加
            if len(current_text) + len(para) <= chunk_size:
                # 如果 current_text 已经有内容，就先加一个换行；如果是空的，就不加换行。
                current_text += ("\n" if current_text else "") + para
                if page_num not in current_pages:
                    current_pages.append(page_num)
            else:
                # 超限：先保存当前旧的 current_text，再处理新来的 para
                if current_text and len(current_text) >= 50:  # 最小长度
                    chunks.append({
                        "chunk_id": f"fin_{chunk_id:04d}",
                        "text": current_text,
                        "title": f"{doc_title} - {current_section}" if current_section else doc_title,
                        "pages": current_pages[:], # 复制当前页码列表，避免后面修改影响已保存的 chunk。
                        "section": current_section,
                    })
                    chunk_id += 1

                # 如果当前段落 para 本身超长，强制切割
                if len(para) > chunk_size: 
                    # 步长是 chunk_size - overlap，保证切块之间有重叠部分，便于后续检索时上下文连续。
                    for start in range(0, len(para), chunk_size - overlap):
                        sub = para[start:start + chunk_size]
                        if len(sub) >= 50:
                            chunks.append({
                                "chunk_id": f"fin_{chunk_id:04d}",
                                "text": sub,
                                "title": f"{doc_title} - {current_section}" if current_section else doc_title,
                                "pages": [page_num],
                                "section": current_section,
                            })
                            chunk_id += 1
                    # 超长段落处理完后为什么清空？
                    # 因为这个段落已经被切成多个 chunk 保存了，后续的段落应该另起一个新的 chunk 来拼接，而不是继续追加到 current_text。
                    current_text = ""
                    current_pages = []
                # 如果 para 自己不长，只是 current_text + para 后会超长：
                else:
                    # 上一步保存旧 chunk 后，用当前段落 para 开启一个新的 chunk。
                    current_text = para
                    current_pages = [page_num]

    # 最后一个 chunk
    # 最后一个 chunk 为什么要单独保存？因为循环结束后 current_text 里可能还有内容没有保存，如果不单独保存，就会丢失最后一部分文本。
    if current_text and len(current_text) >= 50:
        chunks.append({
            "chunk_id": f"fin_{chunk_id:04d}",
            "text": current_text,
            "title": f"{doc_title} - {current_section}" if current_section else doc_title,
            "pages": current_pages[:],
            "section": current_section,
        })

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Parse PDF financial report to corpus.json")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output", default="data/financial/corpus.json")
    parser.add_argument("--chunk-size", type=int, default=500, help="Max chars per chunk")
    parser.add_argument("--doc-title", default="", help="Document title (auto-detect if empty)")
    args = parser.parse_args()

    print(f"Parsing PDF: {args.pdf}")
    pages = extract_text_by_page(args.pdf)
    print(f"Extracted {len(pages)} pages with text")

    # 自动检测标题
    doc_title = args.doc_title
    if not doc_title and pages:
        for line in pages[0]["text"].split("\n"):
            line = line.strip()
            if "公司" in line and len(line) > 5:
                doc_title = line
                break
    print(f"Document title: {doc_title}")

    # 切块
    chunks = chunk_pages(pages, chunk_size=args.chunk_size, doc_title=doc_title)
    print(f"Generated {len(chunks)} chunks")

    # 统计
    lengths = [len(c["text"]) for c in chunks]
    print(f"Chunk length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")

    sections = set(c["section"] for c in chunks if c["section"])
    print(f"Sections detected: {len(sections)}")
    for s in sorted(sections):
        count = sum(1 for c in chunks if c["section"] == s)
        print(f"  [{count:3d}] {s}")

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
