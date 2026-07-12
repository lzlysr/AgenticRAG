#!/usr/bin/env python3
"""统计 eval_agentic 结果文件中的工具类型分布。

用法：
  python scripts/run_agentic_tool.py \
    results/eval_agentic/financial/agentic_185_Qwen3-4B.json

  python scripts/run_agentic_tool.py \
    results/eval_agentic/financial/agentic_185_Qwen3-4B-sft-zh.json \
    --output-dir results/run_agentic_tool/financial

输出：
  results/run_agentic_tool/financial/agentic_{样本数量}_{模型名}_tool.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


DEFAULT_TOOL_ORDER = [
    "keyword_search",
    "hybrid_search",
    "semantic_search",
    "graph_search",
]


def _iter_call_tools(tool_call: dict) -> Iterable[str]:
    """兼容 tool_calls 中的 tool/name 字段，并展开列表型工具名。"""
    tool = tool_call.get("tool") or tool_call.get("name")
    if isinstance(tool, list):
        for item in tool:
            if item:
                yield str(item)
    elif isinstance(tool, str) and tool:
        yield tool


def summarize_file(path: Path) -> dict:
    """统计单个 agentic 结果文件的工具调用次数和占比。"""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    counts = Counter()
    zero_tool_samples = 0

    for sample in results:
        sample_calls = 0
        for call in sample.get("tool_calls", []) or []:
            tools = list(_iter_call_tools(call))
            sample_calls += len(tools)
            counts.update(tools)
        if sample_calls == 0:
            zero_tool_samples += 1

    total = sum(counts.values())
    percentages = {
        tool: (count / total * 100 if total else 0.0)
        for tool, count in counts.items()
    }

    return {
        "source_file": str(path),
        "model": data.get("summary", {}).get("model", path.stem),
        "num_samples": len(results),
        "total_tool_calls": total,
        "zero_tool_samples": zero_tool_samples,
        "counts": dict(sorted(counts.items())),
        "percentages": dict(sorted(percentages.items())),
    }


def _ordered_tools(summary: dict, preferred: list[str]) -> list[str]:
    """按指定工具顺序排列表头，额外工具追加到最后。"""
    seen = []
    for tool in preferred:
        if tool in summary["counts"]:
            seen.append(tool)
    extras = sorted(tool for tool in summary["counts"] if tool not in seen)
    return seen + extras


def _pct(value: float) -> str:
    return f"{round(value):.0f}%"


def build_row(summary: dict, tools: list[str]) -> dict:
    """构造图表可直接使用的单模型百分比行。"""
    row = {"model": summary["model"]}
    for tool in tools:
        row[tool] = _pct(summary["percentages"].get(tool, 0.0))
    return row


def _safe_name(value: str) -> str:
    """将模型名转成适合作为文件名的形式。"""
    return value.replace("/", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 eval_agentic 结果文件中的工具类型分布。"
    )
    parser.add_argument(
        "--result_file",
        default=os.path.join(RESULTS_DIR, "eval_agentic", "financial", "agentic_185_Qwen3-4B.json"),
        help="eval_agentic 生成的单个 JSON 结果文件。",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(RESULTS_DIR, "run_agentic_tool", "financial"),
        help="统计结果输出目录。",
    )
    parser.add_argument(
        "--tool-order",
        nargs="*",
        default=DEFAULT_TOOL_ORDER,
        help="表格中的工具列顺序。",
    )
    args = parser.parse_args()

    result_path = Path(args.result_file)
    if not result_path.exists():
        raise FileNotFoundError(result_path)

    summary = summarize_file(result_path)
    tools = _ordered_tools(summary, args.tool_order)
    table = [build_row(summary, tools)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "table": table,
        "tool_order": tools,
        "model": summary,
    }

    json_path = out_dir / (
        f"agentic_{summary['num_samples']}_{_safe_name(summary['model'])}_tool.json"
    )
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"已保存：{json_path}")
    print()
    print(json.dumps({"table": table}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
