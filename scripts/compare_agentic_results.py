#!/usr/bin/env python3
"""打印 Base、SFT、GRPO 三组 Agentic 评测结果的对比表。

用法：
  python scripts/compare_agentic_results.py

也可以覆盖默认文件：
  python scripts/compare_agentic_results.py \
    --base path/to/base_judged.json \
    --sft path/to/sft_judged.json \
    --grpo path/to/grpo_judged.json
"""

import argparse
import json
import os


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(PROJECT_DIR, "results", "run_agentic_judge", "financial")
TOOL_RESULT_DIR = os.path.join(PROJECT_DIR, "results", "run_agentic_tool", "financial")

DEFAULT_FILES = {
    "Base": os.path.join(RESULT_DIR, "agentic_185_Qwen3-4B_judged.json"),
    "SFT": os.path.join(RESULT_DIR, "agentic_185_Qwen3-4B-sft-zh_judged.json"),
    "GRPO": os.path.join(RESULT_DIR, "agentic_185_Qwen3-4B-grpo-zh-v2_judged.json"),
}

DEFAULT_TOOL_FILES = {
    "Base": os.path.join(TOOL_RESULT_DIR, "agentic_185_Qwen3-4B_tool.json"),
    "SFT": os.path.join(TOOL_RESULT_DIR, "agentic_185_Qwen3-4B-sft-zh_tool.json"),
    "GRPO": os.path.join(TOOL_RESULT_DIR, "agentic_185_Qwen3-4B-grpo-zh-v2_tool.json"),
}

TOOL_COLUMNS = [
    "keyword_search",
    "hybrid_search",
    "semantic_search",
    "graph_search",
]

METRICS = [
    ("EM", ("summary", "avg_em")),
    ("F1", ("summary", "avg_f1")),
    ("Judge_C", ("summary", "judge_correctness")),
    ("Faith", ("summary", "judge_faithfulness")),
    ("CtxP", ("summary", "judge_context_precision")),
    ("Avg Turns", ("summary", "avg_turns")),
    ("Avg Tools", ("summary", "avg_tool_calls")),
    ("hop_recall", ("summary", "diagnostics", "overall", "avg_hop_recall")),
    (
        "premature_collapse",
        ("summary", "diagnostics", "overall", "premature_collapse_rate"),
    ),
    ("over_extension", ("summary", "diagnostics", "overall", "over_extension_rate")),
    ("step_alignment", ("summary", "diagnostics", "overall", "avg_step_alignment")),
]


def _read_metric(data: dict, path: tuple[str, ...], file_path: str) -> float:
    value = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"{file_path} 缺少字段：{'.'.join(path)}")
        value = value[key]
    return float(value)


def _load_row(label: str, file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    values = [_read_metric(data, path, file_path) for _, path in METRICS]
    return [label, *(f"{value:.3f}" for value in values)]


def _print_table(rows: list[list[str]], headers: list[str] | None = None) -> None:
    headers = headers or ["Model", *(name for name, _ in METRICS)]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def _load_tool_row(label: str, file_path: str) -> list[str]:
    """读取 run_agentic_tool.py 生成的单模型工具分布行。"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    table = data.get("table", [])
    if not table or not isinstance(table[0], dict):
        raise KeyError(f"{file_path} 缺少 table[0] 工具分布字段")

    source_row = table[0]
    values = []
    for tool in TOOL_COLUMNS:
        if tool not in source_row:
            raise KeyError(f"{file_path} 缺少工具字段：{tool}")
        values.append(str(source_row[tool]))
    return [label, *values]


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 Base、SFT、GRPO 的 Agentic 评测结果")
    parser.add_argument("--base", default=DEFAULT_FILES["Base"], help="Base judged JSON")
    parser.add_argument("--sft", default=DEFAULT_FILES["SFT"], help="SFT judged JSON")
    parser.add_argument("--grpo", default=DEFAULT_FILES["GRPO"], help="GRPO judged JSON")
    parser.add_argument("--base-tool", default=DEFAULT_TOOL_FILES["Base"], help="Base 工具分布 JSON")
    parser.add_argument("--sft-tool", default=DEFAULT_TOOL_FILES["SFT"], help="SFT 工具分布 JSON")
    parser.add_argument("--grpo-tool", default=DEFAULT_TOOL_FILES["GRPO"], help="GRPO 工具分布 JSON")
    args = parser.parse_args()

    rows = [
        _load_row("Base", args.base),
        _load_row("SFT", args.sft),
        _load_row("GRPO", args.grpo),
    ]
    _print_table(rows)

    tool_rows = [
        _load_tool_row("Base", args.base_tool),
        _load_tool_row("SFT", args.sft_tool),
        _load_tool_row("GRPO", args.grpo_tool),
    ]
    print()
    print("工具分布（按工具调用次数占比）")
    _print_table(tool_rows, headers=["Model", *TOOL_COLUMNS])


if __name__ == "__main__":
    main()
