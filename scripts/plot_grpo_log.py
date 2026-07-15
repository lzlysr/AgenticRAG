#!/usr/bin/env python3
"""从 GRPO 日志提取指标并画训练曲线。

用法:
  python scripts/plot_grpo_log.py \
    --log logs/Qwen3-4B-grpo-zh.log \
    --out-dir training/outputs/Qwen3-4B-grpo-zh

输出:
  metrics.json
  loss.svg
"""
import argparse
import json
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")
METRIC_RE = re.compile(r"([A-Za-z0-9_./@-]+):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)")

METRIC_KEYS = {
    "actor/loss": "loss",
    "actor/kl_loss": "kl_loss",
    "critic/rewards/mean": "reward",
    "val-aux/financial_agentic_rag/reward/mean@1": "val_reward",
    "response_length/max": "response_length/max",
    "response_length/min": "response_length/min",
    "prompt_length/max": "prompt_length/max",
    "prompt_length/min": "prompt_length/min",
    "num_turns/min": "num_turns/min",
    "num_turns/max": "num_turns/max",
    "perf/time_per_step": "time_per_step",
}


def clean(line: str) -> str:
    return ANSI_RE.sub("", line).replace("\r", "").strip()


def parse_log(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = clean(raw)
        match = STEP_RE.search(line)
        if not match:
            continue
        record: dict[str, float] = {"step": float(match.group(1))}
        for key, value in METRIC_RE.findall(match.group(2)):
            try:
                record[key] = float(value)
            except ValueError:
                pass
        records.append(record)
    return records


def save_svg(records: list[dict[str, float]], out_path: Path) -> None:
    width, height = 960, 560
    margin_l, margin_r, margin_t, margin_b = 72, 24, 36, 58
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    series = [
        ("loss", "loss", "#2563eb"),
        ("reward", "reward", "#16a34a"),
        ("val_reward", "val reward", "#dc2626"),
        ("kl_loss", "kl loss", "#9333ea"),
    ]

    points_by_series: list[tuple[str, str, list[tuple[float, float]]]] = []
    values: list[float] = []
    for key, label, color in series:
        pts = [(r["step"], r[key]) for r in records if key in r]
        if pts:
            points_by_series.append((label, color, pts))
            values.extend(y for _, y in pts)

    if not points_by_series:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'><text x='20' y='30'>No metrics</text></svg>")
        return

    steps = [r["step"] for r in records]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        if x_max == x_min:
            return margin_l + plot_w / 2
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin_t + (y_max - y) / (y_max - y_min) * plot_h

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='24' text-anchor='middle' font-family='sans-serif' font-size='18'>GRPO loss / reward</text>",
        f"<line x1='{margin_l}' y1='{margin_t}' x2='{margin_l}' y2='{height-margin_b}' stroke='#111827'/>",
        f"<line x1='{margin_l}' y1='{height-margin_b}' x2='{width-margin_r}' y2='{height-margin_b}' stroke='#111827'/>",
    ]

    for i in range(6):
        y = margin_t + i * plot_h / 5
        value = y_max - i * (y_max - y_min) / 5
        lines.append(f"<line x1='{margin_l}' y1='{y:.1f}' x2='{width-margin_r}' y2='{y:.1f}' stroke='#e5e7eb'/>")
        lines.append(
            f"<text x='{margin_l-8}' y='{y+4:.1f}' text-anchor='end' font-family='monospace' font-size='11'>{value:.3g}</text>"
        )

    for i in range(0, int(x_max) + 1, max(1, int((x_max - x_min) // 8 or 1))):
        if i < x_min:
            continue
        x = sx(float(i))
        lines.append(f"<line x1='{x:.1f}' y1='{height-margin_b}' x2='{x:.1f}' y2='{height-margin_b+5}' stroke='#111827'/>")
        lines.append(
            f"<text x='{x:.1f}' y='{height-margin_b+20}' text-anchor='middle' font-family='monospace' font-size='11'>{i}</text>"
        )

    legend_x = margin_l + 12
    for idx, (label, color, pts) in enumerate(points_by_series):
        y = margin_t + 20 + idx * 22
        lines.append(f"<rect x='{legend_x}' y='{y-10}' width='12' height='12' fill='{color}'/>")
        lines.append(f"<text x='{legend_x+18}' y='{y}' font-family='sans-serif' font-size='12'>{label}</text>")
        coords = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in pts)
        lines.append(f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, v in pts:
            lines.append(f"<circle cx='{sx(x):.1f}' cy='{sy(v):.1f}' r='2.4' fill='{color}'/>")

    lines.append(f"<text x='{width/2}' y='{height-16}' text-anchor='middle' font-family='sans-serif' font-size='13'>global step</text>")
    lines.append("</svg>")
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir) if args.out_dir else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = parse_log(log_path)
    compact_records: list[dict[str, float]] = []
    for raw in records:
        item: dict[str, float] = {"step": raw["step"]}
        for raw_key, out_key in METRIC_KEYS.items():
            if raw_key in raw:
                item[out_key] = raw[raw_key]
        compact_records.append(item)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(compact_records, ensure_ascii=False, indent=2))

    if compact_records:
        plot_path = out_dir / "loss.svg"
        save_svg(compact_records, plot_path)
        print(f"[grpo-summary] metrics: {metrics_path}")
        print(f"[grpo-summary] plot: {plot_path}")
    else:
        print(f"[grpo-summary] no step metrics found in {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
