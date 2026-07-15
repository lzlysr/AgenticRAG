#!/usr/bin/env python3
"""过滤 GRPO 训练日志，让终端只显示进度条和关键指标。

用法:
  python scripts/filter_grpo_console.py < logs/Qwen3-4B-grpo-zh.log

在 start_grpo.sh 中通常这样使用:
  python3 -m verl.trainer.main_ppo ... 2>&1 | tee "$LOG" | python "$PROJECT_DIR/scripts/filter_grpo_console.py"
"""
import re
import sys


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")
METRIC_RE = re.compile(r"([A-Za-z0-9_./@-]+):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)")


def clean(line: str) -> str:
    return ANSI_RE.sub("", line).replace("\r", "").strip()


def parse_metrics(text: str) -> dict[str, str]:
    return {k: v for k, v in METRIC_RE.findall(text)}


def main() -> int:
    for raw in sys.stdin:
        line = clean(raw)
        if not line:
            continue

        if "Training Progress:" in line:
            # tqdm 行已经足够短，直接保留。
            print(line, flush=True)
            continue

        if "Initial validation metrics" in line or "Final validation metrics" in line:
            print(line, flush=True)
            continue

        match = STEP_RE.search(line)
        if match:
            step = match.group(1)
            metrics = parse_metrics(match.group(2))
            fields = [
                ("actor/loss", "loss"),
                ("critic/rewards/mean", "reward"),
                ("val-aux/financial_agentic_rag/reward/mean@1", "val_reward"),
                ("actor/kl_loss", "kl"),
                ("response_length/clip_ratio", "clip"),
                ("timing_s/step", "step_s"),
                ("timing_s/update_actor", "update_s"),
            ]
            parts = [f"step {step}"]
            for key, label in fields:
                if key in metrics:
                    parts.append(f"{label}={metrics[key]}")
            print(" | ".join(parts), flush=True)
            continue

        # 保留真正需要人看的异常摘要。完整栈仍在 LOG 里。
        if any(token in line for token in ("Traceback", "RuntimeError:", "EngineDeadError:", "CUDA out of memory")):
            print(line, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
