from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from tabulate import tabulate


def _aggregate_metrics(outputs: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for output in outputs:
        model = output["model"]
        metrics = output["metrics"]
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                sums[model][name] += float(value)
                counts[model][name] += 1
    averages: Dict[str, Dict[str, float]] = defaultdict(dict)
    for model, metric_values in sums.items():
        for name, total in metric_values.items():
            averages[model][name] = total / counts[model][name]
    return averages


def write_report(path: Path, payload: Dict[str, object]) -> None:
    outputs = payload.get("outputs", [])
    averages = _aggregate_metrics(outputs)

    lines = [
        "# TTS Benchmark Report\n",
        f"Run ID: {payload['run']['run_id']}\n",
    ]

    if not averages:
        lines.append("No outputs generated.\n")
        path.write_text("\n".join(lines))
        return

    headers = ["Model", "avg_rtf", "avg_duration_s", "avg_wer"]
    rows = []
    for model, metrics in averages.items():
        rows.append(
            [
                model,
                f"{metrics.get('rtf', 0.0):.3f}",
                f"{metrics.get('duration_s', 0.0):.3f}",
                f"{metrics.get('wer', 0.0):.3f}",
            ]
        )
    leaderboard = tabulate(rows, headers=headers, tablefmt="github")
    lines.append("## Leaderboard\n")
    lines.append(leaderboard)
    lines.append("\n## Per-model metrics\n")
    for model, metrics in averages.items():
        lines.append(f"### {model}\n")
        metric_rows = [[name, f"{value:.4f}"] for name, value in sorted(metrics.items())]
        lines.append(tabulate(metric_rows, headers=["Metric", "Average"], tablefmt="github"))
        lines.append("")

    path.write_text("\n".join(lines))
