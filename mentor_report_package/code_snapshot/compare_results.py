#!/usr/bin/env python3
"""
Summarize and compare pipeline_results.csv across provider combinations.

Usage:
    python compare_results.py --results output/results/pipeline_results.csv

Notes on manual_success_label:
    If the column has any non-blank values, this script treats the
    (case-insensitive) values "success", "pass", "yes", "true", "1" as a
    success and any other non-blank value as a failure. Blank labels are
    left out of the success-rate calculation (not yet reviewed).
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from src.utils import RESULTS_DIR

SUCCESS_VALUES = {"success", "pass", "yes", "true", "1"}


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_rows(results_path: Path) -> list[dict]:
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with results_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict]) -> list[dict]:
    """Build one summary row per (stt_provider, tts_provider) combination."""
    groups = defaultdict(list)
    for row in rows:
        combo = f"{row.get('stt_provider', '')}+{row.get('tts_provider', '')}"
        groups[combo].append(row)

    summary_rows = []
    for combo, combo_rows in sorted(groups.items()):
        count = len(combo_rows)
        total_latencies = [to_float(r.get("total_latency_seconds")) for r in combo_rows]
        stt_latencies = [to_float(r.get("stt_latency_seconds")) for r in combo_rows]
        llm_latencies = [to_float(r.get("llm_latency_seconds")) for r in combo_rows]
        tts_latencies = [to_float(r.get("tts_latency_seconds")) for r in combo_rows]

        error_count = sum(1 for r in combo_rows if r.get("error"))
        stt_error_count = sum(1 for r in combo_rows if "STT error" in (r.get("error") or ""))
        llm_error_count = sum(1 for r in combo_rows if "LLM error" in (r.get("error") or ""))
        tts_error_count = sum(1 for r in combo_rows if "TTS error" in (r.get("error") or ""))

        labels = [
            (r.get("manual_success_label") or "").strip()
            for r in combo_rows
            if (r.get("manual_success_label") or "").strip()
        ]
        if labels:
            successes = sum(1 for label in labels if label.lower() in SUCCESS_VALUES)
            success_rate = round(successes / len(labels), 3)
        else:
            success_rate = ""

        summary_rows.append(
            {
                "combination": combo,
                "run_count": count,
                "avg_total_latency_seconds": round(sum(total_latencies) / count, 3) if count else 0,
                "avg_stt_latency_seconds": round(sum(stt_latencies) / count, 3) if count else 0,
                "avg_llm_latency_seconds": round(sum(llm_latencies) / count, 3) if count else 0,
                "avg_tts_latency_seconds": round(sum(tts_latencies) / count, 3) if count else 0,
                "error_count": error_count,
                "stt_error_count": stt_error_count,
                "llm_error_count": llm_error_count,
                "tts_error_count": tts_error_count,
                "labeled_run_count": len(labels),
                "manual_success_rate": success_rate,
            }
        )

    return summary_rows


def print_table(summary_rows: list[dict]) -> None:
    if not summary_rows:
        print("No rows to summarize.")
        return

    columns = list(summary_rows[0].keys())
    widths = {col: max(len(col), max(len(str(row[col])) for row in summary_rows)) for col in columns}

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(" | ".join(str(row[col]).ljust(widths[col]) for col in columns))


def save_summary(summary_rows: list[dict], output_path: Path) -> None:
    if not summary_rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize and compare pipeline results by provider combination.")
    parser.add_argument("--results", default=str(RESULTS_DIR / "pipeline_results.csv"), help="Path to pipeline_results.csv")
    args = parser.parse_args()

    try:
        rows = load_rows(Path(args.results))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not rows:
        print("Results file is empty. Run some pipelines first.")
        sys.exit(0)

    summary_rows = summarize(rows)
    print_table(summary_rows)

    summary_path = RESULTS_DIR / "summary.csv"
    save_summary(summary_rows, summary_path)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
