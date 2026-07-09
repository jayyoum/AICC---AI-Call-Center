#!/usr/bin/env python3
"""
Benchmark Ollama routing prompt modes across a directory of audio scenarios.

Runs every .wav file in an audio directory through the full pipeline
(STT -> Ollama -> TTS) once per prompt mode, then summarizes latency and
route-validity differences between the modes so you can compare them.

Example (the 12 AICC scenarios, all local so no Google APIs are called):
    python benchmark_prompt_modes.py \\
      --audio-dir input_audio_16k \\
      --routes config/routes.json \\
      --stt local \\
      --tts local \\
      --model llama3.2 \\
      --modes compact compact_v2 full

Outputs:
    output/results/prompt_mode_benchmark.csv   (one row per audio x mode)
    output/results/prompt_mode_summary.csv     (one row per prompt mode)

This uses the existing src.pipeline.run_pipeline(), which does NOT touch
pipeline_results.csv, so the regular pipeline results file is left alone.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.llm_ollama import VALID_PROMPT_MODES
from src.pipeline import run_pipeline
from src.utils import RESULTS_DIR, ensure_output_dirs

DETAILED_CSV = RESULTS_DIR / "prompt_mode_benchmark.csv"
SUMMARY_CSV = RESULTS_DIR / "prompt_mode_summary.csv"

DETAILED_COLUMNS = [
    "scenario",
    "prompt_mode",
    "router_mode",
    "routing_source",
    "pre_router_handled",
    "rule_name",
    "stt_provider",
    "tts_provider",
    "llm_model",
    "transcript",
    "predicted_top_level_route",
    "predicted_final_route",
    "predicted_action",
    "response_to_customer",
    "route_valid",
    "stt_latency_seconds",
    "llm_latency_seconds",
    "tts_latency_seconds",
    "total_latency_seconds",
    "error",
]

SUMMARY_COLUMNS = [
    "prompt_mode",
    "run_count",
    "error_count",
    "avg_total_latency_seconds",
    "avg_stt_latency_seconds",
    "avg_llm_latency_seconds",
    "avg_tts_latency_seconds",
    "route_valid_true_count",
    "route_valid_false_count",
    "route_valid_rate",
]


def load_routes(routes_arg: str | None) -> dict | None:
    if not routes_arg:
        return None
    routes_path = Path(routes_arg)
    if not routes_path.exists():
        print(f"Error: Routes file not found: {routes_path}")
        sys.exit(1)
    try:
        return json.loads(routes_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: Routes file is not valid JSON: {e}")
        sys.exit(1)


def find_wav_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists() or not audio_dir.is_dir():
        print(f"Error: Audio directory not found: {audio_dir}")
        sys.exit(1)
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        print(f"Error: No .wav files found in {audio_dir}")
        sys.exit(1)
    return wavs


def detailed_row_from_pipeline(row: dict) -> dict:
    """Map a run_pipeline() result row onto the benchmark's detailed columns."""
    return {
        "scenario": Path(row.get("audio_file", "")).name,
        "prompt_mode": row.get("prompt_mode", ""),
        "router_mode": row.get("router_mode", ""),
        "routing_source": row.get("routing_source", ""),
        "pre_router_handled": row.get("pre_router_handled", ""),
        "rule_name": row.get("rule_name", ""),
        "stt_provider": row.get("stt_provider", ""),
        "tts_provider": row.get("tts_provider", ""),
        "llm_model": row.get("llm_model", ""),
        "transcript": row.get("transcript", ""),
        "predicted_top_level_route": row.get("predicted_top_level_route", ""),
        "predicted_final_route": row.get("predicted_final_route", ""),
        "predicted_action": row.get("predicted_action", ""),
        "response_to_customer": row.get("response_to_customer", ""),
        "route_valid": row.get("route_valid", ""),
        "stt_latency_seconds": row.get("stt_latency_seconds", 0.0),
        "llm_latency_seconds": row.get("llm_latency_seconds", 0.0),
        "tts_latency_seconds": row.get("tts_latency_seconds", 0.0),
        "total_latency_seconds": row.get("total_latency_seconds", 0.0),
        "error": row.get("error", ""),
    }


def summarize(detailed_rows: list[dict]) -> list[dict]:
    """Group detailed rows by prompt_mode into summary rows."""
    groups = defaultdict(list)
    for row in detailed_rows:
        groups[row["prompt_mode"]].append(row)

    summary_rows = []
    for mode, mode_rows in sorted(groups.items()):
        count = len(mode_rows)
        error_count = sum(1 for r in mode_rows if r.get("error"))

        totals = [float(r.get("total_latency_seconds") or 0) for r in mode_rows]
        stts = [float(r.get("stt_latency_seconds") or 0) for r in mode_rows]
        llms = [float(r.get("llm_latency_seconds") or 0) for r in mode_rows]
        ttss = [float(r.get("tts_latency_seconds") or 0) for r in mode_rows]

        # route_valid is True/False when a route tree is used, None/"" otherwise.
        true_count = sum(1 for r in mode_rows if r.get("route_valid") is True)
        false_count = sum(1 for r in mode_rows if r.get("route_valid") is False)
        validated = true_count + false_count
        rate = round(true_count / validated, 3) if validated else ""

        summary_rows.append(
            {
                "prompt_mode": mode,
                "run_count": count,
                "error_count": error_count,
                "avg_total_latency_seconds": round(sum(totals) / count, 3) if count else 0,
                "avg_stt_latency_seconds": round(sum(stts) / count, 3) if count else 0,
                "avg_llm_latency_seconds": round(sum(llms) / count, 3) if count else 0,
                "avg_tts_latency_seconds": round(sum(ttss) / count, 3) if count else 0,
                "route_valid_true_count": true_count,
                "route_valid_false_count": false_count,
                "route_valid_rate": rate,
            }
        )
    return summary_rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def print_summary_table(summary_rows: list[dict]) -> None:
    if not summary_rows:
        print("No rows to summarize.")
        return
    widths = {
        col: max(len(col), max(len(str(r[col])) for r in summary_rows))
        for col in SUMMARY_COLUMNS
    }
    header = " | ".join(col.ljust(widths[col]) for col in SUMMARY_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(" | ".join(str(row[col]).ljust(widths[col]) for col in SUMMARY_COLUMNS))


def main():
    parser = argparse.ArgumentParser(description="Benchmark prompt modes across audio scenarios.")
    parser.add_argument("--audio-dir", required=True, help="Directory of .wav files to benchmark.")
    parser.add_argument("--routes", help="Path to a routes JSON file (see config/routes.example.json).")
    parser.add_argument("--stt", default="local", choices=["google", "local"], help="STT backend (default: local).")
    parser.add_argument("--tts", default="local", choices=["google", "local"], help="TTS backend (default: local).")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name (default: llama3.2).")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["compact", "compact_v2", "full"],
        choices=list(VALID_PROMPT_MODES),
        help="Prompt modes to compare (default: compact compact_v2 full).",
    )
    parser.add_argument(
        "--router-mode",
        default="llm_only",
        choices=["llm_only", "pre_router"],
        help="Router mode (default: llm_only, so prompt modes are measured directly without the pre-router).",
    )
    parser.add_argument("--stt-model-size", default="base.en", help="faster-whisper model size for local STT.")
    parser.add_argument("--tts-voice", default="en_US-lessac-medium", help="Piper voice name for local TTS.")
    args = parser.parse_args()

    ensure_output_dirs()

    routes = load_routes(args.routes)
    wav_files = find_wav_files(Path(args.audio_dir))

    print(f"Benchmarking {len(wav_files)} audio file(s) x {len(args.modes)} mode(s) "
          f"= {len(wav_files) * len(args.modes)} run(s).")
    print(f"STT: {args.stt}   TTS: {args.tts}   Model: {args.model}   "
          f"Router mode: {args.router_mode}   Modes: {', '.join(args.modes)}\n")

    detailed_rows = []
    for wav_path in wav_files:
        for mode in args.modes:
            print(f"  {wav_path.name}  [{mode}] ...", flush=True)
            row = run_pipeline(
                audio_path=wav_path,
                stt_provider=args.stt,
                tts_provider=args.tts,
                llm_model=args.model,
                routes=routes,
                stt_model_size=args.stt_model_size,
                tts_voice=args.tts_voice,
                prompt_mode=mode,
                router_mode=args.router_mode,
            )
            detailed = detailed_row_from_pipeline(row)
            detailed_rows.append(detailed)
            note = detailed["error"] or f"{detailed['total_latency_seconds']}s"
            print(f"      -> {detailed['predicted_final_route'] or '(no final route)'} "
                  f"| route_valid={detailed['route_valid']} | {note}")

    write_csv(DETAILED_CSV, DETAILED_COLUMNS, detailed_rows)
    summary_rows = summarize(detailed_rows)
    write_csv(SUMMARY_CSV, SUMMARY_COLUMNS, summary_rows)

    print("\n===== summary by prompt mode =====")
    print_summary_table(summary_rows)
    print(f"\nDetailed rows saved to: {DETAILED_CSV}")
    print(f"Summary saved to:       {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
