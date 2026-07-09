#!/usr/bin/env python3
"""
Benchmark router modes (llm_only vs pre_router) across audio scenarios.

Runs every .wav file in an audio directory through the full pipeline once per
router mode, holding STT/TTS/model/prompt-mode fixed, then summarizes how much
latency the rule-based pre-router saves and how often it handles a request
without calling Ollama.

Example (the 12 AICC scenarios, all local so no Google APIs are called):
    python benchmark_router_modes.py \\
      --audio-dir input_audio_16k \\
      --routes config/routes.json \\
      --stt local \\
      --tts local \\
      --model llama3.2 \\
      --prompt-mode compact_v2 \\
      --router-modes llm_only pre_router

Outputs:
    output/results/router_mode_benchmark.csv   (one row per audio x router mode)
    output/results/router_mode_summary.csv     (one row per router mode)

Uses src.pipeline.run_pipeline(), which does NOT touch pipeline_results.csv or
the other benchmark CSVs.
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

DETAILED_CSV = RESULTS_DIR / "router_mode_benchmark.csv"
SUMMARY_CSV = RESULTS_DIR / "router_mode_summary.csv"

ROUTER_MODES = ("llm_only", "pre_router")

DETAILED_COLUMNS = [
    "scenario",
    "router_mode",
    "routing_source",
    "pre_router_handled",
    "rule_name",
    "model",
    "prompt_mode",
    "stt_provider",
    "tts_provider",
    "transcript",
    "predicted_top_level_route",
    "predicted_final_route",
    "predicted_action",
    "response_to_customer",
    "route_valid",
    "route_normalized",
    "original_final_route",
    "normalized_final_route",
    "stt_latency_seconds",
    "llm_latency_seconds",
    "route_latency_seconds",
    "tts_latency_seconds",
    "total_latency_seconds",
    "error",
]

SUMMARY_COLUMNS = [
    "router_mode",
    "run_count",
    "error_count",
    "avg_total_latency_seconds",
    "avg_stt_latency_seconds",
    "avg_llm_latency_seconds",
    "avg_tts_latency_seconds",
    "pre_router_handled_count",
    "pre_router_handled_rate",
    "route_valid_true_count",
    "route_valid_false_count",
    "route_valid_rate",
    "normalized_count",
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
        "router_mode": row.get("router_mode", ""),
        "routing_source": row.get("routing_source", ""),
        "pre_router_handled": row.get("pre_router_handled", ""),
        "rule_name": row.get("rule_name", ""),
        "model": row.get("llm_model", ""),
        "prompt_mode": row.get("prompt_mode", ""),
        "stt_provider": row.get("stt_provider", ""),
        "tts_provider": row.get("tts_provider", ""),
        "transcript": row.get("transcript", ""),
        "predicted_top_level_route": row.get("predicted_top_level_route", ""),
        "predicted_final_route": row.get("predicted_final_route", ""),
        "predicted_action": row.get("predicted_action", ""),
        "response_to_customer": row.get("response_to_customer", ""),
        "route_valid": row.get("route_valid", ""),
        "route_normalized": row.get("route_normalized", ""),
        "original_final_route": row.get("original_final_route", ""),
        "normalized_final_route": row.get("normalized_final_route", ""),
        "stt_latency_seconds": row.get("stt_latency_seconds", 0.0),
        "llm_latency_seconds": row.get("llm_latency_seconds", 0.0),
        "route_latency_seconds": row.get("route_latency_seconds", 0.0),
        "tts_latency_seconds": row.get("tts_latency_seconds", 0.0),
        "total_latency_seconds": row.get("total_latency_seconds", 0.0),
        "error": row.get("error", ""),
    }


def summarize(detailed_rows: list[dict]) -> list[dict]:
    """Group detailed rows by router_mode, sorted fastest avg-total first."""
    groups = defaultdict(list)
    for row in detailed_rows:
        groups[row["router_mode"]].append(row)

    summary_rows = []
    for mode, mode_rows in groups.items():
        count = len(mode_rows)
        error_count = sum(1 for r in mode_rows if r.get("error"))

        totals = [float(r.get("total_latency_seconds") or 0) for r in mode_rows]
        stts = [float(r.get("stt_latency_seconds") or 0) for r in mode_rows]
        llms = [float(r.get("llm_latency_seconds") or 0) for r in mode_rows]
        ttss = [float(r.get("tts_latency_seconds") or 0) for r in mode_rows]

        handled_count = sum(1 for r in mode_rows if r.get("pre_router_handled") is True)
        handled_rate = round(handled_count / count, 3) if count else ""

        true_count = sum(1 for r in mode_rows if r.get("route_valid") is True)
        false_count = sum(1 for r in mode_rows if r.get("route_valid") is False)
        validated = true_count + false_count
        rate = round(true_count / validated, 3) if validated else ""

        normalized_count = sum(1 for r in mode_rows if r.get("route_normalized") is True)

        summary_rows.append(
            {
                "router_mode": mode,
                "run_count": count,
                "error_count": error_count,
                "avg_total_latency_seconds": round(sum(totals) / count, 3) if count else 0,
                "avg_stt_latency_seconds": round(sum(stts) / count, 3) if count else 0,
                "avg_llm_latency_seconds": round(sum(llms) / count, 3) if count else 0,
                "avg_tts_latency_seconds": round(sum(ttss) / count, 3) if count else 0,
                "pre_router_handled_count": handled_count,
                "pre_router_handled_rate": handled_rate,
                "route_valid_true_count": true_count,
                "route_valid_false_count": false_count,
                "route_valid_rate": rate,
                "normalized_count": normalized_count,
            }
        )

    summary_rows.sort(key=lambda r: r["avg_total_latency_seconds"])
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
    parser = argparse.ArgumentParser(description="Benchmark router modes across audio scenarios.")
    parser.add_argument("--audio-dir", required=True, help="Directory of .wav files to benchmark.")
    parser.add_argument("--routes", help="Path to a routes JSON file (see config/routes.example.json).")
    parser.add_argument("--stt", default="local", choices=["google", "local"], help="STT backend (default: local).")
    parser.add_argument("--tts", default="local", choices=["google", "local"], help="TTS backend (default: local).")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name (default: llama3.2).")
    parser.add_argument(
        "--prompt-mode",
        default="compact_v2",
        choices=list(VALID_PROMPT_MODES),
        help="LLM routing prompt used on the Ollama fallback (default: compact_v2).",
    )
    parser.add_argument(
        "--router-modes",
        nargs="+",
        default=list(ROUTER_MODES),
        choices=list(ROUTER_MODES),
        help="Router modes to compare (default: llm_only pre_router).",
    )
    parser.add_argument("--stt-model-size", default="base.en", help="faster-whisper model size for local STT.")
    parser.add_argument("--tts-voice", default="en_US-lessac-medium", help="Piper voice name for local TTS.")
    args = parser.parse_args()

    ensure_output_dirs()

    routes = load_routes(args.routes)
    wav_files = find_wav_files(Path(args.audio_dir))

    print(f"Benchmarking {len(wav_files)} audio file(s) x {len(args.router_modes)} router mode(s) "
          f"= {len(wav_files) * len(args.router_modes)} run(s).")
    print(f"STT: {args.stt}   TTS: {args.tts}   Model: {args.model}   Prompt mode: {args.prompt_mode}")
    print(f"Router modes: {', '.join(args.router_modes)}\n")

    detailed_rows = []
    for mode in args.router_modes:
        print(f"=== router_mode: {mode} ===", flush=True)
        for wav_path in wav_files:
            print(f"  {wav_path.name} ...", flush=True)
            row = run_pipeline(
                audio_path=wav_path,
                stt_provider=args.stt,
                tts_provider=args.tts,
                llm_model=args.model,
                routes=routes,
                stt_model_size=args.stt_model_size,
                tts_voice=args.tts_voice,
                prompt_mode=args.prompt_mode,
                router_mode=mode,
            )
            detailed = detailed_row_from_pipeline(row)
            detailed_rows.append(detailed)
            if detailed["error"]:
                print(f"      -> error: {detailed['error']}")
            else:
                src = detailed["routing_source"]
                tag = f"{src}:{detailed['rule_name']}" if src == "pre_router" else src
                print(f"      -> [{tag}] {detailed['predicted_final_route'] or '(no final route)'} "
                      f"| route_valid={detailed['route_valid']} | {detailed['total_latency_seconds']}s")

    write_csv(DETAILED_CSV, DETAILED_COLUMNS, detailed_rows)
    summary_rows = summarize(detailed_rows)
    write_csv(SUMMARY_CSV, SUMMARY_COLUMNS, summary_rows)

    print("\n===== summary by router mode (fastest avg total latency first) =====")
    print_summary_table(summary_rows)
    print(f"\nDetailed rows saved to: {DETAILED_CSV}")
    print(f"Summary saved to:       {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
