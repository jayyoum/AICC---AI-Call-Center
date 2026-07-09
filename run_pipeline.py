#!/usr/bin/env python3
"""
Run the full voice pipeline (STT -> Ollama LLM -> TTS) on one audio file.

Usage:
    python run_pipeline.py --audio input_audio/example.wav --stt google --tts google --model llama3.2 --routes config/routes.json
    python run_pipeline.py --audio input_audio/example.wav --stt local --tts local --model llama3.2
    python run_pipeline.py --audio input_audio/example.wav --stt google --tts local --model llama3.2
    python run_pipeline.py --audio input_audio/example.wav --stt local --tts google --model llama3.2
"""

import argparse
import json
import sys
from pathlib import Path

from src.pipeline import run_pipeline
from src.utils import PIPELINE_RESULTS_CSV, append_csv_row, ensure_output_dirs


def main():
    parser = argparse.ArgumentParser(description="Run the full STT -> Ollama LLM -> TTS pipeline.")
    parser.add_argument("--audio", required=True, help="Path to the input audio file.")
    parser.add_argument("--stt", required=True, choices=["google", "local"], help="STT backend to use.")
    parser.add_argument("--llm", default="ollama", choices=["ollama"], help="LLM backend to use (only ollama supported).")
    parser.add_argument("--tts", required=True, choices=["google", "local"], help="TTS backend to use.")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name (default: llama3.2).")
    parser.add_argument("--routes", help="Path to a routes JSON file (see config/routes.example.json).")
    parser.add_argument("--stt-model-size", default="base.en", help="faster-whisper model size for local STT.")
    parser.add_argument("--tts-voice", default="en_US-lessac-medium", help="Piper voice name for local TTS.")
    parser.add_argument(
        "--prompt-mode",
        default="compact_v2",
        choices=["compact", "compact_v2", "full"],
        help="LLM routing prompt: compact (short/fast), compact_v2 (short + stress-case rules, default), or full (verbose).",
    )
    parser.add_argument(
        "--router-mode",
        default="pre_router",
        choices=["llm_only", "pre_router"],
        help="pre_router (default): rule-based pre-router first, Ollama fallback. llm_only: always use Ollama.",
    )
    args = parser.parse_args()

    ensure_output_dirs()

    routes = None
    if args.routes:
        routes_path = Path(args.routes)
        if not routes_path.exists():
            print(f"Error: Routes file not found: {routes_path}")
            sys.exit(1)
        try:
            routes = json.loads(routes_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"Error: Routes file is not valid JSON: {e}")
            sys.exit(1)

    row = run_pipeline(
        audio_path=Path(args.audio),
        stt_provider=args.stt,
        tts_provider=args.tts,
        llm_model=args.model,
        routes=routes,
        stt_model_size=args.stt_model_size,
        tts_voice=args.tts_voice,
        prompt_mode=args.prompt_mode,
        router_mode=args.router_mode,
    )

    print(f"\nRouter mode: {row.get('router_mode')}   Routing source: {row.get('routing_source')}"
          + (f"   Rule: {row.get('rule_name')}" if row.get("routing_source") == "pre_router" else f"   Prompt mode: {row.get('prompt_mode')}"))
    print(f"Transcript: {row['transcript'] or '(none)'}")
    print(f"Predicted route: {row['predicted_top_level_route']} -> {row['predicted_final_route']}")
    print(f"Action: {row['predicted_action']}")
    print(f"Response to customer: {row['response_to_customer'] or '(none)'}")
    if row["output_audio_path"]:
        print(f"Response audio saved to: {row['output_audio_path']}")
    if row["error"]:
        print(f"\nErrors encountered: {row['error']}")

    print(
        f"\nTimings (seconds) - STT: {row['stt_latency_seconds']}  "
        f"Route: {row['route_latency_seconds']} (LLM: {row['llm_latency_seconds']})  "
        f"TTS: {row['tts_latency_seconds']}  Total: {row['total_latency_seconds']}"
    )

    append_csv_row(PIPELINE_RESULTS_CSV, row)
    print(f"\nResult row appended to: {PIPELINE_RESULTS_CSV}")

    if row["error"] and not row["output_audio_path"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
