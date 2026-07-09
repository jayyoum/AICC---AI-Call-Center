#!/usr/bin/env python3
"""
Run pipeline combinations against every scenario in a user-provided manifest CSV.

Usage:
    python run_batch.py --manifest config/manifest.csv
    python run_batch.py --manifest config/manifest.csv --combinations google_google local_local
    python run_batch.py --manifest config/manifest.csv --routes config/routes.json --model llama3.2

See config/manifest.example.csv for the expected manifest schema.
"""

import argparse
import json
import sys
from pathlib import Path

from src.batch_runner import COMBINATIONS, run_batch
from src.utils import ensure_output_dirs


def main():
    parser = argparse.ArgumentParser(description="Run batch pipeline tests from a manifest CSV.")
    parser.add_argument("--manifest", required=True, help="Path to a manifest CSV file.")
    parser.add_argument(
        "--combinations",
        nargs="+",
        default=list(COMBINATIONS.keys()),
        choices=list(COMBINATIONS.keys()),
        help=f"Which provider combinations to run (default: all of {list(COMBINATIONS.keys())}).",
    )
    parser.add_argument("--model", default="llama3.2", help="Ollama model name (default: llama3.2).")
    parser.add_argument("--routes", help="Path to a routes JSON file (see config/routes.example.json).")
    parser.add_argument("--stt-model-size", default="base.en", help="faster-whisper model size for local STT.")
    parser.add_argument("--tts-voice", default="en_US-lessac-medium", help="Piper voice name for local TTS.")
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

    try:
        results = run_batch(
            manifest_path=Path(args.manifest),
            combination_names=args.combinations,
            llm_model=args.model,
            routes=routes,
            stt_model_size=args.stt_model_size,
            tts_voice=args.tts_voice,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    error_count = sum(1 for r in results if r.get("error"))
    print(f"\nBatch complete: {len(results)} run(s), {error_count} with errors.")
    print("Results appended to output/results/pipeline_results.csv")


if __name__ == "__main__":
    main()
