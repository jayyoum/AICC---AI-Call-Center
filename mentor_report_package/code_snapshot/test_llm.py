#!/usr/bin/env python3
"""
Test the Ollama LLM backend on a single transcript or text file.

Usage:
    python test_llm.py --text "customer transcript here" --model llama3.2
    python test_llm.py --file output/transcripts/example.txt --model llama3.2 --routes config/routes.json
"""

import argparse
import json
import sys
from pathlib import Path

from src.llm_ollama import route_with_ollama
from src.utils import (
    COMPONENT_RESULTS_CSV,
    LLM_OUTPUTS_DIR,
    append_csv_row,
    ensure_output_dirs,
    get_text_from_args,
    new_run_id,
    now_iso,
    save_json_log,
)


def main():
    parser = argparse.ArgumentParser(description="Test the Ollama LLM backend on a transcript.")
    parser.add_argument("--text", help="Transcript text directly.")
    parser.add_argument("--file", help="Path to a .txt file containing the transcript.")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name (default: llama3.2).")
    parser.add_argument("--routes", help="Path to a routes JSON file (see config/routes.example.json).")
    args = parser.parse_args()

    ensure_output_dirs()

    try:
        transcript = get_text_from_args(args.text, args.file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

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

    run_id = new_run_id()
    result = route_with_ollama(transcript, routes=routes, model=args.model)

    if result.get("error") and not result["parsed_output"]:
        print(f"Error: {result['error']}")
    elif result.get("error"):
        print(f"Warning: {result['error']}")

    if result["parsed_output"]:
        print("Parsed JSON output:")
        print(json.dumps(result["parsed_output"], indent=2))
    else:
        print("Raw output (could not parse as JSON):")
        print(result["raw_output"])

    print(f"\nProvider: {result['provider']}   Model: {result['model']}   Latency: {result['latency_seconds']}s")
    print(f"Route tree used: {result['route_tree_used']}")

    log_path = save_json_log(
        LLM_OUTPUTS_DIR,
        f"{run_id}_test_llm",
        {"run_id": run_id, "timestamp": now_iso(), **result},
    )
    print(f"Log saved to: {log_path}")

    append_csv_row(
        COMPONENT_RESULTS_CSV,
        {
            "run_id": run_id,
            "timestamp": now_iso(),
            "component": "llm",
            "provider": result["provider"],
            "input_file": args.file or "",
            "output_file": str(log_path),
            "latency_seconds": result["latency_seconds"],
            "error": result.get("error") or "",
        },
    )
    print(f"Result row appended to: {COMPONENT_RESULTS_CSV}")

    if result.get("error") and not result["parsed_output"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
