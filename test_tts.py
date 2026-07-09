#!/usr/bin/env python3
"""
Test a single Text-to-Speech backend on user-provided text.

Usage:
    python test_tts.py --provider google --text "some text" --output response.mp3
    python test_tts.py --provider google --file input_text/response.txt --output response.mp3
    python test_tts.py --provider local --file input_text/response.txt --output response.wav
"""

import argparse
import sys
from pathlib import Path

from src.tts_google import synthesize_google
from src.tts_local import synthesize_piper
from src.utils import (
    AUDIO_DIR,
    COMPONENT_RESULTS_CSV,
    append_csv_row,
    ensure_output_dirs,
    get_text_from_args,
    new_run_id,
    now_iso,
)


def main():
    parser = argparse.ArgumentParser(description="Test a single Text-to-Speech backend.")
    parser.add_argument("--provider", required=True, choices=["google", "local"], help="TTS backend to use.")
    parser.add_argument("--text", help="Text to synthesize directly.")
    parser.add_argument("--file", help="Path to a .txt file containing the text to synthesize.")
    parser.add_argument("--output", required=True, help="Output filename, saved under output/audio/.")
    parser.add_argument("--voice", default="en_US-lessac-medium", help="Piper voice name for local TTS.")
    args = parser.parse_args()

    ensure_output_dirs()

    try:
        text = get_text_from_args(args.text, args.file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    run_id = new_run_id()
    output_path = AUDIO_DIR / args.output

    if args.provider == "google":
        result = synthesize_google(text, output_path)
    else:
        result = synthesize_piper(text, output_path, voice=args.voice)

    if result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"Audio saved to: {result['output_path']}")
    print(f"Provider: {result['provider']}   Latency: {result['latency_seconds']}s   Text length: {result['text_length']}")

    append_csv_row(
        COMPONENT_RESULTS_CSV,
        {
            "run_id": run_id,
            "timestamp": now_iso(),
            "component": "tts",
            "provider": result["provider"],
            "input_file": args.file or "",
            "output_file": result["output_path"],
            "latency_seconds": result["latency_seconds"],
            "error": result.get("error") or "",
        },
    )
    print(f"Result row appended to: {COMPONENT_RESULTS_CSV}")


if __name__ == "__main__":
    main()
