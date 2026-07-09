#!/usr/bin/env python3
"""
Test a single Speech-to-Text backend on one audio file.

Usage:
    python test_stt.py --provider google --audio input_audio/example.wav
    python test_stt.py --provider local --audio input_audio/example.wav --model-size base.en
"""

import argparse
import sys
from pathlib import Path

from src.stt_google import transcribe_google
from src.stt_local import transcribe_local
from src.utils import (
    COMPONENT_RESULTS_CSV,
    LOGS_DIR,
    TRANSCRIPTS_DIR,
    append_csv_row,
    ensure_output_dirs,
    new_run_id,
    now_iso,
    save_json_log,
    save_text,
)


def main():
    parser = argparse.ArgumentParser(description="Test a single Speech-to-Text backend.")
    parser.add_argument("--provider", required=True, choices=["google", "local"], help="STT backend to use.")
    parser.add_argument("--audio", required=True, help="Path to the input audio file.")
    parser.add_argument("--language", default="en-US", help="Language code for Google STT (default: en-US).")
    parser.add_argument("--model-size", default="base.en", help="faster-whisper model size for local STT (default: base.en).")
    args = parser.parse_args()

    ensure_output_dirs()
    audio_path = Path(args.audio)
    run_id = new_run_id()

    if args.provider == "google":
        result = transcribe_google(audio_path, language_code=args.language)
    else:
        result = transcribe_local(audio_path, model_size=args.model_size, language="en")

    if result.get("error") and not result["transcript"]:
        print(f"Error: {result['error']}")
        sys.exit(1)
    elif result.get("error"):
        print(f"Warning: {result['error']}")

    print("Transcript:")
    print(result["transcript"] if result["transcript"] else "(no speech recognized)")
    print(f"\nProvider: {result['provider']}   Latency: {result['latency_seconds']}s")

    transcript_path = save_text(TRANSCRIPTS_DIR, f"{run_id}_{audio_path.stem}.txt", result["transcript"])
    print(f"Transcript saved to: {transcript_path}")

    log_path = save_json_log(LOGS_DIR, f"{run_id}_stt_{args.provider}", {"run_id": run_id, "timestamp": now_iso(), **result})
    print(f"Log saved to: {log_path}")

    append_csv_row(
        COMPONENT_RESULTS_CSV,
        {
            "run_id": run_id,
            "timestamp": now_iso(),
            "component": "stt",
            "provider": result["provider"],
            "input_file": str(audio_path),
            "output_file": str(transcript_path),
            "latency_seconds": result["latency_seconds"],
            "error": result.get("error") or "",
        },
    )
    print(f"Result row appended to: {COMPONENT_RESULTS_CSV}")


if __name__ == "__main__":
    main()
