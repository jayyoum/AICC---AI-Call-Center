"""
Full voice pipeline: audio -> STT -> Ollama LLM routing -> TTS -> response audio.

This module contains the reusable orchestration logic shared by
run_pipeline.py (single audio file) and src/batch_runner.py (manifest of
many audio files x multiple provider combinations).
"""

from pathlib import Path

from src.llm_ollama import route_with_ollama
from src.stt_google import transcribe_google
from src.stt_local import transcribe_local
from src.tts_google import synthesize_google
from src.tts_local import synthesize_piper
from src.utils import (
    AUDIO_DIR,
    LLM_OUTPUTS_DIR,
    TRANSCRIPTS_DIR,
    new_run_id,
    now_iso,
    save_json_log,
    save_text,
)


def run_pipeline(
    audio_path: Path,
    stt_provider: str,
    tts_provider: str,
    llm_model: str = "llama3.2",
    routes: dict | None = None,
    stt_model_size: str = "base.en",
    tts_voice: str = "en_US-lessac-medium",
    extra_metadata: dict | None = None,
) -> dict:
    """Run one full pipeline pass and save transcript / LLM / audio artifacts.

    stt_provider: "google" or "local"
    tts_provider: "google" or "local"
    extra_metadata: optional dict merged into the result (e.g. scenario_id from a manifest row).

    Returns a flat dict suitable for writing straight to pipeline_results.csv.
    """
    audio_path = Path(audio_path)
    run_id = new_run_id()

    row = {
        "run_id": run_id,
        "timestamp": now_iso(),
        "audio_file": str(audio_path),
        "stt_provider": stt_provider,
        "stt_model": stt_model_size if stt_provider == "local" else "",
        "llm_provider": "ollama",
        "llm_model": llm_model,
        "tts_provider": tts_provider,
        "tts_voice": tts_voice if tts_provider == "local" else "",
        "transcript": "",
        "predicted_top_level_route": "",
        "predicted_final_route": "",
        "predicted_action": "",
        "response_to_customer": "",
        "confidence": "",
        "route_valid": "",
        "output_audio_path": "",
        "stt_latency_seconds": 0.0,
        "llm_latency_seconds": 0.0,
        "tts_latency_seconds": 0.0,
        "total_latency_seconds": 0.0,
        "error": "",
    }
    if extra_metadata:
        row.update(extra_metadata)

    errors = []

    # Step 1: Speech-to-Text
    if stt_provider == "google":
        stt_result = transcribe_google(audio_path)
    elif stt_provider == "local":
        stt_result = transcribe_local(audio_path, model_size=stt_model_size)
    else:
        row["error"] = f"Unknown stt_provider: {stt_provider}"
        return row

    row["stt_latency_seconds"] = stt_result["latency_seconds"]
    if stt_result.get("error"):
        errors.append(f"STT error: {stt_result['error']}")
    row["transcript"] = stt_result["transcript"]

    save_text(TRANSCRIPTS_DIR, f"{run_id}_transcript.txt", row["transcript"])

    if not row["transcript"]:
        row["error"] = "; ".join(errors) if errors else "No transcript produced; skipping LLM and TTS steps."
        row["total_latency_seconds"] = row["stt_latency_seconds"]
        return row

    # Step 2: LLM routing decision via Ollama
    llm_result = route_with_ollama(row["transcript"], routes=routes, model=llm_model)
    row["llm_latency_seconds"] = llm_result["latency_seconds"]
    if llm_result.get("error"):
        errors.append(f"LLM error: {llm_result['error']}")

    save_json_log(
        LLM_OUTPUTS_DIR,
        f"{run_id}_llm",
        {
            "raw_output": llm_result["raw_output"],
            "parsed_output": llm_result["parsed_output"],
            "route_tree_used": llm_result["route_tree_used"],
            "route_valid": llm_result["route_valid"],
        },
    )

    parsed = llm_result.get("parsed_output") or {}
    row["predicted_top_level_route"] = parsed.get("top_level_route", "")
    row["predicted_final_route"] = parsed.get("final_route", "")
    row["predicted_action"] = parsed.get("action", "")
    row["response_to_customer"] = parsed.get("response_to_customer", "")
    row["confidence"] = parsed.get("confidence", "")
    row["route_valid"] = llm_result["route_valid"]

    if not row["response_to_customer"]:
        row["error"] = "; ".join(errors) if errors else "LLM produced no response_to_customer; skipping TTS step."
        row["total_latency_seconds"] = row["stt_latency_seconds"] + row["llm_latency_seconds"]
        return row

    # Step 3: Text-to-Speech
    audio_extension = "mp3" if tts_provider == "google" else "wav"
    output_audio_path = AUDIO_DIR / f"{run_id}_response.{audio_extension}"

    if tts_provider == "google":
        tts_result = synthesize_google(row["response_to_customer"], output_audio_path)
    elif tts_provider == "local":
        tts_result = synthesize_piper(row["response_to_customer"], output_audio_path, voice=tts_voice)
    else:
        row["error"] = f"Unknown tts_provider: {tts_provider}"
        row["total_latency_seconds"] = row["stt_latency_seconds"] + row["llm_latency_seconds"]
        return row

    row["tts_latency_seconds"] = tts_result["latency_seconds"]
    if tts_result.get("error"):
        errors.append(f"TTS error: {tts_result['error']}")
    else:
        row["output_audio_path"] = tts_result["output_path"]

    row["total_latency_seconds"] = round(
        row["stt_latency_seconds"] + row["llm_latency_seconds"] + row["tts_latency_seconds"], 3
    )
    row["error"] = "; ".join(errors)

    return row
