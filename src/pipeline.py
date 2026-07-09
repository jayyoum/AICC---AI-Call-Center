"""
Full voice pipeline: audio -> STT -> Ollama LLM routing -> TTS -> response audio.

This module contains the reusable orchestration logic shared by
run_pipeline.py (single audio file) and src/batch_runner.py (manifest of
many audio files x multiple provider combinations).
"""

from pathlib import Path

from src.llm_ollama import route_with_ollama, validate_route
from src.pre_router import pre_route
from src.stt_google import transcribe_google
from src.stt_local import transcribe_local
from src.tts_google import synthesize_google
from src.tts_local import synthesize_piper
from src.utils import (
    AUDIO_DIR,
    LLM_OUTPUTS_DIR,
    TRANSCRIPTS_DIR,
    Timer,
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
    prompt_mode: str = "compact_v2",
    router_mode: str = "pre_router",
    extra_metadata: dict | None = None,
) -> dict:
    """Run one full pipeline pass and save transcript / LLM / audio artifacts.

    stt_provider: "google" or "local"
    tts_provider: "google" or "local"
    router_mode:
        "pre_router" (default): try the rule-based pre-router first, fall back
            to the Ollama compact_v2 router only when no rule matches.
        "llm_only": always use the Ollama router (skip the pre-router).
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
        "prompt_mode": prompt_mode,
        "router_mode": router_mode,
        "routing_source": "",
        "pre_router_handled": "",
        "rule_name": "",
        "tts_provider": tts_provider,
        "tts_voice": tts_voice if tts_provider == "local" else "",
        "transcript": "",
        "predicted_top_level_route": "",
        "predicted_final_route": "",
        "predicted_action": "",
        "response_to_customer": "",
        "confidence": "",
        "route_valid": "",
        "route_normalized": "",
        "original_final_route": "",
        "normalized_final_route": "",
        "output_audio_path": "",
        "stt_latency_seconds": 0.0,
        "llm_latency_seconds": 0.0,
        "route_latency_seconds": 0.0,
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

    # Step 2: Routing decision. In "pre_router" mode, try the rule-based
    # pre-router first and only call Ollama when no rule matches. In "llm_only"
    # mode, always call Ollama.
    pre_result = None
    pre_latency = 0.0
    if router_mode == "pre_router":
        with Timer() as pt:
            pre_result = pre_route(row["transcript"], routes=routes)
        pre_latency = pt.elapsed_seconds

    if pre_result and pre_result.get("handled"):
        # Handled entirely by rules - the Ollama LLM is skipped.
        row["routing_source"] = "pre_router"
        row["pre_router_handled"] = True
        row["rule_name"] = pre_result.get("rule_name", "")
        row["predicted_top_level_route"] = pre_result.get("top_level_route", "")
        row["predicted_final_route"] = pre_result.get("final_route", "")
        row["predicted_action"] = pre_result.get("action", "")
        row["response_to_customer"] = pre_result.get("response_to_customer", "")
        row["confidence"] = pre_result.get("confidence", "")
        row["route_valid"] = validate_route(pre_result, routes)
        # Pre-router returns exact routes, so nothing to normalize.
        row["route_normalized"] = False
        row["original_final_route"] = pre_result.get("final_route", "")
        row["normalized_final_route"] = pre_result.get("final_route", "")
        row["llm_latency_seconds"] = 0.0
        row["route_latency_seconds"] = pre_latency

        save_json_log(
            LLM_OUTPUTS_DIR,
            f"{run_id}_route",
            {
                "routing_source": "pre_router",
                "router_mode": router_mode,
                "pre_router_decision": pre_result,
                "route_valid": row["route_valid"],
            },
        )
    else:
        # llm_only mode, or the pre-router did not match: use the Ollama router.
        if router_mode == "pre_router":
            row["pre_router_handled"] = False
        llm_result = route_with_ollama(
            row["transcript"], routes=routes, model=llm_model, prompt_mode=prompt_mode
        )
        row["routing_source"] = "ollama"
        row["llm_latency_seconds"] = llm_result["latency_seconds"]
        # Routing time includes the (tiny) pre-router probe if it ran.
        row["route_latency_seconds"] = round(pre_latency + llm_result["latency_seconds"], 3)
        if llm_result.get("error"):
            errors.append(f"LLM error: {llm_result['error']}")

        save_json_log(
            LLM_OUTPUTS_DIR,
            f"{run_id}_llm",
            {
                "routing_source": "ollama",
                "router_mode": router_mode,
                "raw_output": llm_result["raw_output"],
                "parsed_output": llm_result["parsed_output"],
                "route_tree_used": llm_result["route_tree_used"],
                "route_valid": llm_result["route_valid"],
                "prompt_mode": llm_result.get("prompt_mode"),
            },
        )

        parsed = llm_result.get("parsed_output") or {}
        row["predicted_top_level_route"] = parsed.get("top_level_route", "")
        row["predicted_final_route"] = parsed.get("final_route", "")
        row["predicted_action"] = parsed.get("action", "")
        row["response_to_customer"] = parsed.get("response_to_customer", "")
        row["confidence"] = parsed.get("confidence", "")
        row["route_normalized"] = llm_result.get("route_normalized", "")
        row["original_final_route"] = llm_result.get("original_final_route", "")
        row["normalized_final_route"] = llm_result.get("normalized_final_route", "")
        row["route_valid"] = llm_result["route_valid"]

    if not row["response_to_customer"]:
        row["error"] = "; ".join(errors) if errors else "Routing produced no response_to_customer; skipping TTS step."
        row["total_latency_seconds"] = round(row["stt_latency_seconds"] + row["route_latency_seconds"], 3)
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
        row["total_latency_seconds"] = round(row["stt_latency_seconds"] + row["route_latency_seconds"], 3)
        return row

    row["tts_latency_seconds"] = tts_result["latency_seconds"]
    if tts_result.get("error"):
        errors.append(f"TTS error: {tts_result['error']}")
    else:
        row["output_audio_path"] = tts_result["output_path"]

    row["total_latency_seconds"] = round(
        row["stt_latency_seconds"] + row["route_latency_seconds"] + row["tts_latency_seconds"], 3
    )
    row["error"] = "; ".join(errors)

    return row
