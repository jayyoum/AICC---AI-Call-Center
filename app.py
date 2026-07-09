"""
Flask web prototype for the AICC voice routing system.

Turn-based flow (not real-time streaming): the browser records one turn of
audio, sends it to /api/voice-turn, and this file runs it through
STT -> Ollama routing -> TTS and returns the result as JSON so the page can
display the transcript/route/response and play the response audio.

This app is additive: it does not modify or depend on the existing CLI
scripts (test_stt.py, test_tts.py, test_llm.py, run_pipeline.py,
run_batch.py), which keep working exactly as before.
"""

import json
import shutil
import subprocess
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from src.llm_ollama import route_with_ollama, validate_route
from src.pre_router import pre_route
from src.stt_google import transcribe_google
from src.stt_local import transcribe_local
from src.tts_google import synthesize_google
from src.tts_local import synthesize_piper
from src.utils import Timer, now_iso, save_json_log

PROJECT_ROOT = Path(__file__).resolve().parent

WEB_UPLOADS_DIR = PROJECT_ROOT / "web_uploads"
WEB_UPLOADS_CONVERTED_DIR = WEB_UPLOADS_DIR / "converted"
WEB_AUDIO_DIR = PROJECT_ROOT / "output" / "web_audio"
WEB_TURN_LOGS_DIR = PROJECT_ROOT / "output" / "logs" / "web_turns"
ROUTES_PATH = PROJECT_ROOT / "config" / "routes.json"

FALLBACK_RESPONSE = (
    "I’m sorry, I had trouble generating a response. "
    "Would you like me to connect you to a human representative?"
)

app = Flask(__name__)


def ensure_web_dirs() -> None:
    for directory in (WEB_UPLOADS_DIR, WEB_UPLOADS_CONVERTED_DIR, WEB_AUDIO_DIR, WEB_TURN_LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_routes() -> dict | None:
    """Load config/routes.json if it exists and is valid JSON, else None."""
    if not ROUTES_PATH.exists():
        return None
    try:
        return json.loads(ROUTES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert a browser-recorded audio file to mono 16kHz LINEAR16 WAV using ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for browser audio conversion. Install it with: brew install ffmpeg"
        )

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        # Keep only the tail of stderr - ffmpeg logs can be long.
        detail = completed.stderr.strip()[-500:]
        raise RuntimeError(f"ffmpeg failed to convert the recorded audio: {detail}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/voice-turn", methods=["POST"])
def voice_turn():
    ensure_web_dirs()
    turn_id = uuid.uuid4().hex[:8]

    try:
        uploaded_file = request.files.get("audio")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"success": False, "error": "No audio file was uploaded."}), 400

        stt_provider = request.form.get("stt_provider", "google")
        tts_provider = request.form.get("tts_provider", "google")
        llm_model = request.form.get("llm_model") or "llama3.2"
        input_mode = request.form.get("input_mode", "manual")
        prompt_mode = request.form.get("prompt_mode", "compact_v2")
        router_mode = request.form.get("router_mode", "pre_router")

        if stt_provider not in ("google", "local"):
            return jsonify({"success": False, "error": f"Unknown stt_provider: {stt_provider}"}), 400
        if tts_provider not in ("google", "local"):
            return jsonify({"success": False, "error": f"Unknown tts_provider: {tts_provider}"}), 400
        if input_mode not in ("manual", "continuous"):
            input_mode = "manual"
        if prompt_mode not in ("compact", "compact_v2", "full"):
            prompt_mode = "compact_v2"
        if router_mode not in ("llm_only", "pre_router"):
            router_mode = "pre_router"

        # Step 1: save the raw browser recording (usually webm/opus).
        upload_suffix = Path(uploaded_file.filename).suffix or ".webm"
        raw_path = WEB_UPLOADS_DIR / f"{turn_id}{upload_suffix}"
        uploaded_file.save(raw_path)

        # Step 2: convert to 16kHz mono WAV, which both STT backends expect.
        wav_path = WEB_UPLOADS_CONVERTED_DIR / f"{turn_id}.wav"
        try:
            convert_to_wav(raw_path, wav_path)
        except RuntimeError as e:
            return jsonify({"success": False, "error": str(e)}), 500

        # Step 3: Speech-to-Text
        if stt_provider == "google":
            stt_result = transcribe_google(wav_path)
        else:
            stt_result = transcribe_local(wav_path)

        transcript = stt_result.get("transcript", "")
        if not transcript:
            error_message = stt_result.get("error") or "No speech was recognized in the recording."
            return jsonify({"success": False, "error": f"STT error: {error_message}"}), 400

        # Step 4: Routing decision. In "pre_router" mode, try the rule-based
        # pre-router first and only call Ollama when no rule matches.
        routes = load_routes()
        pre_result = None
        pre_latency = 0.0
        if router_mode == "pre_router":
            with Timer() as pt:
                pre_result = pre_route(transcript, routes=routes)
            pre_latency = pt.elapsed_seconds

        llm_raw_output = ""
        if pre_result and pre_result.get("handled"):
            routing_source = "pre_router"
            rule_name = pre_result.get("rule_name", "")
            route_valid = validate_route(pre_result, routes)
            parsed_output = {
                "top_level_route": pre_result.get("top_level_route", ""),
                "final_route": pre_result.get("final_route", ""),
                "action": pre_result.get("action", ""),
                "response_type": pre_result.get("response_type", ""),
                "response_to_customer": pre_result.get("response_to_customer", ""),
                "confidence": pre_result.get("confidence", ""),
            }
            llm_latency = 0.0
            route_latency = round(pre_latency, 3)
        else:
            llm_result = route_with_ollama(transcript, routes=routes, model=llm_model, prompt_mode=prompt_mode)
            parsed_output = llm_result.get("parsed_output")
            if parsed_output is None:
                error_message = llm_result.get("error") or "The LLM did not return a usable response."
                return jsonify({"success": False, "error": f"LLM error: {error_message}"}), 500
            routing_source = "ollama"
            rule_name = ""
            route_valid = llm_result.get("route_valid")
            llm_raw_output = llm_result.get("raw_output", "")
            llm_latency = llm_result.get("latency_seconds", 0.0)
            route_latency = round(pre_latency + llm_latency, 3)

        response_to_customer = parsed_output.get("response_to_customer") or FALLBACK_RESPONSE

        # Step 5: Text-to-Speech for the response.
        audio_extension = "mp3" if tts_provider == "google" else "wav"
        output_audio_path = WEB_AUDIO_DIR / f"{turn_id}_response.{audio_extension}"

        if tts_provider == "google":
            tts_result = synthesize_google(response_to_customer, output_audio_path)
        else:
            tts_result = synthesize_piper(response_to_customer, output_audio_path)

        if tts_result.get("error"):
            return jsonify({"success": False, "error": f"TTS error: {tts_result['error']}"}), 500

        latency = {
            "stt": stt_result.get("latency_seconds", 0.0),
            "llm": llm_latency,
            "route": route_latency,
            "tts": tts_result.get("latency_seconds", 0.0),
        }
        latency["total"] = round(latency["stt"] + latency["route"] + latency["tts"], 3)

        providers = {
            "stt": stt_provider,
            "llm": "ollama",
            "llm_model": llm_model,
            "prompt_mode": prompt_mode,
            "router_mode": router_mode,
            "routing_source": routing_source,
            "rule_name": rule_name,
            "tts": tts_provider,
        }

        response_payload = {
            "success": True,
            "input_mode": input_mode,
            "transcript": transcript,
            "llm_output": parsed_output,
            "response_to_customer": response_to_customer,
            "routing_source": routing_source,
            "rule_name": rule_name,
            "route_valid": route_valid,
            "audio_url": f"/generated_audio/{output_audio_path.name}",
            "latency": latency,
            "providers": providers,
        }

        save_json_log(
            WEB_TURN_LOGS_DIR,
            f"{turn_id}_turn",
            {
                "turn_id": turn_id,
                "timestamp": now_iso(),
                "input_mode": input_mode,
                "prompt_mode": prompt_mode,
                "router_mode": router_mode,
                "routing_source": routing_source,
                "rule_name": rule_name,
                "transcript": transcript,
                "llm_raw_output": llm_raw_output,
                "llm_parsed_output": parsed_output,
                "route_valid": route_valid,
                "response_to_customer": response_to_customer,
                "latency": latency,
                "providers": providers,
            },
        )

        return jsonify(response_payload)

    except Exception:
        # Never leak a stack trace to the browser - log it server-side instead.
        print(f"[voice-turn {turn_id}] Unexpected error:\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": "An unexpected server error occurred. Check the server logs."}), 500


@app.route("/generated_audio/<path:filename>")
def generated_audio(filename):
    return send_from_directory(WEB_AUDIO_DIR, filename)


if __name__ == "__main__":
    ensure_web_dirs()
    app.run(debug=True)
