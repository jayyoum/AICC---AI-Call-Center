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

import base64
import json
import shutil
import subprocess
import time
import traceback
import wave
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from src.llm_ollama import OLLAMA_CHAT_URL, route_with_ollama, validate_route
from src.pre_router import pre_route
from src.session_logger import (
    log_session_event,
    log_turn_event,
    new_session_id,
    new_turn_id,
    save_turn_summary,
)
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
STREAMING_LOGS_DIR = PROJECT_ROOT / "output" / "logs" / "streaming_stt"
SESSIONS_LOGS_DIR = PROJECT_ROOT / "output" / "logs" / "sessions"
ROUTES_PATH = PROJECT_ROOT / "config" / "routes.json"

FALLBACK_RESPONSE = (
    "I’m sorry, I had trouble generating a response. "
    "Would you like me to connect you to a human representative?"
)

app = Flask(__name__)

# Optional Socket.IO layer for experimental streaming STT. If flask-socketio is
# not installed the plain HTTP app still works fully - streaming is simply
# unavailable. This keeps all existing routes working regardless.
try:
    from flask_socketio import SocketIO

    from src.streaming_stt_google import GoogleStreamingSession
    from src.streaming_tts_google import (
        DEFAULT_STREAMING_VOICE,
        STREAMING_SAMPLE_RATE_HERTZ,
        stream_google_tts_text,
    )

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    STREAMING_AVAILABLE = True
except Exception as _streaming_import_error:  # pragma: no cover - env dependent
    socketio = None
    GoogleStreamingSession = None
    STREAMING_AVAILABLE = False
    print(f"[streaming] Socket.IO/streaming disabled: {_streaming_import_error}")


def ensure_web_dirs() -> None:
    for directory in (
        WEB_UPLOADS_DIR,
        WEB_UPLOADS_CONVERTED_DIR,
        WEB_AUDIO_DIR,
        WEB_TURN_LOGS_DIR,
        STREAMING_LOGS_DIR,
        SESSIONS_LOGS_DIR,
    ):
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


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate_hertz: int) -> None:
    """Wrap raw mono 16-bit PCM bytes into a .wav file (for inspection/fallback)."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate_hertz)
        wf.writeframes(pcm_bytes)


def route_text_only(transcript, router_mode, prompt_mode, llm_model, on_event=None):
    """Routing ONLY for an already-transcribed message (no STT, no TTS).

    Runs the pre-router (when router_mode="pre_router") and/or the Ollama LLM.
    Used by /api/route-text (routing only, e.g. for streaming TTS) and by
    route_and_synthesize() below, so the routing logic lives in one place.

    on_event: optional callback(event_type, data) used for session logging.
    Passing it keeps the logging out of the routing logic itself.

    Returns a dict of primitives on success, or {"error": ..., "status": ...}.
    """
    def emit(event_type, data=None):
        if on_event:
            on_event(event_type, data or {})

    emit("routing_started", {
        "router_mode": router_mode,
        "prompt_mode": prompt_mode,
        "llm_model": llm_model,
        "transcript": transcript,
    })

    routes = load_routes()

    # Routing: rule-based pre-router first (if enabled), else the Ollama LLM.
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
            message = llm_result.get("error") or "The LLM did not return a usable response."
            emit("routing_error", {"error": message})
            return {"error": f"LLM error: {message}", "status": 500}
        routing_source = "ollama"
        rule_name = ""
        route_valid = llm_result.get("route_valid")
        llm_raw_output = llm_result.get("raw_output", "")
        llm_latency = llm_result.get("latency_seconds", 0.0)
        route_latency = round(pre_latency + llm_latency, 3)

    response_to_customer = parsed_output.get("response_to_customer") or FALLBACK_RESPONSE

    emit("routing_finished", {
        "routing_source": routing_source,
        "rule_name": rule_name,
        "route_valid": route_valid,
        "top_level_route": parsed_output.get("top_level_route", ""),
        "final_route": parsed_output.get("final_route", ""),
        "action": parsed_output.get("action", ""),
        "response_to_customer": response_to_customer,
        "routing_latency_seconds": route_latency,
        "llm_latency_seconds": llm_latency,
    })

    return {
        "routing_source": routing_source,
        "rule_name": rule_name,
        "route_valid": route_valid,
        "parsed_output": parsed_output,
        "response_to_customer": response_to_customer,
        "route_latency": route_latency,
        "llm_latency": llm_latency,
        "llm_raw_output": llm_raw_output,
    }


def route_and_synthesize(transcript, turn_id, tts_provider, llm_model, prompt_mode, router_mode, on_event=None):
    """Shared routing + file-based TTS for an already-transcribed message.

    Calls route_text_only() and then synthesizes the reply to an audio file.
    Does NOT run STT. Used by /api/voice-turn (after its STT step) and
    /api/text-turn (which skips STT).

    Returns a dict of primitives on success, or {"error": ..., "status": ...}.
    """
    def emit(event_type, data=None):
        if on_event:
            on_event(event_type, data or {})

    result = route_text_only(transcript, router_mode, prompt_mode, llm_model, on_event=on_event)
    if result.get("error"):
        return result

    response_to_customer = result["response_to_customer"]

    # Text-to-Speech for the reply (file-based, stable path).
    audio_extension = "mp3" if tts_provider == "google" else "wav"
    output_audio_path = WEB_AUDIO_DIR / f"{turn_id}_response.{audio_extension}"

    emit("tts_started", {"tts_provider": tts_provider, "text_length": len(response_to_customer)})
    if tts_provider == "google":
        tts_result = synthesize_google(response_to_customer, output_audio_path)
    else:
        tts_result = synthesize_piper(response_to_customer, output_audio_path)
    if tts_result.get("error"):
        emit("tts_error", {"tts_provider": tts_provider, "error": tts_result["error"]})
        return {"error": f"TTS error: {tts_result['error']}", "status": 500}

    result["tts_latency"] = tts_result.get("latency_seconds", 0.0)
    result["audio_url"] = f"/generated_audio/{output_audio_path.name}"

    emit("tts_finished", {
        "tts_provider": tts_provider,
        "tts_latency_seconds": result["tts_latency"],
        "audio_url": result["audio_url"],
    })
    return result


@app.route("/")
@app.route("/dev")
def index():
    """Developer/debug dashboard (also available at /dev)."""
    return render_template("index.html")


@app.route("/customer")
def customer():
    """Mobile-first customer call screen (open this one from a phone)."""
    return render_template("customer.html")


@app.route("/api/voice-turn", methods=["POST"])
def voice_turn():
    ensure_web_dirs()
    # Session/turn ids tie every log line for this call together. The browser
    # sends session_id; if it doesn't, we start a new session.
    session_id = (request.form.get("session_id") or "").strip() or new_session_id()
    turn_id = new_turn_id()

    def log_event(event_type, data=None):
        log_turn_event(session_id, turn_id, event_type, data)

    try:
        uploaded_file = request.files.get("audio")
        if uploaded_file is None or uploaded_file.filename == "":
            log_event("voice_turn_error", {"error": "No audio file was uploaded."})
            return jsonify({"success": False, "error": "No audio file was uploaded.", "session_id": session_id, "turn_id": turn_id}), 400

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
        if input_mode not in ("manual", "continuous", "customer_continuous"):
            input_mode = "manual"
        if prompt_mode not in ("compact", "compact_v2", "full"):
            prompt_mode = "compact_v2"
        if router_mode not in ("llm_only", "pre_router"):
            router_mode = "pre_router"

        log_event("voice_turn_received", {
            "input_mode": input_mode,
            "stt_provider": stt_provider,
            "tts_provider": tts_provider,
            "router_mode": router_mode,
            "prompt_mode": prompt_mode,
            "llm_model": llm_model,
        })

        # Step 1: save the raw browser recording (usually webm/opus).
        upload_suffix = Path(uploaded_file.filename).suffix or ".webm"
        raw_path = WEB_UPLOADS_DIR / f"{turn_id}{upload_suffix}"
        uploaded_file.save(raw_path)
        log_event("audio_saved", {"filename": raw_path.name, "bytes": raw_path.stat().st_size})

        # Step 2: convert to 16kHz mono WAV, which both STT backends expect.
        wav_path = WEB_UPLOADS_CONVERTED_DIR / f"{turn_id}.wav"
        try:
            convert_to_wav(raw_path, wav_path)
        except RuntimeError as e:
            log_event("voice_turn_error", {"stage": "audio_convert", "error": str(e)})
            return jsonify({"success": False, "error": str(e), "session_id": session_id, "turn_id": turn_id}), 500
        log_event("audio_converted", {"filename": wav_path.name, "bytes": wav_path.stat().st_size})

        # Step 3: Speech-to-Text
        log_event("stt_started", {"stt_provider": stt_provider})
        if stt_provider == "google":
            stt_result = transcribe_google(wav_path)
        else:
            stt_result = transcribe_local(wav_path)

        transcript = stt_result.get("transcript", "")
        log_event("stt_finished", {
            "stt_provider": stt_provider,
            "stt_latency_seconds": stt_result.get("latency_seconds", 0.0),
            "transcript": transcript,
            "error": stt_result.get("error") or "",
        })
        if not transcript:
            error_message = stt_result.get("error") or "No speech was recognized in the recording."
            log_event("voice_turn_error", {"stage": "stt", "error": error_message})
            return jsonify({"success": False, "error": f"STT error: {error_message}", "session_id": session_id, "turn_id": turn_id}), 400

        # Steps 4 & 5: routing + TTS (shared with /api/text-turn).
        result = route_and_synthesize(
            transcript, turn_id, tts_provider, llm_model, prompt_mode, router_mode, on_event=log_event
        )
        if result.get("error"):
            log_event("voice_turn_error", {"stage": "routing_or_tts", "error": result["error"]})
            return jsonify({"success": False, "error": result["error"], "session_id": session_id, "turn_id": turn_id}), result.get("status", 500)

        routing_source = result["routing_source"]
        rule_name = result["rule_name"]
        route_valid = result["route_valid"]
        parsed_output = result["parsed_output"]
        response_to_customer = result["response_to_customer"]
        llm_latency = result["llm_latency"]
        route_latency = result["route_latency"]
        llm_raw_output = result["llm_raw_output"]
        audio_url = result["audio_url"]

        latency = {
            "stt": stt_result.get("latency_seconds", 0.0),
            "llm": llm_latency,
            "route": route_latency,
            "tts": result["tts_latency"],
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
            "session_id": session_id,
            "turn_id": turn_id,
            "input_mode": input_mode,
            "transcript": transcript,
            "llm_output": parsed_output,
            "response_to_customer": response_to_customer,
            "routing_source": routing_source,
            "rule_name": rule_name,
            "route_valid": route_valid,
            "prompt_mode": prompt_mode,
            "router_mode": router_mode,
            "audio_url": audio_url,
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

        log_event("voice_turn_completed", {"latency": latency, "audio_url": audio_url})
        save_turn_summary(session_id, turn_id, {
            "kind": "voice_turn",
            "input_mode": input_mode,
            "transcript": transcript,
            "response_to_customer": response_to_customer,
            "routing_source": routing_source,
            "rule_name": rule_name,
            "prompt_mode": prompt_mode,
            "router_mode": router_mode,
            "top_level_route": parsed_output.get("top_level_route", ""),
            "final_route": parsed_output.get("final_route", ""),
            "action": parsed_output.get("action", ""),
            "route_valid": route_valid,
            "latency": latency,
            "providers": providers,
            "audio_url": audio_url,
            "error": "",
        })

        return jsonify(response_payload)

    except Exception:
        # Never leak a stack trace to the browser - log it server-side instead.
        print(f"[voice-turn {turn_id}] Unexpected error:\n{traceback.format_exc()}")
        log_event("voice_turn_error", {"stage": "unexpected", "error": "unexpected server error"})
        return jsonify({"success": False, "error": "An unexpected server error occurred. Check the server logs.", "session_id": session_id, "turn_id": turn_id}), 500


@app.route("/api/text-turn", methods=["POST"])
def text_turn():
    """Route + TTS for an already-transcribed message (no STT re-run).

    Used by streaming STT mode: once Google returns a final transcript, the
    browser calls this instead of /api/voice-turn to avoid transcribing again.
    Accepts JSON or form fields.
    """
    ensure_web_dirs()
    data_early = request.get_json(silent=True) or request.form
    session_id = (data_early.get("session_id") or "").strip() or new_session_id()
    turn_id = new_turn_id()

    def log_event(event_type, data=None):
        log_turn_event(session_id, turn_id, event_type, data)

    try:
        data = data_early
        transcript = (data.get("transcript") or "").strip()
        if not transcript:
            log_event("text_turn_error", {"error": "No transcript was provided."})
            return jsonify({"success": False, "error": "No transcript was provided.", "session_id": session_id, "turn_id": turn_id}), 400

        tts_provider = data.get("tts_provider", "local")
        llm_model = data.get("llm_model") or "llama3.2"
        prompt_mode = data.get("prompt_mode", "compact_v2")
        router_mode = data.get("router_mode", "pre_router")
        input_mode = data.get("input_mode", "streaming_customer_text")

        if tts_provider not in ("google", "local"):
            tts_provider = "local"
        if prompt_mode not in ("compact", "compact_v2", "full"):
            prompt_mode = "compact_v2"
        if router_mode not in ("llm_only", "pre_router"):
            router_mode = "pre_router"

        log_event("text_turn_received", {
            "input_mode": input_mode,
            "tts_provider": tts_provider,
            "router_mode": router_mode,
            "prompt_mode": prompt_mode,
            "llm_model": llm_model,
            "transcript": transcript,
        })

        result = route_and_synthesize(
            transcript, turn_id, tts_provider, llm_model, prompt_mode, router_mode, on_event=log_event
        )
        if result.get("error"):
            log_event("text_turn_error", {"stage": "routing_or_tts", "error": result["error"]})
            return jsonify({"success": False, "error": result["error"], "session_id": session_id, "turn_id": turn_id}), result.get("status", 500)

        parsed_output = result["parsed_output"]
        latency = {
            "routing": result["route_latency"],
            "llm": result["llm_latency"],
            "tts": result["tts_latency"],
            "total": round(result["route_latency"] + result["tts_latency"], 3),
        }
        route = {
            "top_level_route": parsed_output.get("top_level_route", ""),
            "final_route": parsed_output.get("final_route", ""),
            "action": parsed_output.get("action", ""),
        }

        payload = {
            "success": True,
            "session_id": session_id,
            "turn_id": turn_id,
            "input_mode": input_mode,
            "transcript": transcript,
            "response_to_customer": result["response_to_customer"],
            "audio_url": result["audio_url"],
            "latency": latency,
            "routing_source": result["routing_source"],
            "rule_name": result["rule_name"],
            "router_mode": router_mode,
            "prompt_mode": prompt_mode,
            "route": route,
            "route_valid": result["route_valid"],
            "llm_output": parsed_output,
        }

        save_json_log(
            STREAMING_LOGS_DIR,
            f"{turn_id}_text_turn",
            {
                "turn_id": turn_id,
                "timestamp": now_iso(),
                "input_mode": input_mode,
                "prompt_mode": prompt_mode,
                "router_mode": router_mode,
                "routing_source": result["routing_source"],
                "rule_name": result["rule_name"],
                "transcript": transcript,
                "llm_raw_output": result["llm_raw_output"],
                "llm_parsed_output": parsed_output,
                "route_valid": result["route_valid"],
                "response_to_customer": result["response_to_customer"],
                "latency": latency,
            },
        )

        log_event("text_turn_completed", {"latency": latency, "audio_url": result["audio_url"]})
        save_turn_summary(session_id, turn_id, {
            "kind": "text_turn",
            "input_mode": input_mode,
            "transcript": transcript,
            "response_to_customer": result["response_to_customer"],
            "routing_source": result["routing_source"],
            "rule_name": result["rule_name"],
            "prompt_mode": prompt_mode,
            "router_mode": router_mode,
            "top_level_route": route["top_level_route"],
            "final_route": route["final_route"],
            "action": route["action"],
            "route_valid": result["route_valid"],
            "latency": latency,
            "audio_url": result["audio_url"],
            "error": "",
        })

        return jsonify(payload)

    except Exception:
        print(f"[text-turn {turn_id}] Unexpected error:\n{traceback.format_exc()}")
        log_event("text_turn_error", {"stage": "unexpected", "error": "unexpected server error"})
        return jsonify({"success": False, "error": "An unexpected server error occurred. Check the server logs.", "session_id": session_id, "turn_id": turn_id}), 500


@app.route("/api/route-text", methods=["POST"])
def route_text_endpoint():
    """Routing ONLY for an already-transcribed message (no STT, no TTS).

    Used by the experimental streaming-TTS flow: the browser gets the routed
    response_to_customer here, then streams the TTS audio separately over
    Socket.IO. /api/text-turn is unchanged and still does routing + file TTS.
    """
    ensure_web_dirs()
    data_early = request.get_json(silent=True) or request.form
    session_id = (data_early.get("session_id") or "").strip() or new_session_id()
    turn_id = new_turn_id()

    def log_event(event_type, data=None):
        log_turn_event(session_id, turn_id, event_type, data)

    try:
        data = data_early
        transcript = (data.get("transcript") or "").strip()
        if not transcript:
            log_event("route_text_error", {"error": "No transcript was provided."})
            return jsonify({"success": False, "error": "No transcript was provided.", "session_id": session_id, "turn_id": turn_id}), 400

        llm_model = data.get("llm_model") or "llama3.2"
        prompt_mode = data.get("prompt_mode", "compact_v2")
        router_mode = data.get("router_mode", "pre_router")
        input_mode = data.get("input_mode", "streaming_customer_route_only")

        if prompt_mode not in ("compact", "compact_v2", "full"):
            prompt_mode = "compact_v2"
        if router_mode not in ("llm_only", "pre_router"):
            router_mode = "pre_router"

        log_event("route_text_received", {
            "input_mode": input_mode,
            "router_mode": router_mode,
            "prompt_mode": prompt_mode,
            "llm_model": llm_model,
            "transcript": transcript,
        })

        result = route_text_only(transcript, router_mode, prompt_mode, llm_model, on_event=log_event)
        if result.get("error"):
            log_event("route_text_error", {"stage": "routing", "error": result["error"]})
            return jsonify({"success": False, "error": result["error"], "session_id": session_id, "turn_id": turn_id}), result.get("status", 500)

        parsed_output = result["parsed_output"]
        latency = {
            "routing": result["route_latency"],
            "llm": result["llm_latency"],
            "total": result["route_latency"],  # routing already includes the LLM time
        }
        route = {
            "top_level_route": parsed_output.get("top_level_route", ""),
            "final_route": parsed_output.get("final_route", ""),
            "action": parsed_output.get("action", ""),
        }

        payload = {
            "success": True,
            "session_id": session_id,
            "turn_id": turn_id,
            "input_mode": input_mode,
            "transcript": transcript,
            "response_to_customer": result["response_to_customer"],
            "routing_source": result["routing_source"],
            "rule_name": result["rule_name"],
            "router_mode": router_mode,
            "prompt_mode": prompt_mode,
            "route": route,
            "route_valid": result["route_valid"],
            "llm_output": parsed_output,
            "latency": latency,
        }

        save_json_log(
            STREAMING_LOGS_DIR,
            f"{turn_id}_route_text",
            {
                "turn_id": turn_id,
                "timestamp": now_iso(),
                "input_mode": input_mode,
                "prompt_mode": prompt_mode,
                "router_mode": router_mode,
                "routing_source": result["routing_source"],
                "rule_name": result["rule_name"],
                "transcript": transcript,
                "llm_raw_output": result["llm_raw_output"],
                "llm_parsed_output": parsed_output,
                "route_valid": result["route_valid"],
                "response_to_customer": result["response_to_customer"],
                "latency": latency,
            },
        )

        log_event("route_text_completed", {"latency": latency})
        save_turn_summary(session_id, turn_id, {
            "kind": "route_text",
            "input_mode": input_mode,
            "transcript": transcript,
            "response_to_customer": result["response_to_customer"],
            "routing_source": result["routing_source"],
            "rule_name": result["rule_name"],
            "prompt_mode": prompt_mode,
            "router_mode": router_mode,
            "top_level_route": route["top_level_route"],
            "final_route": route["final_route"],
            "action": route["action"],
            "route_valid": result["route_valid"],
            "latency": latency,
            "error": "",
        })

        return jsonify(payload)

    except Exception:
        print(f"[route-text {turn_id}] Unexpected error:\n{traceback.format_exc()}")
        log_event("route_text_error", {"stage": "unexpected", "error": "unexpected server error"})
        return jsonify({"success": False, "error": "An unexpected server error occurred. Check the server logs.", "session_id": session_id, "turn_id": turn_id}), 500


@app.route("/api/health")
def health():
    """Cheap health/readiness check. Makes no network calls by default.

    Pass ?check_ollama=true for a lightweight Ollama /api/tags probe.
    """
    ollama_base = OLLAMA_CHAT_URL.replace("/api/chat", "")

    try:
        from src.streaming_tts_google import streaming_tts_available
        tts_stream_ok = streaming_tts_available()
    except Exception:
        tts_stream_ok = False

    output_dirs = (WEB_AUDIO_DIR, WEB_TURN_LOGS_DIR, STREAMING_LOGS_DIR, SESSIONS_LOGS_DIR)

    payload = {
        "success": True,
        "server": "ok",
        "routes_loaded": load_routes() is not None,
        "ollama_url": ollama_base,
        "output_dirs_exist": all(d.exists() for d in output_dirs),
        "streaming_stt_available": STREAMING_AVAILABLE,
        "streaming_tts_available": bool(tts_stream_ok),
        "timestamp": now_iso(),
    }

    # Optional, explicitly requested: a quick check that Ollama is reachable.
    if request.args.get("check_ollama", "").lower() in ("1", "true", "yes"):
        try:
            import requests

            response = requests.get(f"{ollama_base}/api/tags", timeout=2)
            payload["ollama"] = "ok" if response.ok else f"http {response.status_code}"
        except Exception:
            payload["ollama"] = "unreachable"

    return jsonify(payload)


@app.route("/generated_audio/<path:filename>")
def generated_audio(filename):
    return send_from_directory(WEB_AUDIO_DIR, filename)


# ---------------------------------------------------------------------------
# Experimental streaming STT over Socket.IO (only if flask-socketio is present).
# One GoogleStreamingSession per connected client, keyed by Socket.IO sid.
# ---------------------------------------------------------------------------
_streaming_sessions = {}  # sid -> GoogleStreamingSession
_session_ids_by_sid = {}  # Socket.IO sid -> our logging session_id


def _cleanup_streaming_session(sid: str) -> None:
    session = _streaming_sessions.pop(sid, None)
    if session is not None:
        try:
            session.stop()
        except Exception:
            pass


if STREAMING_AVAILABLE:

    @socketio.on("stt_start")
    def on_stt_start(options=None):
        """Start a Google streaming-recognition session for this client."""
        sid = request.sid
        options = options or {}
        language_code = options.get("language_code", "en-US")
        try:
            sample_rate = int(options.get("sample_rate_hertz", 16000))
        except (TypeError, ValueError):
            sample_rate = 16000

        # Track the browser's session id so streaming events land in its logs.
        session_id = (options.get("session_id") or "").strip() or _session_ids_by_sid.get(sid) or new_session_id()
        _session_ids_by_sid[sid] = session_id
        stt_turn_id = new_turn_id()
        first_interim_seen = {"done": False}

        # Replace any previous session for this client.
        _cleanup_streaming_session(sid)

        def on_interim(text):
            if not first_interim_seen["done"]:
                first_interim_seen["done"] = True
                log_turn_event(session_id, stt_turn_id, "streaming_stt_first_interim", {"transcript": text})
            socketio.emit("stt_interim", {"transcript": text, "is_final": False}, to=sid)

        def on_final(text):
            log_turn_event(session_id, stt_turn_id, "streaming_stt_final", {"transcript": text})
            socketio.emit("stt_final", {"transcript": text, "is_final": True}, to=sid)

        def on_error(message):
            log_turn_event(session_id, stt_turn_id, "streaming_error", {"stage": "stt", "error": message})
            socketio.emit("stt_error", {"error": message}, to=sid)

        session = GoogleStreamingSession(
            language_code=language_code,
            sample_rate_hertz=sample_rate,
            on_interim=on_interim,
            on_final=on_final,
            on_error=on_error,
        )
        if not session.start():
            log_turn_event(session_id, stt_turn_id, "streaming_error",
                           {"stage": "stt_start", "error": "Could not start streaming STT."})
            socketio.emit("stt_error", {"error": "Could not start streaming STT on the server."}, to=sid)
            return

        _streaming_sessions[sid] = session
        log_turn_event(session_id, stt_turn_id, "streaming_stt_started",
                       {"language_code": language_code, "sample_rate_hertz": sample_rate})
        socketio.emit(
            "stt_started",
            {"language_code": language_code, "sample_rate_hertz": sample_rate},
            to=sid,
        )

    @socketio.on("audio_chunk")
    def on_audio_chunk(data):
        """Receive one binary PCM chunk and queue it for recognition."""
        session = _streaming_sessions.get(request.sid)
        if session is None or not data:
            return
        # Socket.IO delivers binary as bytes/bytearray; normalize to bytes.
        session.add_chunk(bytes(data) if not isinstance(data, bytes) else data)

    @socketio.on("stt_stop")
    def on_stt_stop():
        sid = request.sid
        session_id = _session_ids_by_sid.get(sid)
        if session_id:
            log_session_event(session_id, "streaming_stt_stopped", {})
        _cleanup_streaming_session(sid)
        socketio.emit("stt_stopped", {}, to=sid)

    @socketio.on("disconnect")
    def on_disconnect():
        sid = request.sid
        session_id = _session_ids_by_sid.pop(sid, None)
        if session_id:
            log_session_event(session_id, "streaming_client_disconnected", {})
        _cleanup_streaming_session(sid)

    # --- Experimental streaming TTS ---------------------------------------
    # The browser emits "tts_stream_start" with the routed response text; the
    # server streams Google TTS audio chunks back and the browser plays them.

    def _run_tts_stream(sid, text, language_code, voice_name, input_mode, session_id=None):
        """Background task: stream Google TTS audio chunks to one client."""
        ensure_web_dirs()
        turn_id = new_turn_id()

        def log_event(event_type, data=None):
            if session_id:
                log_turn_event(session_id, turn_id, event_type, data)

        start = time.perf_counter()
        first_chunk_at = None
        chunk_count = 0
        total_bytes = 0
        pcm_parts = []

        try:
            socketio.emit(
                "tts_stream_started",
                {
                    "voice_name": voice_name,
                    "language_code": language_code,
                    "mime_type": "audio/wav",  # browser wraps the raw PCM into WAV
                    "sample_rate_hertz": STREAMING_SAMPLE_RATE_HERTZ,
                },
                to=sid,
            )
            log_event("streaming_tts_started", {
                "voice_name": voice_name,
                "language_code": language_code,
                "text_length": len(text),
            })

            for chunk in stream_google_tts_text(text, language_code=language_code, voice_name=voice_name):
                if not chunk:
                    continue
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    log_event("streaming_tts_first_chunk", {
                        "time_to_first_audio_chunk_seconds": round(first_chunk_at - start, 3),
                    })
                chunk_count += 1
                total_bytes += len(chunk)
                pcm_parts.append(chunk)
                # Send raw PCM as base64 (robust across Socket.IO transports).
                socketio.emit(
                    "tts_audio_chunk",
                    {"chunk_base64": base64.b64encode(chunk).decode("ascii"), "mime_type": "audio/l16"},
                    to=sid,
                )
                socketio.sleep(0)  # yield so the emit is flushed promptly

            # Save the full streamed audio to a .wav for inspection / fallback.
            saved_audio_url = ""
            if pcm_parts:
                wav_path = WEB_AUDIO_DIR / f"{turn_id}_tts_stream.wav"
                _write_wav(wav_path, b"".join(pcm_parts), STREAMING_SAMPLE_RATE_HERTZ)
                saved_audio_url = f"/generated_audio/{wav_path.name}"

            total_latency = round(time.perf_counter() - start, 3)
            time_to_first = round(first_chunk_at - start, 3) if first_chunk_at else None

            socketio.emit(
                "tts_stream_done",
                {
                    "chunk_count": chunk_count,
                    "total_audio_bytes": total_bytes,
                    "latency_seconds": total_latency,
                    "time_to_first_audio_chunk_seconds": time_to_first,
                    "sample_rate_hertz": STREAMING_SAMPLE_RATE_HERTZ,
                    "saved_audio_url": saved_audio_url,
                },
                to=sid,
            )

            save_json_log(
                STREAMING_LOGS_DIR,
                f"{turn_id}_tts_stream",
                {
                    "turn_id": turn_id,
                    "timestamp": now_iso(),
                    "input_mode": input_mode,
                    "voice_name": voice_name,
                    "language_code": language_code,
                    "chunk_count": chunk_count,
                    "total_audio_bytes": total_bytes,
                    "time_to_first_audio_chunk_seconds": time_to_first,
                    "total_tts_stream_seconds": total_latency,
                    "saved_audio_url": saved_audio_url,
                    "playback_mode": "streaming_buffered",
                },
            )

            log_event("streaming_tts_done", {
                "chunk_count": chunk_count,
                "total_audio_bytes": total_bytes,
                "time_to_first_audio_chunk_seconds": time_to_first,
                "total_tts_stream_seconds": total_latency,
                "saved_audio_url": saved_audio_url,
                "playback_mode": "streaming_buffered",
            })
            if session_id:
                save_turn_summary(session_id, turn_id, {
                    "kind": "streaming_tts",
                    "input_mode": input_mode,
                    "response_to_customer": text,
                    "voice_name": voice_name,
                    "chunk_count": chunk_count,
                    "total_audio_bytes": total_bytes,
                    "latency": {
                        "time_to_first_audio_chunk": time_to_first,
                        "total_tts_stream": total_latency,
                    },
                    "audio_url": saved_audio_url,
                    "playback_mode": "streaming_buffered",
                    "error": "",
                })

        except Exception as e:
            # Readable message only - never a raw traceback to the client.
            print(f"[tts-stream {turn_id}] error: {e}")
            log_event("streaming_error", {"stage": "tts", "error": str(e)})
            socketio.emit("tts_stream_error", {"error": f"Streaming TTS failed: {e}"}, to=sid)

    @socketio.on("tts_stream_start")
    def on_tts_stream_start(options=None):
        sid = request.sid
        options = options or {}
        text = (options.get("text") or "").strip()
        language_code = options.get("language_code", "en-US")
        voice_name = options.get("voice_name") or DEFAULT_STREAMING_VOICE
        input_mode = options.get("input_mode", "streaming_customer_tts")

        session_id = (options.get("session_id") or "").strip() or _session_ids_by_sid.get(sid)
        if session_id:
            _session_ids_by_sid[sid] = session_id

        if not text:
            socketio.emit("tts_stream_error", {"error": "No text provided for streaming TTS."}, to=sid)
            return

        # Run in a background task so the Socket.IO server is not blocked.
        socketio.start_background_task(
            _run_tts_stream, sid, text, language_code, voice_name, input_mode, session_id
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the AICC Flask voice server.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind. Use 0.0.0.0 to allow your phone / other LAN devices to connect.",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000).")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode (auto-reload + in-browser debugger). Leave off when binding 0.0.0.0.",
    )
    args = parser.parse_args()

    ensure_web_dirs()
    if STREAMING_AVAILABLE:
        # socketio.run serves both the HTTP routes and the WebSocket layer.
        # allow_unsafe_werkzeug lets the simple dev server run with Socket.IO.
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            allow_unsafe_werkzeug=True,
        )
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)
