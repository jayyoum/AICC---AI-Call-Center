"""
Google Cloud Speech-to-Text backend.

Authentication:
    Uses Application Default Credentials from gcloud. Run once per machine:
        gcloud init
        gcloud auth application-default login

Notes:
    - First version supports WAV / LINEAR16 audio under 60 seconds
      (uses the synchronous `recognize` API, not `long_running_recognize`).
"""

from pathlib import Path

from src.utils import Timer


def transcribe_google(
    audio_path: Path, language_code: str = "en-US", sample_rate_hertz: int = 16000
) -> dict:
    """Transcribe an audio file using Google Cloud Speech-to-Text.

    Assumes the audio is WAV, LINEAR16, mono, at `sample_rate_hertz` (default 16000 Hz).

    Returns a dict with: transcript, provider, language_code, latency_seconds,
    raw_response_summary, error (None if successful).
    """
    audio_path = Path(audio_path)

    result = {
        "transcript": "",
        "provider": "google",
        "language_code": language_code,
        "latency_seconds": 0.0,
        "raw_response_summary": "",
        "error": None,
    }

    if not audio_path.exists():
        result["error"] = f"Audio file not found: {audio_path}"
        return result

    try:
        from google.cloud import speech
    except ImportError:
        result["error"] = (
            "google-cloud-speech is not installed. Run: pip install -r requirements.txt"
        )
        return result

    try:
        client = speech.SpeechClient()
        audio_bytes = audio_path.read_bytes()
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hertz,
            language_code=language_code,
            audio_channel_count=1,
            enable_automatic_punctuation=True,
        )

        with Timer() as t:
            response = client.recognize(config=config, audio=audio)
        result["latency_seconds"] = t.elapsed_seconds

        if not response.results:
            result["raw_response_summary"] = "No speech detected in audio."
            return result

        transcript_parts = [r.alternatives[0].transcript for r in response.results]
        result["transcript"] = " ".join(transcript_parts).strip()
        result["raw_response_summary"] = f"{len(response.results)} result segment(s) returned."

    except Exception as e:
        result["error"] = f"Google Speech-to-Text call failed: {e}"

    return result
