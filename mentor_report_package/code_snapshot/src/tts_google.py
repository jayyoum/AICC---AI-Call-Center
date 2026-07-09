"""
Google Cloud Text-to-Speech backend.

Authentication:
    Uses Application Default Credentials from gcloud. Run once per machine:
        gcloud init
        gcloud auth application-default login
"""

from pathlib import Path

from src.utils import Timer


def synthesize_google(
    text: str,
    output_path: Path,
    language_code: str = "en-US",
    voice_name: str = "en-US-Standard-C",
) -> dict:
    """Synthesize speech from text using Google Cloud Text-to-Speech, saved as MP3.

    Returns a dict with: output_path, provider, latency_seconds, text_length,
    error (None if successful).
    """
    output_path = Path(output_path)

    result = {
        "output_path": str(output_path),
        "provider": "google",
        "latency_seconds": 0.0,
        "text_length": len(text),
        "error": None,
    }

    if not text.strip():
        result["error"] = "No text provided to synthesize."
        return result

    try:
        from google.cloud import texttospeech
    except ImportError:
        result["error"] = (
            "google-cloud-texttospeech is not installed. Run: pip install -r requirements.txt"
        )
        return result

    try:
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        with Timer() as t:
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
        result["latency_seconds"] = t.elapsed_seconds

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.audio_content)

    except Exception as e:
        result["error"] = f"Google Text-to-Speech call failed: {e}"

    return result
