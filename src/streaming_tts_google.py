"""
Experimental streaming Text-to-Speech using Google Cloud streaming synthesis.

Google's bidirectional streaming TTS is a preview-style feature and works with
Chirp 3 HD voices (e.g. "en-US-Chirp3-HD-Charon"). This module yields raw
LINEAR16 PCM audio chunks as they are produced, so the caller can forward them
to the browser over Socket.IO for early playback.

Authentication uses Application Default Credentials (gcloud auth
application-default login) - no API keys or service-account files.

This is a prototype and an OPTIONAL path: file-based TTS (src/tts_google.py and
src/tts_local.py) remains the stable default.
"""

# Chirp 3 HD streaming voices produce 24kHz audio; we request LINEAR16 at this
# rate so the browser can wrap the raw PCM into a WAV container for playback.
DEFAULT_STREAMING_VOICE = "en-US-Chirp3-HD-Charon"
STREAMING_SAMPLE_RATE_HERTZ = 24000


def streaming_tts_available() -> bool:
    """True if the installed google-cloud-texttospeech exposes streaming synth."""
    try:
        from google.cloud import texttospeech as tts

        return hasattr(tts.TextToSpeechClient, "streaming_synthesize") and hasattr(
            tts, "StreamingSynthesizeConfig"
        )
    except Exception:
        return False


def stream_google_tts_text(
    text: str,
    language_code: str = "en-US",
    voice_name: str = DEFAULT_STREAMING_VOICE,
    sample_rate_hertz: int = STREAMING_SAMPLE_RATE_HERTZ,
):
    """Yield raw LINEAR16 PCM audio chunks (bytes) from Google streaming TTS.

    Sends exactly one initial StreamingSynthesizeRequest carrying the config,
    then one request with the text to synthesize, and yields each response's
    audio_content as it arrives.

    Raises RuntimeError with a readable message if streaming TTS is not
    available in the installed library (the caller should fall back to
    file-based TTS). Other Google errors propagate as their own exceptions and
    should be caught by the caller and reported as a readable message - never a
    raw traceback.
    """
    try:
        from google.cloud import texttospeech as tts
    except ImportError:
        raise RuntimeError(
            "google-cloud-texttospeech is not installed. Run: pip install -r requirements.txt"
        )

    if not streaming_tts_available():
        raise RuntimeError(
            "Google streaming TTS is not available in the installed "
            "google-cloud-texttospeech version. Try upgrading "
            "google-cloud-texttospeech or use file-based TTS."
        )

    if not text.strip():
        raise RuntimeError("No text provided for streaming TTS.")

    client = tts.TextToSpeechClient()

    streaming_config = tts.StreamingSynthesizeConfig(
        voice=tts.VoiceSelectionParams(language_code=language_code, name=voice_name),
        streaming_audio_config=tts.StreamingAudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hertz,
        ),
    )

    def request_generator():
        # 1) First request: streaming config only (no audio input yet).
        yield tts.StreamingSynthesizeRequest(streaming_config=streaming_config)
        # 2) Then the text to synthesize.
        yield tts.StreamingSynthesizeRequest(input=tts.StreamingSynthesisInput(text=text))

    responses = client.streaming_synthesize(request_generator())
    for response in responses:
        if response.audio_content:
            yield response.audio_content
