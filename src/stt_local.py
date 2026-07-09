"""
Local Speech-to-Text backend using faster-whisper.

Runs entirely on-device, no network calls or credentials required.
Defaults to CPU with int8 compute for broad compatibility; pass
device="cuda" if you have a supported GPU set up.
"""

from pathlib import Path

from src.utils import Timer

# Cache loaded models by (model_size, device, compute_type) so repeated calls
# in the same process (e.g. batch runs) don't reload the model from disk each time.
_MODEL_CACHE: dict = {}


def _get_model(model_size: str, device: str, compute_type: str):
    key = (model_size, device, compute_type)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _MODEL_CACHE[key]


def transcribe_local(
    audio_path: Path,
    model_size: str = "base.en",
    language: str = "en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> dict:
    """Transcribe an audio file using a local faster-whisper model.

    Returns a dict with: transcript, provider, model_size, latency_seconds,
    segments (list of {start, end, text}), error (None if successful).
    """
    audio_path = Path(audio_path)

    result = {
        "transcript": "",
        "provider": "faster-whisper",
        "model_size": model_size,
        "latency_seconds": 0.0,
        "segments": [],
        "error": None,
    }

    if not audio_path.exists():
        result["error"] = f"Audio file not found: {audio_path}"
        return result

    try:
        from faster_whisper import WhisperModel  # noqa: F401  (import check only)
    except ImportError:
        result["error"] = "faster-whisper is not installed. Run: pip install -r requirements.txt"
        return result

    try:
        model = _get_model(model_size, device, compute_type)

        with Timer() as t:
            segments_iter, info = model.transcribe(str(audio_path), language=language)
            segments = []
            transcript_parts = []
            for segment in segments_iter:
                segments.append(
                    {
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                        "text": segment.text.strip(),
                    }
                )
                transcript_parts.append(segment.text.strip())
        result["latency_seconds"] = t.elapsed_seconds
        result["segments"] = segments
        result["transcript"] = " ".join(transcript_parts).strip()

        if not result["transcript"]:
            result["error"] = None  # not an error, just no speech detected

    except Exception as e:
        result["error"] = f"faster-whisper transcription failed: {e}"

    return result
