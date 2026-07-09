"""
Local Text-to-Speech backend using Piper (via `python -m piper`).

Runs entirely on-device, no network calls or credentials required.

Voice models are NOT downloaded automatically. If the model files are
missing, this module returns a clear error telling you how to get one
yourself (see README.md for the piper voice download command).
"""

import subprocess
import sys
from pathlib import Path

from src.utils import Timer


def synthesize_piper(
    text: str,
    output_path: Path,
    voice: str = "en_US-lessac-medium",
    data_dir: Path = Path("models/piper"),
) -> dict:
    """Synthesize speech from text using local Piper TTS, saved as WAV.

    Returns a dict with: output_path, provider, voice, latency_seconds,
    text_length, error (None if successful).
    """
    output_path = Path(output_path)
    data_dir = Path(data_dir)

    result = {
        "output_path": str(output_path),
        "provider": "piper",
        "voice": voice,
        "latency_seconds": 0.0,
        "text_length": len(text),
        "error": None,
    }

    if not text.strip():
        result["error"] = "No text provided to synthesize."
        return result

    model_path = data_dir / f"{voice}.onnx"
    config_path = data_dir / f"{voice}.onnx.json"

    if not model_path.exists() or not config_path.exists():
        result["error"] = (
            f"Piper voice model not found for '{voice}' in {data_dir}. "
            f"Expected files: {model_path.name} and {config_path.name}. "
            "Download a voice manually (see README.md) before running this again."
        )
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "piper",
        "-m",
        str(model_path),
        "-c",
        str(config_path),
        "-f",
        str(output_path),
    ]

    try:
        with Timer() as t:
            completed = subprocess.run(
                command,
                input=text,
                text=True,
                capture_output=True,
            )
        result["latency_seconds"] = t.elapsed_seconds

        if completed.returncode != 0:
            result["error"] = f"Piper failed (exit code {completed.returncode}): {completed.stderr.strip()}"
        elif not output_path.exists():
            result["error"] = "Piper reported success but no output file was created."

    except FileNotFoundError:
        result["error"] = "Could not run 'python -m piper'. Is piper-tts installed in this environment?"
    except Exception as e:
        result["error"] = f"Piper synthesis failed: {e}"

    return result
