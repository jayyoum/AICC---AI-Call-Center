"""
Shared helper functions used across the STT/TTS/LLM test scripts.

Keeps the individual test scripts small by centralizing:
- project path resolution
- output directory creation
- CSV row appending (with automatic header creation)
- JSON log saving
- timestamp / run id helpers
- reading text from either --text or --file command-line args
"""

import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Project root is the parent of the src/ directory this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output"
AUDIO_DIR = OUTPUT_DIR / "audio"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
LLM_OUTPUTS_DIR = OUTPUT_DIR / "llm_outputs"
LOGS_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

COMPONENT_RESULTS_CSV = RESULTS_DIR / "component_results.csv"
PIPELINE_RESULTS_CSV = RESULTS_DIR / "pipeline_results.csv"


def ensure_output_dirs() -> None:
    """Create all output subdirectories if they don't already exist."""
    for directory in (AUDIO_DIR, TRANSCRIPTS_DIR, LLM_OUTPUTS_DIR, LOGS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """Current UTC timestamp as an ISO 8601 string, safe for filenames and CSV cells."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_for_filename() -> str:
    """Current UTC timestamp formatted for use inside a filename."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def new_run_id() -> str:
    """Short unique id to tie together files/logs/rows produced by one run."""
    return uuid.uuid4().hex[:8]


class Timer:
    """Simple context manager for measuring elapsed wall-clock seconds.

    Usage:
        with Timer() as t:
            do_work()
        print(t.elapsed_seconds)
    """

    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed_seconds = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_seconds = round(time.perf_counter() - self._start, 3)
        return False


def append_csv_row(csv_path: Path, row: dict) -> None:
    """Append one row to a CSV file, writing the header row first if the file is new.

    The column order is taken from `row.keys()` the first time the file is created.
    If the file already exists, we reuse its existing header order and fill any
    missing columns with an empty string so the file never breaks.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    if file_exists:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            existing_fieldnames = next(csv.reader(f))
        fieldnames = existing_fieldnames
    else:
        fieldnames = list(row.keys())

    safe_row = {key: row.get(key, "") for key in fieldnames}

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(safe_row)


def save_json_log(directory: Path, name: str, data: dict) -> Path:
    """Save a dict as a pretty-printed JSON file under `directory` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{name}.json"
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def save_text(directory: Path, name: str, text: str) -> Path:
    """Save a plain text string under `directory` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / name
    output_path.write_text(text, encoding="utf-8")
    return output_path


def get_text_from_args(text_arg: str | None, file_arg: str | None) -> str:
    """Resolve input text from either a direct --text argument or a --file path.

    Never invents text: raises a clear error if neither is provided, or if the
    provided file doesn't exist or is empty.
    """
    if text_arg and file_arg:
        raise ValueError("Provide only one of --text or --file, not both.")

    if text_arg:
        return text_arg.strip()

    if file_arg:
        file_path = Path(file_arg)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Text file is empty: {file_path}")
        return text

    raise ValueError("You must provide either --text or --file.")
