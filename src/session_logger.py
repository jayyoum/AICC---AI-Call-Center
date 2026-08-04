"""
Unified session + turn logging for the AICC web app.

One phone/browser call is a "session"; each user utterance inside it is a
"turn". Events are appended as JSON Lines so a long call stays readable and
nothing is overwritten.

Layout:
    output/logs/sessions/
      <session_id>/
        session_events.jsonl
        turns/
          <turn_id>_events.jsonl
          <turn_id>_summary.json

Safety notes:
- Values are sanitized before writing: credential-looking keys are redacted and
  very long strings (e.g. base64 audio) are truncated, so logs never contain
  secrets or raw audio blobs.
- Logging never raises. If a write fails we print a warning and carry on, so a
  logging problem can never break a live call.
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = PROJECT_ROOT / "output" / "logs" / "sessions"

# Anything longer than this is truncated (keeps audio/base64 out of the logs).
MAX_STRING_LENGTH = 2000

# Keys whose values are never written out.
_SENSITIVE_KEY_PARTS = (
    "credential", "api_key", "apikey", "token", "password", "secret", "authorization",
)

_write_lock = threading.Lock()


# --- IDs --------------------------------------------------------------------

def new_session_id() -> str:
    """Short id for one call/conversation, e.g. 's_9f2a1c4b77de'."""
    return "s_" + uuid.uuid4().hex[:12]


def new_turn_id() -> str:
    """Short id for one utterance/turn, e.g. 't_1a2b3c4d'."""
    return "t_" + uuid.uuid4().hex[:8]


def _safe_id(value) -> str:
    """Keep ids filesystem-safe (also blocks path traversal from user input)."""
    text = str(value or "unknown")
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in ("-", "_"))
    return cleaned[:64] or "unknown"


def _now_iso() -> str:
    """UTC timestamp with milliseconds, e.g. '2026-07-21T10:15:03.123Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# --- Paths ------------------------------------------------------------------

def session_dir(session_id) -> Path:
    return SESSIONS_DIR / _safe_id(session_id)


def turns_dir(session_id) -> Path:
    return session_dir(session_id) / "turns"


# --- Sanitization -----------------------------------------------------------

def _sanitize(value, depth: int = 0):
    """Recursively redact secrets, shrink blobs, and keep values JSON-safe."""
    if depth > 6:
        return "<nested>"

    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_text = str(key)
            if any(part in key_text.lower() for part in _SENSITIVE_KEY_PARTS):
                out[key_text] = "<redacted>"
            else:
                out[key_text] = _sanitize(item, depth + 1)
        return out

    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth + 1) for item in list(value)[:50]]

    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"

    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            return value[:MAX_STRING_LENGTH] + f"… <truncated, {len(value)} chars total>"
        return value

    if value is None or isinstance(value, (int, float, bool)):
        return value

    return str(value)[:MAX_STRING_LENGTH]


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# --- Public logging API -----------------------------------------------------

def log_session_event(session_id, event_type: str, data=None) -> dict | None:
    """Append one session-level event (call started/ended, errors, etc.)."""
    record = {
        "timestamp": _now_iso(),
        "session_id": _safe_id(session_id),
        "turn_id": None,
        "event_type": event_type,
        "data": _sanitize(data or {}),
    }
    try:
        _append_jsonl(session_dir(session_id) / "session_events.jsonl", record)
        return record
    except Exception as e:  # logging must never break a call
        print(f"[session_logger] could not write session event: {e}")
        return None


def log_turn_event(session_id, turn_id, event_type: str, data=None) -> dict | None:
    """Append one turn-level event (stt_started, routing_finished, ...)."""
    record = {
        "timestamp": _now_iso(),
        "session_id": _safe_id(session_id),
        "turn_id": _safe_id(turn_id),
        "event_type": event_type,
        "data": _sanitize(data or {}),
    }
    try:
        _append_jsonl(turns_dir(session_id) / f"{_safe_id(turn_id)}_events.jsonl", record)
        return record
    except Exception as e:
        print(f"[session_logger] could not write turn event: {e}")
        return None


def save_turn_summary(session_id, turn_id, summary: dict) -> Path | None:
    """Write the one-file-per-turn summary used by inspect_session.py."""
    payload = {
        "session_id": _safe_id(session_id),
        "turn_id": _safe_id(turn_id),
        "timestamp": _now_iso(),
    }
    payload.update(_sanitize(summary or {}))
    try:
        path = turns_dir(session_id) / f"{_safe_id(turn_id)}_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception as e:
        print(f"[session_logger] could not write turn summary: {e}")
        return None


# --- Read helpers (used by inspect_session.py) ------------------------------

def list_session_ids() -> list[str]:
    """All session ids on disk, most recently modified first."""
    if not SESSIONS_DIR.exists():
        return []
    dirs = [d for d in SESSIONS_DIR.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in dirs]


def latest_session_id() -> str | None:
    """The most recently modified session id, or None if there are no sessions."""
    ids = list_session_ids()
    return ids[0] if ids else None


def read_session_events(session_id) -> list[dict]:
    path = session_dir(session_id) / "session_events.jsonl"
    return _read_jsonl(path)


def read_turn_events(session_id, turn_id) -> list[dict]:
    path = turns_dir(session_id) / f"{_safe_id(turn_id)}_events.jsonl"
    return _read_jsonl(path)


def read_turn_summaries(session_id) -> list[dict]:
    """All turn summaries for a session, oldest first (by file mtime)."""
    directory = turns_dir(session_id)
    if not directory.exists():
        return []
    files = sorted(directory.glob("*_summary.json"), key=lambda p: p.stat().st_mtime)
    summaries = []
    for path in files:
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return summaries


def list_turn_ids(session_id) -> list[str]:
    """Turn ids for a session, oldest first (based on their event files)."""
    directory = turns_dir(session_id)
    if not directory.exists():
        return []
    files = sorted(directory.glob("*_events.jsonl"), key=lambda p: p.stat().st_mtime)
    return [p.name.replace("_events.jsonl", "") for p in files]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
