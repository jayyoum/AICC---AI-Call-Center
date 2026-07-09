"""
Batch runner: reads a user-provided manifest CSV and runs one or more
pipeline provider combinations against every audio file listed in it.

Manifest columns (see config/manifest.example.csv):
    scenario_id, audio_path, expected_route, expected_action, stress_type, notes
"""

import csv
from pathlib import Path

from src.pipeline import run_pipeline
from src.utils import PIPELINE_RESULTS_CSV, append_csv_row

# Maps a short combination name to (stt_provider, tts_provider).
COMBINATIONS = {
    "google_google": ("google", "google"),
    "local_local": ("local", "local"),
    "google_local": ("google", "local"),
    "local_google": ("local", "google"),
}

MANIFEST_REQUIRED_COLUMNS = [
    "scenario_id",
    "audio_path",
    "expected_route",
    "expected_action",
    "stress_type",
    "notes",
]


def read_manifest(manifest_path: Path) -> list[dict]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if rows:
        missing_columns = [c for c in MANIFEST_REQUIRED_COLUMNS if c not in rows[0]]
        if missing_columns:
            raise ValueError(
                f"Manifest is missing required columns: {', '.join(missing_columns)}. "
                "See config/manifest.example.csv for the expected schema."
            )

    return rows


def run_batch(
    manifest_path: Path,
    combination_names: list[str],
    llm_model: str = "llama3.2",
    routes: dict | None = None,
    stt_model_size: str = "base.en",
    tts_voice: str = "en_US-lessac-medium",
) -> list[dict]:
    """Run the given pipeline combinations against every row of the manifest.

    Appends every result to output/results/pipeline_results.csv and also
    returns the list of result rows for further processing (e.g. printing).
    """
    unknown = [name for name in combination_names if name not in COMBINATIONS]
    if unknown:
        raise ValueError(f"Unknown combination name(s): {', '.join(unknown)}. Valid: {', '.join(COMBINATIONS)}")

    manifest_rows = read_manifest(manifest_path)
    results = []

    for manifest_row in manifest_rows:
        scenario_id = manifest_row.get("scenario_id", "")
        audio_path = Path(manifest_row.get("audio_path", ""))

        extra_metadata = {
            "scenario_id": scenario_id,
            "expected_route": manifest_row.get("expected_route", ""),
            "expected_action": manifest_row.get("expected_action", ""),
            "stress_type": manifest_row.get("stress_type", ""),
            "notes": manifest_row.get("notes", ""),
            "manual_success_label": "",
        }

        for combo_name in combination_names:
            stt_provider, tts_provider = COMBINATIONS[combo_name]
            print(f"Running scenario '{scenario_id}' with combination '{combo_name}'...")

            row = run_pipeline(
                audio_path=audio_path,
                stt_provider=stt_provider,
                tts_provider=tts_provider,
                llm_model=llm_model,
                routes=routes,
                stt_model_size=stt_model_size,
                tts_voice=tts_voice,
                extra_metadata={**extra_metadata, "combination": combo_name},
            )

            if row.get("error"):
                print(f"  -> Completed with error: {row['error']}")
            else:
                print(f"  -> Completed in {row['total_latency_seconds']}s")

            append_csv_row(PIPELINE_RESULTS_CSV, row)
            results.append(row)

    return results
