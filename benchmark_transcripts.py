#!/usr/bin/env python3
"""
Replay saved transcripts through routing only - no STT, no TTS, no audio.

This is much faster than the audio benchmarks because it skips speech
processing entirely: it measures routing latency, route validity, and route
normalization across router modes / prompt modes / models.

It is NOT semantic scoring. It does not judge whether a route is "correct" -
only whether routing was fast, valid against the route tree, and whether the
LLM's route label needed normalizing.

Transcript sources (pick one):
    --transcript-dir output/transcripts     every .txt file in a folder
    --csv output/results/router_mode_benchmark.csv   the 'transcript' column
    --text "I lost my debit card."          a single inline transcript

Examples:
    python benchmark_transcripts.py \\
      --transcript-dir output/transcripts \\
      --routes config/routes.json \\
      --router-modes llm_only pre_router \\
      --prompt-modes compact_v2 full \\
      --model llama3.2

    python benchmark_transcripts.py \\
      --text "I lost my debit card" \\
      --routes config/routes.json \\
      --router-modes llm_only pre_router \\
      --prompt-modes compact_v2 full \\
      --model llama3.2

Outputs:
    output/results/transcript_replay_benchmark.csv   (one row per replay)
    output/results/transcript_replay_summary.csv     (grouped summary)

Note: llm_only (and pre_router fall-through) calls your local Ollama.
pre_router runs are pure string matching and need nothing external.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.llm_ollama import VALID_PROMPT_MODES, normalize_route_decision, route_with_ollama, validate_route
from src.pre_router import pre_route
from src.utils import RESULTS_DIR, Timer, ensure_output_dirs

DETAILED_CSV = RESULTS_DIR / "transcript_replay_benchmark.csv"
SUMMARY_CSV = RESULTS_DIR / "transcript_replay_summary.csv"

ROUTER_MODES = ("llm_only", "pre_router")

DETAILED_COLUMNS = [
    "transcript_source",
    "transcript",
    "router_mode",
    "routing_source",
    "pre_router_handled",
    "rule_name",
    "prompt_mode",
    "model",
    "predicted_top_level_route",
    "predicted_final_route",
    "predicted_action",
    "response_to_customer",
    "route_valid",
    "route_normalized",
    "original_final_route",
    "normalized_final_route",
    "routing_latency_seconds",
    "llm_latency_seconds",
    "error",
]

SUMMARY_COLUMNS = [
    "router_mode",
    "prompt_mode",
    "model",
    "run_count",
    "error_count",
    "avg_routing_latency_seconds",
    "avg_llm_latency_seconds",
    "pre_router_handled_count",
    "pre_router_handled_rate",
    "route_valid_true_count",
    "route_valid_false_count",
    "route_valid_rate",
    "normalized_count",
]


def load_routes(routes_arg):
    if not routes_arg:
        return None
    path = Path(routes_arg)
    if not path.exists():
        print(f"Error: Routes file not found: {path}")
        sys.exit(1)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: Routes file is not valid JSON: {e}")
        sys.exit(1)


def collect_transcripts(args) -> list[tuple[str, str]]:
    """Return a list of (source_label, transcript) pairs."""
    items: list[tuple[str, str]] = []

    if args.text:
        items.append(("--text", args.text.strip()))

    if args.transcript_dir:
        directory = Path(args.transcript_dir)
        if not directory.is_dir():
            print(f"Error: Transcript directory not found: {directory}")
            sys.exit(1)
        for path in sorted(directory.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                items.append((path.name, text))

    if args.csv:
        path = Path(args.csv)
        if not path.exists():
            print(f"Error: CSV not found: {path}")
            sys.exit(1)
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = (row.get("transcript") or "").strip()
                if text:
                    label = row.get("scenario") or path.name
                    items.append((label, text))

    # De-duplicate identical transcripts while keeping the first source label.
    seen = set()
    unique = []
    for label, text in items:
        if text in seen:
            continue
        seen.add(text)
        unique.append((label, text))
    return unique


def replay_one(transcript, source, router_mode, prompt_mode, model, routes) -> dict:
    """Run one transcript through routing only and return a detailed row."""
    row = {
        "transcript_source": source,
        "transcript": transcript,
        "router_mode": router_mode,
        "routing_source": "",
        "pre_router_handled": "",
        "rule_name": "",
        "prompt_mode": prompt_mode,
        "model": model,
        "predicted_top_level_route": "",
        "predicted_final_route": "",
        "predicted_action": "",
        "response_to_customer": "",
        "route_valid": "",
        "route_normalized": "",
        "original_final_route": "",
        "normalized_final_route": "",
        "routing_latency_seconds": 0.0,
        "llm_latency_seconds": 0.0,
        "error": "",
    }

    # Pre-router first (when enabled). Pure string matching, no network.
    pre_result = None
    pre_latency = 0.0
    if router_mode == "pre_router":
        with Timer() as t:
            pre_result = pre_route(transcript, routes=routes)
        pre_latency = t.elapsed_seconds

    if pre_result and pre_result.get("handled"):
        row.update({
            "routing_source": "pre_router",
            "pre_router_handled": True,
            "rule_name": pre_result.get("rule_name", ""),
            "predicted_top_level_route": pre_result.get("top_level_route", ""),
            "predicted_final_route": pre_result.get("final_route", ""),
            "predicted_action": pre_result.get("action", ""),
            "response_to_customer": pre_result.get("response_to_customer", ""),
            "route_valid": validate_route(pre_result, routes),
            # The pre-router emits exact routes, so nothing is normalized.
            "route_normalized": False,
            "original_final_route": pre_result.get("final_route", ""),
            "normalized_final_route": pre_result.get("final_route", ""),
            "routing_latency_seconds": pre_latency,
            "llm_latency_seconds": 0.0,
        })
        return row

    # Otherwise ask the LLM (this is where normalization can kick in).
    if router_mode == "pre_router":
        row["pre_router_handled"] = False

    llm_result = route_with_ollama(transcript, routes=routes, model=model, prompt_mode=prompt_mode)
    llm_latency = llm_result.get("latency_seconds", 0.0)
    row["routing_source"] = "ollama"
    row["llm_latency_seconds"] = llm_latency
    row["routing_latency_seconds"] = round(pre_latency + llm_latency, 3)

    parsed = llm_result.get("parsed_output")
    if parsed is None:
        row["error"] = llm_result.get("error") or "LLM returned no usable output."
        return row

    row.update({
        "predicted_top_level_route": parsed.get("top_level_route", ""),
        "predicted_final_route": parsed.get("final_route", ""),
        "predicted_action": parsed.get("action", ""),
        "response_to_customer": parsed.get("response_to_customer", ""),
        "route_valid": llm_result.get("route_valid"),
        "route_normalized": llm_result.get("route_normalized", ""),
        "original_final_route": llm_result.get("original_final_route", ""),
        "normalized_final_route": llm_result.get("normalized_final_route", ""),
        "error": llm_result.get("error") or "",
    })
    return row


def summarize(rows) -> list[dict]:
    """Group rows by (router_mode, prompt_mode, model)."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["router_mode"], row["prompt_mode"], row["model"])].append(row)

    summary_rows = []
    for (router_mode, prompt_mode, model), group in groups.items():
        count = len(group)
        errors = sum(1 for r in group if r.get("error"))
        routing = [float(r.get("routing_latency_seconds") or 0) for r in group]
        llm = [float(r.get("llm_latency_seconds") or 0) for r in group]

        handled = sum(1 for r in group if r.get("pre_router_handled") is True)
        valid_true = sum(1 for r in group if r.get("route_valid") is True)
        valid_false = sum(1 for r in group if r.get("route_valid") is False)
        validated = valid_true + valid_false
        normalized = sum(1 for r in group if r.get("route_normalized") is True)

        summary_rows.append({
            "router_mode": router_mode,
            "prompt_mode": prompt_mode,
            "model": model,
            "run_count": count,
            "error_count": errors,
            "avg_routing_latency_seconds": round(sum(routing) / count, 3) if count else 0,
            "avg_llm_latency_seconds": round(sum(llm) / count, 3) if count else 0,
            "pre_router_handled_count": handled,
            "pre_router_handled_rate": round(handled / count, 3) if count else "",
            "route_valid_true_count": valid_true,
            "route_valid_false_count": valid_false,
            "route_valid_rate": round(valid_true / validated, 3) if validated else "",
            "normalized_count": normalized,
        })

    summary_rows.sort(key=lambda r: r["avg_routing_latency_seconds"])
    return summary_rows


def write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def print_table(rows, columns) -> None:
    if not rows:
        print("No rows to summarize.")
        return
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row[c]).ljust(widths[c]) for c in columns))


def main():
    parser = argparse.ArgumentParser(description="Replay transcripts through routing only (no STT/TTS).")
    parser.add_argument("--transcript-dir", help="Directory of .txt transcripts (e.g. output/transcripts).")
    parser.add_argument("--csv", help="CSV with a 'transcript' column to replay.")
    parser.add_argument("--text", help="A single transcript to replay.")
    parser.add_argument("--routes", help="Path to routes JSON (see config/routes.example.json).")
    parser.add_argument("--router-modes", nargs="+", default=list(ROUTER_MODES),
                        choices=list(ROUTER_MODES), help="Router modes to compare.")
    parser.add_argument("--prompt-modes", nargs="+", default=["compact_v2"],
                        choices=list(VALID_PROMPT_MODES), help="Prompt modes to compare.")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name (default: llama3.2).")
    parser.add_argument("--limit", type=int, help="Only replay the first N transcripts.")
    args = parser.parse_args()

    if not (args.transcript_dir or args.csv or args.text):
        print("Error: provide one of --transcript-dir, --csv, or --text.")
        sys.exit(1)

    ensure_output_dirs()
    routes = load_routes(args.routes)
    transcripts = collect_transcripts(args)
    if args.limit:
        transcripts = transcripts[: args.limit]

    if not transcripts:
        print("Error: no transcripts found to replay.")
        sys.exit(1)

    total_runs = len(transcripts) * len(args.router_modes) * len(args.prompt_modes)
    print(f"Replaying {len(transcripts)} transcript(s) x {len(args.router_modes)} router mode(s) "
          f"x {len(args.prompt_modes)} prompt mode(s) = {total_runs} run(s).")
    print(f"Model: {args.model}   (routing only - no STT, no TTS)\n")

    rows = []
    for router_mode in args.router_modes:
        for prompt_mode in args.prompt_modes:
            print(f"=== router_mode={router_mode}  prompt_mode={prompt_mode} ===", flush=True)
            for source, transcript in transcripts:
                row = replay_one(transcript, source, router_mode, prompt_mode, args.model, routes)
                rows.append(row)
                if row["error"]:
                    print(f"  {source}: error: {row['error']}")
                else:
                    tag = row["routing_source"]
                    if row["routing_source"] == "pre_router" and row["rule_name"]:
                        tag += f":{row['rule_name']}"
                    print(f"  {source}: [{tag}] {row['predicted_final_route'] or '(clarify)'} "
                          f"| valid={row['route_valid']} | {row['routing_latency_seconds']}s")

    write_csv(DETAILED_CSV, DETAILED_COLUMNS, rows)
    summary_rows = summarize(rows)
    write_csv(SUMMARY_CSV, SUMMARY_COLUMNS, summary_rows)

    print("\n===== summary (fastest average routing latency first) =====")
    print_table(summary_rows, SUMMARY_COLUMNS)
    print(f"\nDetailed rows saved to: {DETAILED_CSV}")
    print(f"Summary saved to:       {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
