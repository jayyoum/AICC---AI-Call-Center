#!/usr/bin/env python3
"""
Print a readable timeline for one logged AICC call (session).

Reads output/logs/sessions/<session_id>/ and prints, per turn: the transcript,
routing decision, agent response, latency breakdown, and any errors.

Usage:
    python inspect_session.py --session-id latest
    python inspect_session.py --session-id s_9f2a1c4b77de
    python inspect_session.py --list
    python inspect_session.py --session-id latest --events   # full event timeline
"""

import argparse
import sys

from src.session_logger import (
    latest_session_id,
    list_session_ids,
    list_turn_ids,
    read_session_events,
    read_turn_events,
    read_turn_summaries,
    session_dir,
)


def fmt_seconds(value):
    """Format a latency value as '1.234s', or '-' when missing."""
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return str(value)


def print_latency(latency):
    """Print whichever latency keys are present (shapes differ per turn kind)."""
    if not isinstance(latency, dict) or not latency:
        return
    labels = [
        ("stt", "STT"),
        ("route", "Routing"),
        ("routing", "Routing"),
        ("llm", "LLM"),
        ("tts", "TTS"),
        ("time_to_first_audio_chunk", "First audio chunk"),
        ("total_tts_stream", "TTS stream total"),
        ("total", "TOTAL"),
    ]
    parts = [f"{label} {fmt_seconds(latency[key])}" for key, label in labels if key in latency]
    if parts:
        print("    latency : " + "  |  ".join(parts))


def print_turn(index, summary):
    """Print one turn summary block."""
    kind = summary.get("kind", "turn")
    turn_id = summary.get("turn_id", "?")
    print(f"\n  [{index}] {kind}  ({turn_id})  {summary.get('timestamp', '')}")

    transcript = summary.get("transcript")
    if transcript:
        print(f"    you     : {transcript}")

    source = summary.get("routing_source", "")
    rule = summary.get("rule_name", "")
    routed_by = f"{source} ({rule})" if source == "pre_router" and rule else source
    detail = f"router={summary.get('router_mode', '-')} prompt={summary.get('prompt_mode', '-')}"
    if routed_by:
        print(f"    routed  : {routed_by}   [{detail}]")

    final_route = summary.get("final_route") or summary.get("top_level_route")
    if final_route or summary.get("action"):
        valid = summary.get("route_valid")
        valid_text = "" if valid in (None, "") else f"   route_valid={valid}"
        print(f"    route   : {final_route or '(clarify)'}   action={summary.get('action', '-')}{valid_text}")

    response = summary.get("response_to_customer")
    if response:
        print(f"    agent   : {response}")

    if summary.get("audio_url"):
        print(f"    audio   : {summary['audio_url']}")

    print_latency(summary.get("latency"))

    if summary.get("error"):
        print(f"    ERROR   : {summary['error']}")


def print_events(session_id, turn_ids):
    """Optional full event timeline (every logged event, in order)."""
    print("\n--- full event timeline ---")
    for turn_id in turn_ids:
        events = read_turn_events(session_id, turn_id)
        if not events:
            continue
        print(f"\n  turn {turn_id}:")
        for event in events:
            data = event.get("data") or {}
            # Keep the line short: show a couple of the most useful fields.
            hints = []
            for key in ("transcript", "routing_source", "rule_name", "error",
                        "stt_latency_seconds", "routing_latency_seconds",
                        "tts_latency_seconds", "time_to_first_audio_chunk_seconds"):
                if key in data and data[key] not in ("", None):
                    hints.append(f"{key}={data[key]}")
            suffix = ("  " + "  ".join(hints[:3])) if hints else ""
            print(f"    {event.get('timestamp', '')}  {event.get('event_type', '')}{suffix}")


def main():
    parser = argparse.ArgumentParser(description="Inspect one logged AICC session.")
    parser.add_argument("--session-id", help="Session id, or 'latest' for the most recent.")
    parser.add_argument("--list", action="store_true", help="List available session ids and exit.")
    parser.add_argument("--events", action="store_true", help="Also print the full event timeline.")
    args = parser.parse_args()

    if args.list or not args.session_id:
        ids = list_session_ids()
        if not ids:
            print("No sessions found under output/logs/sessions/.")
            print("Make a call in the web UI first, then re-run this.")
            return
        print(f"{len(ids)} session(s), most recent first:")
        for sid in ids[:40]:
            print(f"  {sid}")
        if not args.session_id:
            print("\nRun: python inspect_session.py --session-id latest")
        return

    session_id = args.session_id
    if session_id == "latest":
        session_id = latest_session_id()
        if not session_id:
            print("No sessions found under output/logs/sessions/.")
            sys.exit(1)

    directory = session_dir(session_id)
    if not directory.exists():
        print(f"Error: session not found: {directory}")
        print("Try: python inspect_session.py --list")
        sys.exit(1)

    print("=" * 70)
    print(f"SESSION {session_id}")
    print(f"path: {directory}")
    print("=" * 70)

    session_events = read_session_events(session_id)
    if session_events:
        print("\nsession events:")
        for event in session_events:
            print(f"  {event.get('timestamp', '')}  {event.get('event_type', '')}")

    summaries = read_turn_summaries(session_id)
    turn_ids = list_turn_ids(session_id)

    if not summaries:
        print("\nNo completed turn summaries in this session.")
        if turn_ids:
            print(f"({len(turn_ids)} turn(s) have events but no summary - likely errored mid-turn.)")
    else:
        print(f"\n{len(summaries)} turn(s):")
        for index, summary in enumerate(summaries, 1):
            print_turn(index, summary)

    # Session-level totals across turns that recorded a total latency.
    totals = []
    for summary in summaries:
        latency = summary.get("latency") or {}
        if isinstance(latency, dict) and latency.get("total") not in (None, ""):
            try:
                totals.append(float(latency["total"]))
            except (TypeError, ValueError):
                pass
    if totals:
        print(f"\n  turns with timing: {len(totals)}"
              f"   avg total {sum(totals) / len(totals):.3f}s"
              f"   min {min(totals):.3f}s   max {max(totals):.3f}s")

    if args.events:
        print_events(session_id, turn_ids)

    print()


if __name__ == "__main__":
    main()
