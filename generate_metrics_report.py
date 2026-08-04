#!/usr/bin/env python3
"""
Build a single static HTML report from the benchmark summary CSVs.

Reads whichever of these exist in output/results/ and renders them as plain
HTML tables plus a few automatically-derived takeaways:

    prompt_mode_summary.csv        (prompt mode comparison)
    model_summary.csv              (Ollama model comparison)
    router_mode_summary.csv        (pre-router vs LLM-only)
    transcript_replay_summary.csv  (routing-only transcript replay)

Usage:
    python generate_metrics_report.py
    python generate_metrics_report.py --results-dir output/results --output output/reports/metrics_report.html
    python generate_metrics_report.py --csv output/results/model_summary.csv

Output:
    output/reports/metrics_report.html   (open directly in a browser)

No server, no JavaScript frameworks, and no internet access required.
"""

import argparse
import csv
import html
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "output" / "results"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "reports" / "metrics_report.html"

# (filename, section title, short description)
KNOWN_SUMMARIES = [
    ("prompt_mode_summary.csv", "Prompt mode comparison",
     "Same audio and model; only the LLM routing prompt changes."),
    ("model_summary.csv", "Model comparison",
     "Same prompt mode; only the Ollama model changes."),
    ("router_mode_summary.csv", "Router mode comparison",
     "Rule-based pre-router (with LLM fallback) vs LLM-only routing."),
    ("transcript_replay_summary.csv", "Transcript replay (routing only)",
     "Saved transcripts re-routed without STT or TTS."),
]

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 2rem 1.5rem 4rem; background: #f6f7f9; color: #1f2430; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { margin: 0 0 .25rem; font-size: 1.7rem; }
.sub { color: #667085; margin: 0 0 1.75rem; font-size: .95rem; }
section { background: #fff; border: 1px solid #e4e7ec; border-radius: 10px;
          padding: 1.1rem 1.25rem 1.35rem; margin-bottom: 1.25rem; }
h2 { margin: 0 0 .2rem; font-size: 1.1rem; }
.desc { color: #667085; font-size: .88rem; margin: 0 0 .8rem; }
.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .87rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #eef0f3; white-space: nowrap; }
th { background: #fafbfc; color: #475467; font-weight: 600; }
tr:last-child td { border-bottom: none; }
td.num { font-variant-numeric: tabular-nums; }
ul.takeaways { margin: .4rem 0 0; padding-left: 1.2rem; }
ul.takeaways li { margin: .3rem 0; }
.missing { color: #98a2b3; font-style: italic; font-size: .9rem; }
.badge { display: inline-block; background: #eef4ff; color: #1849a9; border-radius: 999px;
         padding: .08rem .55rem; font-size: .78rem; margin-left: .35rem; }
footer { color: #98a2b3; font-size: .82rem; text-align: center; margin-top: 1.5rem; }
"""


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="missing">No rows.</p>'
    columns = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            css = "num" if to_float(value) is not None else ""
            cells.append(f'<td class="{css}">{html.escape(str(value))}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return ('<div class="table-scroll"><table><thead><tr>' + head +
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


def best_row(rows, column, lowest=True):
    """Return (row, value) with the min/max numeric value in `column`."""
    candidates = [(r, to_float(r.get(column))) for r in rows]
    candidates = [(r, v) for r, v in candidates if v is not None]
    if not candidates:
        return None, None
    return (min if lowest else max)(candidates, key=lambda pair: pair[1])


def label_for(row) -> str:
    """Human label for a summary row, using whichever key columns exist."""
    for key in ("prompt_mode", "model", "router_mode", "combination"):
        if row.get(key):
            parts = [row[key]]
            # Transcript replay rows are keyed by three columns.
            if key == "router_mode" and row.get("prompt_mode"):
                parts.append(row["prompt_mode"])
            if key == "router_mode" and row.get("model"):
                parts.append(row["model"])
            return " / ".join(str(p) for p in parts)
    return "(unknown)"


def build_takeaways(datasets: dict[str, list[dict]]) -> list[str]:
    """Derive a few plain-language findings from whatever CSVs are present."""
    notes = []

    # Fastest average TOTAL latency across any summary that reports it.
    total_candidates = []
    for name, rows in datasets.items():
        row, value = best_row(rows, "avg_total_latency_seconds", lowest=True)
        if row is not None:
            total_candidates.append((value, label_for(row), name))
    if total_candidates:
        value, label, name = min(total_candidates, key=lambda t: t[0])
        notes.append(f"Fastest average <strong>total</strong> latency: <strong>{label}</strong> "
                     f"at <strong>{value:.3f}s</strong> <span class='badge'>{name}</span>")

    # Fastest routing step (transcript replay reports routing latency directly).
    routing_candidates = []
    for name, rows in datasets.items():
        row, value = best_row(rows, "avg_routing_latency_seconds", lowest=True)
        if row is not None:
            routing_candidates.append((value, label_for(row), name))
    if routing_candidates:
        value, label, name = min(routing_candidates, key=lambda t: t[0])
        notes.append(f"Fastest average <strong>routing</strong> step: <strong>{label}</strong> "
                     f"at <strong>{value:.3f}s</strong> <span class='badge'>{name}</span>")

    # Route validity: best and worst across everything that reports it.
    valid_rows = []
    for name, rows in datasets.items():
        for row in rows:
            rate = to_float(row.get("route_valid_rate"))
            if rate is not None:
                valid_rows.append((rate, label_for(row), name))
    if valid_rows:
        best = max(valid_rows, key=lambda t: t[0])
        worst = min(valid_rows, key=lambda t: t[0])
        notes.append(f"Highest <strong>route_valid_rate</strong>: <strong>{best[1]}</strong> "
                     f"at <strong>{best[0]:.3f}</strong> <span class='badge'>{best[2]}</span>")
        if worst[1] != best[1]:
            notes.append(f"Lowest route_valid_rate: <strong>{worst[1]}</strong> "
                         f"at <strong>{worst[0]:.3f}</strong> <span class='badge'>{worst[2]}</span>")

    # How often the pre-router answered without the LLM.
    handled_rows = []
    for name, rows in datasets.items():
        for row in rows:
            rate = to_float(row.get("pre_router_handled_rate"))
            if rate is not None and rate > 0:
                handled_rows.append((rate, label_for(row), name))
    if handled_rows:
        best = max(handled_rows, key=lambda t: t[0])
        notes.append(f"Highest <strong>pre_router_handled_rate</strong>: <strong>{best[1]}</strong> "
                     f"at <strong>{best[0]:.3f}</strong> (LLM skipped that often) "
                     f"<span class='badge'>{best[2]}</span>")

    if not notes:
        notes.append("No numeric summary columns found yet - run a benchmark first.")
    return notes


def build_html(sections: list[tuple[str, str, list[dict]]], takeaways: list[str]) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>AICC metrics report</title>",
        f"<style>{CSS}</style></head><body><div class='wrap'>",
        "<h1>AICC metrics report</h1>",
        f"<p class='sub'>Generated {html.escape(generated)} from output/results/ &middot; "
        "latency, route validity, and normalization only (no semantic scoring).</p>",
        "<section><h2>Key takeaways</h2>",
        "<p class='desc'>Derived automatically from the summary CSVs present.</p>",
        "<ul class='takeaways'>",
    ]
    parts += [f"<li>{note}</li>" for note in takeaways]
    parts.append("</ul></section>")

    for title, description, rows in sections:
        parts.append("<section>")
        parts.append(f"<h2>{html.escape(title)}</h2>")
        parts.append(f"<p class='desc'>{html.escape(description)}</p>")
        parts.append(render_table(rows) if rows else
                     "<p class='missing'>Not available - the CSV for this section does not exist yet.</p>")
        parts.append("</section>")

    parts.append("<footer>AICC prototype &middot; static report, no server required</footer>")
    parts.append("</div></body></html>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate a static HTML metrics report.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                        help="Folder holding the summary CSVs (default: output/results).")
    parser.add_argument("--csv", nargs="+",
                        help="Explicit CSV paths to include instead of auto-discovery.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Where to write the HTML report.")
    args = parser.parse_args()

    sections: list[tuple[str, str, list[dict]]] = []
    datasets: dict[str, list[dict]] = {}

    if args.csv:
        for raw in args.csv:
            path = Path(raw)
            rows = read_csv_rows(path)
            sections.append((path.name, f"Loaded from {path}", rows))
            if rows:
                datasets[path.name] = rows
    else:
        results_dir = Path(args.results_dir)
        for filename, title, description in KNOWN_SUMMARIES:
            rows = read_csv_rows(results_dir / filename)
            sections.append((title, description, rows))
            if rows:
                datasets[filename] = rows

    takeaways = build_takeaways(datasets)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(sections, takeaways), encoding="utf-8")

    found = len(datasets)
    print(f"Included {found} summary CSV(s).")
    for name in datasets:
        print(f"  - {name}")
    if not found:
        print("  (none found - run a benchmark, then re-run this script)")
    print(f"\nReport written to: {output_path}")
    print(f"Open it with:      open {output_path}")


if __name__ == "__main__":
    main()
