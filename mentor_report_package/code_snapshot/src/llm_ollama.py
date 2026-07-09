"""
Ollama LLM backend for routing/response generation (default model: llama3.2).

Calls the local Ollama HTTP API. No API key is needed since Ollama runs
entirely on your machine. Make sure Ollama is running first:

    ollama run llama3.2

(or `ollama serve` if you just want the background server running).
"""

import json
import re
from pathlib import Path

import requests

from src.utils import Timer

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

REQUIRED_FIELDS = [
    "top_level_route",
    "final_route",
    "action",
    "response_to_customer",
    "confidence",
    "explanation_short",
    "route_tree_used",
]

# Built-in fallback route, always valid even if it isn't in the user's route tree.
OUT_OF_SCOPE_TOP_LEVEL = "Out of Scope"
OUT_OF_SCOPE_FINAL = "Out of Scope > Unsupported Service"


def _route_name(entry) -> str | None:
    """Pull a display name out of a route entry that may be a plain string or a dict."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("name") or entry.get("route") or entry.get("label")
    return None


def _sub_routes(entry) -> list:
    """Pull a list of sub-route entries out of a top-level route dict, if any."""
    if not isinstance(entry, dict):
        return []
    return (
        entry.get("sub_routes")
        or entry.get("subroutes")
        or entry.get("children")
        or entry.get("routes")
        or []
    )


def _extract_allowed_routes(routes: dict | None) -> tuple[set, set]:
    """Walk the user-supplied route tree and collect the exact allowed route names.

    Returns (allowed_top_level_routes, allowed_final_routes), where each final
    route is formatted as "Top Level > Sub Route". The built-in "Out of Scope"
    fallback is always included in both sets.

    Expected route tree shape (see config/routes.example.json):
        {"routes": [{"name": "Card Services", "sub_routes": [{"name": "Lost or Stolen Card"}, ...]}, ...]}
    Plain strings are also accepted for top-level entries and sub-routes.
    """
    top_level = {OUT_OF_SCOPE_TOP_LEVEL}
    final = {OUT_OF_SCOPE_FINAL}

    if not routes:
        return top_level, final

    for entry in routes.get("routes", []) or []:
        top_name = _route_name(entry)
        if not top_name:
            continue
        top_level.add(top_name)

        subs = _sub_routes(entry)
        if not subs:
            # A top-level route with no children can also stand alone as a final route.
            final.add(top_name)
            continue

        for sub in subs:
            sub_name = _route_name(sub)
            if sub_name:
                final.add(f"{top_name} > {sub_name}")

    return top_level, final


def _format_allowed_routes_section(top_level: set, final: set) -> str:
    top_level_lines = "\n".join(f"- {name}" for name in sorted(top_level))
    final_lines = "\n".join(f"- {name}" for name in sorted(final))
    return (
        f"Allowed top_level_route values:\n{top_level_lines}\n\n"
        f"Allowed final_route values (must match one of these exactly):\n{final_lines}"
    )


def _build_prompt(transcript: str, routes: dict | None) -> str:
    route_tree_text = (
        json.dumps(routes, indent=2) if routes else "No route tree was provided."
    )
    top_level, final = _extract_allowed_routes(routes)
    allowed_routes_section = _format_allowed_routes_section(top_level, final)

    return f"""You are a call routing assistant for a customer service voice pipeline.

Customer transcript:
\"\"\"{transcript}\"\"\"

Route tree (may be empty):
{route_tree_text}

allowed_routes:
{allowed_routes_section}

Rules you MUST follow:
1. Choose top_level_route and final_route ONLY from the allowed_routes list
   above. Do not invent, rename, translate, or abbreviate a route. Copy the
   exact text.
2. If the customer's request is not covered by any route in allowed_routes,
   return exactly:
   - top_level_route: "{OUT_OF_SCOPE_TOP_LEVEL}"
   - final_route: "{OUT_OF_SCOPE_FINAL}"
   - action: "offer_human_or_fallback"
3. If the request is vague or ambiguous and you cannot confidently pick a
   final_route, do NOT guess. Return the most relevant top_level_route, leave
   final_route as an empty string "", and set action: "clarify".
4. If the request involves a suspicious transaction, an unauthorized
   purchase, a lost or stolen card combined with fraud, or any other card
   security risk, prioritize routing to a fraud- or safety-related route from
   allowed_routes (for example one containing "Fraud" or "Safety" or
   "High-Risk" in its name) over a generic card-replacement route.
5. Respond with ONLY a single JSON object. No markdown, no code fences, no
   commentary before or after it.

Respond with exactly this JSON schema:
{{
  "top_level_route": "...",
  "final_route": "...",
  "action": "route | clarify | escalate | offer_human_or_fallback",
  "response_to_customer": "...",
  "confidence": 0.0,
  "explanation_short": "...",
  "route_tree_used": true
}}

If no route tree was provided, still make a reasonable decision using your
own judgment based on the transcript alone, and set route_tree_used to false.
"""


def _extract_json_block(text: str) -> str | None:
    """Try to pull a JSON object out of text that may be wrapped in markdown fences."""
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return None


def _parse_model_json(raw_output: str) -> tuple[dict | None, str | None]:
    """Try direct json.loads, then fall back to extracting a JSON block from markdown."""
    try:
        return json.loads(raw_output), None
    except json.JSONDecodeError:
        pass

    candidate = _extract_json_block(raw_output)
    if candidate:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as e:
            return None, f"Found a JSON-like block but it failed to parse: {e}"

    return None, "Model output did not contain valid JSON."


def _validate_routes(parsed_output: dict, allowed_top_level: set, allowed_final: set) -> bool:
    """Check the model's chosen routes against the allowed sets, exactly (no fuzzy matching).

    A blank final_route (used for the "clarify" action) is treated as valid,
    since rule 3 explicitly allows omitting it when the model is unsure.
    """
    top_ok = parsed_output.get("top_level_route") in allowed_top_level
    final_value = parsed_output.get("final_route")
    final_ok = final_value == "" or final_value in allowed_final
    return top_ok and final_ok


def route_with_ollama(transcript: str, routes: dict | None = None, model: str = "llama3.2") -> dict:
    """Send a transcript to Ollama and get back a structured routing decision.

    Returns a dict with: parsed_output, raw_output, provider, model,
    latency_seconds, route_tree_used, route_valid, error (None if successful).

    route_valid is:
    - None if no route tree was provided (nothing to validate against)
    - True/False if a route tree was provided, based on an exact match check
      against the route tree's top-level and final route names.

    Invalid routes are never silently replaced or corrected — parsed_output
    and raw_output always reflect exactly what the model returned, so
    failures can be inspected.
    """
    result = {
        "parsed_output": None,
        "raw_output": "",
        "provider": "ollama",
        "model": model,
        "latency_seconds": 0.0,
        "route_tree_used": bool(routes),
        "route_valid": None,
        "error": None,
    }

    if not transcript.strip():
        result["error"] = "No transcript/text provided to route."
        return result

    prompt = _build_prompt(transcript, routes)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
    }

    try:
        with Timer() as t:
            response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
        result["latency_seconds"] = t.elapsed_seconds
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        result["error"] = (
            "Could not connect to Ollama at http://localhost:11434. "
            "Is Ollama running? Try: ollama run llama3.2"
        )
        return result
    except requests.exceptions.RequestException as e:
        result["error"] = f"Ollama API call failed: {e}"
        return result

    try:
        body = response.json()
        raw_output = body.get("message", {}).get("content", "")
    except Exception as e:
        result["error"] = f"Could not parse Ollama response envelope: {e}"
        return result

    result["raw_output"] = raw_output

    parsed_output, parse_error = _parse_model_json(raw_output)
    if parse_error:
        result["error"] = parse_error
        return result

    missing_fields = [field for field in REQUIRED_FIELDS if field not in parsed_output]
    if missing_fields:
        result["error"] = f"Model JSON is missing expected fields: {', '.join(missing_fields)}"

    result["parsed_output"] = parsed_output
    # route_tree_used reflects what we actually gave the model, not the model's own claim.
    parsed_output["route_tree_used"] = bool(routes)

    if routes:
        allowed_top_level, allowed_final = _extract_allowed_routes(routes)
        result["route_valid"] = _validate_routes(parsed_output, allowed_top_level, allowed_final)

    return result
