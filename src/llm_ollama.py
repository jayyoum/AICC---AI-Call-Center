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

# Fields the "full" prompt asks the model to produce directly.
FULL_REQUIRED_FIELDS = [
    "top_level_route",
    "final_route",
    "action",
    "response_to_customer",
    "confidence",
    "explanation_short",
    "route_tree_used",
]

# Fields the "compact" prompts ask the model to produce. They intentionally do
# NOT ask for response_to_customer / confidence / explanation_short - those are
# filled in afterwards (response_to_customer via generate_template_response())
# to keep the prompt short and the LLM step fast.
COMPACT_REQUIRED_FIELDS = [
    "top_level_route",
    "final_route",
    "action",
]

# "compact_v2" is a compact prompt with a few extra targeted clarification/safety
# rules for the AICC stress cases. It behaves like "compact" everywhere else
# (minimal fields, template-generated reply).
COMPACT_MODES = ("compact", "compact_v2")
VALID_PROMPT_MODES = ("compact", "compact_v2", "full")
DEFAULT_PROMPT_MODE = "compact_v2"

# Built-in fallback route, always valid even if it isn't in the user's route tree.
OUT_OF_SCOPE_TOP_LEVEL = "Out of Scope"
OUT_OF_SCOPE_FINAL = "Out of Scope > Unsupported Service"

# Short customer-facing response used when routing fails entirely.
FALLBACK_TEMPLATE_RESPONSE = (
    "I'm sorry, I had trouble routing that request. "
    "Would you like me to connect you with a human representative?"
)

# Specific customer-facing lines for a few high-value routes. Any allowed route
# not listed here gets a generic line built from its leaf name (see
# generate_template_response). These are shared by compact-mode LLM routing and
# the rule-based pre-router so both produce identical wording for the same route.
_ROUTE_RESPONSE_TEMPLATES = {
    "Card Services > Lost or Stolen Card":
        "I can help with that. I'll route you to lost or stolen card support.",
    "Card Services > Fraud or Suspicious Transaction":
        "Because this may involve unauthorized activity, I recommend connecting "
        "you to card security and fraud support.",
    "Transfers and Payments > Failed Transfer":
        "I can help with that. I'll route you to failed transfer support.",
    "Transfers and Payments > Scheduled Transfer":
        "I can help with that. I'll route you to scheduled transfer support.",
    "Account Access > Locked Account":
        "I can help with that. I'll route you to locked account support.",
    "Deposits and ATM > ATM Withdrawal Issue":
        "I can help with that. I'll route you to ATM withdrawal support.",
    "Out of Scope > Unsupported Service":
        "That request appears to be outside the services I can handle here. "
        "Would you like me to connect you with a human representative?",
}


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


def _ordered_allowed_routes(routes: dict | None) -> tuple[list, list]:
    """Like _extract_allowed_routes but preserves the route tree's order for display.

    Returns (top_level_routes, final_routes) as ordered, de-duplicated lists so
    the compact prompt can show a clean numbered list. The built-in Out of Scope
    fallback is appended if the tree doesn't already include it.
    """
    top_level: list = []
    final: list = []

    if routes:
        for entry in routes.get("routes", []) or []:
            top_name = _route_name(entry)
            if not top_name:
                continue
            if top_name not in top_level:
                top_level.append(top_name)

            subs = _sub_routes(entry)
            if not subs:
                if top_name not in final:
                    final.append(top_name)
                continue

            for sub in subs:
                sub_name = _route_name(sub)
                if sub_name:
                    label = f"{top_name} > {sub_name}"
                    if label not in final:
                        final.append(label)

    if OUT_OF_SCOPE_TOP_LEVEL not in top_level:
        top_level.append(OUT_OF_SCOPE_TOP_LEVEL)
    if OUT_OF_SCOPE_FINAL not in final:
        final.append(OUT_OF_SCOPE_FINAL)

    return top_level, final


def _build_compact_prompt(transcript: str, routes: dict | None) -> str:
    """Short routing prompt focused only on fast route/action classification.

    Deliberately omits the long role description, full JSON route tree, and the
    response-generation schema used by the full prompt, to reduce token count
    and LLM latency. The customer-facing reply is generated afterwards via
    generate_template_response(), not by the model.
    """
    top_level, final = _ordered_allowed_routes(routes)
    top_level_line = ", ".join(top_level)
    final_lines = "\n".join(f"{i}. {name}" for i, name in enumerate(final, 1))

    return f"""Classify the customer message into one allowed banking route.

Message: "{transcript}"

Allowed top-level routes: {top_level_line}

Allowed final routes:
{final_lines}

Rules:
- Pick top_level_route and final_route only from the Allowed lists; copy the text exactly.
- If unclear, set final_route="" and action="clarify".
- If outside allowed routes, set top_level_route="Out of Scope", final_route="Out of Scope > Unsupported Service", action="offer_human_or_fallback".
- If suspicious, unauthorized, or lost card combined with fraud, prefer the "Card Services > Fraud or Suspicious Transaction" route.
- Return JSON only, no other text.

JSON schema:
{{"top_level_route":"...","final_route":"...","action":"route|clarify|escalate|offer_human_or_fallback","response_type":"route|clarify|escalate|fallback"}}"""


def _build_compact_v2_prompt(transcript: str, routes: dict | None) -> str:
    """Compact prompt plus a few targeted clarification/safety rules.

    Same short format and minimal JSON as _build_compact_prompt, but with extra
    rules aimed at the AICC stress cases where the plain compact prompt tended to
    over-route instead of asking for clarification (e.g. "a payment didn't go
    through" with no payment type given). The rules are phrased generically in
    terms of what the user says, not tied to any scenario filename.
    """
    top_level, final = _ordered_allowed_routes(routes)
    top_level_line = ", ".join(top_level)
    final_lines = "\n".join(f"{i}. {name}" for i, name in enumerate(final, 1))

    return f"""Classify the customer message into one allowed banking route.

Message: "{transcript}"

Allowed top-level routes: {top_level_line}

Allowed final routes:
{final_lines}

Rules:
- Pick top_level_route and final_route only from the Allowed lists; copy the text exactly.
- If outside allowed routes, set top_level_route="Out of Scope", final_route="Out of Scope > Unsupported Service", action="offer_human_or_fallback".
- If unclear, use the most relevant top_level_route, final_route="", action="clarify".
- If the user says a payment did not go through but does not specify whether it was a transfer, bill payment, scheduled transfer, person-to-person payment, or card payment, use top_level_route="Transfers and Payments", final_route="", action="clarify".
- If the user vaguely says something is wrong or weird with their card but gives no specific issue, use top_level_route="Card Services", final_route="", action="clarify".
- If the user mentions suspicious, unauthorized, or unrecognized purchases/charges, fraud, or a lost/stolen card plus purchases they did not make, use final_route="Card Services > Fraud or Suspicious Transaction", action="escalate".
- If the user says the card is lost or stolen with no suspicious charge mentioned, use final_route="Card Services > Lost or Stolen Card", action="route".
- If the user says an ATM charged the account but did not dispense cash, use final_route="Deposits and ATM > ATM Withdrawal Issue", action="route".
- If the user mentions logging in too many times or the account being locked, use final_route="Account Access > Locked Account", action="route".
- If the user mentions changing or canceling a scheduled transfer, use final_route="Transfers and Payments > Scheduled Transfer", action="route".
- Return JSON only, no other text.

JSON schema:
{{"top_level_route":"...","final_route":"...","action":"route|clarify|escalate|offer_human_or_fallback","response_type":"route|clarify|escalate|fallback"}}"""


def generate_template_response(parsed_output: dict, transcript: str = "") -> str:
    """Build a short customer-facing response from a routing decision.

    Used by compact mode, which does not ask the LLM to write the reply. The
    logic keys off action and final_route so the wording stays consistent and
    fast without a second LLM call. `transcript` is accepted for future use
    (e.g. echoing details back) but is not required.
    """
    action = (parsed_output.get("action") or "").strip()
    final_route = (parsed_output.get("final_route") or "").strip()

    # Ask-for-more-detail case comes first: no confident final route yet.
    if action == "clarify" or not final_route:
        return (
            "I can help with that. Could you provide a little more detail so I "
            "can route you correctly?"
        )

    # Curated wording for specific high-value routes.
    if final_route in _ROUTE_RESPONSE_TEMPLATES:
        return _ROUTE_RESPONSE_TEMPLATES[final_route]

    # Out-of-scope / human handoff.
    if action == "offer_human_or_fallback":
        return (
            "That request appears to be outside the services I can handle here. "
            "Would you like me to connect you with a human representative?"
        )
    if action == "escalate":
        return (
            "I understand. I'll connect you with a representative who can help "
            "you right away."
        )

    # Generic route confirmation built from the final route's leaf name, e.g.
    # "Card Services > PIN or Card Activation" -> "...pin or card activation support."
    leaf = final_route.split(">")[-1].strip()
    if leaf:
        return f"I can help with that. I'll route you to {leaf.lower()} support."

    return FALLBACK_TEMPLATE_RESPONSE


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


def validate_route(decision: dict, routes: dict | None) -> bool | None:
    """Public route-tree compliance check, reusable by any router.

    Accepts any dict with "top_level_route" and "final_route" keys (an LLM
    parsed_output or a pre-router decision). Returns None if no route tree was
    provided (nothing to validate against), else True/False for exact match. A
    blank final_route (clarify case) is treated as valid.
    """
    if not routes:
        return None
    allowed_top_level, allowed_final = _extract_allowed_routes(routes)
    return _validate_routes(decision, allowed_top_level, allowed_final)


def _clean_route_string(value: str) -> str:
    """Tidy a route label: '->' becomes '>', collapse whitespace, normalize spacing.

    'Transfers and Payments -> Failed Transfer' -> 'Transfers and Payments > Failed Transfer'
    '  Failed   Transfer ' -> 'Failed Transfer'
    """
    value = (value or "").replace("->", ">")
    parts = [re.sub(r"\s+", " ", part).strip() for part in value.split(">")]
    parts = [part for part in parts if part]
    return " > ".join(parts)


def _build_leaf_index(routes: dict | None) -> dict:
    """Map each lowercased sub-route leaf name to the list of full paths it appears under.

    A top-level route with no sub-routes maps its own name to itself, so it can
    stand alone as a final route. Used to expand sub-route-only labels to the
    exact full 'Top > Sub' path when the leaf is unambiguous.
    """
    leaf_to_paths: dict = {}
    if not routes:
        return leaf_to_paths

    def add(leaf: str, full: str) -> None:
        leaf_to_paths.setdefault(leaf.lower(), [])
        if full not in leaf_to_paths[leaf.lower()]:
            leaf_to_paths[leaf.lower()].append(full)

    for entry in routes.get("routes", []) or []:
        top_name = _route_name(entry)
        if not top_name:
            continue
        subs = _sub_routes(entry)
        if not subs:
            add(top_name, top_name)
            continue
        for sub in subs:
            sub_name = _route_name(sub)
            if sub_name:
                add(sub_name, f"{top_name} > {sub_name}")
    return leaf_to_paths


def normalize_route_decision(decision: dict, routes: dict | None) -> dict:
    """Repair a routing decision's final_route to an exact full route path.

    Handles: '->' instead of '>', extra whitespace, and sub-route-only labels
    like 'Failed Transfer' -> 'Transfers and Payments > Failed Transfer'.

    Conservative by design: a sub-route name is only expanded when it appears
    under exactly one top-level route (or can be disambiguated by the decision's
    own top_level_route). Ambiguous or unknown labels are left unchanged - never
    invented.

    Returns a shallow copy of `decision` with a possibly-updated final_route,
    plus metadata: route_normalized (bool), original_final_route,
    normalized_final_route.
    """
    result = dict(decision)
    original = decision.get("final_route") or ""
    result["original_final_route"] = original

    cleaned = _clean_route_string(original)
    candidate = cleaned

    if routes and cleaned:
        _, allowed_final = _extract_allowed_routes(routes)
        if cleaned not in allowed_final:
            leaf = cleaned.split(">")[-1].strip()
            paths = _build_leaf_index(routes).get(leaf.lower(), [])
            if len(paths) == 1:
                candidate = paths[0]
            elif len(paths) > 1:
                # Disambiguate using the decision's own top_level_route, if it fits.
                top = (decision.get("top_level_route") or "").strip()
                matches = [p for p in paths if p.split(">")[0].strip() == top]
                if len(matches) == 1:
                    candidate = matches[0]
                # else: still ambiguous -> leave cleaned value unchanged

        # When the resolved final is a real full path, align top_level_route to
        # its parent. The leaf uniquely determines the top-level, so this fixes a
        # missing/mismatched top_level_route without inventing anything.
        if " > " in candidate and candidate in allowed_final:
            result["top_level_route"] = candidate.split(" > ")[0].strip()

    result["final_route"] = candidate
    result["normalized_final_route"] = candidate
    result["route_normalized"] = candidate != original
    return result


def route_with_ollama(
    transcript: str,
    routes: dict | None = None,
    model: str = "llama3.2",
    prompt_mode: str = DEFAULT_PROMPT_MODE,
) -> dict:
    """Send a transcript to Ollama and get back a structured routing decision.

    prompt_mode:
    - "compact": short prompt for faster classification. The customer reply is
      generated locally via generate_template_response() and inserted into
      parsed_output["response_to_customer"] for backward compatibility.
    - "compact_v2" (default): like compact, plus a few targeted clarification/
      safety rules for the AICC stress cases (e.g. asking for clarification when
      a payment type is unspecified). Same minimal fields and template reply.
    - "full": the original longer prompt where the model also writes the reply,
      confidence, and explanation. Kept for comparison/debugging.

    Returns a dict with: parsed_output, raw_output, provider, model,
    latency_seconds, route_tree_used, route_valid, prompt_mode, error
    (None if successful).

    route_valid is:
    - None if no route tree was provided (nothing to validate against)
    - True/False if a route tree was provided, based on an exact match check
      against the route tree's top-level and final route names.

    Invalid routes are never silently replaced or corrected — parsed_output
    and raw_output always reflect exactly what the model returned, so
    failures can be inspected.
    """
    if prompt_mode not in VALID_PROMPT_MODES:
        prompt_mode = DEFAULT_PROMPT_MODE

    result = {
        "parsed_output": None,
        "raw_output": "",
        "provider": "ollama",
        "model": model,
        "latency_seconds": 0.0,
        "route_tree_used": bool(routes),
        "route_valid": None,
        "route_normalized": False,
        "original_final_route": "",
        "normalized_final_route": "",
        "prompt_mode": prompt_mode,
        "error": None,
    }

    if not transcript.strip():
        result["error"] = "No transcript/text provided to route."
        return result

    if prompt_mode == "full":
        prompt = _build_prompt(transcript, routes)
    elif prompt_mode == "compact_v2":
        prompt = _build_compact_v2_prompt(transcript, routes)
    else:
        prompt = _build_compact_prompt(transcript, routes)
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

    required_fields = COMPACT_REQUIRED_FIELDS if prompt_mode in COMPACT_MODES else FULL_REQUIRED_FIELDS
    missing_fields = [field for field in required_fields if field not in parsed_output]
    if missing_fields:
        result["error"] = f"Model JSON is missing expected fields: {', '.join(missing_fields)}"

    result["parsed_output"] = parsed_output
    # route_tree_used reflects what we actually gave the model, not the model's own claim.
    parsed_output["route_tree_used"] = bool(routes)

    # Normalize the LLM's final_route (fix '->', whitespace, sub-route-only labels)
    # to an exact full path before generating the reply or validating.
    normalized = normalize_route_decision(parsed_output, routes)
    parsed_output["final_route"] = normalized["final_route"]
    result["route_normalized"] = normalized["route_normalized"]
    result["original_final_route"] = normalized["original_final_route"]
    result["normalized_final_route"] = normalized["normalized_final_route"]

    # Compact modes don't ask the model for a customer reply, so build one from
    # templates and store it under the same key existing TTS/Flask code expects.
    if prompt_mode in COMPACT_MODES and not parsed_output.get("response_to_customer"):
        parsed_output["response_to_customer"] = generate_template_response(parsed_output, transcript)

    if routes:
        allowed_top_level, allowed_final = _extract_allowed_routes(routes)
        result["route_valid"] = _validate_routes(parsed_output, allowed_top_level, allowed_final)

    return result
