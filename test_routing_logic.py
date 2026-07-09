#!/usr/bin/env python3
"""
Offline tests for the routing logic: route normalization and the pre-router.

These tests are pure string/dict logic - they do NOT require Google, Ollama,
STT, or TTS. Run directly:

    python test_routing_logic.py

Exits non-zero if any assertion fails.
"""

import json
import sys
from pathlib import Path

from src.llm_ollama import normalize_route_decision, validate_route
from src.pre_router import pre_route

ROUTES = json.loads((Path(__file__).resolve().parent / "config" / "routes.json").read_text())

_failures = []


def check(label, got, expected):
    ok = got == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"        expected: {expected!r}")
        print(f"        got:      {got!r}")
        _failures.append(label)
    return ok


def norm(final_route, top_level_route=""):
    decision = {"top_level_route": top_level_route, "final_route": final_route}
    return normalize_route_decision(decision, ROUTES)


def rule_of(transcript):
    d = pre_route(transcript)
    return d.get("rule_name") if d.get("handled") else "no_rule_match"


def test_normalization():
    print("Route normalization:")

    # 1. Bare sub-route -> full path.
    r = norm("Failed Transfer")
    check("1. 'Failed Transfer' -> full path", r["final_route"], "Transfers and Payments > Failed Transfer")
    check("1. route_normalized flag is True", r["route_normalized"], True)

    # 2. Another bare sub-route.
    r = norm("Scheduled Transfer")
    check("2. 'Scheduled Transfer' -> full path", r["final_route"], "Transfers and Payments > Scheduled Transfer")

    # 3. Arrow + whitespace variant.
    r = norm("Transfers and Payments -> Failed Transfer")
    check("3. '-> ' arrow normalized to ' > '", r["final_route"], "Transfers and Payments > Failed Transfer")

    # Extra: already-correct route is left unchanged and not flagged.
    r = norm("Card Services > Lost or Stolen Card")
    check("extra: exact route unchanged", r["final_route"], "Card Services > Lost or Stolen Card")
    check("extra: route_normalized False when unchanged", r["route_normalized"], False)

    # Extra: blank final_route (clarify) stays blank and valid.
    r = norm("")
    check("extra: blank final_route stays blank", r["final_route"], "")
    check("extra: normalized clarify still route_valid", validate_route({"top_level_route": "Card Services", "final_route": r["final_route"]}, ROUTES), True)

    # Extra: unknown leaf is left unchanged (never invented).
    r = norm("Totally Made Up Route")
    check("extra: unknown leaf unchanged", r["final_route"], "Totally Made Up Route")

    # Extra: normalized routes pass validation.
    check("extra: normalized 'Failed Transfer' is route_valid",
          validate_route(norm("Failed Transfer"), ROUTES), True)


def test_pre_router():
    print("Pre-router rules:")

    # 4. Vague payment stays clarify (falls under vague_payment_issue), not routed.
    check("4a. 'my payment did not go through' -> vague_payment_issue",
          rule_of("My payment did not go through"), "vague_payment_issue")
    d = pre_route("My payment did not go through")
    check("4b. vague payment action is clarify", d.get("action"), "clarify")

    # 5. Clear transfer-from-checking-to-savings failure -> Failed Transfer.
    check("5a. transfer checking->savings that failed -> failed_transfer_clear",
          rule_of("I tried to transfer money from checking to savings but it did not go through"),
          "failed_transfer_clear")
    check("5b. send money checking->savings that failed -> failed_transfer_clear",
          rule_of("I was trying to send money from checking to savings and it failed"),
          "failed_transfer_clear")

    # 6. Scheduled transfer + change intent -> Scheduled Transfer.
    check("6a. 'scheduled a transfer ... need to change it' -> scheduled_transfer_change",
          rule_of("I scheduled a transfer for July fifteenth and need to change it"),
          "scheduled_transfer_change")
    check("6b. 'transfer scheduled for Friday ... cancel it' -> scheduled_transfer_change",
          rule_of("I have a transfer scheduled for Friday and want to cancel it"),
          "scheduled_transfer_change")

    # 7. Lost card + fraud terms -> Fraud (not Lost Card).
    check("7. lost card + 'purchases I did not make' -> fraud_or_suspicious",
          rule_of("I lost my card and there are purchases I did not make"),
          "fraud_or_suspicious")
    check("7b. 'someone used my card' -> fraud_or_suspicious",
          rule_of("Someone used my card without my permission"), "fraud_or_suspicious")

    # 8. 'card declined' does NOT trigger the vague card issue rule.
    got = rule_of("My card was declined at the store")
    check("8. 'card declined' is not vague_card_issue", got != "vague_card_issue", True)

    # Guard: short fraud cue 'not me' must not false-match inside 'do not mention'.
    check("guard: 'do not mention' does not match fraud",
          rule_of("I did not mention my new address yet"), "no_rule_match")

    # Guard: plain lost card (no fraud) still routes to lost card.
    check("guard: plain lost card -> lost_or_stolen_card",
          rule_of("I lost my debit card yesterday"), "lost_or_stolen_card")


def main():
    test_normalization()
    test_pre_router()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        sys.exit(1)
    print("All routing logic tests passed.")


if __name__ == "__main__":
    main()
