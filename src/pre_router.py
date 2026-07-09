"""
Rule-based fast pre-router that runs before the Ollama LLM.

Goal: skip the ~2.5s LLM call for obvious, high-confidence customer requests
by matching simple lowercase keywords/phrases. Rules are intentionally
conservative - when in doubt, pre_route() returns handled=False and the
pipeline falls back to the existing compact_v2 Ollama router.

This module never calls Ollama or any network service; it is pure string
matching, so it adds effectively no latency.
"""

import re

from src.llm_ollama import generate_template_response


def _contains_any(text: str, terms) -> bool:
    return any(term in text for term in terms)


def _matches_any_word(text: str, patterns) -> bool:
    """Word-boundary match, for short phrases where substring matching is unsafe
    (e.g. plain 'not me' must not match inside 'do not mention')."""
    return any(re.search(p, text) for p in patterns)


# --- Rule keyword sets (all lowercase) --------------------------------------

_FRAUD_TERMS = [
    "suspicious charge",
    "suspicious transaction",
    "unauthorized",
    "unrecognized",
    "do not recognize",
    "don't recognize",
    "didn't make",
    "did not make",
    "fraud",
    "purchase i did not make",
    "purchases i did not make",
    "purchases that i did not make",
    "charge i did not make",
    "charges i did not make",
    "charges i didn't make",
    "someone used my card",
    "someone has been using my card",
    "someone is using my card",
]

# Short fraud cues matched on word boundaries to avoid false positives such as
# "not me" inside "do not mention".
_FRAUD_WORD_PATTERNS = [
    r"\bnot me\b",
    r"\bwasn't me\b",
    r"\bwas not me\b",
]

_LOST_CARD_TERMS = [
    "lost my card",
    "lost my debit card",
    "lost my credit card",
    "stolen card",
    "card was stolen",
    "card has been stolen",
    "card got stolen",
]

_LOCKED_ACCOUNT_TERMS = [
    "account is locked",
    "account was locked",
    "account got locked",
    "locked out",
    "logging in too many times",
    "too many login attempts",
    "too many times",
]

_ATM_CASH_MISSING_TERMS = [
    "did not give me the money",
    "didn't give me the money",
    "did not dispense",
    "didn't dispense",
    "no cash",
    "account was charged",
    "charged my account",
]

# References to a scheduled transfer (any of these phrasings).
_SCHEDULED_TRANSFER_REF_TERMS = [
    "scheduled transfer",
    "scheduled a transfer",
    "scheduled my transfer",
    "transfer scheduled for",
    "scheduled transfer for",
]

# Change/cancel intent. "need to change it" / "want to cancel it" are covered by
# the "change" / "cancel" substrings.
_SCHEDULED_TRANSFER_ACTION_TERMS = ["change", "cancel", "modify"]

# Phrases that clearly say a transfer failed (contain both "transfer" and a
# failure), enough on their own to route.
_FAILED_TRANSFER_EXPLICIT_TERMS = [
    "transfer failed",
    "failed transfer",
    "transfer did not go through",
    "transfer didn't go through",
]

# Generic failure phrases that only route to Failed Transfer when combined with
# clear transfer/send-money context (below).
_GENERIC_FAILURE_TERMS = [
    "did not go through",
    "didn't go through",
    "it failed",
    "money did not go through",
    "money didn't go through",
]

# Clear transfer / send-money context.
_TRANSFER_CONTEXT_TERMS = [
    "checking to savings",
    "from checking to savings",
    "savings to checking",
    "checking to my savings",
    "checking account to my savings account",
    "transfer from checking",
    "transfer to savings",
    "bank transfer",
    "transfer money",
    "send money",
    "sending money",
]

_VAGUE_CARD_TERMS = [
    "something weird with my card",
    "something wrong with my card",
    "card issue",
]

# Specific card keywords that mean the request is NOT vague (so rule 7 should
# not fire and the more specific rule/LLM should handle it).
_CARD_SPECIFIC_TERMS = [
    "fraud",
    "unauthorized",
    "suspicious",
    "lost",
    "stolen",
    "declined",
    "not working",
    "activate",
    "activation",
    "pin",
]

_VAGUE_PAYMENT_TERMS = [
    "payment did not go through",
    "payment didn't go through",
    "payment failed",
]

# If any of these appear, the payment request is specific enough that rule 8
# should not fire.
_PAYMENT_SPECIFIC_TERMS = [
    "transfer",
    "bill payment",
    "scheduled",
    "person-to-person",
    "person to person",
    "card",
    "debit",
    "credit",
]

_OUT_OF_SCOPE_TERMS = [
    "book a flight",
    "flight to",
    "book a hotel",
    "hotel reservation",
    "order food",
    "food delivery",
]


def _decision(top_level, final_route, action, response_type, rule_name, confidence, reason):
    """Build a handled decision dict, filling response_to_customer from templates."""
    decision = {
        "handled": True,
        "top_level_route": top_level,
        "final_route": final_route,
        "action": action,
        "response_type": response_type,
        "rule_name": rule_name,
        "confidence": confidence,
        "reason": reason,
    }
    # Reuse the shared template logic so wording matches the LLM compact path.
    decision["response_to_customer"] = generate_template_response(
        {"action": action, "final_route": final_route}
    )
    return decision


def pre_route(transcript: str, routes: dict | None = None) -> dict:
    """Try to route a transcript using conservative keyword rules.

    Returns a decision dict with handled=True when a high-confidence rule
    matches, otherwise {"handled": False, "rule_name": "no_rule_match",
    "confidence": 0.0}. `routes` is accepted for signature compatibility and
    possible future use; matching itself does not depend on it.
    """
    text = (transcript or "").lower()

    # Rule 1: Fraud / suspicious / unauthorized. Checked first so it wins over
    # the lost-card rule when both fraud and lost-card terms are present.
    if _contains_any(text, _FRAUD_TERMS) or _matches_any_word(text, _FRAUD_WORD_PATTERNS):
        return _decision(
            "Card Services", "Card Services > Fraud or Suspicious Transaction",
            "escalate", "escalate", "fraud_or_suspicious", 0.9,
            "Matched fraud/unauthorized/suspicious terms.",
        )

    # Rule 2: Lost/stolen card without fraud keywords.
    if _contains_any(text, _LOST_CARD_TERMS):
        return _decision(
            "Card Services", "Card Services > Lost or Stolen Card",
            "route", "route", "lost_or_stolen_card", 0.9,
            "Matched lost/stolen card terms with no fraud terms.",
        )

    # Rule 3: Locked account.
    if _contains_any(text, _LOCKED_ACCOUNT_TERMS):
        return _decision(
            "Account Access", "Account Access > Locked Account",
            "route", "route", "locked_account", 0.9,
            "Matched locked-account / too-many-login terms.",
        )

    # Rule 4: ATM charged the account but did not dispense cash.
    if "atm" in text and _contains_any(text, _ATM_CASH_MISSING_TERMS):
        return _decision(
            "Deposits and ATM", "Deposits and ATM > ATM Withdrawal Issue",
            "route", "route", "atm_no_cash", 0.9,
            "Matched ATM plus cash-not-dispensed / account-charged terms.",
        )

    # Rule 5: Scheduled transfer, with a change/cancel/modify intent.
    if _contains_any(text, _SCHEDULED_TRANSFER_REF_TERMS) and _contains_any(text, _SCHEDULED_TRANSFER_ACTION_TERMS):
        return _decision(
            "Transfers and Payments", "Transfers and Payments > Scheduled Transfer",
            "route", "route", "scheduled_transfer_change", 0.85,
            "Matched scheduled transfer reference plus change/cancel/modify.",
        )

    # Rule 6: Clear failed transfer. Either an explicit "transfer failed" phrase,
    # or a generic failure phrase together with clear transfer/send-money context.
    # Vague "payment did not go through" (no transfer context) is intentionally
    # left to Rule 8 -> clarify.
    explicit_failed = _contains_any(text, _FAILED_TRANSFER_EXPLICIT_TERMS)
    generic_failed_with_context = (
        _contains_any(text, _GENERIC_FAILURE_TERMS) and _contains_any(text, _TRANSFER_CONTEXT_TERMS)
    )
    if explicit_failed or generic_failed_with_context:
        return _decision(
            "Transfers and Payments", "Transfers and Payments > Failed Transfer",
            "route", "route", "failed_transfer_clear", 0.85,
            "Matched failed-transfer phrase or generic failure with transfer context.",
        )

    # Rule 7: Vague card issue (no specific card keywords) -> clarify.
    if _contains_any(text, _VAGUE_CARD_TERMS) and not _contains_any(text, _CARD_SPECIFIC_TERMS):
        return _decision(
            "Card Services", "", "clarify", "clarify", "vague_card_issue", 0.6,
            "Vague card complaint with no specific card issue mentioned.",
        )

    # Rule 8: Vague payment issue (no specific payment type) -> clarify.
    if _contains_any(text, _VAGUE_PAYMENT_TERMS) and not _contains_any(text, _PAYMENT_SPECIFIC_TERMS):
        return _decision(
            "Transfers and Payments", "", "clarify", "clarify", "vague_payment_issue", 0.6,
            "Vague payment complaint with no specific payment type mentioned.",
        )

    # Rule 9: Clearly out-of-scope, non-banking requests.
    if _contains_any(text, _OUT_OF_SCOPE_TERMS):
        return _decision(
            "Out of Scope", "Out of Scope > Unsupported Service",
            "offer_human_or_fallback", "fallback", "out_of_scope", 0.85,
            "Matched clearly non-banking out-of-scope request.",
        )

    # No confident rule: let the LLM handle it.
    return {"handled": False, "rule_name": "no_rule_match", "confidence": 0.0}
