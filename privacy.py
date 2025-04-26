"""On-device PII redaction for the draft-generation step.

The privacy-preserving contract pipeline is built on a simple invariant: any
content that identifies a real party or quotes a real financial figure must
not leave the user's machine.  Refinement and validation already run on local
Ollama models, but the draft-generation call to Gemini was historically the
weak link - it sent the full `user_inputs` dict, including party names and
fees, to the cloud LLM.

This module fixes that by interposing a deterministic, field-aware redactor:

    redacted, token_map = redact_inputs(user_inputs, contract_type)
    # ...send only `redacted` to Gemini, never `user_inputs`...
    restored = restore_pii(gemini_output, token_map)

Tokens are stable for the lifetime of a single generate_contract() call and
are chosen to be obviously non-substantive so the cloud model treats them as
opaque placeholders ("[PARTY_1]", "[AMOUNT_2]", "[ADDRESS_1]").  The full
mapping never leaves the local process.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, Iterable, Tuple

# ---------------------------------------------------------------------------
# Field classification.  Keys come from app.py's *_FIELDS definitions; new
# fields should be added here when introduced so they're redacted by default.
# ---------------------------------------------------------------------------

_PARTY_NAME_FIELDS = frozenset({
    "client_name", "provider_name",
    "employer_name", "employee_name",
    "landlord_name", "tenant_name",
})

_ADDRESS_FIELDS = frozenset({
    "client_address", "provider_address",
    "employer_address", "employee_address",
    "landlord_address", "tenant_address",
    "property_address",
})

_AMOUNT_FIELDS = frozenset({
    "fees", "salary", "bonus",
    "rent_amount", "security_deposit", "late_fees",
})

# Fields whose value is *not* sensitive: dates, generic policy text, free-text
# describing duties / services.  These pass through unchanged.  We never
# attempt to NER-redact inside free-text fields because the local validation
# models need to see the real obligations to evaluate consistency.
_PASSTHROUGH_FIELDS = frozenset({
    "start_date", "end_date", "rent_due_date",
    "position", "duties", "services_description", "property_description",
    "termination", "intellectual_property", "confidentiality",
    "dispute_resolution", "benefits", "non_compete",
    "utilities", "pets_policy", "smoking_policy",
})

# ---------------------------------------------------------------------------
# Regex catch-all for stray PII inside free-text fields.  These run *after*
# field-level redaction so a salary mentioned inside a duties paragraph still
# gets masked.
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?(?:\s?(?:USD|usd))?"
    r"|\b\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?\s?(?:dollars|USD)\b",
    re.IGNORECASE,
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(
    r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


class _TokenIssuer:
    """Issues stable tokens of the form [LABEL_N] for distinct values."""

    def __init__(self) -> None:
        self._by_value: Dict[Tuple[str, str], str] = {}
        self._counters: Dict[str, int] = {}

    def issue(self, label: str, value: str) -> str:
        key = (label, value)
        if key in self._by_value:
            return self._by_value[key]
        self._counters[label] = self._counters.get(label, 0) + 1
        token = f"[{label}_{self._counters[label]}]"
        self._by_value[key] = token
        return token

    def mapping(self) -> Dict[str, str]:
        """token -> original value, ordered by issue order."""
        return {token: value for (_, value), token in self._by_value.items()}


def _classify(field_key: str) -> str | None:
    """Map a user_inputs key to its PII label, or None if not sensitive."""
    if field_key in _PARTY_NAME_FIELDS:
        return "PARTY"
    if field_key in _ADDRESS_FIELDS:
        return "ADDRESS"
    if field_key in _AMOUNT_FIELDS:
        return "AMOUNT"
    return None


def _redact_free_text(text: str, issuer: _TokenIssuer) -> str:
    """Scrub stray money/SSN/phone/email from a free-text field."""
    def replace_money(match: re.Match[str]) -> str:
        return issuer.issue("AMOUNT", match.group(0).strip())

    def replace_ssn(match: re.Match[str]) -> str:
        return issuer.issue("SSN", match.group(0))

    def replace_phone(match: re.Match[str]) -> str:
        return issuer.issue("PHONE", match.group(0))

    def replace_email(match: re.Match[str]) -> str:
        return issuer.issue("EMAIL", match.group(0))

    text = _MONEY_RE.sub(replace_money, text)
    text = _SSN_RE.sub(replace_ssn, text)
    text = _PHONE_RE.sub(replace_phone, text)
    text = _EMAIL_RE.sub(replace_email, text)
    return text


def redact_inputs(
    user_inputs: Dict[str, Any],
    contract_type: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Return (redacted_inputs, token_map).

    The redacted dict is safe to send to a cloud LLM: every party identity,
    address, and financial figure has been replaced with an opaque token.  The
    token_map lets the caller restore originals via :func:`restore_pii` once
    the cloud response is back on-device.
    """
    _ = contract_type  # reserved for future per-type rules
    issuer = _TokenIssuer()
    redacted: Dict[str, Any] = {}

    for key, value in user_inputs.items():
        # Dates and other non-string values pass through.  We stringify dates
        # the same way generate_contract() does so the cloud-bound payload
        # matches what the prompt expects.
        if isinstance(value, date):
            redacted[key] = value.strftime("%Y-%m-%d")
            continue
        if not isinstance(value, str) or not value.strip():
            redacted[key] = value
            continue

        label = _classify(key)
        if label is not None:
            redacted[key] = issuer.issue(label, value.strip())
            continue

        if key in _PASSTHROUGH_FIELDS:
            redacted[key] = _redact_free_text(value, issuer)
            continue

        # Unknown field: be conservative and scrub stray PII regardless.
        redacted[key] = _redact_free_text(value, issuer)

    return redacted, issuer.mapping()


def restore_pii(text: str, token_map: Dict[str, str]) -> str:
    """Replace tokens in ``text`` with their original values.

    Longer tokens are replaced first to avoid `[PARTY_1]` matching inside
    `[PARTY_10]`.  This is a plain string substitution - we deliberately do
    not use regex so the original values (which may contain regex
    metacharacters like ``$`` or ``.``) survive intact.
    """
    if not token_map:
        return text
    for token in sorted(token_map.keys(), key=len, reverse=True):
        text = text.replace(token, token_map[token])
    return text


def describe_redaction(token_map: Dict[str, str]) -> str:
    """Human-readable summary of what got redacted, for logging / UI."""
    if not token_map:
        return "No sensitive values detected; nothing redacted."
    counts: Dict[str, int] = {}
    for token in token_map:
        label = token.strip("[]").split("_", 1)[0]
        counts[label] = counts.get(label, 0) + 1
    parts = [f"{n} {label.lower()}{'s' if n != 1 else ''}"
             for label, n in sorted(counts.items())]
    return "Kept on-device: " + ", ".join(parts) + "."


def iter_sensitive_field_keys() -> Iterable[str]:
    """Iterator over all field keys treated as PII (for tests / docs)."""
    return iter(_PARTY_NAME_FIELDS | _ADDRESS_FIELDS | _AMOUNT_FIELDS)
