"""Unit tests for the eval-set redactor (Plan 3 Lane A — A3.5).

Covers:
- allowlist precedence (fields outside allowlist are dropped)
- NER scrub of free-text PII (PERSON, GPE, ORG)
- Presidio scrub of structured PII (SSN, credit card, email, IP)
- allowlist-but-PII-inside (the combined ``redact`` path runs all three)
- non-string leaves pass through unchanged
- malformed allowlist entries are skipped, not crashing
"""

from __future__ import annotations

import pytest
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from sampler.redactor import (
    apply_allowlist,
    redact,
    scrub_payload_strings,
    scrub_pii,
    scrub_text,
    set_analyzer_for_testing,
    set_nlp_for_testing,
)


@pytest.fixture(scope="module")
def nlp():
    """Load en_core_web_sm once per test module."""
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def analyzer():
    """Load Presidio AnalyzerEngine once per test module (~1 s cold start).

    Pinned to en_core_web_sm so test setup never tries to download
    en_core_web_lg (~400 MB) — that download blew the 60s CI timeout
    on clean runners.
    """
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    return AnalyzerEngine(nlp_engine=provider.create_engine())


@pytest.fixture(autouse=True)
def _inject_nlp(nlp, analyzer):
    """Inject both spaCy + Presidio so the redactor uses test-scoped instances."""
    set_nlp_for_testing(nlp)
    set_analyzer_for_testing(analyzer)
    yield
    set_nlp_for_testing(None)
    set_analyzer_for_testing(None)


# ---------- allowlist ----------


def test_apply_allowlist_drops_unlisted_fields():
    payload = {"keep": "ok", "drop": "secret", "also_drop": {"nested": 1}}
    out = apply_allowlist(payload, ["$.keep"])
    assert out == {"keep": "ok"}


def test_apply_allowlist_empty_drops_everything():
    assert apply_allowlist({"a": 1, "b": 2}, []) == {}


def test_apply_allowlist_preserves_nested_path():
    payload = {"user": {"name": "Alice", "ssn": "123-45-6789"}, "msg": "hi"}
    out = apply_allowlist(payload, ["$.user.name", "$.msg"])
    assert out == {"user": {"name": "Alice"}, "msg": "hi"}
    assert "ssn" not in out["user"]


def test_apply_allowlist_array_index():
    payload = {"items": [{"title": "first"}, {"title": "second", "secret": "x"}]}
    out = apply_allowlist(payload, ["$.items[*].title"])
    assert out == {"items": [{"title": "first"}, {"title": "second"}]}


def test_apply_allowlist_malformed_path_is_skipped():
    out = apply_allowlist({"k": "v"}, ["$.k", "this is not a path["])
    assert out == {"k": "v"}


# ---------- scrub_text ----------


def test_scrub_text_redacts_person():
    out = scrub_text("Hello Barack Obama, please review.")
    assert "Barack Obama" not in out
    assert "[REDACTED:PERSON]" in out


def test_scrub_text_redacts_place():
    out = scrub_text("Meeting in Paris next week.")
    assert "Paris" not in out
    assert "[REDACTED:GPE]" in out


def test_scrub_text_passthrough_non_pii():
    text = "the quick brown fox jumps"
    assert scrub_text(text) == text


def test_scrub_text_handles_empty_and_non_string():
    assert scrub_text("") == ""
    # Non-string leaves pass through unchanged — the walker uses this
    # for generic recursion over arbitrary JSON-shaped inputs.
    assert scrub_text(None) is None
    assert scrub_text(42) == 42


# ---------- scrub_payload_strings ----------


def test_scrub_payload_strings_walks_nested_dict():
    payload = {
        "user": {"note": "Barack Obama lives in Tokyo"},
        "items": [{"label": "the quick brown fox"}, {"label": "Contact Microsoft"}],
        "count": 5,
    }
    out = scrub_payload_strings(payload)
    assert out["count"] == 5
    assert "Barack Obama" not in out["user"]["note"]
    assert "Tokyo" not in out["user"]["note"]
    assert "Microsoft" not in out["items"][1]["label"]
    assert out["items"][0]["label"] == "the quick brown fox"


# ---------- redact (combined entry point) ----------


def test_redact_allowlists_then_scrubs():
    payload = {
        "message": "Barack Obama spoke in London yesterday.",
        "ssn": "123-45-6789",
        "credit_card": "4111-1111-1111-1111",
    }
    out = redact(payload, ["$.message"])
    # ssn / credit_card dropped by allowlist
    assert "ssn" not in out
    assert "credit_card" not in out
    # message survived but PII inside is scrubbed
    assert "Barack Obama" not in out["message"]
    assert "London" not in out["message"]
    assert "[REDACTED:" in out["message"]


def test_redact_no_allowlist_returns_empty():
    out = redact({"anything": "Alice in Paris"}, [])
    assert out == {}


# ---------- scrub_pii (Presidio) ----------


def test_scrub_pii_redacts_credit_card():
    # 4111-1111-1111-1111 is a Luhn-valid Visa test number — Presidio's
    # CreditCardRecognizer applies Luhn validation, so an invalid number
    # would silently pass through.
    out = scrub_pii("Charge to 4111-1111-1111-1111 today.")
    assert "4111" not in out
    assert "[REDACTED:CREDIT_CARD]" in out


def test_scrub_pii_redacts_us_ssn():
    # 100-23-4567: verified-detected by UsSsnRecognizer (some test SSNs
    # like 123-45-6789 are rejected by Presidio's area-number heuristics).
    out = scrub_pii("Has SSN 100-23-4567 on file")
    assert "100-23-4567" not in out
    assert "[REDACTED:US_SSN]" in out


def test_scrub_pii_redacts_email():
    out = scrub_pii("Reach me at jane.doe@example.com please.")
    assert "jane.doe@example.com" not in out
    assert "[REDACTED:EMAIL_ADDRESS]" in out


def test_scrub_pii_redacts_ip_address():
    out = scrub_pii("Origin IP was 192.168.7.42 last night.")
    assert "192.168.7.42" not in out
    assert "[REDACTED:IP_ADDRESS]" in out


def test_scrub_pii_passthrough_clean_text():
    text = "the quick brown fox jumps over"
    assert scrub_pii(text) == text


def test_scrub_pii_handles_non_string():
    assert scrub_pii(None) is None
    assert scrub_pii(42) == 42


# ---------- combined: spaCy + Presidio ----------


def test_redact_combines_ner_and_pii():
    """End-to-end: allowlist passes a free-text field that contains BOTH a
    named entity (PERSON) and structured PII (credit card). Both passes
    must fire — neither should be left in the persisted payload."""
    payload = {
        "message": "Barack Obama paid with 4111-1111-1111-1111 in Paris.",
        "ssn_field": "100-23-4567",
    }
    out = redact(payload, ["$.message"])
    assert "ssn_field" not in out  # dropped by allowlist
    msg = out["message"]
    assert "Barack Obama" not in msg  # spaCy
    assert "Paris" not in msg  # spaCy
    assert "4111" not in msg  # Presidio
    assert "[REDACTED:PERSON]" in msg or "[REDACTED:GPE]" in msg
    assert "[REDACTED:CREDIT_CARD]" in msg
