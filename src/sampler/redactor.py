"""Per-stream payload redactor for the eval-set sampler.

Three stages, applied in order to every prod-sampled execution row:

1. ``apply_allowlist`` — copy ONLY the JSONPath-matched fields from the
   raw payload into a fresh dict. Anything not in the allowlist is dropped.
   This is the primary defence; the allowlist is per-stream and operator-
   editable.

2. ``scrub_text`` (spaCy NER) — for free-text values that survived (1),
   run spaCy NER and replace named entities (PERSON, GPE, LOC, ORG) with
   ``[REDACTED:<label>]``. Catches names, places, organisations.

3. ``scrub_pii`` (Presidio) — secondary structured-PII pass on the same
   text. Catches the patterns spaCy misses: SSN, credit card, IBAN, IP
   address, US passport / driver licence / ITIN, crypto wallet, etc.
   Belt-and-braces — anything either pass matches gets redacted before
   persistence.

All three stages run before an ``eval_tuples`` row is written, and the
row is flagged ``redaction_review_required = true`` so an operator can
clear it in the UI before the tuple becomes usable.

Design spec §9.1.
"""

from __future__ import annotations

import threading
from typing import Any

from jsonpath_ng.exceptions import JSONPathError
from jsonpath_ng.ext import parse as jsonpath_parse

# Entity labels we always scrub from free text. spaCy's en_core_web_sm
# emits these labels for named entities (people, places, orgs).
# Structured-PII patterns (SSN, credit card, …) are handled by Presidio
# in scrub_pii — keeping the responsibility split clean.
_DEFAULT_NER_LABELS: frozenset[str] = frozenset({"PERSON", "GPE", "LOC", "ORG"})

# Presidio entity types to redact. Covers structured PII spaCy misses.
# EMAIL_ADDRESS + PHONE_NUMBER lightly overlap with spaCy but Presidio's
# regex+context recognisers catch shapes spaCy's NER doesn't.
_DEFAULT_PRESIDIO_ENTITIES: frozenset[str] = frozenset(
    {
        "CREDIT_CARD",
        "CRYPTO",
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        "PHONE_NUMBER",
        "US_BANK_NUMBER",
        "US_DRIVER_LICENSE",
        "US_ITIN",
        "US_PASSPORT",
        "US_SSN",
    }
)

# Lazy module-level caches: both spaCy and Presidio are heavy on
# cold-start. We hold one instance per process; tests inject their own
# via the ``*_for_testing`` setters.
_nlp_lock = threading.Lock()
_nlp: Any | None = None

_analyzer_lock = threading.Lock()
_analyzer: Any | None = None


def get_nlp() -> Any:
    """Return the process-wide spaCy ``nlp`` pipeline (lazy-loaded).

    The first call on a process pays the model-load cost (~200 ms for
    ``en_core_web_sm``); subsequent calls are O(1).
    """
    global _nlp
    if _nlp is not None:
        return _nlp
    with _nlp_lock:
        if _nlp is None:
            import spacy  # noqa: PLC0415 — defer heavy import until first use

            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def set_nlp_for_testing(nlp: Any | None) -> None:
    """Override the cached spaCy pipeline. Tests only."""
    global _nlp
    _nlp = nlp


def get_analyzer() -> Any:
    """Return the process-wide Presidio ``AnalyzerEngine`` (lazy-loaded).

    Cold-start cost is ~1 s (loads its own NLP engine + recognizer
    registry). Subsequent calls are O(1).

    Pins Presidio's NLP engine to ``en_core_web_sm`` (the same spaCy
    model the rest of this module uses and a project dependency).
    Without this pin Presidio's default constructor tries to download
    ``en_core_web_lg`` (~400 MB) on first use, which hangs CI runners
    on a clean cache.
    """
    global _analyzer
    if _analyzer is not None:
        return _analyzer
    with _analyzer_lock:
        if _analyzer is None:
            from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415
            from presidio_analyzer.nlp_engine import NlpEngineProvider  # noqa: PLC0415

            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
            )
            _analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
    return _analyzer


def set_analyzer_for_testing(analyzer: Any | None) -> None:
    """Override the cached Presidio analyzer. Tests only."""
    global _analyzer
    _analyzer = analyzer


def apply_allowlist(payload: dict[str, Any], allowlist: list[str]) -> dict[str, Any]:
    """Copy only allowlisted JSONPaths from ``payload`` into a fresh dict.

    Empty allowlist returns an empty dict (drop everything). Each allowlist
    entry is a JSONPath expression evaluated against ``payload``; matched
    values are placed into the output under the same path. Paths that don't
    match are silently skipped.

    Output structure mirrors the matched paths — e.g. allowlisting
    ``$.user.name`` and ``$.message`` produces
    ``{"user": {"name": ...}, "message": ...}``.
    """
    if not allowlist:
        return {}
    out: dict[str, Any] = {}
    for path_expr in allowlist:
        try:
            expr = jsonpath_parse(path_expr)
        except ValueError, TypeError, AttributeError, SyntaxError, JSONPathError:
            # malformed allowlist entry; jsonpath-ng raises a mix of these
            continue
        for match in expr.find(payload):
            # jsonpath-ng's match.full_path is itself a JSONPath expression
            # that knows how to materialise the right shape (dict-vs-list)
            # into a target. Use it instead of stringifying + reparsing.
            match.full_path.update_or_create(out, match.value)
    return out


def scrub_text(
    text: Any,
    *,
    nlp: Any | None = None,
    labels: frozenset[str] = _DEFAULT_NER_LABELS,
) -> Any:
    """Replace PII spans in ``text`` with ``[REDACTED:<label>]``.

    Uses spaCy NER (``en_core_web_sm`` by default). Spans are replaced
    right-to-left so that span offsets stay valid during substitution.
    Non-string inputs (None, numbers, etc.) are returned unchanged so
    the function can be used inside generic walkers without type juggling.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    pipeline = nlp if nlp is not None else get_nlp()
    doc = pipeline(text)
    spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ in labels]
    if not spans:
        return text
    spans.sort(reverse=True)
    out = text
    for start, end, label in spans:
        out = f"{out[:start]}[REDACTED:{label}]{out[end:]}"
    return out


def scrub_pii(
    text: Any,
    *,
    analyzer: Any | None = None,
    entities: frozenset[str] = _DEFAULT_PRESIDIO_ENTITIES,
    language: str = "en",
) -> Any:
    """Replace structured-PII spans in ``text`` with ``[REDACTED:<label>]``.

    Uses Presidio's ``AnalyzerEngine`` to detect SSN, credit-card, IBAN,
    IP, phone, US passport / driver licence / ITIN, crypto wallet etc.
    Spans replaced right-to-left so offsets remain valid. Non-string
    inputs are returned unchanged.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    engine = analyzer if analyzer is not None else get_analyzer()
    results = engine.analyze(text=text, entities=list(entities), language=language)
    if not results:
        return text
    # Presidio returns RecognizerResult with .start, .end, .entity_type.
    spans = [(r.start, r.end, r.entity_type) for r in results]
    spans.sort(reverse=True)
    out = text
    for start, end, label in spans:
        out = f"{out[:start]}[REDACTED:{label}]{out[end:]}"
    return out


def scrub_payload_strings(
    payload: dict[str, Any],
    *,
    nlp: Any | None = None,
    analyzer: Any | None = None,
    labels: frozenset[str] = _DEFAULT_NER_LABELS,
    presidio_entities: frozenset[str] = _DEFAULT_PRESIDIO_ENTITIES,
) -> dict[str, Any]:
    """Walk ``payload`` and apply both NER + Presidio scrubs to every
    string leaf.

    Order: spaCy first (catches names/places), Presidio second (catches
    structured patterns). Both passes work — anything either marks as PII
    is redacted before the value lands in ``eval_tuples``. Returns a new
    dict; the input is not mutated.
    """
    pipeline = nlp if nlp is not None else get_nlp()
    engine = analyzer if analyzer is not None else get_analyzer()
    return _walk(payload, pipeline, engine, labels, presidio_entities)


def _walk(
    node: Any,
    pipeline: Any,
    engine: Any,
    labels: frozenset[str],
    presidio_entities: frozenset[str],
) -> Any:
    if isinstance(node, dict):
        return {k: _walk(v, pipeline, engine, labels, presidio_entities) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, pipeline, engine, labels, presidio_entities) for v in node]
    if isinstance(node, str):
        # spaCy first — catches named entities; Presidio second — catches
        # structured patterns. The double-pass is intentional, see module
        # docstring.
        ner_scrubbed = scrub_text(node, nlp=pipeline, labels=labels)
        return scrub_pii(ner_scrubbed, analyzer=engine, entities=presidio_entities)
    return node


def redact(
    payload: dict[str, Any],
    allowlist: list[str],
    *,
    nlp: Any | None = None,
    analyzer: Any | None = None,
) -> dict[str, Any]:
    """Run all three stages: allowlist filter → spaCy NER → Presidio PII.

    The conventional entry point used by the sampler.
    """
    filtered = apply_allowlist(payload, allowlist)
    return scrub_payload_strings(filtered, nlp=nlp, analyzer=analyzer)
