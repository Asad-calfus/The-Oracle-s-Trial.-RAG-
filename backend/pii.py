import logging
import re

logger = logging.getLogger(__name__)

# Tier 1 only (PLANNING.md 19.1): structured, near-universally-not-the-
# actual-answer PII. Deliberately does NOT touch names/addresses — this
# project's own test history includes a legitimate lookup like "What was
# Ajinkya's role at Airports Authority of India?", and redacting names
# would break exactly that kind of core functionality.
PII_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    # Requires an actual separator (space/dash/dot) BETWEEN digit groups —
    # deliberately does not match a bare run of digits, so plain numbers in
    # a table (e.g. a currency amount like "15,000" or a chunk count) don't
    # get mistaken for a phone number.
    "PHONE": re.compile(
        r"(?<!\d)(?:\+\d{1,3}[-.\s]?)?(?:\(\d{2,4}\)[-.\s]?)?"
        r"\d{3,4}[-.\s]\d{3,4}(?:[-.\s]?\d{2,4})?(?!\d)"
    ),
    # Indian PAN card format: 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F).
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    # Indian Aadhaar number: 12 digits, in the commonly-written 4-4-4 grouping.
    "AADHAAR": re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b"),
    # Credit-card-like numbers, requiring the standard 4-digit grouping
    # (with a separator) rather than any bare 13-16 digit run.
    "CREDIT_CARD": re.compile(r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{1,4}\b"),
}

# Order matters: more specific patterns first, so e.g. a PAN-like string
# isn't first chewed up by the looser PHONE/CREDIT_CARD patterns.
PATTERN_ORDER = ["EMAIL", "PAN", "AADHAAR", "CREDIT_CARD", "PHONE"]


def redact_text(text: str) -> tuple[str, int]:
    """Mask structured PII in text, returning (redacted_text, count).

    Runs on every page's text at ingestion, before chunking/embedding —
    once redacted here, the original value can never appear in a chunk,
    an embedding, an answer, or a citation excerpt.
    """
    redacted = text
    total = 0

    for label in PATTERN_ORDER:
        pattern = PII_PATTERNS[label]
        redacted, count = pattern.subn(f"[REDACTED_{label}]", redacted)
        total += count

    if total:
        logger.debug("redact_text: made %d redaction(s)", total)

    return redacted, total
