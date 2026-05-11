"""
guardrails.py
=============

Four checkpoints that run at different stages of the pipeline.

CHECKPOINT 1 — pre_flight()
    Runs BEFORE retrieval.
    Catches prompt injection, malicious requests, PII, nonsense.
    If triggered → return escalation result immediately, skip everything else.

CHECKPOINT 2 — is_hard_escalation()
    Runs BEFORE retrieval (after pre-flight passes).
    Hardcoded sensitive keywords that must ALWAYS escalate.
    No corpus can help with identity theft or security breaches.

CHECKPOINT 3 — post_retrieval()
    Runs AFTER retrieval, BEFORE Gemini.
    Checks if retrieved chunks are good enough to answer.
    If top cross_score < 0.55 → not enough corpus coverage → escalate.

CHECKPOINT 4 — post_generation()
    Runs AFTER Gemini responds.
    Validates output structure, enum values, hallucinated URLs.
    If invalid → returns cleaned/fixed version or flags for retry.

HOW THESE FIT IN agent.py:
    ticket
      → pre_flight()          # CHECKPOINT 1
      → is_hard_escalation()  # CHECKPOINT 2
      → retrieve()            # retriever.py
      → post_retrieval()      # CHECKPOINT 3
      → gemini_generate()     # agent.py
      → post_generation()     # CHECKPOINT 4
      → output.csv
"""

from __future__ import annotations

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Minimum cross-encoder score to consider retrieval good enough.
# Below this → corpus doesn't cover the ticket well → safer to escalate.
# Why 0.55? Cross-encoder ms-marco scores above 0.5 generally indicate
# meaningful relevance. Below 0.3 is noise. 0.55 is a conservative middle ground.
CONFIDENCE_THRESHOLD = 0.55

# Maximum ticket length we'll process.
# Tickets over 5000 chars are suspicious (likely prompt injection or spam).
MAX_TICKET_LENGTH = 5000

# Valid enum values for the output CSV
VALID_STATUS       = {"replied", "escalated"}
VALID_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 1 — PRE-FLIGHT PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

# Prompt injection attempts — phrases that try to hijack the LLM
# These are exact substring matches (case-insensitive)
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard previous",
    "disregard all previous",
    "forget your instructions",
    "you are now",
    "act as if",
    "pretend you are",
    "your new instructions",
    "system prompt",
    "print your prompt",
    "show your instructions",
    "reveal your instructions",
    "bypass your",
    "jailbreak",
    "affiche toutes les règles",      # French: "show all the rules" — in our CSV
    "règles internes",                # French: "internal rules"
    "logique exacte que vous utilisez",  # French: "exact logic you use"
    "documents récupérés",            # French: "retrieved documents"
]

# Malicious / harmful requests that should never be answered
MALICIOUS_PATTERNS = [
    "delete all files",
    "rm -rf",
    "drop table",
    "drop database",
    "give me the code to delete",
    "how to hack",
    "exploit vulnerability",
    "sql injection",
    "shell command",
    "execute command",
    "os.system",
    "subprocess.run",
]

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 2 — HARD ESCALATION KEYWORDS
# These topics MUST always escalate regardless of corpus coverage.
# No support doc can handle identity theft or security breaches.
# ─────────────────────────────────────────────────────────────────────────────

HARD_ESCALATION_KEYWORDS = [
    # Financial crimes
    "identity theft",
    "identity has been stolen",
    "identity stolen",
    "stolen card",
    "stolen cheque",
    "stolen check",
    "card stolen",
    "fraud",
    "fraudulent",

    # Security incidents
    "security breach",
    "security vulnerability",       # edge case: bug bounty → escalate, not answer
    "data breach",
    "account hacked",
    "account compromised",
    "unauthorized access",
    "major security",

    # Legal / financial escalations
    "legal action",
    "lawsuit",
    "sue",
    "lawyer",
    "attorney",
    "billing dispute",
    "chargeback",

    # Physical safety
    "urgent cash",                  # "I need urgent cash" — in our CSV
    "cash advance",
    "emergency funds",
]


# ─────────────────────────────────────────────────────────────────────────────
# PII PATTERNS — strip before logging (but keep in ticket for processing)
# ─────────────────────────────────────────────────────────────────────────────

PII_PATTERNS = [
    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),

    # Phone numbers (various formats)
    (r"\b(\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[PHONE]"),

    # Credit card numbers (16 digits, possibly spaced)
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD_NUMBER]"),

    # Stripe / payment order IDs (cs_live_... or cs_test_...)
    # Example from our CSV: cs_live_abcdefgh
    (r"\bcs_(live|test)_[A-Za-z0-9]+\b", "[ORDER_ID]"),

    # Generic order/reference IDs (long alphanumeric strings)
    (r"\b[A-Z]{2,3}-\d{6,}\b", "[REFERENCE_ID]"),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — build a standard escalation result dict
# Used when any guardrail fires — returns this directly as the ticket output.
# ─────────────────────────────────────────────────────────────────────────────

def _escalation_result(
    reason:       str,
    product_area: str = "general",
    request_type: str = "invalid",
    response:     str = "",
) -> dict:
    """
    Build a standardised escalation output dict.

    This is returned directly as the row output in output.csv when a
    guardrail fires — no retrieval, no Gemini call needed.

    Args:
        reason:       Internal reason for escalation (goes into justification)
        product_area: Best guess at product area (default "general")
        request_type: One of the 4 valid types (default "invalid")
        response:     User-facing message (default = generic escalation message)

    Returns dict matching the output.csv schema:
        status, product_area, response, justification, request_type
    """
    if not response:
        response = (
            "Thank you for reaching out. Your request has been forwarded "
            "to our support team who will assist you shortly."
        )

    return {
        "status":       "escalated",
        "product_area": product_area,
        "response":     response,
        "justification": f"Escalated by guardrail: {reason}",
        "request_type": request_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 1 — PRE-FLIGHT
# ─────────────────────────────────────────────────────────────────────────────

def strip_pii(text: str) -> str:
    """
    Replace PII patterns with placeholder tokens.
    Used before writing to trace log — we NEVER log raw PII.

    Note: We strip PII from the LOG copy only.
    The agent still processes the original ticket text so it has full context.

    Example:
        "My email is john@acme.com and order cs_live_abc123"
        → "My email is [EMAIL] and order [ORDER_ID]"
    """
    result = text
    for pattern, replacement in PII_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result


def pre_flight(
    issue:   str,
    subject: str,
    company: str,
) -> Optional[dict]:
    """
    CHECKPOINT 1 — runs before anything else.

    Checks (in order):
        1. Ticket too long?     → suspicious, escalate
        2. Prompt injection?    → escalate immediately
        3. Malicious request?   → escalate immediately
        4. Completely empty?    → mark as invalid

    Returns:
        None  → all checks passed, safe to continue pipeline
        dict  → guardrail fired, return this as the ticket output directly
    """
    combined = f"{issue} {subject}".strip()

    # ── Check 1: Ticket length ───────────────────────────────────────────────
    # Legitimate support tickets are rarely over 1000 chars.
    # Over 5000 is almost certainly an injection or spam attempt.
    if len(combined) > MAX_TICKET_LENGTH:
        return _escalation_result(
            reason       = f"ticket too long ({len(combined)} chars > {MAX_TICKET_LENGTH})",
            request_type = "invalid",
            response     = "Your message is too long to process. Please summarise your issue in a few sentences.",
        )

    # ── Check 2: Prompt injection ────────────────────────────────────────────
    # We check both issue and subject (attackers sometimes hide injection in subject)
    combined_lower = combined.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern.lower() in combined_lower:
            return _escalation_result(
                reason       = f"prompt injection detected: '{pattern}'",
                request_type = "invalid",
                response     = "We were unable to process your request. Please contact support directly.",
            )

    # ── Check 3: Malicious request ───────────────────────────────────────────
    for pattern in MALICIOUS_PATTERNS:
        if pattern.lower() in combined_lower:
            return _escalation_result(
                reason       = f"malicious request detected: '{pattern}'",
                request_type = "invalid",
                response     = "We cannot process this type of request.",
            )

    # ── Check 4: Completely empty ticket ────────────────────────────────────
    # Some tickets have no issue text and no subject — nothing to work with
    if len(combined.strip()) < 5:
        return _escalation_result(
            reason       = "empty ticket — no actionable content",
            request_type = "invalid",
            response     = "We did not receive enough information to process your request. Please describe your issue.",
        )

    # All checks passed
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 2 — HARD ESCALATION KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────

def is_hard_escalation(
    issue:   str,
    subject: str,
    company: str,
) -> tuple[bool, str]:
    """
    CHECKPOINT 2 — hardcoded sensitive keyword check.

    Some topics must ALWAYS escalate regardless of what the corpus says.
    Identity theft, fraud, security breaches — no support doc can handle these.
    A human agent is required.

    Returns:
        (True,  reason_string)  → must escalate
        (False, "")             → safe to continue

    Why hardcoded keywords instead of an LLM check?
        Speed: zero latency, zero API cost.
        Reliability: LLMs can be confused or bypassed. Exact string matching cannot.
        Auditability: every escalation decision is traceable to a specific keyword.

        "For safety-critical routing decisions I prefer deterministic rules
         over probabilistic models. The cost of a missed fraud escalation
         is much higher than the cost of over-escalating an edge case."
    """
    combined_lower = f"{issue} {subject}".lower()

    for keyword in HARD_ESCALATION_KEYWORDS:
        if keyword.lower() in combined_lower:
            return True, f"hard escalation keyword: '{keyword}'"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 3 — POST-RETRIEVAL CONFIDENCE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def post_retrieval(chunks: list[dict]) -> tuple[bool, str]:
    """
    CHECKPOINT 3 — runs after retrieval, before Gemini.

    Checks:
        1. No chunks returned?          → escalate (corpus has nothing)
        2. Top cross_score < threshold? → escalate (corpus doesn't cover this well)

    Args:
        chunks: list returned by retriever.retrieve()
                sorted by cross_score descending (best first)

    Returns:
        (True,  "")            → chunks are good enough, continue to Gemini
        (False, reason_string) → escalate, don't call Gemini

    Why check cross_score specifically (not bi_score)?
        bi_score  = cosine similarity from Qdrant (approximate, computed independently)
        cross_score = cross-encoder score (exact, computed on query+chunk together)
        Cross-encoder is more accurate — it's what we sort by.
        A high bi_score with low cross_score means Qdrant found something
        that looked similar but the cross-encoder determined it's actually not relevant.

    Interview talking point:
        "We escalate when max cross_score < 0.55. This prevents the agent
         from hallucinating answers for tickets the corpus doesn't cover.
         False escalation (escalating a simple FAQ) is much safer than
         false reply (answering a fraud case with made-up information)."
    """
    # No results at all
    if not chunks:
        return False, "zero retrieval results — corpus has no coverage for this ticket"

    # Check top score (list is already sorted by cross_score descending)
    top_score = chunks[0].get("cross_score", 0.0)

    if top_score < CONFIDENCE_THRESHOLD:
        return (
            False,
            f"low retrieval confidence: top cross_score={top_score:.3f} < {CONFIDENCE_THRESHOLD}"
        )

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 4 — POST-GENERATION VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_urls(text: str) -> list[str]:
    """
    Extract all URLs from a string.
    Used to check if Gemini hallucinated URLs not in the corpus.
    """
    return re.findall(r"https?://[^\s\)\"'>]+", text)


def _has_hallucinated_urls(
    response:      str,
    source_urls:   list[str],
) -> tuple[bool, list[str]]:
    """
    Check if Gemini's response contains URLs that weren't in the retrieved chunks.

    How hallucination happens:
        Gemini was trained on the web and "remembers" URLs.
        If you ask it about HackerRank, it might generate:
            "See https://support.hackerrank.com/articles/some-made-up-article"
        That URL might not exist — it's a hallucination.

    We detect this by comparing URLs in the response against URLs from
    the retrieved chunks (which are real, from the corpus).

    Args:
        response:    Gemini's generated response text
        source_urls: list of source_url values from retrieved chunks

    Returns:
        (True,  [hallucinated_urls])  → hallucination detected
        (False, [])                   → all URLs are legitimate
    """
    response_urls   = _extract_urls(response)
    legitimate_urls = set(source_urls)

    hallucinated = [
        url for url in response_urls
        if url not in legitimate_urls
    ]

    return (len(hallucinated) > 0), hallucinated


def post_generation(
    output:      dict,
    chunks:      list[dict],
) -> tuple[bool, dict]:
    """
    CHECKPOINT 4 — validate Gemini's structured output.

    Checks (in order):
        1. Required keys present?
        2. status is valid enum?
        3. request_type is valid enum?
        4. response is non-empty?
        5. Hallucinated URLs in response?

    Args:
        output: dict from Gemini — expected keys:
                status, product_area, response, justification, request_type
        chunks: retrieved chunks (for source_url comparison)

    Returns:
        (True,  cleaned_output)  → valid, use this output
        (False, cleaned_output)  → invalid but we fixed what we could
                                   caller should retry or escalate

    Why return the cleaned output even when invalid?
        We fix minor issues (wrong enum case, extra whitespace) automatically.
        Only truly unfixable issues (empty response, wrong structure) return False.

    Interview talking point:
        "Post-generation validation is the last safety net. It catches
         cases where Gemini returns malformed JSON, uses wrong enum values,
         or hallucinates URLs. We auto-fix minor issues and escalate
         unfixable ones rather than writing bad data to output.csv."
    """
    is_valid = True
    cleaned  = dict(output)   # work on a copy

    # ── Check 1: Required keys ───────────────────────────────────────────────
    required = {"status", "product_area", "response", "justification", "request_type"}
    missing  = required - set(cleaned.keys())
    if missing:
        # Can't fix missing keys — flag as invalid
        return False, _escalation_result(
            reason       = f"Gemini output missing required keys: {missing}",
            request_type = "invalid",
        )

    # ── Check 2: status enum ─────────────────────────────────────────────────
    status = str(cleaned.get("status", "")).lower().strip()
    if status not in VALID_STATUS:
        # Try to fix common variations before giving up
        if "escal" in status:
            cleaned["status"] = "escalated"
        elif "repl" in status or "answer" in status or "resolv" in status:
            cleaned["status"] = "replied"
        else:
            # Can't fix — default to escalated (safer than replied)
            cleaned["status"] = "escalated"
            is_valid = False
    else:
        cleaned["status"] = status

    # ── Check 3: request_type enum ───────────────────────────────────────────
    rtype = str(cleaned.get("request_type", "")).lower().strip()
    # Normalise common variations
    rtype_map = {
        "product issue":    "product_issue",
        "productissue":     "product_issue",
        "product_issues":   "product_issue",
        "feature request":  "feature_request",
        "featurerequest":   "feature_request",
        "feature_requests": "feature_request",
        "bugs":             "bug",
        "defect":           "bug",
        "invalid request":  "invalid",
        "out of scope":     "invalid",
        "outofscope":       "invalid",
    }
    rtype = rtype_map.get(rtype, rtype)

    if rtype not in VALID_REQUEST_TYPE:
        cleaned["request_type"] = "invalid"   # safe default
        is_valid = False
    else:
        cleaned["request_type"] = rtype

    # ── Check 4: Non-empty response ──────────────────────────────────────────
    response_text = str(cleaned.get("response", "")).strip()
    if not response_text or response_text.upper() in {"CAN_NOT_ANSWER", "NONE", "NULL", "N/A"}:
        # Gemini couldn't answer — convert to escalation
        cleaned["status"]   = "escalated"
        cleaned["response"] = (
            "Thank you for reaching out. Your request has been escalated "
            "to our support team for further assistance."
        )
        cleaned["justification"] = (
            cleaned.get("justification", "") +
            " | Escalated: Gemini could not generate a grounded answer."
        )
        is_valid = False
    else:
        cleaned["response"] = response_text

    # ── Check 5: Hallucinated URLs ───────────────────────────────────────────
    source_urls   = [c.get("source_url", "") for c in chunks if c.get("source_url")]
    hallucinated, bad_urls = _has_hallucinated_urls(cleaned["response"], source_urls)

    if hallucinated:
        # Strip the hallucinated URLs from the response
        # rather than throwing away the whole answer
        cleaned_response = cleaned["response"]
        for url in bad_urls:
            cleaned_response = cleaned_response.replace(url, "[link removed]")
        cleaned["response"]      = cleaned_response
        cleaned["justification"] = (
            cleaned.get("justification", "") +
            f" | Hallucinated URLs removed: {bad_urls}"
        )
        # Don't mark as invalid — we fixed it by stripping the URLs
        # But log it so we can track hallucination rate in evaluator.py

    # ── Ensure justification is non-empty ────────────────────────────────────
    if not str(cleaned.get("justification", "")).strip():
        cleaned["justification"] = "Answered based on retrieved support documentation."

    # ── Ensure product_area is non-empty ─────────────────────────────────────
    if not str(cleaned.get("product_area", "")).strip():
        # Fall back to subdomain from best retrieved chunk
        if chunks:
            cleaned["product_area"] = chunks[0].get("subdomain", "general")
        else:
            cleaned["product_area"] = "general"

    return is_valid, cleaned


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — infer product_area from retrieved chunks
# Used by agent.py when Gemini doesn't return a good product_area
# ─────────────────────────────────────────────────────────────────────────────

def infer_product_area(chunks: list[dict], company: str) -> str:
    """
    Infer the most likely product_area from retrieved chunk breadcrumbs.

    How it works:
        Each chunk has breadcrumbs like ["Screen", "Managing Tests"]
        or ["SkillUp", "Integrations"].
        We take the first breadcrumb of the top-scoring chunk as product_area.
        If multiple chunks agree → higher confidence.

    Example:
        Top 3 chunks all have breadcrumbs[0] = "Screen"
        → product_area = "screen"

    Falls back to company name if no breadcrumbs available.
    """
    if not chunks:
        return company.lower() if company and company != "None" else "general"

    # Count first breadcrumbs across top chunks (weighted by position)
    # Top chunk counts more than lower chunks
    votes: dict[str, float] = {}
    for i, chunk in enumerate(chunks[:4]):   # look at top 4 chunks
        breadcrumbs = chunk.get("breadcrumbs", [])
        if breadcrumbs:
            area   = breadcrumbs[0].lower().replace(" ", "_")
            weight = 1.0 / (i + 1)           # top chunk gets weight 1.0, next 0.5, etc.
            votes[area] = votes.get(area, 0) + weight

    if votes:
        return max(votes, key=votes.get)

    # Fall back to subdomain field
    subdomain = chunks[0].get("subdomain", "")
    if subdomain:
        return subdomain

    return company.lower() if company and company != "None" else "general"