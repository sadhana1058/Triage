"""
agent.py
========

THE BRAIN — orchestrates every step for one ticket.

PUBLIC API (called by main.py):
    process_ticket(issue, subject, company, row_idx, client, trace_path) → dict
    run_agent(tickets_df, client, trace_path) → output_df

INTERNAL FUNCTIONS:
    rewrite_query()       Gemini call #1 — rephrase ticket as clean search query
    build_prompt()        assemble retrieved chunks + ticket into Gemini prompt
    call_gemini()         Gemini call #2 — generate structured JSON answer
    parse_gemini_output() safely extract JSON from Gemini response
    process_ticket()      orchestrate all steps for one ticket
    run_agent()           loop over all tickets, write output.csv

GEMINI RATE LIMIT STRATEGY:
    Free tier = 15 RPM (requests per minute)
    Per ticket: 2 calls (rewrite + generate)
    sleep(4) between tickets
    56 tickets × 2 = 112 calls → ~4 minutes total
    Retry on invalid output = 3 calls max → still safe
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient

import guardrails
import retriever
import tracer

# ─────────────────────────────────────────────────────────────────────────────
# SETUP — load API key, configure Gemini
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

# WHY configure once at module level?
#   genai.configure() is a global call — doing it per-ticket wastes time
#   and risks race conditions if we ever parallelise.
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# WHY gemini-2.0-flash?
#   Fastest model on free tier. For structured output tasks (JSON extraction)
#   it performs nearly identically to larger models.
#   Pro models are slower and have lower RPM on free tier.
GEMINI_MODEL = "gemini-2.0-flash"

# Rate limiting
SLEEP_BETWEEN_TICKETS = 4    # seconds — keeps us under 15 RPM
MAX_RETRIES           = 1    # retry invalid output once before escalating

# How many retrieved chunks to include in the prompt
# Top 3 after cross-encoder ranking are the most relevant.
# Adding more chunks adds noise and burns prompt tokens.
TOP_CHUNKS_FOR_PROMPT = 3

# Paths
TICKETS_PATH = Path(__file__).parent.parent / "support_tickets" / "support_tickets.csv"
OUTPUT_PATH  = Path(__file__).parent.parent / "support_tickets" / "output.csv"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — QUERY REWRITE
# Gemini call #1. Rephrases ticket noise into a clean retrieval query.
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_query(issue: str, subject: str, company: str) -> str:
    """
    Rephrase the support ticket as a clean, short retrieval query.

    WHY rewrite at all?
        Raw tickets have noise that hurts retrieval:
            "I would like to request a rescheduling of my company HackerRank
             assessment due to unforeseen circumstances. Thank you."
        The words "I would like", "due to", "Thank you" are not in any support doc.
        They dilute the query vector and hurt BM25 scoring.

        Rewritten:
            "reschedule HackerRank assessment candidate alternative date"
        These words ARE in support docs → better retrieval.

    WHY a separate Gemini call for this (not do it in the generate call)?
        We want to search BEFORE generating. The generate call needs the
        retrieved docs as input. So rewrite → retrieve → generate.

    FALLBACK:
        If Gemini fails or times out, we fall back to the raw issue text.
        Better to search with noisy text than skip retrieval entirely.

    Args:
        issue:   full ticket text
        subject: ticket subject line (may be empty or noisy)
        company: "HackerRank", "Claude", "Visa", or "None"

    Returns:
        Clean search query string (1-2 sentences max)
    """
    # Build a short, focused prompt — we don't need a long system instruction here
    subject_line = f"Subject: {subject}\n" if subject.strip() else ""
    prompt = f"""Rephrase this support ticket as a short, clean search query (max 15 words).
Remove filler words, greetings, and emotional language.
Keep only the core technical issue and product names.
Return only the search query — no explanation, no punctuation at the end.

{subject_line}Ticket: {issue[:500]}
Company: {company}

Search query:"""

    try:
        model    = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,      # low temperature = deterministic, focused
                max_output_tokens=50, # we only want a short query
            ),
        )
        rewritten = response.text.strip()

        # Sanity check — if Gemini returned something weird, fall back
        if not rewritten or len(rewritten) > 200:
            return issue[:300]

        return rewritten

    except Exception as e:
        # Network error, rate limit, etc. — fall back gracefully
        print(f"[agent] Query rewrite failed ({e}), using raw issue text")
        return issue[:300]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD PROMPT
# Assembles retrieved chunks + ticket into the Gemini generate prompt.
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(
    issue:   str,
    subject: str,
    company: str,
    chunks:  list[dict],
    retry:   bool = False,
) -> str:
    """
    Assemble the full prompt for Gemini's answer generation call.

    STRUCTURE:
        [1] System instruction — who you are, rules, constraints
        [2] Retrieved documentation — top N chunks formatted cleanly
        [3] Support ticket — the actual user request
        [4] Output format — exact JSON schema with field descriptions

    WHY only top 3 chunks?
        8 chunks × 500 chars = 4000 extra chars in prompt
        3 chunks × 500 chars = 1500 extra chars
        Cross-encoder already ranked them — top 3 are best 3.
        Lower chunks add noise and cost tokens with little benefit.

    WHY number the chunks [1], [2], [3]?
        Gemini can reference them in the justification:
        "Based on [1] from HackerRank documentation..."
        This makes the justification traceable.

    WHY include source_url?
        Prevents hallucination — Gemini sees the real URL and is less
        likely to invent a different one.
        Also cited in the response: "See support.hackerrank.com/..."

    WHY `retry=True` mode?
        On retry we make the instruction stricter:
        "You MUST return valid JSON. Nothing else."
        Sometimes Gemini adds prose on the first try.

    Args:
        issue:   full ticket text
        subject: ticket subject
        company: company name
        chunks:  retrieved chunks from retriever.retrieve() (sorted by cross_score)
        retry:   if True, use stricter JSON-only instruction

    Returns:
        Complete prompt string ready to send to Gemini.
    """
    # ── Section 1: System instruction ────────────────────────────────────────
    system_instruction = f"""You are a support agent for {company if company != 'None' else 'a software/financial services company'}.

RULES (follow strictly):
1. Answer using ONLY the provided documentation below.
2. Do NOT use outside knowledge or make up information.
3. If the documentation does not cover the ticket, set response to exactly: CAN_NOT_ANSWER
4. Never invent URLs, policies, prices, or steps not in the documentation.
5. Be concise and helpful. Address the user directly.
6. If the ticket asks for something impossible or inappropriate, classify as invalid."""

    if retry:
        system_instruction += "\n\nCRITICAL: Return ONLY the JSON object. No prose, no markdown, no explanation."

    # ── Section 2: Retrieved documentation ───────────────────────────────────
    # Format each chunk clearly with its source context
    docs_section = "─── RETRIEVED DOCUMENTATION ───────────────────────────────────────────────\n\n"

    # Use only top N chunks — already sorted by cross_score descending
    top_chunks = chunks[:TOP_CHUNKS_FOR_PROMPT]

    for i, chunk in enumerate(top_chunks, 1):
        # Build the source label from breadcrumbs
        breadcrumbs = chunk.get("breadcrumbs", [])
        source_label = " > ".join(breadcrumbs) if breadcrumbs else chunk.get("domain", "").title()

        # Build the header for this doc
        title      = chunk.get("title", "")
        section    = chunk.get("section", "")
        source_url = chunk.get("source_url", "")

        doc_header = f"[{i}] {source_label}"
        if section:
            doc_header += f" — {section}"
        if source_url:
            doc_header += f"\n    Source: {source_url}"

        docs_section += f"{doc_header}\n\n{chunk['text']}\n\n{'─'*40}\n\n"

    # ── Section 3: The support ticket ────────────────────────────────────────
    ticket_section = "─── SUPPORT TICKET ─────────────────────────────────────────────────────────\n\n"
    if subject.strip():
        ticket_section += f"Subject: {subject}\n"
    ticket_section += f"Issue:   {issue}\n"
    ticket_section += f"Company: {company}\n"

    # ── Section 4: Output format ─────────────────────────────────────────────
    # We describe each field so Gemini knows exactly what to put there.
    # This is called "schema prompting" — very effective for structured output.
    output_format = """─── RESPOND IN THIS EXACT JSON FORMAT ──────────────────────────────────────

{
  "status": "replied" or "escalated",
  "product_area": "the specific product area or support category (e.g. screen, skillup, interviews, billing, privacy, amazon_bedrock, travel_support, dispute_resolution)",
  "response": "your complete response to the user — friendly, helpful, grounded in the docs above. If unanswerable write: CAN_NOT_ANSWER",
  "justification": "1-2 sentences explaining your routing decision and which doc supported your answer",
  "request_type": "product_issue" or "feature_request" or "bug" or "invalid",
  "confidence": 0.0
}

FIELD RULES:
- status:       "escalated" if sensitive, out-of-scope, or CAN_NOT_ANSWER. Otherwise "replied".
- product_area: use lowercase_underscore format. Infer from the documentation category.
- response:     must be grounded in [1][2][3] above. Never invent steps or URLs.
- request_type: "invalid" if ticket is spam, malicious, or completely off-topic.
- confidence:   your confidence in this answer from 0.0 (not confident) to 1.0 (very confident).

Return ONLY the JSON object. No markdown fences, no explanation text."""

    # Combine all sections
    return "\n\n".join([
        system_instruction,
        docs_section,
        ticket_section,
        output_format,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CALL GEMINI
# Send the prompt, get back text, measure latency and tokens.
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> tuple[str, int, int, int]:
    """
    Send a prompt to Gemini and return the response with metadata.

    WHY measure tokens here?
        Token counts tell us how much of the context window we're using.
        If prompt_tokens > 8000 we know our prompt is too long —
        Gemini might truncate it which causes bad answers.
        We log this in tracer.py for post-run analysis.

    WHY temperature=0.2 for generation (not 0.0)?
        0.0 = fully deterministic. Identical tickets → identical answers.
        0.2 = very low randomness. Slightly more natural language variation
        while still being reliable and consistent.
        Retrieval and routing decisions are deterministic (guardrails).
        Response phrasing can have tiny natural variation.

    Returns:
        (response_text, prompt_tokens, completion_tokens, latency_ms)
        response_text = raw text from Gemini (may include markdown fences)

    On failure:
        Returns ("", 0, 0, 0) — caller handles empty string as failure.
    """
    t0 = time.time()

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1000,
            ),
        )

        latency_ms = round((time.time() - t0) * 1000)

        # Extract token counts from response metadata
        # (available in Gemini API response object)
        usage          = response.usage_metadata
        prompt_tokens  = usage.prompt_token_count     if usage else 0
        output_tokens  = usage.candidates_token_count if usage else 0

        return response.text.strip(), prompt_tokens, output_tokens, latency_ms

    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000)
        print(f"[agent] Gemini call failed ({e})")
        return "", 0, 0, latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PARSE GEMINI OUTPUT
# Safely extract JSON from Gemini's response text.
# ─────────────────────────────────────────────────────────────────────────────

def parse_gemini_output(raw_text: str) -> Optional[dict]:
    """
    Safely extract a JSON dict from Gemini's raw response text.

    WHY is this non-trivial?
        Gemini sometimes wraps JSON in markdown fences:
            ```json
            {"status": "replied", ...}
            ```
        Or adds prose before the JSON:
            "Here is my response:\n{"status": "replied", ...}"
        Or uses single quotes instead of double quotes.

        We handle all of these cases.

    APPROACH (in order):
        1. Try direct json.loads() — works if Gemini behaved
        2. Strip ```json ... ``` fences and try again
        3. Find the first { and last } and extract that substring
        4. If all fail → return None → caller will escalate

    WHY not use Gemini's structured output mode (response_schema)?
        That feature requires a fixed schema at the API level.
        Our prompt already enforces the schema via instructions.
        Handling parsing ourselves gives more control and visibility.

    Returns:
        dict if parsing succeeded, None if all attempts failed.
    """
    if not raw_text:
        return None

    # Attempt 1 — direct parse (Gemini followed instructions perfectly)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Attempt 2 — strip markdown fences
    # Handles: ```json\n{...}\n``` or ```\n{...}\n```
    stripped = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    stripped = stripped.replace("```", "").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Attempt 3 — find JSON object boundaries
    # Handles: "Sure! Here is the response:\n{...}\nLet me know if..."
    start = raw_text.find("{")
    end   = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # All attempts failed
    print(f"[agent] Could not parse Gemini output: {raw_text[:200]}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR — process one ticket
# ─────────────────────────────────────────────────────────────────────────────

def process_ticket(
    issue:      str,
    subject:    str,
    company:    str,
    row_idx:    int,
    client:     QdrantClient,
    trace_path: Path,
) -> dict:
    """
    Process one support ticket through the full 6-layer pipeline.

    PIPELINE:
        1. start_trace()          — create trace skeleton
        2. pre_flight()           — prompt injection, malicious, PII checks
        3. is_hard_escalation()   — fraud, identity theft, security keywords
        4. rewrite_query()        — Gemini call #1: clean up ticket for search
        5. retrieve()             — hybrid search + cross-encoder re-rank
        6. post_retrieval()       — confidence threshold check (0.55)
        7. build_prompt()         — assemble context + ticket into Gemini prompt
        8. call_gemini()          — Gemini call #2: generate structured answer
        9. parse_gemini_output()  — safely extract JSON
        10. post_generation()     — validate enums, URLs, empty responses
        11. [retry if invalid]    — one retry with stricter prompt
        12. finish_trace()        — append JSONL line to trace file

    Each step can exit early with an escalation result.
    Only Steps 7-11 are skippable — everything else always runs.

    Args:
        issue:      ticket body text
        subject:    ticket subject line (may be empty)
        company:    "HackerRank", "Claude", "Visa", or "None"
        row_idx:    0-based row index (for ticket_id in trace)
        client:     connected QdrantClient from build_index()
        trace_path: Path to the JSONL trace file for this run

    Returns:
        dict with keys: status, product_area, response, justification, request_type
        Always returns a valid dict — never raises an exception to the caller.
    """
    print(f"\n[agent] ── Ticket {row_idx} | company={company} ──")
    print(f"[agent] Issue: {issue[:80]}...")

    # ── STEP 1: Create trace skeleton ────────────────────────────────────────
    # Build the trace dict at the very start.
    # If we crash mid-ticket, finish_trace() will still write what we have.
    trace = tracer.start_trace(row_idx, issue, subject, company)

    # ── STEP 2: Pre-flight guardrail ─────────────────────────────────────────
    # Runs in microseconds. Catches injection, malicious, empty tickets.
    # Returns None if passed, or an escalation dict to return immediately.
    t0 = time.time()
    preflight_result = guardrails.pre_flight(issue, subject, company)
    preflight_ms     = round((time.time() - t0) * 1000)

    if preflight_result:
        # Guardrail fired — log and return immediately
        # No retrieval, no Gemini call needed
        print(f"[agent] Pre-flight FAILED: {preflight_result['justification']}")
        tracer.add_guardrail_result(trace, "pre_flight", {
            "passed":     False,
            "reason":     preflight_result["justification"],
            "latency_ms": preflight_ms,
        })
        tracer.finish_trace(trace, preflight_result, trace_path)
        return preflight_result

    tracer.add_guardrail_result(trace, "pre_flight", {
        "passed": True, "reason": "", "latency_ms": preflight_ms,
    })

    # ── STEP 3: Hard escalation check ────────────────────────────────────────
    # Hardcoded keywords that ALWAYS escalate.
    # Fraud, identity theft, security breaches — no doc can handle these.
    triggered, keyword = guardrails.is_hard_escalation(issue, subject, company)

    tracer.add_guardrail_result(trace, "hard_escalation", {
        "triggered": triggered, "keyword": keyword,
    })

    if triggered:
        print(f"[agent] Hard escalation: {keyword}")

        # Infer product_area from company even without retrieval
        product_area = company.lower() if company != "None" else "general"

        # Craft a specific response based on the keyword type
        if any(k in keyword for k in ["fraud", "stolen", "identity"]):
            response = (
                "We understand this is an urgent situation. "
                "For security-related issues like this, please contact "
                "your card issuer or bank directly — they have dedicated "
                "fraud teams available 24/7. Your case has been escalated "
                "to our support team as well."
            )
        elif "security vulnerability" in keyword or "security breach" in keyword:
            response = (
                "Thank you for bringing this to our attention. "
                "Security reports require immediate specialist review. "
                "Your report has been escalated to our security team."
            )
        else:
            response = (
                "Thank you for reaching out. Your request requires "
                "specialist attention and has been escalated to our "
                "support team who will contact you shortly."
            )

        result = {
            "status":        "escalated",
            "product_area":  product_area,
            "response":      response,
            "justification": f"Hard escalation triggered by keyword: '{keyword}'",
            "request_type":  "product_issue",
        }
        tracer.finish_trace(trace, result, trace_path)
        return result

    # ── STEP 4: Query rewrite ─────────────────────────────────────────────────
    # Gemini call #1 — clean the ticket for retrieval.
    # Falls back to raw issue text if Gemini fails.
    print("[agent] Rewriting query...")
    query_rewritten = rewrite_query(issue, subject, company)
    print(f"[agent] Query: '{query_rewritten}'")

    # ── STEP 5: Hybrid retrieval ──────────────────────────────────────────────
    # Dense + sparse → Qdrant RRF → cross-encoder re-rank
    t0 = time.time()
    chunks = retriever.retrieve(
        query   = query_rewritten,
        company = company,
        client  = client,
    )
    retrieval_ms = round((time.time() - t0) * 1000)

    tracer.add_retrieval_result(
        trace, chunks, issue, query_rewritten, retrieval_ms
    )

    # ── STEP 6: Post-retrieval confidence check ───────────────────────────────
    # If top cross_score < 0.55 → corpus doesn't cover this → escalate
    good_enough, reason = guardrails.post_retrieval(chunks)

    tracer.add_guardrail_result(trace, "post_retrieval", {
        "passed":     good_enough,
        "reason":     reason,
        "top_cross":  chunks[0].get("cross_score", 0) if chunks else 0,
        "num_chunks": len(chunks),
    })

    if not good_enough:
        print(f"[agent] Post-retrieval FAILED: {reason}")

        # Infer product_area from best chunk even if confidence is low
        product_area = guardrails.infer_product_area(chunks, company)

        result = {
            "status":        "escalated",
            "product_area":  product_area,
            "response":      (
                "Thank you for reaching out. We don't have enough information "
                "in our knowledge base to fully address your request. "
                "Your ticket has been escalated to a specialist."
            ),
            "justification": f"Escalated: {reason}. Insufficient corpus coverage to answer safely.",
            "request_type":  "product_issue",
        }
        tracer.finish_trace(trace, result, trace_path)
        return result

    # ── STEP 7 + 8: Build prompt and call Gemini ─────────────────────────────
    # This is the generation step. We build the prompt with top 3 chunks
    # and call Gemini for a structured JSON response.
    print("[agent] Calling Gemini for answer generation...")

    prompt         = build_prompt(issue, subject, company, chunks, retry=False)
    raw_text, tokens_in, tokens_out, gen_ms = call_gemini(prompt)

    # ── STEP 9: Parse Gemini output ───────────────────────────────────────────
    parsed = parse_gemini_output(raw_text)

    # ── STEP 10: Post-generation validation ───────────────────────────────────
    if parsed:
        is_valid, cleaned = guardrails.post_generation(parsed, chunks)
    else:
        is_valid, cleaned = False, None

    # ── STEP 11: Retry once if invalid ───────────────────────────────────────
    # On retry: stricter prompt, log it, try once more.
    # If still fails: force escalate.
    if not is_valid or cleaned is None:
        print("[agent] Output invalid — retrying with stricter prompt...")

        retry_prompt = build_prompt(issue, subject, company, chunks, retry=True)
        raw_text2, tokens_in2, tokens_out2, gen_ms2 = call_gemini(retry_prompt)

        # Accumulate token counts from both calls
        tokens_in  += tokens_in2
        tokens_out += tokens_out2
        gen_ms     += gen_ms2

        parsed2 = parse_gemini_output(raw_text2)
        if parsed2:
            is_valid2, cleaned2 = guardrails.post_generation(parsed2, chunks)
            if is_valid2 or cleaned2:
                cleaned  = cleaned2
                is_valid = is_valid2

    # Log generation metadata (covers both original + retry calls)
    tracer.add_generation_result(
        trace,
        prompt_tokens     = tokens_in,
        completion_tokens = tokens_out,
        latency_ms        = gen_ms,
        self_confidence   = float(cleaned.get("confidence", 0)) if cleaned else 0.0,
    )

    tracer.add_guardrail_result(trace, "post_generation", {
        "valid": is_valid,
        "had_retry": not is_valid,
    })

    # If still no valid output after retry — force escalate
    if not cleaned:
        print("[agent] Retry failed — forcing escalation")
        product_area = guardrails.infer_product_area(chunks, company)
        cleaned = {
            "status":        "escalated",
            "product_area":  product_area,
            "response":      (
                "We were unable to generate a reliable answer for your request. "
                "Your ticket has been escalated to our support team."
            ),
            "justification": "Escalated: Gemini failed to produce valid structured output after retry.",
            "request_type":  "product_issue",
        }

    # ── Final cleanup ─────────────────────────────────────────────────────────
    # If product_area is empty/missing, infer from chunks
    if not cleaned.get("product_area", "").strip():
        cleaned["product_area"] = guardrails.infer_product_area(chunks, company)

    # Normalise status capitalisation to match expected output
    # Sample CSV uses "Replied"/"Escalated" but problem statement says lowercase
    # We use lowercase throughout — main.py capitalises if needed
    cleaned["status"] = cleaned.get("status", "escalated").lower()

    # Remove internal "confidence" key — not part of output.csv schema
    cleaned.pop("confidence", None)

    print(f"[agent] ✓ status={cleaned['status']} | area={cleaned['product_area']} | type={cleaned['request_type']}")

    # ── STEP 12: Write trace ──────────────────────────────────────────────────
    tracer.finish_trace(trace, cleaned, trace_path)

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# RUN AGENT — loop over all tickets
# Called by main.py
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(
    tickets_df: pd.DataFrame,
    client:     QdrantClient,
    trace_path: Path,
) -> pd.DataFrame:
    """
    Process all tickets in the DataFrame and return output DataFrame.

    WHY process sequentially (not in parallel)?
        Gemini free tier = 15 RPM. Parallel calls would instantly hit
        the rate limit. Sequential + sleep(4) keeps us safe.
        With 56 tickets × ~2.5s each = ~140s total. Acceptable.

    WHY sleep between tickets (not between Gemini calls)?
        The two Gemini calls within one ticket happen <2s apart —
        well within the per-minute window. The sleep is between tickets
        to space them out across the minute.

    Args:
        tickets_df: DataFrame with columns Issue, Subject, Company
        client:     connected QdrantClient
        trace_path: Path to trace JSONL file for this run

    Returns:
        DataFrame with all output columns:
        status, product_area, response, justification, request_type
    """
    results = []

    for row_idx, row in tickets_df.iterrows():
        issue   = str(row.get("Issue", "")).strip()
        subject = str(row.get("Subject", "")).strip()
        company = str(row.get("Company", "None")).strip()

        # Replace NaN or empty company with "None"
        if not company or company.lower() in {"nan", "none", ""}:
            company = "None"

        try:
            output = process_ticket(
                issue      = issue,
                subject    = subject,
                company    = company,
                row_idx    = int(row_idx),
                client     = client,
                trace_path = trace_path,
            )
        except Exception as e:
            # Catch-all — never let one bad ticket crash the whole run
            print(f"[agent] Unexpected error on row {row_idx}: {e}")
            output = {
                "status":        "escalated",
                "product_area":  "general",
                "response":      "An unexpected error occurred. Your ticket has been escalated.",
                "justification": f"System error: {str(e)[:100]}",
                "request_type":  "invalid",
            }

        results.append(output)

        # Rate limit sleep between tickets
        # WHY here and not inside process_ticket?
        #   process_ticket doesn't know if it's the last ticket.
        #   The loop knows — we don't sleep after the last one.
        if int(row_idx) < len(tickets_df) - 1:
            print(f"[agent] Sleeping {SLEEP_BETWEEN_TICKETS}s (rate limit)...")
            time.sleep(SLEEP_BETWEEN_TICKETS)

    return pd.DataFrame(results)