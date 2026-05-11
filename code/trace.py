"""
tracer.py
=========

Per-ticket JSONL trace logger.

WHAT IT DOES:
    Records every decision the agent makes for one ticket into a structured
    JSON line. One line per ticket. One file per run. Appended as each
    ticket finishes — crash-safe and streamable.

WHY IT MATTERS:
    The evaluation criteria explicitly checks for traceable decisions.
    "justification: concise, accurate, traceable to the corpus."
    The trace log is what makes every decision auditable — not just the
    final answer, but HOW the agent got there.

FILE LOCATION:
    traces/run_YYYYMMDD_HHMMSS.jsonl
    Created fresh each run. Old runs are preserved.

USAGE PATTERN IN agent.py:
    # Step 1 — create partial trace at start of ticket
    trace = tracer.start_trace(row_idx, issue, subject, company)

    # Step 2 — add to trace as each step completes
    tracer.add_guardrail_result(trace, "pre_flight", {"passed": True})
    tracer.add_retrieval_result(trace, chunks, query_rewritten, latency_ms)
    tracer.add_generation_result(trace, tokens_in, tokens_out, latency_ms)

    # Step 3 — write to disk when ticket is done
    tracer.finish_trace(trace, output, trace_file_path)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from guardrails import strip_pii

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TRACES_DIR = Path(__file__).parent.parent / "traces"


# ─────────────────────────────────────────────────────────────────────────────
# RUN FILE MANAGEMENT
# One file per run — named by timestamp so old runs are preserved.
# ─────────────────────────────────────────────────────────────────────────────

def make_trace_path() -> Path:
    """
    Create the traces/ directory if it doesn't exist and return
    a unique file path for this run.

    Format: traces/run_20260510_142301.jsonl
    Using timestamp means:
        - Old runs are never overwritten
        - You can compare runs side by side
        - Sorting by filename = sorting by time

    Returns Path object ready to pass into finish_trace().
    """
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return TRACES_DIR / f"run_{timestamp}.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# TRACE LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

def start_trace(
    row_idx: int,
    issue:   str,
    subject: str,
    company: str,
) -> dict:
    """
    Create a partial trace dict at the START of processing a ticket.

    WHY create it at the start (not end)?
        If the agent crashes mid-ticket, we still have a partial trace
        to write. finish_trace() checks for missing keys and fills
        defaults so even a partial trace is valid JSONL.

    WHAT IS issue_safe?
        We store the PII-stripped version of the ticket.
        The agent still processes issue (full text) — it needs context.
        But the LOG only ever sees issue_safe — PII never touches disk.

        Example:
            issue     = "My email john@acme.com order cs_live_abc123"
            issue_safe = "My email [EMAIL] order [ORDER_ID]"

    Args:
        row_idx: 0-based row index in support_tickets.csv
        issue:   full ticket text (NOT logged directly)
        subject: ticket subject line (NOT logged directly)
        company: "HackerRank", "Claude", "Visa", or "None"

    Returns:
        Partial trace dict. Pass this to add_* functions and finish_trace().
    """
    return {
        # ── Identity ──────────────────────────────────────────────────────
        # ticket_id matches the row so you can cross-reference with output.csv
        "ticket_id":  f"row_{row_idx}",
        "row_idx":    row_idx,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "company":    company,

        # ── Input ─────────────────────────────────────────────────────────
        # issue_safe: PII stripped — this is what we persist
        # subject_safe: same treatment
        # WHY: GDPR + basic privacy hygiene. Trace files may be
        #      read by judges, stored in git, or uploaded to W&B.
        #      They should never contain raw PII.
        "issue_safe":   strip_pii(issue),
        "subject_safe": strip_pii(subject),

        # ── Placeholders ──────────────────────────────────────────────────
        # These get filled in as the agent progresses.
        # Having them here means finish_trace() always finds a key —
        # no KeyError if a step was skipped due to early escalation.
        "guardrails":   {},
        "retrieval":    {},
        "generation":   {},
        "output":       {},

        # ── Timing ────────────────────────────────────────────────────────
        # _start_time is internal — used to compute total_latency_ms.
        # Leading underscore = convention for "don't serialize this".
        "_start_time":  time.time(),
    }


def add_guardrail_result(
    trace:        dict,
    checkpoint:   str,
    result:       dict,
) -> None:
    """
    Add the result of one guardrail checkpoint to the trace.

    WHY a separate function per step (not just dict update)?
        1. Each step has a defined schema — this function enforces it.
        2. Agent code stays clean: one call per step, no inline dict juggling.
        3. Easy to add validation or side effects later (e.g. alert on injection).

    Args:
        trace:      the trace dict from start_trace()
        checkpoint: which guardrail fired — "pre_flight", "hard_escalation",
                    "post_retrieval", "post_generation"
        result:     dict describing what happened at this checkpoint

    Example calls from agent.py:
        add_guardrail_result(trace, "pre_flight", {
            "passed":  True,
            "reason":  "",
            "latency_ms": 0.3,
        })
        add_guardrail_result(trace, "hard_escalation", {
            "triggered": False,
            "keyword":   "",
        })
        add_guardrail_result(trace, "post_retrieval", {
            "passed":     True,
            "top_cross":  0.87,
            "num_chunks": 8,
        })
    """
    trace["guardrails"][checkpoint] = result


def add_retrieval_result(
    trace:            dict,
    chunks:           list[dict],
    query_original:   str,
    query_rewritten:  str,
    latency_ms:       int,
) -> None:
    """
    Add retrieval results to the trace.

    WHY log query_rewritten separately from query_original?
        The agent rewrites the query before searching (Gemini call #1).
        Logging both lets you audit whether the rewrite helped or hurt.
        If original = "it broke" and rewritten = "HackerRank assessment
        submission error" — you can see the rewrite added value.

    WHY log chunk IDs not full chunk text?
        Chunk text is 500 chars × 8 chunks = 4KB per ticket.
        For 56 tickets that's 224KB of duplicate data in the trace.
        The chunk_id + source path is sufficient to look up the exact
        text in Qdrant or the original .md file if needed.

    Args:
        trace:           trace dict from start_trace()
        chunks:          list of chunk dicts from retriever.retrieve()
        query_original:  the raw ticket text used as initial query
        query_rewritten: the Gemini-rewritten query used for actual search
        latency_ms:      time spent in retriever.retrieve()
    """
    trace["retrieval"] = {
        "query_original":  strip_pii(query_original),
        "query_rewritten": strip_pii(query_rewritten),

        # Log lightweight chunk summaries — not full text
        "chunks": [
            {
                "chunk_id":    c.get("chunk_id", ""),
                "source":      c.get("source", ""),
                "title":       c.get("title", ""),
                "section":     c.get("section", ""),
                "cross_score": c.get("cross_score", 0),
                "rrf_score":   c.get("rrf_score", 0),
                "bi_score":    c.get("bi_score", 0),
            }
            for c in chunks
        ],

        # Summary stats (useful for quick scanning)
        "num_chunks":     len(chunks),
        "top_cross":      chunks[0].get("cross_score", 0) if chunks else 0,
        "top_source_url": chunks[0].get("source_url", "") if chunks else "",
        "latency_ms":     latency_ms,
    }


def add_generation_result(
    trace:             dict,
    prompt_tokens:     int,
    completion_tokens: int,
    latency_ms:        int,
    self_confidence:   float = 0.0,
    query_rewritten:   str   = "",
) -> None:
    """
    Add Gemini generation metadata to the trace.

    WHY log token counts?
        1. Cost tracking — even on free tier, knowing prompt size helps
           you optimise the prompt to stay under context limits.
        2. Debugging — if a ticket gets a bad answer, seeing 3000 prompt
           tokens tells you the prompt was probably overstuffed with chunks.
        3. Interview — shows you thought about compute efficiency.

    WHY log self_confidence?
        We ask Gemini to rate its own confidence (0.0-1.0) in the response.
        This is different from cross_score (retrieval confidence).
        A ticket can have great retrieval but Gemini still says
        "I'm not confident" — which might trigger a retry.
        Logging both lets you analyse each independently.

    Args:
        trace:             trace dict from start_trace()
        prompt_tokens:     tokens in the Gemini prompt
        completion_tokens: tokens in the Gemini response
        latency_ms:        time spent waiting for Gemini
        self_confidence:   Gemini's self-reported confidence (0.0-1.0)
        query_rewritten:   the rewritten query (for reference)
    """
    trace["generation"] = {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      prompt_tokens + completion_tokens,
        "latency_ms":        latency_ms,
        "self_confidence":   self_confidence,
    }


def finish_trace(
    trace:      dict,
    output:     dict,
    trace_path: Path,
) -> None:
    """
    Finalise the trace and APPEND it as one JSON line to the trace file.

    WHY append (not overwrite)?
        JSONL is append-only by design. Each line is one complete JSON object.
        Appending means:
            - Crash-safe: previous tickets' traces survive a crash
            - Streamable: `tail -f traces/run_xyz.jsonl` shows live progress
            - No file locking needed (append is atomic on most OS)

    WHY pop _start_time before writing?
        _start_time is an internal float used to compute total_latency_ms.
        It's not meaningful to the reader and floats like 1715350981.234
        add noise. We compute the duration and discard the raw timestamp.

    Args:
        trace:      trace dict built up during agent processing
        output:     final output dict (status, product_area, response, etc.)
        trace_path: Path from make_trace_path() — same path for all tickets in a run
    """
    # Record the final output
    trace["output"] = {
        "status":        output.get("status", ""),
        "product_area":  output.get("product_area", ""),
        "request_type":  output.get("request_type", ""),
        # We DON'T log full response/justification text —
        # those can contain PII from the ticket context.
        # Log length instead so you can spot truncated responses.
        "response_len":     len(str(output.get("response", ""))),
        "justification_len": len(str(output.get("justification", ""))),
    }

    # Compute total latency before removing _start_time
    start_time = trace.pop("_start_time", time.time())
    trace["total_latency_ms"] = round((time.time() - start_time) * 1000)

    # Serialise to JSON and append one line to the trace file
    # ensure_ascii=False → preserves non-English characters (our French ticket)
    # default=str → safely handles any non-serialisable type (datetime etc.)
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False, default=str))
        f.write("\n")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — READ TRACES BACK
# Useful in evaluator.py to analyse trace files after a run.
# ─────────────────────────────────────────────────────────────────────────────

def read_traces(trace_path: Path) -> list[dict]:
    """
    Read a JSONL trace file back into a list of dicts.

    WHY a reader in tracer.py?
        evaluator.py needs to read traces to compute:
            - avg retrieval latency
            - avg generation latency
            - guardrail trigger rate
            - self_confidence distribution
        Putting the reader here keeps I/O logic in one place.

    Returns list of trace dicts, one per ticket.
    Skips malformed lines (e.g. partial write from a crash) silently.
    """
    traces = []
    if not trace_path.exists():
        return traces

    with open(trace_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                # Malformed line — partial write from crash or corruption
                # Log and skip rather than crashing the whole read
                print(f"[tracer] Warning: skipping malformed line {line_num} in {trace_path}")

    return traces


def get_latest_trace_path() -> Path | None:
    """
    Find the most recent trace file in the traces/ directory.
    Used by evaluator.py when no specific trace path is provided.

    Returns Path to latest file, or None if no traces exist.
    """
    if not TRACES_DIR.exists():
        return None

    trace_files = sorted(TRACES_DIR.glob("run_*.jsonl"))
    return trace_files[-1] if trace_files else None


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — QUICK SUMMARY
# Print a human-readable summary of a run to the terminal.
# Called by main.py after a run completes.
# ─────────────────────────────────────────────────────────────────────────────

def print_run_summary(trace_path: Path) -> None:
    """
    Print a quick summary of a completed run to the terminal.

    Shows:
        - Total tickets processed
        - replied vs escalated counts
        - Guardrail trigger counts
        - Average latency
        - Trace file location

    WHY in tracer.py and not main.py?
        tracer.py already knows how to read trace files.
        The summary logic is just a read + aggregate — it belongs here.
        main.py stays thin (just calls this after the run).
    """
    traces = read_traces(trace_path)
    if not traces:
        print("[tracer] No traces found.")
        return

    total     = len(traces)
    replied   = sum(1 for t in traces if t.get("output", {}).get("status") == "replied")
    escalated = total - replied

    # Guardrail counts
    injections  = sum(
        1 for t in traces
        if not t.get("guardrails", {}).get("pre_flight", {}).get("passed", True)
    )
    hard_esc    = sum(
        1 for t in traces
        if t.get("guardrails", {}).get("hard_escalation", {}).get("triggered", False)
    )
    low_conf    = sum(
        1 for t in traces
        if not t.get("guardrails", {}).get("post_retrieval", {}).get("passed", True)
    )

    # Latency
    latencies   = [t.get("total_latency_ms", 0) for t in traces]
    avg_latency = round(sum(latencies) / len(latencies)) if latencies else 0

    # Retrieval confidence
    top_crosses = [
        t.get("retrieval", {}).get("top_cross", 0)
        for t in traces
        if t.get("retrieval")
    ]
    avg_cross   = round(sum(top_crosses) / len(top_crosses), 3) if top_crosses else 0

    print("\n" + "─" * 50)
    print(f"  Run Summary — {trace_path.name}")
    print("─" * 50)
    print(f"  Tickets processed : {total}")
    print(f"  Replied           : {replied}  ({100*replied//total}%)")
    print(f"  Escalated         : {escalated}  ({100*escalated//total}%)")
    print(f"  ─ Injection       : {injections}")
    print(f"  ─ Hard escalation : {hard_esc}")
    print(f"  ─ Low confidence  : {low_conf}")
    print(f"  Avg latency       : {avg_latency}ms/ticket")
    print(f"  Avg top cross     : {avg_cross}")
    print(f"  Trace file        : {trace_path}")
    print("─" * 50 + "\n")