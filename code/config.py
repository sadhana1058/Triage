"""
config.py
=========

Single source of truth for every constant and tuneable in the system.

HOW TO USE:
    from config import COLLECTION_NAME, TOP_K, GEMINI_MODEL, ...

WHY ONE FILE?
    Previously constants were scattered across retriever.py, agent.py,
    guardrails.py, tracer.py, and main.py. Changing a threshold meant
    hunting across 5 files. This file centralises all of them so you
    change a value once and it propagates everywhere.

SECTIONS:
    1. Paths          — all filesystem locations derived from ROOT_DIR
    2. Vector DB      — Qdrant collection & vector config
    3. Embedding      — model names, dimensions
    4. Chunking       — chunk size, overlap, min length
    5. Retrieval      — top-K, score floors
    6. Generation     — Gemini model, rate limits, prompt knobs
    7. Guardrails     — thresholds, valid enums, keyword lists, PII patterns
"""

from __future__ import annotations

from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS
# All paths derived from ROOT_DIR — never hardcode absolute paths.
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR        = Path(__file__).parent.parent          # project root (Triage/)
DATA_ROOT       = ROOT_DIR / "data"                     # markdown corpus
QDRANT_PATH     = str(ROOT_DIR / "qdrant_db")           # Qdrant on-disk storage
BM25_CACHE_PATH = ROOT_DIR / "qdrant_db" / "bm25_state.pkl"
TRACES_DIR      = ROOT_DIR / "traces"
TICKETS_PATH    = ROOT_DIR / "support_tickets" / "support_tickets.csv"
OUTPUT_PATH     = ROOT_DIR / "support_tickets" / "output.csv"
SAMPLE_PATH     = ROOT_DIR / "support_tickets" / "sample_support_tickets.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 2. VECTOR DB
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "support_corpus"
DENSE_VEC       = "dense"
SPARSE_VEC      = "sparse"

# Domain → folder mapping (must match subdirectories under data/)
DOMAIN_FOLDER_MAP: dict[str, str] = {
    "hackerrank": "hackerrank",
    "claude":     "claude",
    "visa":       "visa",
}

# Company name (from CSV) → domain key used for Qdrant filter
COMPANY_TO_DOMAIN: dict[str, str | None] = {
    "HackerRank": "hackerrank",
    "Claude":     "claude",
    "Visa":       "visa",
    "None":       None,         # search all domains
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBEDDING MODELS
# ─────────────────────────────────────────────────────────────────────────────

BIENCODER_MODEL    = "all-MiniLM-L6-v2"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM      = 384        # output dimension of all-MiniLM-L6-v2


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

MIN_CHUNK_CHARS = 60            # discard fragments shorter than this
MAX_CHUNK_CHARS = 1200          # split sections longer than this
OVERLAP_CHARS   = 200           # tail of previous chunk prepended to next chunk
                                # WHY: answers that span two sections need shared
                                # context so the cross-encoder can score them fairly.


# ─────────────────────────────────────────────────────────────────────────────
# 5. RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

TOP_K          = 15             # candidates pulled from Qdrant before re-ranking (per query variant)
BI_SCORE_FLOOR = 0.20           # lenient Qdrant pre-filter (bi-encoder score)


# ─────────────────────────────────────────────────────────────────────────────
# 6. GENERATION
# ─────────────────────────────────────────────────────────────────────────────

OPENAI_MODEL           = "gpt-4o-mini"
SLEEP_BETWEEN_TICKETS  = 1      # gentle pacing — OpenAI has generous RPM limits
MAX_RETRIES            = 1      # retry invalid output once before escalating
TOP_CHUNKS_FOR_PROMPT  = 5      # top-N chunks sent to LLM (after cross-encoder rank)
NUM_QUERY_VARIANTS     = 3      # multi-query: number of query phrasings to generate


# ─────────────────────────────────────────────────────────────────────────────
# 7. GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

# Minimum cross-encoder score to proceed to Gemini generation.
# Below this → corpus doesn't cover the ticket → safer to escalate.
CONFIDENCE_THRESHOLD = 0.40

# Tickets over this length are suspicious (likely injection or spam).
MAX_TICKET_LENGTH = 5000

# Valid enum values for output.csv
VALID_STATUS       = {"replied", "escalated"}
VALID_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}

# BM25 stopwords — stripped from tokens before building sparse vectors
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "this", "that", "it", "its",
    "i", "we", "you", "he", "she", "they", "my", "your", "our",
}

# Phrases that indicate the user is trying to override the agent's instructions
INJECTION_PATTERNS: list[str] = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard previous",
    "disregard all previous",
    "forget your instructions",
    "pretend you are",
    "your new instructions",
    "system prompt",
    "print your prompt",
    "show your instructions",
    "reveal your instructions",
    "bypass your",
    "jailbreak",
]

# Requests that should never be processed regardless of corpus coverage
MALICIOUS_PATTERNS: list[str] = [
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

# Topics that always require a human agent — no doc can resolve these
HARD_ESCALATION_KEYWORDS: list[str] = [
    # Financial crimes
    "identity theft",
    "identity has been stolen",
    "identity stolen",
    "fraud",
    "fraudulent",

    # Security incidents
    "security breach",
    "security vulnerability",
    "data breach",
    "account hacked",
    "account compromised",
    "unauthorized access",
    "major security",

    # Legal
    "legal action",
    "lawsuit",
    "lawyer",
    "attorney",

    # Physical safety / financial urgency
    "urgent cash",
    "cash advance",
    "emergency funds",
]

# Regex patterns for stripping PII from trace logs.
# Format: (pattern, replacement_token)
PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b(\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",  "[PHONE]"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",                   "[CARD_NUMBER]"),
    (r"\bcs_(live|test)_[A-Za-z0-9]+\b",                                "[ORDER_ID]"),
    (r"\b[A-Z]{2,3}-\d{6,}\b",                                          "[REFERENCE_ID]"),
]
