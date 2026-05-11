"""
retriever.py
============

INDEX TIME  (runs once, persists to disk):
    774 .md files
    → parse frontmatter  (title, breadcrumbs, source_url, last_updated)
    → markdown-aware chunking  (split at # headings, not blind char count)
    → build context string  (title + breadcrumbs + section + chunk text)
    → MiniLM embeds the context string  → 384-dim vector
    → store vector + production payload in Qdrant

QUERY TIME  (runs per ticket):
    query string + company
    → MiniLM embeds query
    → Qdrant: domain-filtered vector search → top-8 candidates
    → cross-encoder re-ranks candidates  (reads query+chunk together)
    → return ranked list to agent.py

WHY TWO THINGS ARE DIFFERENT:
    What we EMBED  = context-rich string (title + breadcrumbs + section + text)
                     → better vector, better retrieval accuracy
    What we STORE  = clean chunk text only
                     → goes into Gemini prompt without noise
"""

from __future__ import annotations

import re
import hashlib
import time
from pathlib import Path
from typing import Optional

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from sentence_transformers import SentenceTransformer, CrossEncoder


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT        = Path(__file__).parent.parent / "data"
QDRANT_PATH      = str(Path(__file__).parent.parent / "qdrant_db")
COLLECTION_NAME  = "support_corpus"
EMBEDDING_DIM    = 384      # all-MiniLM-L6-v2 output dimension

# Chunking
MIN_CHUNK_CHARS  = 60       # skip sections shorter than this (e.g. empty headings)
MAX_CHUNK_CHARS  = 1200     # hard cap — split oversized sections at paragraph boundary

# Retrieval
TOP_K            = 8        # candidates to pull from Qdrant before re-ranking
BI_SCORE_FLOOR   = 0.25     # Qdrant drops results below this (lenient pre-filter)
                             # The strict 0.55 guardrail is applied in agent.py

# Domain mappings
DOMAIN_FOLDER_MAP = {
    "hackerrank": "hackerrank",
    "claude":     "claude",
    "visa":       "visa",
}

# CSV company field → domain label used in Qdrant filter
# None means no filter → search all domains
COMPANY_TO_DOMAIN = {
    "HackerRank": "hackerrank",
    "Claude":     "claude",
    "Visa":       "visa",
    "None":       None,
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SINGLETONS
# Load once on first call, reuse for all 56 tickets.
# Loading MiniLM takes ~3s, cross-encoder ~5s.
# ─────────────────────────────────────────────────────────────────────────────

_biencoder:    Optional[SentenceTransformer] = None
_crossencoder: Optional[CrossEncoder]        = None


def get_biencoder() -> SentenceTransformer:
    """
    Bi-encoder = all-MiniLM-L6-v2
    Converts any text → 384-number vector.
    Fast because query and documents are encoded independently.
    Used at:
        - index time: encode every chunk (runs once)
        - query time: encode each ticket query (runs per ticket)
    """
    global _biencoder
    if _biencoder is None:
        print("[retriever] Loading bi-encoder (all-MiniLM-L6-v2)...")
        _biencoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _biencoder


def get_crossencoder() -> CrossEncoder:
    """
    Cross-encoder = ms-marco-MiniLM-L-6-v2
    Reads (query + chunk) TOGETHER → single relevance score.
    Slower than bi-encoder but much more accurate.
    Used only on top-8 Qdrant results (not all 774 chunks).
    """
    global _crossencoder
    if _crossencoder is None:
        print("[retriever] Loading cross-encoder (ms-marco-MiniLM-L-6-v2)...")
        _crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _crossencoder


# ─────────────────────────────────────────────────────────────────────────────
# FRONTMATTER PARSER
# Each .md file starts with a YAML block between --- markers.
# We extract structured metadata from it instead of treating it as text.
# ─────────────────────────────────────────────────────────────────────────────

def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """
    Split a markdown file into (frontmatter_dict, body_text).

    Frontmatter looks like:
        ---
        title: "Setting Up SCIM for SkillUp"
        source_url: "https://support.hackerrank.com/articles/..."
        last_updated_exact: "Mar 11, 2026"
        breadcrumbs:
          - "SkillUp"
          - "Integrations"
        ---

    Returns:
        meta  = {"title": "...", "source_url": "...", "breadcrumbs": [...], ...}
        body  = everything after the closing ---

    If no frontmatter found, returns ({}, raw)
    """
    # Match the opening --- ... closing --- block
    match = re.match(r"^---\n([\s\S]*?)\n---\n([\s\S]*)$", raw.strip())
    if not match:
        return {}, raw

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}

    body = match.group(2).strip()
    return meta, body


def extract_meta_fields(meta: dict, domain: str) -> dict:
    """
    Normalize frontmatter fields across all three domains.

    Problem: each domain uses slightly different field names:
        HackerRank: last_updated_exact, breadcrumbs, article_slug
        Claude:     last_updated_iso,   breadcrumbs, article_id
        Visa:       last_modified,      no breadcrumbs, description

    We normalize everything into a consistent set of fields.

    Returns a clean dict with guaranteed keys (empty string if missing).
    """
    # Title — all domains have this
    title = meta.get("title", "").strip()
    # Remove repeated words that appear in HackerRank titles
    # e.g. "End an Interview Ending an Interview" → "End an Interview"
    title = re.sub(r"(.{20,})\s+\1", r"\1", title).strip()

    # Source URL — the canonical public URL (cited in responses)
    source_url = meta.get("source_url", meta.get("final_url", "")).strip()

    # Breadcrumbs — navigation path, used to infer product_area
    # e.g. ["SkillUp", "Integrations"] or ["Screen", "Managing Tests"]
    breadcrumbs = meta.get("breadcrumbs", [])
    if isinstance(breadcrumbs, str):
        breadcrumbs = [breadcrumbs]

    # For Visa docs that have no breadcrumbs, infer from folder path
    # We'll fill this from the filepath in chunk_document()

    # Subdomain = first breadcrumb (most specific product area)
    subdomain = breadcrumbs[0].lower().replace(" ", "_") if breadcrumbs else domain

    # Last updated — normalize to ISO date string
    last_updated = (
        meta.get("last_updated_exact")        # HackerRank format
        or meta.get("last_updated_iso")       # Claude format
        or meta.get("last_modified")          # Visa format
        or ""
    )
    # Keep only the date part (drop time component for cleanliness)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2}|\w+ \d+, \d{4})", str(last_updated))
    last_updated = date_match.group(1) if date_match else ""

    # Doc ID — extracted from filename (e.g. "9005750838-setting-up-scim.md" → "9005750838")
    # Filled in later by chunk_document() which has the filepath
    doc_id = meta.get("article_slug", meta.get("article_id", "")).strip()

    # Description — Visa docs have this, others don't
    description = meta.get("description", "").strip()

    return {
        "title":        title,
        "source_url":   source_url,
        "breadcrumbs":  breadcrumbs,
        "subdomain":    subdomain,
        "last_updated": last_updated,
        "doc_id":       doc_id,
        "description":  description,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN-AWARE CHUNKER
#
# Instead of blindly splitting every 500 chars, we split at heading boundaries.
# Each H1/H2/H3 section becomes its own chunk.
#
# Why this is better for support docs:
#   - Each heading = one topic = one potential answer
#   - Sections are already self-contained by the author
#   - No sentence is ever split at a chunk boundary
# ─────────────────────────────────────────────────────────────────────────────

def split_into_sections(body: str) -> list[tuple[str, str]]:
    """
    Split markdown body into (heading, content) pairs at H1/H2/H3 boundaries.

    Example input:
        # Prerequisites
        You need Azure AD configured.

        ## Step 1 — Enable SCIM
        Go to Settings > Integrations.

        ## Step 2 — Add token
        Copy the token from...

    Example output:
        [
            ("Prerequisites",        "You need Azure AD configured."),
            ("Step 1 — Enable SCIM", "Go to Settings > Integrations."),
            ("Step 2 — Add token",   "Copy the token from..."),
        ]

    Content before the first heading is kept as section ("", content).
    """
    # Split at any # heading (H1, H2, H3)
    # re.split with a capture group keeps the delimiter in the result list
    parts = re.split(r"\n(#{1,3} .+)\n", body)

    sections = []
    # parts[0] = content before first heading (if any)
    if parts[0].strip():
        sections.append(("", parts[0].strip()))

    # Remaining parts come in pairs: [heading, content, heading, content, ...]
    i = 1
    while i < len(parts) - 1:
        heading = parts[i].lstrip("#").strip()   # "## Step 1" → "Step 1"
        content = parts[i + 1].strip()
        if content:
            sections.append((heading, content))
        i += 2

    return sections


def split_oversized(section_text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    If a section is too long (e.g. a huge list or table), split it at
    paragraph boundaries (\n\n) rather than blindly at char count.

    This preserves paragraph integrity even when sections are large.

    Example: a 3000-char section with 5 paragraphs →
        returns 3 chunks, each under max_chars, each ending at \n\n boundary.
    """
    if len(section_text) <= max_chars:
        return [section_text]

    paragraphs = section_text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para.strip()

    if current:
        chunks.append(current)

    return chunks if chunks else [section_text[:max_chars]]


def chunk_document(
    raw_text: str,
    filepath: Path,
    domain: str,
) -> list[dict]:
    """
    Full pipeline for one markdown file:
        1. Parse frontmatter → extract metadata
        2. Split body into heading sections
        3. Split oversized sections at paragraph boundaries
        4. Build production payload for each chunk
        5. Build context string for embedding (different from stored text)

    Returns list of chunk dicts, each ready to become a Qdrant PointStruct.

    Each dict has:
        embed_text   : context-rich string → what MiniLM encodes
        payload      : clean production payload → stored in Qdrant
    """
    meta, body = parse_frontmatter(raw_text)
    fields     = extract_meta_fields(meta, domain)

    # Extract doc_id from filename if not in frontmatter
    # e.g. "9005750838-setting-up-scim-provisioning.md" → "9005750838"
    stem = filepath.stem   # filename without extension
    id_match = re.match(r"^(\d+)-", stem)
    doc_id = fields["doc_id"] or (id_match.group(1) if id_match else stem)

    # For Visa docs without breadcrumbs, infer from folder structure
    # e.g. data/visa/support/small-business/fraud-protection.md
    #      → breadcrumbs = ["Small Business", "Fraud Protection"]
    if not fields["breadcrumbs"] and domain == "visa":
        rel_parts = filepath.relative_to(DATA_ROOT).parts
        # Skip "visa", "support" → take remaining folder names
        crumb_parts = [p.replace("-", " ").title() for p in rel_parts[2:-1]]
        fields["breadcrumbs"] = crumb_parts
        fields["subdomain"]   = crumb_parts[0].lower().replace(" ", "_") if crumb_parts else "visa"

    sections = split_into_sections(body)
    chunks   = []
    chunk_idx = 0

    for heading, content in sections:
        # Split oversized sections at paragraph boundaries
        sub_chunks = split_oversized(content)

        for sub in sub_chunks:
            if len(sub) < MIN_CHUNK_CHARS:
                continue   # skip tiny fragments (empty sections, single lines)

            # ── Unique ID ───────────────────────────────────────────────────
            # Deterministic: same file + same chunk index → same ID every run
            # Qdrant requires int (or UUID) as point ID
            raw_id   = f"{doc_id}::{chunk_idx}"
            chunk_id = int(hashlib.md5(raw_id.encode()).hexdigest(), 16) % (10 ** 12)

            # ── Context string for EMBEDDING ────────────────────────────────
            # This is what MiniLM encodes.
            # We prepend document context so the vector "knows" where it came from.
            # The title + breadcrumbs + section words enrich the semantic meaning
            # of the chunk even when those words don't appear in the chunk text.
            context_parts = []
            if fields["title"]:
                context_parts.append(f"Document: {fields['title']}")
            if fields["breadcrumbs"]:
                context_parts.append(f"Category: {' > '.join(fields['breadcrumbs'])}")
            if heading:
                context_parts.append(f"Section: {heading}")
            context_parts.append(f"Domain: {domain}")
            context_parts.append("")         # blank line separator
            context_parts.append(sub)        # actual chunk content

            embed_text = "\n".join(context_parts)

            # ── Production payload for STORAGE ─────────────────────────────
            # This is what gets stored in Qdrant and returned to agent.py.
            # It does NOT include the context header — that was only for embedding.
            payload = {
                # ── Identity ───────────────────────────────────────────────
                "doc_id":        doc_id,
                "chunk_id":      raw_id,                  # "9005750838::2"
                "chunk_idx":     chunk_idx,
                "source":        str(filepath),           # local path for debugging

                # ── Content ────────────────────────────────────────────────
                "text":          sub,                     # clean chunk → sent to Gemini
                "title":         fields["title"],         # doc title → shown in prompt header
                "section":       heading,                 # H2/H3 heading → shown in prompt
                "breadcrumbs":   fields["breadcrumbs"],   # ["SkillUp", "Integrations"]
                "source_url":    fields["source_url"],    # cited in response
                "description":   fields["description"],   # Visa docs have this

                # ── Operational ────────────────────────────────────────────
                "domain":        domain,                  # primary Qdrant filter key
                "subdomain":     fields["subdomain"],     # "skillup", "screen", "interviews"
                "last_updated":  fields["last_updated"],  # freshness signal
                "language":      "en",                    # all docs are English
                "chunk_length":  len(sub),                # char count of this chunk
            }

            chunks.append({
                "id":         chunk_id,
                "embed_text": embed_text,
                "payload":    payload,
            })

            chunk_idx += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT STRING BUILDER (for query time)
# At query time we also want to prepend any known context to the query
# before embedding — so the query vector matches the same "language" as chunks.
# ─────────────────────────────────────────────────────────────────────────────

def build_query_embed_text(query: str, company: str) -> str:
    """
    Prepend domain context to the query before embedding.

    Why:
        Our chunk vectors were built with "Domain: hackerrank" prepended.
        If the query vector has no domain context, there's a slight mismatch.
        Adding domain context to the query brings them into the same space.

    Example:
        query   = "how do I sync users automatically"
        company = "HackerRank"
        output  = "Domain: HackerRank\n\nhow do I sync users automatically"
    """
    if company and company != "None":
        return f"Domain: {company}\n\n{query}"
    return query


# ─────────────────────────────────────────────────────────────────────────────
# INDEX TIME — build the Qdrant collection from the corpus
# ─────────────────────────────────────────────────────────────────────────────

def build_index(force_rebuild: bool = False) -> QdrantClient:
    """
    Full indexing pipeline: read corpus → chunk → embed → store in Qdrant.

    force_rebuild=False (default):
        If collection already exists, return client immediately.
        Makes `python main.py run` instant on 2nd+ invocation.

    force_rebuild=True:
        Deletes collection and rebuilds from scratch.
        Use when corpus changes or chunking logic is updated.

    Returns a connected QdrantClient ready for searching.
    """
    client   = QdrantClient(path=QDRANT_PATH)
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing and not force_rebuild:
        count = client.get_collection(COLLECTION_NAME).points_count
        print(f"[retriever] Index exists ({count} points). Skipping rebuild.")
        return client

    if COLLECTION_NAME in existing:
        print("[retriever] Deleting existing collection for rebuild...")
        client.delete_collection(COLLECTION_NAME)

    # Distance.COSINE:
    #   similarity = 1.0 → vectors point in the same direction → very similar text
    #   similarity = 0.0 → perpendicular → unrelated text
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[retriever] Created collection '{COLLECTION_NAME}'")

    model    = get_biencoder()
    md_files = list(DATA_ROOT.rglob("*.md"))
    print(f"[retriever] Chunking {len(md_files)} markdown files...")

    all_chunks = []
    for filepath in md_files:
        rel       = filepath.relative_to(DATA_ROOT)
        domain    = DOMAIN_FOLDER_MAP.get(rel.parts[0].lower(), "unknown")
        raw_text  = filepath.read_text(encoding="utf-8", errors="ignore")
        chunks    = chunk_document(raw_text, filepath, domain)
        all_chunks.extend(chunks)

    print(f"[retriever] Total chunks to embed: {len(all_chunks)}")

    # Batch embedding — MiniLM handles batches efficiently
    BATCH = 64
    total_upserted = 0

    for i in range(0, len(all_chunks), BATCH):
        batch      = all_chunks[i : i + BATCH]
        # Embed the context-rich string (not the stored text)
        embed_texts = [c["embed_text"] for c in batch]
        vectors    = model.encode(embed_texts, show_progress_bar=False).tolist()

        points = [
            PointStruct(
                id      = c["id"],
                vector  = v,
                payload = c["payload"],   # production payload stored, not embed_text
            )
            for c, v in zip(batch, vectors)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_upserted += len(points)

        done = min(i + BATCH, len(all_chunks))
        if done % 500 == 0 or done == len(all_chunks):
            print(f"[retriever]   {done}/{len(all_chunks)} chunks indexed...")

    print(f"[retriever] ✓ Index complete — {total_upserted} vectors stored.")
    return client


# ─────────────────────────────────────────────────────────────────────────────
# QUERY TIME — search Qdrant + cross-encoder re-rank
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query:   str,
    company: str,
    client:  QdrantClient,
    top_k:   int = TOP_K,
) -> list[dict]:
    """
    Find the most relevant corpus chunks for a support ticket query.

    Pipeline:
        1. Prepend domain context to query (matches the chunk embedding space)
        2. Embed with bi-encoder (MiniLM) → query vector
        3. Qdrant search filtered by domain → top_k candidates
        4. Cross-encoder re-ranks candidates (reads query + chunk together)
        5. Sort by cross_score, return

    Args:
        query:   Support ticket text (may be rewritten by agent before calling)
        company: CSV company field — "HackerRank", "Claude", "Visa", or "None"
        client:  Connected QdrantClient (from build_index())
        top_k:   Candidates to retrieve before re-ranking

    Returns list of result dicts sorted by cross_score (best first):
        [
          {
            "text":        "...",            ← sent to Gemini as context
            "title":       "...",            ← shown as context header in prompt
            "section":     "Prerequisites", ← shown in prompt
            "breadcrumbs": ["SkillUp", ...],← used for product_area
            "source_url":  "https://...",   ← cited in response
            "domain":      "hackerrank",
            "subdomain":   "skillup",
            "last_updated":"2026-03-11",
            "source":      "data/...",      ← for trace log
            "bi_score":    0.78,            ← Qdrant cosine similarity
            "cross_score": 0.91,            ← cross-encoder score (sort key)
          },
          ...
        ]
    """
    t0 = time.time()

    # Step 1 — resolve domain filter
    domain = COMPANY_TO_DOMAIN.get(company)   # None = no filter = search all domains

    # Step 2 — embed query with domain context prepended
    query_embed = build_query_embed_text(query, company)
    query_vector = get_biencoder().encode(query_embed).tolist()

    # Step 3 — Qdrant search with optional domain filter
    qdrant_filter = None
    if domain:
        # Only return chunks where payload.domain == domain
        # This is the domain routing step — HackerRank tickets never touch Visa docs
        qdrant_filter = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        )

    results = client.search(
        collection_name  = COLLECTION_NAME,
        query_vector     = query_vector,
        query_filter     = qdrant_filter,
        limit            = top_k,
        score_threshold  = BI_SCORE_FLOOR,   # lenient pre-filter
        with_payload     = True,
    )

    if not results:
        print(f"[retriever] ⚠ Zero results for: '{query[:70]}'")
        return []

    # Step 4 — cross-encoder re-ranking
    # Cross-encoder reads (query, chunk_text) TOGETHER — not the embed_text
    # We use the clean stored text here, not the context string
    pairs        = [(query, r.payload["text"]) for r in results]
    cross_scores = get_crossencoder().predict(pairs).tolist()

    # Step 5 — combine + sort by cross_score
    combined = []
    for result, cs in zip(results, cross_scores):
        p = result.payload
        combined.append({
            # Content fields (used by agent.py to build Gemini prompt)
            "text":        p.get("text", ""),
            "title":       p.get("title", ""),
            "section":     p.get("section", ""),
            "breadcrumbs": p.get("breadcrumbs", []),
            "source_url":  p.get("source_url", ""),
            "description": p.get("description", ""),

            # Operational fields (used for product_area, tracing, guardrails)
            "domain":      p.get("domain", ""),
            "subdomain":   p.get("subdomain", ""),
            "last_updated":p.get("last_updated", ""),
            "source":      p.get("source", ""),
            "chunk_id":    p.get("chunk_id", ""),

            # Scores (used by guardrails.py for confidence threshold)
            "bi_score":    round(result.score, 4),
            "cross_score": round(float(cs), 4),
        })

    combined.sort(key=lambda x: x["cross_score"], reverse=True)

    ms  = round((time.time() - t0) * 1000)
    top = combined[0]["cross_score"] if combined else "N/A"
    print(f"[retriever] {len(combined)} chunks | top_cross={top} | {ms}ms")

    return combined