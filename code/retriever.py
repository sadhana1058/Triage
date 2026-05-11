"""
retriever.py
============

HYBRID SEARCH — Dense + Sparse vectors, fused with RRF.

  Dense  (all-MiniLM-L6-v2):   captures semantic meaning
                                "sync users" matches "provision accounts"
  Sparse (BM25 via rank_bm25):  captures exact keywords
                                "SCIM", "OAuth", "cs_live_abcdefgh"
  Fusion (RRF):                 combines rankings — no weight tuning needed

INDEX TIME (runs once, persists to disk):
    774 .md files
    → parse frontmatter  (title, breadcrumbs, source_url, last_updated)
    → markdown-aware chunking  (split at # headings)
    → build context string  (title + breadcrumbs + section + text)
    → MiniLM encodes context string → dense vector  (384 floats)
    → BM25 encodes chunk text      → sparse vector  ({word_id: idf_score})
    → both stored in one Qdrant point with production payload

QUERY TIME (runs per ticket):
    query string + company
    → MiniLM encodes query        → dense query vector
    → BM25 encodes query          → sparse query vector
    → Qdrant RRF fusion search    → top-8 candidates
    → cross-encoder re-ranks      → final ranked list
    → return to agent.py          (same interface as before)

KEY DESIGN:
    embed_text  ≠  stored text
    We embed (title + breadcrumbs + section + chunk) for better retrieval.
    We store only the clean chunk text for the Gemini prompt.
"""

from __future__ import annotations

import re
import hashlib
import time
import pickle
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
from rank_bm25 import BM25Okapi

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector,
    Filter, FieldCondition, MatchValue,
    Prefetch, FusionQuery, Fusion,
    NamedVector, NamedSparseVector,
)
from sentence_transformers import SentenceTransformer, CrossEncoder


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT        = Path(__file__).parent.parent / "data"
QDRANT_PATH      = str(Path(__file__).parent.parent / "qdrant_db")
BM25_CACHE_PATH  = Path(__file__).parent.parent / "qdrant_db" / "bm25_state.pkl"
COLLECTION_NAME  = "support_corpus"
EMBEDDING_DIM    = 384        # all-MiniLM-L6-v2 output size

# Chunking
MIN_CHUNK_CHARS  = 60         # skip tiny fragments
MAX_CHUNK_CHARS  = 1200       # split oversized sections at paragraph boundary

# Retrieval
TOP_K            = 8          # candidates before cross-encoder re-ranking
BI_SCORE_FLOOR   = 0.20       # lenient Qdrant pre-filter (strict 0.55 in agent.py)

# Qdrant vector names (used in hybrid search calls)
DENSE_VEC   = "dense"
SPARSE_VEC  = "sparse"

# Domain mappings
DOMAIN_FOLDER_MAP = {
    "hackerrank": "hackerrank",
    "claude":     "claude",
    "visa":       "visa",
}

COMPANY_TO_DOMAIN = {
    "HackerRank": "hackerrank",
    "Claude":     "claude",
    "Visa":       "visa",
    "None":       None,        # search all domains
}

# BM25 stopwords — common words with no discriminative value
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "this", "that", "it", "its",
    "i", "we", "you", "he", "she", "they", "my", "your", "our",
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SINGLETONS — load once, reuse for all tickets
# ─────────────────────────────────────────────────────────────────────────────

_biencoder:    Optional[SentenceTransformer] = None
_crossencoder: Optional[CrossEncoder]        = None
_bm25:         Optional[BM25Okapi]           = None   # built from full corpus at index time
_vocab:        Optional[dict[str, int]]      = None   # word → integer index for sparse vectors


def get_biencoder() -> SentenceTransformer:
    """
    Bi-encoder = all-MiniLM-L6-v2
    Text → 384-float dense vector.
    Fast: query and docs encoded independently.
    Used at index time (all chunks) and query time (each ticket).
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
    Slower than bi-encoder but far more accurate.
    Only applied to top-8 Qdrant results, not all chunks.
    """
    global _crossencoder
    if _crossencoder is None:
        print("[retriever] Loading cross-encoder (ms-marco-MiniLM-L-6-v2)...")
        _crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _crossencoder


def get_bm25_state() -> tuple[BM25Okapi, dict[str, int]]:
    """
    Returns (bm25_model, vocab) — loaded from cache if available.
    Built over the full corpus at index time.
    Vocab maps word → integer index (Qdrant sparse vectors need integer indices).
    """
    global _bm25, _vocab
    if _bm25 is None or _vocab is None:
        if BM25_CACHE_PATH.exists():
            print("[retriever] Loading BM25 state from cache...")
            with open(BM25_CACHE_PATH, "rb") as f:
                state = pickle.load(f)
            _bm25  = state["bm25"]
            _vocab = state["vocab"]
        else:
            raise RuntimeError(
                "BM25 state not found. Run build_index() first."
            )
    return _bm25, _vocab


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER (shared by BM25 at index time and query time)
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Lowercase + extract alphanumeric tokens + remove stopwords.

    Why this tokenizer?
        - Lowercasing: "SCIM" and "scim" should match
        - Alphanumeric only: strip punctuation that adds noise
        - Stopwords removed: "the", "is", "of" add no discriminative value
          and inflate sparse vector size
        - Keep numbers: "2.0", "oauth2", "cs_live_abcdefgh" are meaningful

    Example:
        "Setting up SCIM 2.0 for SkillUp users"
        → ["setting", "up", "scim", "2", "0", "skillup", "users"]
        → after stopwords: ["setting", "scim", "2", "0", "skillup", "users"]
    """
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE VECTOR BUILDER
# Converts text → SparseVector using BM25 IDF weights
# ─────────────────────────────────────────────────────────────────────────────

def text_to_sparse(
    text: str,
    vocab: dict[str, int],
    bm25:  BM25Okapi,
) -> SparseVector:
    """
    Convert text to a Qdrant SparseVector using BM25 IDF weights.

    What a sparse vector is:
        A normal dense vector has a value at every position (384 floats).
        A sparse vector only stores positions where the value is non-zero.
        For a 30,000-word vocabulary, most documents use ~50-200 unique words,
        so 99%+ of positions are zero — sparse format saves memory.

    SparseVector format:
        indices = [4821, 2034, 9103]   ← which vocab positions are non-zero
        values  = [0.91, 0.76, 0.54]   ← BM25 IDF weight for each word

    Why IDF as the weight?
        IDF (Inverse Document Frequency) = log(N / df)
        High IDF = word appears in few documents = very discriminative
        Low IDF  = word appears in many documents = common, less useful
        "SCIM" → high IDF (rare term, very specific)
        "setting" → low IDF (appears in many docs)

    Returns SparseVector(indices=[...], values=[...])
    """
    tokens  = tokenize(text)
    seen    = set()
    indices = []
    values  = []

    for token in tokens:
        if token in seen:
            continue
        seen.add(token)

        if token in vocab and token in bm25.idf:
            idf = float(bm25.idf[token])
            if idf > 0:
                indices.append(vocab[token])
                values.append(idf)

    return SparseVector(indices=indices, values=values)


# ─────────────────────────────────────────────────────────────────────────────
# FRONTMATTER PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """
    Split a markdown file into (frontmatter_dict, body_text).

    Frontmatter is the YAML block between --- markers at the top of each file:
        ---
        title: "Setting Up SCIM for SkillUp"
        source_url: "https://support.hackerrank.com/articles/..."
        breadcrumbs:
          - "SkillUp"
          - "Integrations"
        ---

    Returns:
        meta = {"title": "...", "breadcrumbs": [...], ...}
        body = everything after the closing ---

    If no frontmatter, returns ({}, raw).
    """
    match = re.match(r"^---\n([\s\S]*?)\n---\n([\s\S]*)$", raw.strip())
    if not match:
        return {}, raw
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, match.group(2).strip()


def extract_meta_fields(meta: dict, domain: str) -> dict:
    """
    Normalize frontmatter across all three domains into one consistent schema.

    Problem: each domain uses different field names:
        HackerRank → last_updated_exact, breadcrumbs, article_slug
        Claude     → last_updated_iso,   breadcrumbs, article_id
        Visa       → last_modified,      no breadcrumbs, description

    We map all of these to the same output keys so downstream code
    never needs to know which domain a chunk came from.
    """
    # Title
    title = meta.get("title", "").strip()
    # Fix HackerRank duplicate titles: "End Interview Ending Interview" → "End Interview"
    title = re.sub(r"(.{20,})\s+\1", r"\1", title).strip()

    # Source URL (canonical public link, cited in responses)
    source_url = meta.get("source_url", meta.get("final_url", "")).strip()

    # Breadcrumbs (navigation path → used to infer product_area)
    breadcrumbs = meta.get("breadcrumbs", [])
    if isinstance(breadcrumbs, str):
        breadcrumbs = [breadcrumbs]

    # Subdomain = first breadcrumb (most specific product area)
    subdomain = breadcrumbs[0].lower().replace(" ", "_") if breadcrumbs else domain

    # Last updated — normalize across formats
    last_updated = (
        meta.get("last_updated_exact")   # HackerRank
        or meta.get("last_updated_iso")  # Claude
        or meta.get("last_modified")     # Visa
        or ""
    )
    date_match   = re.search(r"(\d{4}-\d{2}-\d{2}|\w+ \d+, \d{4})", str(last_updated))
    last_updated = date_match.group(1) if date_match else ""

    # Doc ID — from frontmatter if available, else parsed from filename
    doc_id      = str(meta.get("article_slug", meta.get("article_id", ""))).strip()
    description = meta.get("description", "").strip()   # Visa docs only

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
# Split at heading boundaries instead of blind character count.
# Each H1/H2/H3 section = one chunk = one answerable topic.
# ─────────────────────────────────────────────────────────────────────────────

def split_into_sections(body: str) -> list[tuple[str, str]]:
    """
    Split markdown body into (heading, content) pairs at H1/H2/H3 headings.

    Example:
        "# Prerequisites\nYou need Azure AD.\n\n## Step 1\nGo to Settings."
        → [("Prerequisites", "You need Azure AD."), ("Step 1", "Go to Settings.")]

    Content before the first heading → ("", content).
    """
    parts    = re.split(r"\n(#{1,3} .+)\n", body)
    sections = []

    if parts[0].strip():
        sections.append(("", parts[0].strip()))

    i = 1
    while i < len(parts) - 1:
        heading = parts[i].lstrip("#").strip()
        content = parts[i + 1].strip()
        if content:
            sections.append((heading, content))
        i += 2

    return sections


def split_oversized(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    If a section exceeds max_chars, split at paragraph boundaries (\n\n).
    Preserves paragraph integrity — never cuts mid-sentence.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para.strip()

    if current:
        chunks.append(current)

    return chunks or [text[:max_chars]]


def chunk_document(
    raw_text: str,
    filepath: Path,
    domain:   str,
) -> list[dict]:
    """
    Full pipeline for one markdown file:
        1. Parse frontmatter → extract metadata
        2. Split body at headings → sections
        3. Split oversized sections at paragraph boundaries
        4. For each chunk:
             - build embed_text  (context-rich, for MiniLM)
             - build payload     (clean, for Qdrant storage and Gemini)

    Returns list of chunk dicts:
        {
            "id":         int (deterministic hash),
            "embed_text": str (what MiniLM encodes),
            "text":       str (what gets stored + sent to Gemini),
            "payload":    dict (full production payload),
        }

    WHY embed_text ≠ stored text:
        embed_text = "Document: SCIM Setup\nCategory: SkillUp > Integrations\n\nYou need Azure AD..."
        stored     = "You need Azure AD..."

        The vector carries full document context → better retrieval.
        The stored text is clean → better Gemini prompt (no noise).
    """
    meta, body = parse_frontmatter(raw_text)
    fields     = extract_meta_fields(meta, domain)

    # Extract doc_id from filename if frontmatter doesn't have it
    stem     = filepath.stem
    id_match = re.match(r"^(\d+)-", stem)
    doc_id   = fields["doc_id"] or (id_match.group(1) if id_match else stem)

    # Visa docs have no breadcrumbs — infer from folder structure
    # data/visa/support/small-business/fraud-protection.md
    # → ["Small Business"]
    if not fields["breadcrumbs"] and domain == "visa":
        rel_parts           = filepath.relative_to(DATA_ROOT).parts
        crumb_parts         = [p.replace("-", " ").title() for p in rel_parts[2:-1]]
        fields["breadcrumbs"] = crumb_parts
        fields["subdomain"]   = crumb_parts[0].lower().replace(" ", "_") if crumb_parts else "visa"

    sections  = split_into_sections(body)
    chunks    = []
    chunk_idx = 0

    for heading, content in sections:
        for sub in split_oversized(content):
            if len(sub) < MIN_CHUNK_CHARS:
                continue

            # Deterministic integer ID (Qdrant requires int or UUID)
            raw_id   = f"{doc_id}::{chunk_idx}"
            chunk_id = int(hashlib.md5(raw_id.encode()).hexdigest(), 16) % (10 ** 12)

            # ── embed_text: context-rich, used by MiniLM ──────────────────
            # Prepending title + breadcrumbs + section means the vector
            # "knows" the document context even when those words aren't in
            # the chunk text itself.
            ctx = []
            if fields["title"]:
                ctx.append(f"Document: {fields['title']}")
            if fields["breadcrumbs"]:
                ctx.append(f"Category: {' > '.join(fields['breadcrumbs'])}")
            if heading:
                ctx.append(f"Section: {heading}")
            ctx.append(f"Domain: {domain}")
            ctx.append("")
            ctx.append(sub)
            embed_text = "\n".join(ctx)

            # ── payload: production fields stored in Qdrant ───────────────
            payload = {
                # Identity
                "doc_id":       doc_id,
                "chunk_id":     raw_id,           # "9005750838::2"
                "chunk_idx":    chunk_idx,
                "source":       str(filepath),

                # Content (agent.py reads these to build Gemini prompt)
                "text":         sub,              # clean chunk text
                "title":        fields["title"],
                "section":      heading,
                "breadcrumbs":  fields["breadcrumbs"],
                "source_url":   fields["source_url"],
                "description":  fields["description"],

                # Operational (filtering, product_area, tracing)
                "domain":       domain,
                "subdomain":    fields["subdomain"],
                "last_updated": fields["last_updated"],
                "language":     "en",
                "chunk_length": len(sub),
            }

            chunks.append({
                "id":         chunk_id,
                "embed_text": embed_text,   # → MiniLM → dense vector
                "text":       sub,          # → BM25   → sparse vector (raw text, not embed_text)
                "payload":    payload,
            })
            chunk_idx += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# INDEX TIME
# ─────────────────────────────────────────────────────────────────────────────

def build_index(force_rebuild: bool = False) -> QdrantClient:
    """
    Full indexing pipeline.

    Steps:
        1. Walk all 774 .md files, chunk each one
        2. Build BM25 model over all chunk texts → save to disk
        3. Build vocab (word → int index) → save to disk
        4. Create Qdrant collection with BOTH dense + sparse vector configs
        5. For each chunk:
             dense  = MiniLM.encode(embed_text)
             sparse = BM25 IDF weights for words in chunk text
        6. Upsert both vectors + payload into Qdrant

    force_rebuild=False (default):
        Skip if collection already exists. Makes startup instant on run 2+.
    force_rebuild=True:
        Delete and rebuild. Use when corpus or chunking logic changes.

    Returns connected QdrantClient for query time.
    """
    client   = QdrantClient(path=QDRANT_PATH)
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing and not force_rebuild:
        count = client.get_collection(COLLECTION_NAME).points_count
        print(f"[retriever] Index exists ({count} points). Skipping rebuild.")
        # Load BM25 state into memory for query time
        get_bm25_state()
        return client

    if COLLECTION_NAME in existing:
        print("[retriever] Deleting collection for rebuild...")
        client.delete_collection(COLLECTION_NAME)

    # ── Step 1: Chunk all documents ──────────────────────────────────────────
    md_files   = list(DATA_ROOT.rglob("*.md"))
    print(f"[retriever] Chunking {len(md_files)} markdown files...")

    all_chunks = []
    for filepath in md_files:
        rel      = filepath.relative_to(DATA_ROOT)
        domain   = DOMAIN_FOLDER_MAP.get(rel.parts[0].lower(), "unknown")
        raw_text = filepath.read_text(encoding="utf-8", errors="ignore")
        all_chunks.extend(chunk_document(raw_text, filepath, domain))

    print(f"[retriever] Total chunks: {len(all_chunks)}")

    # ── Step 2: Build BM25 model over all chunk texts ────────────────────────
    # BM25 needs to see ALL documents to compute IDF correctly.
    # IDF = log(total_docs / docs_containing_word)
    # A word appearing in only 2 of 5000 chunks → high IDF → very discriminative
    # A word appearing in 4000 of 5000 chunks → low IDF → common, less useful
    print("[retriever] Building BM25 model over full corpus...")
    tokenized_corpus = [tokenize(c["text"]) for c in all_chunks]
    bm25             = BM25Okapi(tokenized_corpus)

    # ── Step 3: Build vocab ──────────────────────────────────────────────────
    # Qdrant sparse vectors need integer indices, not strings.
    # We build a global word → int mapping and save it.
    vocab: dict[str, int] = {}
    for tokens in tokenized_corpus:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    print(f"[retriever] Vocabulary size: {len(vocab)} unique tokens")

    # Save BM25 + vocab to disk so query time can load without rebuilding
    BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "vocab": vocab}, f)
    print(f"[retriever] BM25 state saved to {BM25_CACHE_PATH}")

    # Cache in memory
    global _bm25, _vocab
    _bm25, _vocab = bm25, vocab

    # ── Step 4: Create Qdrant collection with hybrid vector config ───────────
    # Named vectors: "dense" for MiniLM, "sparse" for BM25
    # This is different from single-vector collections.
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VEC: VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VEC: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),  # keep in RAM for speed
            ),
        },
    )
    print(f"[retriever] Created hybrid collection '{COLLECTION_NAME}'")

    # ── Step 5 + 6: Embed + upsert in batches ───────────────────────────────
    biencoder      = get_biencoder()
    BATCH          = 64
    total_upserted = 0

    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i : i + BATCH]

        # Dense vectors: MiniLM encodes the context-rich embed_text
        dense_vecs = biencoder.encode(
            [c["embed_text"] for c in batch],
            show_progress_bar=False,
        ).tolist()

        # Sparse vectors: BM25 IDF weights for words in clean chunk text
        sparse_vecs = [
            text_to_sparse(c["text"], vocab, bm25)
            for c in batch
        ]

        # Build PointStructs with NAMED vectors
        points = [
            PointStruct(
                id      = c["id"],
                vector  = {
                    DENSE_VEC:  dv,    # list of 384 floats
                    SPARSE_VEC: sv,    # SparseVector(indices, values)
                },
                payload = c["payload"],
            )
            for c, dv, sv in zip(batch, dense_vecs, sparse_vecs)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_upserted += len(points)

        done = min(i + BATCH, len(all_chunks))
        if done % 500 == 0 or done == len(all_chunks):
            print(f"[retriever]   {done}/{len(all_chunks)} chunks indexed...")

    print(f"[retriever] ✓ Index complete — {total_upserted} vectors stored.")
    return client


# ─────────────────────────────────────────────────────────────────────────────
# QUERY TIME
# ─────────────────────────────────────────────────────────────────────────────

def build_query_embed_text(query: str, company: str) -> str:
    """
    Prepend domain context to query before dense embedding.

    Why:
        Chunk vectors were built with "Domain: hackerrank" prepended.
        Prepending the same context to the query brings it into the same
        semantic space → better cosine similarity scores.
    """
    if company and company != "None":
        return f"Domain: {company}\n\n{query}"
    return query


def retrieve(
    query:   str,
    company: str,
    client:  QdrantClient,
    top_k:   int = TOP_K,
) -> list[dict]:
    """
    Hybrid search: dense + sparse → RRF fusion → cross-encoder re-rank.

    Pipeline:
        1. Build dense query vector  (MiniLM on context-prepended query)
        2. Build sparse query vector (BM25 IDF weights for query tokens)
        3. Qdrant prefetch: run dense search → top_k candidates
        4. Qdrant prefetch: run sparse search → top_k candidates
        5. RRF fusion: merge both candidate lists by rank position
        6. Apply domain filter (HackerRank tickets never search Visa docs)
        7. Cross-encoder re-ranks the fused top_k results
        8. Return sorted list, best chunk first

    WHY RRF (Reciprocal Rank Fusion):
        Dense score = 0.91 (cosine, 0-1 scale)
        Sparse score = 12.4 (BM25, unbounded scale)
        You CANNOT add these directly — they're on different scales.
        RRF works on RANK POSITIONS, not raw scores:
            score = 1/(rank_dense + 60) + 1/(rank_sparse + 60)
        Scale doesn't matter. No weight tuning needed.
        60 is a standard constant that prevents top-rank dominance.

    Returns list of dicts sorted by cross_score (best first).
    Same interface as before — agent.py doesn't know about hybrid internals.
    """
    t0 = time.time()

    bm25, vocab = get_bm25_state()
    domain      = COMPANY_TO_DOMAIN.get(company)

    # Step 1 — dense query vector
    query_embed  = build_query_embed_text(query, company)
    dense_qvec   = get_biencoder().encode(query_embed).tolist()

    # Step 2 — sparse query vector
    sparse_qvec  = text_to_sparse(query, vocab, bm25)

    # Step 3+4+5 — Qdrant hybrid search with RRF fusion
    # Prefetch = run each search independently first, then fuse.
    # This is Qdrant's native hybrid search pattern.
    qdrant_filter = None
    if domain:
        qdrant_filter = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        )

    results = client.query_points(
        collection_name = COLLECTION_NAME,
        prefetch        = [
            # Dense arm: semantic search
            Prefetch(
                query        = NamedVector(name=DENSE_VEC, vector=dense_qvec),
                filter       = qdrant_filter,
                limit        = top_k,
                score_threshold = BI_SCORE_FLOOR,
            ),
            # Sparse arm: keyword search
            Prefetch(
                query        = NamedSparseVector(name=SPARSE_VEC, vector=sparse_qvec),
                filter       = qdrant_filter,
                limit        = top_k,
            ),
        ],
        # RRF fusion: merge the two candidate lists by rank position
        query  = FusionQuery(fusion=Fusion.RRF),
        limit  = top_k,
        with_payload = True,
    ).points

    if not results:
        print(f"[retriever] ⚠ Zero results for: '{query[:70]}'")
        return []

    # Step 6 — cross-encoder re-ranking
    # Cross-encoder reads (query, clean_chunk_text) together
    # → much more accurate relevance score than either bi-encoder gives
    pairs        = [(query, r.payload["text"]) for r in results]
    cross_scores = get_crossencoder().predict(pairs).tolist()

    # Step 7 — combine + sort by cross_score
    combined = sorted(
        [
            {
                # Content (agent.py uses these for Gemini prompt)
                "text":        r.payload.get("text", ""),
                "title":       r.payload.get("title", ""),
                "section":     r.payload.get("section", ""),
                "breadcrumbs": r.payload.get("breadcrumbs", []),
                "source_url":  r.payload.get("source_url", ""),
                "description": r.payload.get("description", ""),

                # Operational (product_area, trace log, guardrails)
                "domain":      r.payload.get("domain", ""),
                "subdomain":   r.payload.get("subdomain", ""),
                "last_updated":r.payload.get("last_updated", ""),
                "source":      r.payload.get("source", ""),
                "chunk_id":    r.payload.get("chunk_id", ""),

                # Scores
                "rrf_score":   round(r.score, 6),    # RRF fusion score
                "cross_score": round(float(cs), 4),  # cross-encoder (final sort key)
            }
            for r, cs in zip(results, cross_scores)
        ],
        key     = lambda x: x["cross_score"],
        reverse = True,
    )

    ms  = round((time.time() - t0) * 1000)
    top = combined[0]["cross_score"] if combined else "N/A"
    print(f"[retriever] {len(combined)} chunks | top_cross={top} | {ms}ms")

    return combined