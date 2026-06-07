"""
retriever.py
============

HYBRID SEARCH — Dense + Sparse vectors, fused with RRF.

  Dense  (all-MiniLM-L6-v2):   captures semantic meaning
                                "sync users" matches "provision accounts"
  Sparse (BM25 via rank_bm25):  captures exact keywords
                                "SCIM", "OAuth", "cs_live_abcdefgh"
  Fusion (RRF):                 combines rankings — no weight tuning needed

CHANGES FROM V1:
    1. Fixed Prefetch API for qdrant-client >= 1.10
       (using= instead of NamedVector/NamedSparseVector)
    2. Added overlapping chunking (200-char overlap between chunks)
       so answers that span section boundaries are not missed
    3. Removed broken NamedVector / NamedSparseVector imports

INDEX TIME (runs once, persists to disk):
    774 .md files
    → parse frontmatter  (title, breadcrumbs, source_url, last_updated)
    → markdown-aware chunking with 200-char overlap
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
)
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    DATA_ROOT, QDRANT_PATH, BM25_CACHE_PATH, COLLECTION_NAME,
    EMBEDDING_DIM, BIENCODER_MODEL, CROSSENCODER_MODEL,
    MIN_CHUNK_CHARS, MAX_CHUNK_CHARS, OVERLAP_CHARS,
    TOP_K, BI_SCORE_FLOOR,
    DENSE_VEC, SPARSE_VEC,
    DOMAIN_FOLDER_MAP, COMPANY_TO_DOMAIN, STOPWORDS,
)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────

_biencoder:    Optional[SentenceTransformer] = None
_crossencoder: Optional[CrossEncoder]        = None
_bm25:         Optional[BM25Okapi]           = None
_vocab:        Optional[dict[str, int]]      = None


def get_biencoder() -> SentenceTransformer:
    global _biencoder
    if _biencoder is None:
        print(f"[retriever] Loading bi-encoder ({BIENCODER_MODEL})...")
        _biencoder = SentenceTransformer(BIENCODER_MODEL)
    return _biencoder


def get_crossencoder() -> CrossEncoder:
    global _crossencoder
    if _crossencoder is None:
        print(f"[retriever] Loading cross-encoder ({CROSSENCODER_MODEL})...")
        _crossencoder = CrossEncoder(CROSSENCODER_MODEL)
    return _crossencoder


def get_bm25_state() -> tuple[BM25Okapi, dict[str, int]]:
    global _bm25, _vocab
    if _bm25 is None or _vocab is None:
        if BM25_CACHE_PATH.exists():
            print("[retriever] Loading BM25 state from cache...")
            with open(BM25_CACHE_PATH, "rb") as f:
                state = pickle.load(f)
            _bm25  = state["bm25"]
            _vocab = state["vocab"]
        else:
            raise RuntimeError("BM25 state not found. Run build_index() first.")
    return _bm25, _vocab


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE VECTOR BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def text_to_sparse(
    text: str,
    vocab: dict[str, int],
    bm25:  BM25Okapi,
) -> SparseVector:
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
    match = re.match(r"^---\n([\s\S]*?)\n---\n([\s\S]*)$", raw.strip())
    if not match:
        return {}, raw
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, match.group(2).strip()


def extract_meta_fields(meta: dict, domain: str) -> dict:
    title = meta.get("title", "").strip()
    title = re.sub(r"(.{20,})\s+\1", r"\1", title).strip()

    source_url  = meta.get("source_url", meta.get("final_url", "")).strip()
    breadcrumbs = meta.get("breadcrumbs", [])
    if isinstance(breadcrumbs, str):
        breadcrumbs = [breadcrumbs]

    subdomain    = breadcrumbs[0].lower().replace(" ", "_") if breadcrumbs else domain
    last_updated = (
        meta.get("last_updated_exact")
        or meta.get("last_updated_iso")
        or meta.get("last_modified")
        or ""
    )
    date_match   = re.search(r"(\d{4}-\d{2}-\d{2}|\w+ \d+, \d{4})", str(last_updated))
    last_updated = date_match.group(1) if date_match else ""
    doc_id       = str(meta.get("article_slug", meta.get("article_id", ""))).strip()
    description  = meta.get("description", "").strip()

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
# MARKDOWN-AWARE CHUNKER WITH OVERLAP
# ─────────────────────────────────────────────────────────────────────────────

def split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (heading, content) pairs at H1/H2/H3 headings."""
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


def split_with_overlap(
    text:          str,
    max_chars:     int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[str]:
    """
    Split text into overlapping chunks at paragraph boundaries.

    WHY OVERLAP?
        Without overlap: answer spans sections A and B → neither chunk alone
        has enough info → low cross_score → escalated.
        With overlap: chunk 1 ends with last 200 chars of section A,
        chunk 2 starts with those same 200 chars → cross-encoder sees
        full context → higher score → replied with correct answer.

    Example (overlap_chars=200):
        Section A: "...Configure the timeout in Settings > Interviews.
                    The default is 20 minutes for both parties."       ← last 200 chars
        Section B: "...The default is 20 minutes for both parties.     ← first 200 chars (overlap)
                    To extend it, go to Admin > Inactivity Settings."

        Query: "extend inactivity timeout"
        Without overlap: chunk B misses the "Settings > Interviews" context
        With overlap:    chunk B has it → better answer

    ALGORITHM:
        Split at paragraph boundaries (\n\n) to avoid cutting mid-sentence.
        When current chunk exceeds max_chars:
            1. Save current chunk
            2. Start new chunk with last overlap_chars of previous chunk
               (trimmed to nearest paragraph boundary)
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks     = []
    current    = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)

            # Build overlap from the tail of the previous chunk
            # Find a clean paragraph boundary within the overlap window
            overlap_text = current[-overlap_chars:] if len(current) > overlap_chars else current
            # Try to start at a paragraph boundary within the overlap
            para_boundary = overlap_text.find("\n\n")
            if para_boundary != -1 and para_boundary < len(overlap_text) - 50:
                overlap_text = overlap_text[para_boundary:].strip()

            current = (overlap_text + "\n\n" + para).strip()

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
        3. Split oversized sections with overlap
        4. For each chunk: build embed_text + payload
    """
    meta, body = parse_frontmatter(raw_text)
    fields     = extract_meta_fields(meta, domain)

    stem     = filepath.stem
    id_match = re.match(r"^(\d+)-", stem)
    doc_id   = fields["doc_id"] or (id_match.group(1) if id_match else stem)

    # Visa docs: infer breadcrumbs from folder structure
    if not fields["breadcrumbs"] and domain == "visa":
        rel_parts             = filepath.relative_to(DATA_ROOT).parts
        crumb_parts           = [p.replace("-", " ").title() for p in rel_parts[2:-1]]
        fields["breadcrumbs"] = crumb_parts
        fields["subdomain"]   = crumb_parts[0].lower().replace(" ", "_") if crumb_parts else "visa"

    sections  = split_into_sections(body)
    chunks    = []
    chunk_idx = 0

    for heading, content in sections:
        # Use overlapping split instead of non-overlapping split_oversized
        for sub in split_with_overlap(content):
            if len(sub) < MIN_CHUNK_CHARS:
                continue

            raw_id   = f"{doc_id}::{chunk_idx}"
            chunk_id = int(hashlib.md5(raw_id.encode()).hexdigest(), 16) % (10 ** 12)

            # embed_text: context-rich string for MiniLM
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

            payload = {
                "doc_id":       doc_id,
                "chunk_id":     raw_id,
                "chunk_idx":    chunk_idx,
                "source":       str(filepath),
                "text":         sub,
                "title":        fields["title"],
                "section":      heading,
                "breadcrumbs":  fields["breadcrumbs"],
                "source_url":   fields["source_url"],
                "description":  fields["description"],
                "domain":       domain,
                "subdomain":    fields["subdomain"],
                "last_updated": fields["last_updated"],
                "language":     "en",
                "chunk_length": len(sub),
            }

            chunks.append({
                "id":         chunk_id,
                "embed_text": embed_text,
                "text":       sub,
                "payload":    payload,
            })
            chunk_idx += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# INDEX TIME
# ─────────────────────────────────────────────────────────────────────────────

def build_index(force_rebuild: bool = False) -> QdrantClient:
    """
    Full indexing pipeline. Skips rebuild if collection already exists.
    Use force_rebuild=True only when corpus or chunking logic changes.
    """
    client   = QdrantClient(path=QDRANT_PATH)
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing and not force_rebuild:
        count = client.get_collection(COLLECTION_NAME).points_count
        print(f"[retriever] Index exists ({count} points). Skipping rebuild.")
        get_bm25_state()
        return client

    if COLLECTION_NAME in existing:
        print("[retriever] Deleting collection for rebuild...")
        client.delete_collection(COLLECTION_NAME)

    # Step 1: Chunk all documents
    md_files   = list(DATA_ROOT.rglob("*.md"))
    print(f"[retriever] Chunking {len(md_files)} markdown files...")

    all_chunks = []
    for filepath in md_files:
        rel      = filepath.relative_to(DATA_ROOT)
        domain   = DOMAIN_FOLDER_MAP.get(rel.parts[0].lower(), "unknown")
        raw_text = filepath.read_text(encoding="utf-8", errors="ignore")
        all_chunks.extend(chunk_document(raw_text, filepath, domain))

    print(f"[retriever] Total chunks: {len(all_chunks)}")

    # Step 2: Build BM25
    print("[retriever] Building BM25 model over full corpus...")
    tokenized_corpus = [tokenize(c["text"]) for c in all_chunks]
    bm25             = BM25Okapi(tokenized_corpus)

    # Step 3: Build vocab
    vocab: dict[str, int] = {}
    for tokens in tokenized_corpus:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    print(f"[retriever] Vocabulary size: {len(vocab)} unique tokens")

    BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "vocab": vocab}, f)
    print(f"[retriever] BM25 state saved to {BM25_CACHE_PATH}")

    global _bm25, _vocab
    _bm25, _vocab = bm25, vocab

    # Step 4: Create Qdrant collection
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
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    print(f"[retriever] Created hybrid collection '{COLLECTION_NAME}'")

    # Step 5+6: Embed + upsert in batches
    biencoder      = get_biencoder()
    BATCH          = 64
    total_upserted = 0

    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i : i + BATCH]

        dense_vecs = biencoder.encode(
            [c["embed_text"] for c in batch],
            show_progress_bar=False,
        ).tolist()

        sparse_vecs = [
            text_to_sparse(c["text"], vocab, bm25)
            for c in batch
        ]

        points = [
            PointStruct(
                id      = c["id"],
                vector  = {
                    DENSE_VEC:  dv,
                    SPARSE_VEC: sv,
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
    if company and company != "None":
        return f"Domain: {company}\n\n{query}"
    return query


def multi_retrieve(
    queries:  list[str],
    company:  str,
    client:   QdrantClient,
    top_k:    int = TOP_K,
) -> list[dict]:
    """
    Multi-query retrieval: run each query variant through Qdrant, merge the
    candidate pools by chunk_id deduplication, then cross-encode the merged
    pool once and return ranked results.

    WHY:
        A single query rewrite can miss relevant chunks when the ticket uses
        different vocabulary than the documentation. Running 2-3 rephrased
        variants and merging their candidates before the cross-encoder
        significantly improves recall without changing the reranking logic.

    DEDUP STRATEGY:
        Keep the first occurrence of each chunk_id (RRF score from the
        query that retrieved it first). The cross-encoder will re-score
        all of them on the primary query, so order doesn't matter here.

    Args:
        queries:  list of query strings (primary + variants), max ~5
        company:  used for domain filter
        client:   QdrantClient instance
        top_k:    candidates to pull per query from Qdrant

    Returns:
        Same format as retrieve() — list of chunk dicts sorted by cross_score.
    """
    t0 = time.time()

    bm25, vocab = get_bm25_state()
    domain      = COMPANY_TO_DOMAIN.get(company)

    qdrant_filter = None
    if domain:
        qdrant_filter = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        )

    # Collect raw Qdrant results from all query variants
    seen_chunk_ids: set[str] = set()
    merged_results = []

    for q in queries:
        embed_text = build_query_embed_text(q, company)
        dense_qvec = get_biencoder().encode(embed_text).tolist()
        sparse_qvec = text_to_sparse(q, vocab, bm25)

        points = client.query_points(
            collection_name = COLLECTION_NAME,
            prefetch        = [
                Prefetch(
                    query           = dense_qvec,
                    using           = DENSE_VEC,
                    filter          = qdrant_filter,
                    limit           = top_k,
                    score_threshold = BI_SCORE_FLOOR,
                ),
                Prefetch(
                    query  = sparse_qvec,
                    using  = SPARSE_VEC,
                    filter = qdrant_filter,
                    limit  = top_k,
                ),
            ],
            query        = FusionQuery(fusion=Fusion.RRF),
            limit        = top_k,
            with_payload = True,
        ).points

        for p in points:
            cid = p.payload.get("chunk_id", str(p.id))
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                merged_results.append(p)

    if not merged_results:
        print(f"[retriever] ⚠ Zero results across {len(queries)} queries")
        return []

    # Cross-encode the merged pool using the primary query
    primary_query = queries[0]
    pairs        = [(primary_query, r.payload["text"]) for r in merged_results]
    cross_scores = get_crossencoder().predict(pairs).tolist()

    combined = sorted(
        [
            {
                "text":        r.payload.get("text", ""),
                "title":       r.payload.get("title", ""),
                "section":     r.payload.get("section", ""),
                "breadcrumbs": r.payload.get("breadcrumbs", []),
                "source_url":  r.payload.get("source_url", ""),
                "description": r.payload.get("description", ""),
                "domain":      r.payload.get("domain", ""),
                "subdomain":   r.payload.get("subdomain", ""),
                "last_updated":r.payload.get("last_updated", ""),
                "source":      r.payload.get("source", ""),
                "chunk_id":    r.payload.get("chunk_id", ""),
                "rrf_score":   round(r.score, 6),
                "cross_score": round(float(cs), 4),
            }
            for r, cs in zip(merged_results, cross_scores)
        ],
        key     = lambda x: x["cross_score"],
        reverse = True,
    )

    ms  = round((time.time() - t0) * 1000)
    top = combined[0]["cross_score"] if combined else "N/A"
    print(f"[retriever] {len(combined)} chunks (from {len(queries)} queries) | top_cross={top} | {ms}ms")
    return combined


def retrieve(
    query:   str,
    company: str,
    client:  QdrantClient,
    top_k:   int = TOP_K,
) -> list[dict]:
    """
    Hybrid search: dense + sparse → RRF fusion → cross-encoder re-rank.

    FIXED in v2:
        Prefetch now uses `using=` parameter (qdrant-client >= 1.10 API).
        Old NamedVector/NamedSparseVector wrappers are removed.
    """
    t0 = time.time()

    bm25, vocab = get_bm25_state()
    domain      = COMPANY_TO_DOMAIN.get(company)

    # Step 1 — dense query vector
    query_embed = build_query_embed_text(query, company)
    dense_qvec  = get_biencoder().encode(query_embed).tolist()

    # Step 2 — sparse query vector
    sparse_qvec = text_to_sparse(query, vocab, bm25)

    # Step 3+4+5 — Qdrant hybrid search with RRF fusion
    qdrant_filter = None
    if domain:
        qdrant_filter = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        )

    results = client.query_points(
        collection_name = COLLECTION_NAME,
        prefetch        = [
            # Dense arm — uses `using=` (fixed for qdrant-client >= 1.10)
            Prefetch(
                query           = dense_qvec,
                using           = DENSE_VEC,
                filter          = qdrant_filter,
                limit           = top_k,
                score_threshold = BI_SCORE_FLOOR,
            ),
            # Sparse arm
            Prefetch(
                query  = sparse_qvec,
                using  = SPARSE_VEC,
                filter = qdrant_filter,
                limit  = top_k,
            ),
        ],
        query        = FusionQuery(fusion=Fusion.RRF),
        limit        = top_k,
        with_payload = True,
    ).points

    if not results:
        print(f"[retriever] ⚠ Zero results for: '{query[:70]}'")
        return []

    # Step 6 — cross-encoder re-ranking
    pairs        = [(query, r.payload["text"]) for r in results]
    cross_scores = get_crossencoder().predict(pairs).tolist()

    # Step 7 — combine + sort by cross_score descending
    combined = sorted(
        [
            {
                "text":        r.payload.get("text", ""),
                "title":       r.payload.get("title", ""),
                "section":     r.payload.get("section", ""),
                "breadcrumbs": r.payload.get("breadcrumbs", []),
                "source_url":  r.payload.get("source_url", ""),
                "description": r.payload.get("description", ""),
                "domain":      r.payload.get("domain", ""),
                "subdomain":   r.payload.get("subdomain", ""),
                "last_updated":r.payload.get("last_updated", ""),
                "source":      r.payload.get("source", ""),
                "chunk_id":    r.payload.get("chunk_id", ""),
                "rrf_score":   round(r.score, 6),
                "cross_score": round(float(cs), 4),
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