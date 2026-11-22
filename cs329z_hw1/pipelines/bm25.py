"""BM25 over an inverted index. k1 = 1.5, b = 0.75, Lucene IDF.

score(q, d) = Σ_t IDF(t) · tf · (k1 + 1) / (tf + k1 · (1 - b + b · |d| / avgdl))
IDF(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))

The sum is over the distinct query terms. n_t is the number of documents
that contain the term, tf is how often it occurs in this document, |d| is
the document's length in tokens, and avgdl is the average length. The +1
inside the log is Lucene's form: a term that occurs in more than half the
documents still gets a positive IDF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cs329z_hw1.tokenizer import tokenize

# How fast term frequency saturates, and how much document length weighs
# in the denominator. The tests fix both values.
K1 = 1.5
B = 0.75


@dataclass(slots=True)
class Index:
    # term -> [(doc_id, tf), ...]. One entry per document, in ascending doc_id.
    postings: dict[str, list[tuple[int, int]]]
    n: int  # Document count. Documents with no tokens still count.
    avgdl: float  # Mean document length, in tokens.
    # norms[doc_id] = k1 * (1 - b + b * |d| / avgdl): the part of the
    # denominator that does not depend on the term.
    norms: list[float]


def build_index(docs: list[str]) -> Index:
    """Index ``docs`` in one pass. A document's id is its list position."""
    postings: dict[str, list[tuple[int, int]]] = {}
    lengths: list[int] = []
    for doc_id, text in enumerate(docs):
        # Count tf within this document. The postings list stores one entry
        # per document, not one entry per occurrence.
        counts: dict[str, int] = {}
        for token in tokenize(text or ""):
            counts[token] = counts.get(token, 0) + 1
        # An empty document has length 0: no postings, but it keeps its
        # doc_id and still counts toward N and avgdl.
        lengths.append(sum(counts.values()))
        for term, tf in counts.items():
            bucket = postings.get(term)
            if bucket is None:
                postings[term] = [(doc_id, tf)]
            else:
                bucket.append((doc_id, tf))

    n = len(docs)
    total = sum(lengths)
    avgdl = total / n if n else 0.0
    if avgdl == 0.0:
        # Every document was empty, so skip the division. This index has no
        # postings, and a search returns [].
        norms = [0.0] * n
    else:
        base = K1 * (1.0 - B)
        scale = K1 * B / avgdl
        norms = [base + scale * length for length in lengths]
    return Index(postings, n, avgdl, norms)


def search(index: Index, query: str, k: int) -> list[tuple[int, float]]:
    """Top-k ``(doc_id, score)`` pairs. Ties break toward the lower doc id."""
    if k <= 0 or index.n == 0:
        return []
    # tokenize lowercases. dict.fromkeys drops repeats, so "apple apple APPLE"
    # contributes once.
    terms = list(dict.fromkeys(tokenize(query or "")))
    if not terms:
        return []

    # Only documents that share a query term receive a score, so a document
    # with no term in common is never returned.
    scores: dict[int, float] = {}
    weight = K1 + 1.0
    for term in terms:
        hits = index.postings.get(term)
        if not hits:
            # An unknown query term is skipped; the other terms still count.
            continue
        n_t = len(hits)
        idf = math.log((index.n - n_t + 0.5) / (n_t + 0.5) + 1.0)
        boost = idf * weight
        norms = index.norms
        # Walk this term's postings only, not the whole collection.
        for doc_id, tf in hits:
            scores[doc_id] = scores.get(doc_id, 0.0) + boost * tf / (tf + norms[doc_id])
    if not scores:
        return []

    # Highest score first; equal scores go to the smaller doc_id. When k
    # exceeds the number of hits, return every hit.
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if k < len(ranked):
        ranked = ranked[:k]
    return [(doc_id, float(score)) for doc_id, score in ranked]
