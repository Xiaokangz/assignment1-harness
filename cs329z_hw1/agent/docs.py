"""Document search over one BM25 index of lines, plus bounded reading."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from cs329z_hw1.pipelines.bm25 import Index, build_index, search
from cs329z_hw1.tokens import count_tokens
from cs329z_hw1.tokenizer import tokenize

# Under the 1500-token adapter limit, so a few reads fit in one conversation.
_WINDOW_TOKENS = 800
_SNIPPET_TOKENS = 150

_lock = threading.Lock()
_shared: dict = {}


@dataclass(slots=True)
class DocIndex:
    bm25: Index
    # Parallel to the BM25 documents: the source line of each indexed line.
    chunks: list[tuple[str, int]]
    docs: dict[str, dict]


def build_doc_index(docs: list[dict]) -> DocIndex:
    """Index one line per BM25 document. Empty lines are stored but not indexed."""
    stored: dict[str, dict] = {}
    chunks: list[tuple[str, int]] = []
    texts: list[str] = []
    for doc in docs:
        lines = doc["text"].splitlines()
        stored[doc["doc_id"]] = {"title": doc["title"], "lines": lines}
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            chunks.append((doc["doc_id"], number))
            texts.append(line)
    return DocIndex(build_index(texts), chunks, stored)


def shared_index() -> DocIndex:
    """The company-document index, built on first use and then reused."""
    cached = _shared.get("index")
    if cached is not None:
        return cached
    with _lock:
        cached = _shared.get("index")
        if cached is None:
            from cs329z_hw1.data import load_docs

            cached = build_doc_index(load_docs())
            _shared["index"] = cached
        return cached


def search_documents(index: DocIndex, query: str, k: int = 5) -> list[dict]:
    """Top matching lines. Each snippet is a short excerpt of its line."""
    hits = search(index.bm25, query, k)
    results = []
    for chunk_id, _score in hits:
        doc_id, line_no = index.chunks[chunk_id]
        doc = index.docs[doc_id]
        line = doc["lines"][line_no - 1]
        results.append(
            {
                "doc_id": doc_id,
                "title": doc["title"],
                "snippet": _snippet(doc["lines"], line_no, query),
                "start_line": line_no,
            }
        )
    return results


def read_document(index: DocIndex, doc_id: str, start_line: int = 1) -> dict:
    """Lines from start_line onward, packed until the token budget."""
    doc = index.docs.get(doc_id)
    if doc is None:
        raise KeyError(f"Unknown document: {doc_id}")
    lines = doc["lines"]
    total = len(lines)
    if start_line < 1 or start_line > total:
        raise ValueError(f"start_line {start_line} is outside 1..{total} for {doc_id}")
    chosen: list[str] = []
    end = start_line
    for offset, line in enumerate(lines[start_line - 1 :], start=start_line):
        candidate = chosen + [line]
        if chosen and count_tokens("\n".join(candidate)) > _WINDOW_TOKENS:
            break
        chosen.append(line)
        end = offset
    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": end,
        "total_lines": total,
        "text": "\n".join(chosen),
    }


def _snippet(lines: list[str], line_no: int, query: str) -> str:
    """The matching line, extended to neighboring lines until it can quote four words."""
    start = line_no - 1
    end = line_no
    snippet = lines[start]
    while len(tokenize(snippet)) < 4:
        if end < len(lines):
            end += 1
        elif start > 0:
            start -= 1
        else:
            break
        snippet = "\n".join(lines[start:end])
    if count_tokens(snippet) <= _SNIPPET_TOKENS:
        return snippet
    return _trim_snippet(lines[line_no - 1], query)


def _trim_snippet(line: str, query: str) -> str:
    """A slice of a long line around the first query term, within the token limit."""
    lower = line.lower()
    pos = 0
    for term in tokenize(query):
        found = lower.find(term)
        if found >= 0:
            pos = found
            break
    start = max(0, pos - 200)
    end = min(len(line), pos + 200)
    snippet = line[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(line):
        snippet = snippet + "..."
    while count_tokens(snippet) > _SNIPPET_TOKENS and len(snippet) > 40:
        snippet = snippet[:-20]
    return snippet
