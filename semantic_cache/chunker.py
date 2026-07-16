"""Context chunking: split long documents/transcripts into overlapping chunks
and extract lightweight entities used to build graph edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-\.]*")
_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have if in into is it its of on
    or that the their then there these this to was were will with not can what
    which when we you they he she i his her our your my me him them us do does
    did so than too very just also been being had would could should may might
    must shall about over under between after before while because how why all
    any each other some such no nor only own same s t don now more most"""
    .split()
)


@dataclass
class Chunk:
    chunk_id: int
    text: str
    doc_id: str
    position: int                      # index within its source document
    entities: Set[str] = field(default_factory=set)
    tokens: List[str] = field(default_factory=list)


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def extract_entities(text: str, tokens: List[str]) -> Set[str]:
    """Cheap entity extraction: capitalized/CamelCase/dotted terms plus rare
    lowercase tokens. Good enough to knit co-occurrence edges; swap in a real
    NER model behind the same signature if you need better graphs.
    """
    entities = {
        m.group(0)
        for m in _WORD_RE.finditer(text)
        if m.group(0)[0].isupper() or "_" in m.group(0) or "." in m.group(0)
    }
    entities |= {t for t in tokens if len(t) >= 8 and t not in _STOPWORDS}
    return {e.lower() for e in entities if e.lower() not in _STOPWORDS}


def chunk_text(
    text: str,
    doc_id: str = "doc",
    chunk_size: int = 120,
    overlap: int = 20,
    start_id: int = 0,
) -> List[Chunk]:
    """Split `text` into chunks of ~`chunk_size` words with `overlap` words of
    continuity between neighbors (so sentences cut at a boundary survive).
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must exceed overlap")
    words = text.split()
    chunks: List[Chunk] = []
    step = chunk_size - overlap
    for pos, start in enumerate(range(0, max(len(words), 1), step)):
        piece = " ".join(words[start : start + chunk_size])
        if not piece:
            break
        tokens = tokenize(piece)
        chunks.append(
            Chunk(
                chunk_id=start_id + pos,
                text=piece,
                doc_id=doc_id,
                position=pos,
                entities=extract_entities(piece, tokens),
                tokens=tokens,
            )
        )
        if start + chunk_size >= len(words):
            break
    return chunks
