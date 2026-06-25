"""PDF loader — extract text from a PDF and index it into the same vector store.

Lets the system ingest a real uploaded document (e.g., a regulator complaints
report) alongside the synthetic corpus. pdfplumber for text; chunked and added
with the same hash-incremental contract as the corpus.
"""
from __future__ import annotations

from pathlib import Path

from core.tools.vector_store import _sha, chunk_text, get_collection


def load_pdf_text(path: str | Path) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def index_pdf(path: str | Path, embed_fn=None, persist: bool = True) -> dict:
    path = Path(path)
    text = load_pdf_text(path)
    coll = get_collection(embed_fn=embed_fn, persist=persist)
    digest = _sha(text)
    existing = coll.get(where={"source": path.name}, include=["metadatas"])
    if existing["ids"]:
        coll.delete(ids=existing["ids"])
    chunks = chunk_text(text)
    if chunks:
        coll.add(
            ids=[f"{path.name}:{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": path.name, "hash": digest} for _ in chunks],
        )
    return {"source": path.name, "chunks": len(chunks)}
