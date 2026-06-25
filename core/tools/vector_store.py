"""Vector store tool — Chroma + local embeddings over the synthetic CX corpus.

Reuses learning_AgenticAI/module4 conventions: PersistentClient, the local
all-MiniLM-L6-v2 SentenceTransformer embedding function (keyless, CPU), cosine
space, and content-hash incremental indexing (skip unchanged files).

Production swap: the get_collection() seam is where Chroma is replaced by
Pinecone/Weaviate — the retrieve() contract above it does not change.

Testability: embed_fn is injectable. Tests pass a deterministic local stub and
an in-memory (Ephemeral) client, so index+retrieve run fully offline / keyless
(no model download). Real use defaults to all-MiniLM-L6-v2.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from core.config import settings

COLLECTION_NAME = "cx_corpus"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

_collection = None  # process-wide cache


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _default_embed_fn():
    # Imported lazily so module import never triggers a model download.
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def get_collection(embed_fn=None, persist: bool = True):
    import chromadb
    client = (chromadb.PersistentClient(path=str(settings.chroma_dir))
              if persist else chromadb.EphemeralClient())
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn or _default_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def index_corpus(corpus_dir: Path | None = None, embed_fn=None, persist: bool = True,
                 verbose: bool = False) -> dict:
    """Hash-incremental index of every .md/.txt under corpus_dir into the store."""
    corpus_dir = corpus_dir or settings.corpus_dir
    coll = get_collection(embed_fn=embed_fn, persist=persist)
    stats = {"new": 0, "unchanged": 0, "chunks": 0}
    for path in sorted(Path(corpus_dir).glob("*")):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        digest = _sha(text)
        existing = coll.get(where={"source": path.name}, include=["metadatas"])
        if existing["ids"] and all(m.get("hash") == digest for m in existing["metadatas"]):
            stats["unchanged"] += 1
            continue
        if existing["ids"]:
            coll.delete(ids=existing["ids"])
        chunks = chunk_text(text)
        coll.add(
            ids=[f"{path.name}:{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": path.name, "hash": digest} for _ in chunks],
        )
        stats["new"] += 1
        stats["chunks"] += len(chunks)
    if verbose:
        print(f"[index] {stats}")
    return stats


def _ensure_indexed(embed_fn=None, persist: bool = True):
    global _collection
    if _collection is None:
        coll = get_collection(embed_fn=embed_fn, persist=persist)
        if coll.count() == 0:
            index_corpus(embed_fn=embed_fn, persist=persist)
        _collection = coll
    return _collection


def _retrieve_sync(query: str, k: int, embed_fn=None, persist: bool = True) -> list[dict]:
    try:
        coll = _ensure_indexed(embed_fn=embed_fn, persist=persist)
        res = coll.query(query_texts=[query], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [{"text": d, "source": (m or {}).get("source", "corpus")}
                for d, m in zip(docs, metas, strict=False)]
    except Exception:
        return []


async def rag_retrieve(query: str, k: int = 4) -> list[dict]:
    """Return [{text, source}] from the corpus (possibly empty). Never raises."""
    return await asyncio.to_thread(_retrieve_sync, query, k)
