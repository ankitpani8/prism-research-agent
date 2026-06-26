"""Vector store tool — Chroma + local embeddings over the synthetic CX corpus.

Reuses learning_AgenticAI/module4 conventions: PersistentClient, the local
all-MiniLM-L6-v2 ONNX embedding function (keyless, CPU, no torch), cosine
space, content-hash incremental indexing.

Failures are VISIBLE: index/retrieve print a [rag-error] line to stderr with the
real exception, plus the corpus path / file count / collection count — so an
empty result is diagnosable, never silent. Production swap point: get_collection().
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
import threading
import traceback
from pathlib import Path

from core.config import settings

COLLECTION_NAME = "cx_corpus"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

_collection = None
_init_lock = threading.Lock()  # serialize Chroma init across the async fan-out's threads


def _warn(msg: str) -> None:
    print(f"[rag-error] {msg}", file=sys.stderr)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def default_embed_fn():
    """Local all-MiniLM-L6-v2 as ONNX (Chroma's DefaultEmbeddingFunction).

    Uses onnxruntime, NOT torch/sentence-transformers — ~80MB download once, no
    2GB torch dependency, and it sidesteps torch import-incompatibility crashes.
    Lighter on the RAM budget and faster for graders to install. Lazy import.
    """
    from chromadb.utils import embedding_functions
    return embedding_functions.DefaultEmbeddingFunction()


def get_collection(embed_fn=None, persist: bool = True):
    import chromadb
    client = (chromadb.PersistentClient(path=str(settings.chroma_dir))
              if persist else chromadb.EphemeralClient())
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn or default_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def _index_into(coll, corpus_dir: Path, verbose: bool = False) -> dict:
    """Index every .md/.txt under corpus_dir INTO the given collection."""
    files = [p for p in sorted(Path(corpus_dir).glob("*")) if p.suffix.lower() in (".md", ".txt")]
    stats = {"files_found": len(files), "new": 0, "unchanged": 0, "chunks": 0}
    for path in files:
        text = path.read_text(encoding="utf-8")
        digest = _sha(text)
        existing = coll.get(where={"source": path.name}, include=["metadatas"])
        if existing["ids"] and all((m or {}).get("hash") == digest for m in existing["metadatas"]):
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
        print(f"[rag] indexed from {corpus_dir}: {stats} | collection count={coll.count()}")
    return stats


def index_corpus(corpus_dir: Path | None = None, embed_fn=None, persist: bool = True,
                 verbose: bool = False) -> dict:
    corpus_dir = corpus_dir or settings.corpus_dir
    coll = get_collection(embed_fn=embed_fn, persist=persist)
    return _index_into(coll, Path(corpus_dir), verbose=verbose)


def _ensure_indexed(embed_fn=None, persist: bool = True):
    # Double-checked locking: concurrent rag_retrieve calls in the async fan-out
    # run in separate threads (asyncio.to_thread); Chroma client/collection init
    # is NOT thread-safe, so serialize the first-time init behind a lock.
    global _collection
    if _collection is not None:
        return _collection
    with _init_lock:
        if _collection is None:
            coll = get_collection(embed_fn=embed_fn, persist=persist)
            if coll.count() == 0:
                stats = _index_into(coll, settings.corpus_dir, verbose=True)
                if coll.count() == 0:
                    _warn(f"collection still EMPTY after indexing. corpus_dir={settings.corpus_dir} "
                          f"exists={Path(settings.corpus_dir).exists()} files_found={stats['files_found']}")
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
    except Exception as e:
        _warn(f"{type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc(limit=2, file=sys.stderr)
        return []


async def rag_retrieve(query: str, k: int = 4) -> list[dict]:
    """Return [{text, source}] from the corpus (possibly empty). Never raises."""
    return await asyncio.to_thread(_retrieve_sync, query, k)
