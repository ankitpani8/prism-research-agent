"""Tool contracts — offline / keyless.

vector store: index the real synthetic corpus with a DETERMINISTIC local embedding
stub + an in-memory client, so index+retrieve mechanics run with no model download.
web search: returns a list of dicts and never raises (contract), via a stub.
"""
import asyncio
from types import SimpleNamespace

from chromadb import Documents, EmbeddingFunction, Embeddings

import core.tools.vector_store as vs
import core.tools.web_search as ws
from core.tools.vector_store import chunk_text


class FakeEmbed(EmbeddingFunction):
    """Deterministic 16-dim hashing embedding — offline, no model download."""

    def __init__(self):
        pass

    def name(self) -> str:
        return "fake-embed-16"

    def __call__(self, input: Documents) -> Embeddings:
        import hashlib
        out = []
        for t in input:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([b / 255.0 for b in h[:16]])
        return out


def test_chunk_text_overlaps():
    chunks = chunk_text("x" * 1500, size=600, overlap=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 600 for c in chunks)


def test_corpus_index_and_retrieve_offline():
    vs._collection = None
    results = vs._retrieve_sync("billing incorrect charges", k=3,
                                embed_fn=FakeEmbed(), persist=False)
    assert isinstance(results, list) and len(results) >= 1
    assert all("text" in r and "source" in r for r in results)
    vs._collection = None


def test_web_search_returns_list(monkeypatch):
    monkeypatch.setattr(ws, "settings", SimpleNamespace(tavily_api_key=None))
    monkeypatch.setattr(ws, "_ddgs_sync",
                        lambda q, n: [{"title": "t", "url": "u", "content": "c"}])
    out = asyncio.run(ws.web_search("anything"))
    assert isinstance(out, list) and out and "content" in out[0]


def test_web_search_empty_is_graceful(monkeypatch):
    monkeypatch.setattr(ws, "settings", SimpleNamespace(tavily_api_key=None))
    monkeypatch.setattr(ws, "_ddgs_sync", lambda q, n: [])
    out = asyncio.run(ws.web_search("anything"))
    assert out == []
