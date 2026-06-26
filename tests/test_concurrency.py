"""Concurrent rag_retrieve must not hit the Chroma tenant-init race that the
async research fan-out exposed (RustBindings 'bindings' / tenant error)."""
import concurrent.futures as cf
from pathlib import Path
from types import SimpleNamespace

import core.tools.vector_store as vs
from tests.conftest import FakeEmbed

REAL_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


def test_concurrent_retrieve_no_tenant_race(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "settings",
                        SimpleNamespace(chroma_dir=str(tmp_path / "chroma"),
                                        corpus_dir=str(REAL_CORPUS)))
    vs._collection = None
    ef = FakeEmbed()

    def call(_):
        return vs._retrieve_sync("billing complaints categories", 3,
                                 embed_fn=ef, persist=True)

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(call, range(8)))

    # With the init lock, every concurrent caller gets the indexed collection.
    assert all(isinstance(r, list) for r in results)
    assert all(len(r) >= 1 for r in results), "a race would leave some callers empty"
    vs._collection = None
