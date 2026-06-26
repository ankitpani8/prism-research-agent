"""Shared offline test helpers."""
from chromadb import Documents, EmbeddingFunction, Embeddings


class FakeEmbed(EmbeddingFunction):
    """Deterministic 16-dim hashing embedding — offline, no model download."""

    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "fake-embed-16"

    def get_config(self) -> dict:
        return {}

    @classmethod
    def build_from_config(cls, config):
        return cls()

    def __call__(self, input: Documents) -> Embeddings:
        import hashlib
        return [[b / 255.0 for b in hashlib.sha256(t.encode("utf-8")).digest()[:16]]
                for t in input]
