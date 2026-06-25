"""Pydantic boundary validation at every agent edge.

Phase 2 implements these. Marked skip until then so `make test` stays green
and the rail's growth is visible on video.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Implemented in Phase 2")


def test_placeholder():
    pytest.fail("replace in Phase 2")
