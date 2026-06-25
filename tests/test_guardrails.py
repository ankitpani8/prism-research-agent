"""Critic flags a PLANTED unsupported claim; cites_source catches a missing source.

Phase 5 implements these. Marked skip until then so `make test` stays green
and the rail's growth is visible on video.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Implemented in Phase 5")


def test_placeholder():
    pytest.fail("replace in Phase 5")
