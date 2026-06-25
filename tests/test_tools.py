"""Tool contracts: web/vector/pdf return errors-as-strings.

Phase 3 implements these. Marked skip until then so `make test` stays green
and the rail's growth is visible on video.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Implemented in Phase 3")


def test_placeholder():
    pytest.fail("replace in Phase 3")
