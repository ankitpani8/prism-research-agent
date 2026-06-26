"""Pure display-helper tests."""
from core.format import badge_html, confidence_tier


def test_confidence_tiers():
    assert confidence_tier(0.95) == "high"
    assert confidence_tier(0.7) == "high"
    assert confidence_tier(0.5) == "medium"
    assert confidence_tier(0.4) == "medium"
    assert confidence_tier(0.2) == "low"


def test_badge_html_contains_value_and_color():
    html = badge_html(0.83)
    assert "0.83" in html and "#1a7f37" in html  # high -> green
    assert "#cf222e" in badge_html(0.1)           # low -> red
