"""Phase 0 smoke tests — the seed of the test rail.

These run GREEN immediately, with only light deps (pydantic, dotenv), so CI is
green from commit #1 and there is always something real to show on video. Each
later phase ADDS a test file here; nothing is back-loaded.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_core_imports():
    import core  # noqa: F401
    from core.config import settings
    assert settings.ollama_host.startswith("http")


def test_required_repo_files_exist():
    for f in [".env.example", "requirements.txt", "README.md", "LICENSE", "Makefile"]:
        assert (ROOT / f).exists(), f"missing {f}"


def _active_rules(gitignore_text: str) -> set[str]:
    """Ignore-rule lines only — strip blanks and comments."""
    return {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def test_dotenv_ignored_but_example_committed():
    rules = _active_rules((ROOT / ".gitignore").read_text())
    assert ".env" in rules, ".env must be gitignored"
    assert ".env.example" not in rules, (
        ".env.example must NOT have an ignore rule — graders need it committed"
    )
    assert "CLAUDE.md" in rules, "CLAUDE.md must be gitignored per the brief"
