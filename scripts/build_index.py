"""Build/refresh the Chroma index over data/corpus/ (and optionally a PDF).

    python scripts/build_index.py
    python scripts/build_index.py --pdf path/to/report.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tools.vector_store import index_corpus  # noqa: E402


def main() -> int:
    stats = index_corpus(verbose=True)
    print(f"corpus indexed: {stats}")
    if "--pdf" in sys.argv:
        from core.tools.pdf_loader import index_pdf
        pdf = sys.argv[sys.argv.index("--pdf") + 1]
        print(f"pdf indexed: {index_pdf(pdf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
