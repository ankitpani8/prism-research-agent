"""Generate a synthetic prepaid-CX dashboard image for the vision agent.

    python scripts/make_dashboard.py

Two panels: (1) complaint-category breakdown (%), (2) NPS trend by quarter.
All data SYNTHETIC, grounded in the same taxonomy as data/corpus (billing the
single largest driver, NPS trending down ~6 pts). Deterministic + reproducible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_dashboard.png"

CATEGORIES = [
    ("Billing & Charges", 27),
    ("Network & Coverage", 24),
    ("Recharge Failures", 18),
    ("Data Speed", 14),
    ("Provisioning", 10),
    ("Other", 7),
]
NPS_QUARTERS = ["Q2 FY24", "Q3 FY24", "Q4 FY24", "Q1 FY25"]
NPS_VALUES = [28, 26, 24, 22]  # down 6 over the year, -2 QoQ


def main() -> int:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Prepaid Customer Experience — Complaints & NPS (Q1 FY25, synthetic)",
                 fontsize=14, fontweight="bold")

    # Panel 1 — complaint categories
    labels = [c[0] for c in CATEGORIES]
    vals = [c[1] for c in CATEGORIES]
    colors = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#2980b9", "#7f8c8d"]
    bars = ax1.barh(labels[::-1], vals[::-1], color=colors[::-1])
    ax1.set_title("Complaint drivers (% of prepaid complaints)", fontsize=11)
    ax1.set_xlabel("% of complaints")
    ax1.set_xlim(0, 32)
    for bar, v in zip(bars, vals[::-1], strict=False):
        ax1.text(v + 0.6, bar.get_y() + bar.get_height() / 2, f"{v}%",
                 va="center", fontsize=10, fontweight="bold")

    # Panel 2 — NPS trend
    ax2.plot(NPS_QUARTERS, NPS_VALUES, marker="o", linewidth=2.5, color="#c0392b")
    ax2.set_title("Net Promoter Score (NPS) trend", fontsize=11)
    ax2.set_ylabel("NPS")
    ax2.set_ylim(15, 32)
    for x, y in zip(NPS_QUARTERS, NPS_VALUES, strict=False):
        ax2.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=10, fontweight="bold")
    ax2.annotate("-6 pts YoY", xy=(3, 22), xytext=(1.3, 19),
                 arrowprops=dict(arrowstyle="->", color="#c0392b"),
                 color="#c0392b", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
