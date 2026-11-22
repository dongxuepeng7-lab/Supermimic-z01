"""Rebuild the two compact figures adapted from the 2026-11-22 draft.

The replay chart uses archived, full-record metrics, not digitized curves.
The paired-rollout drawing describes the implemented simulator supervision.
Run with a Python installation providing matplotlib.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAVY, TEAL, GRAY = "#183849", "#00796b", "#516d7b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, name):
    for extension in ("pdf", "svg"):
        fig.savefig(HERE / f"{name}.{extension}", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def replay_contrast():
    audit = json.loads((ROOT / "release/ral/manuscript_evidence_20261122.json").read_text())
    rows = audit["archived_hardware_rows"]
    fig, ax = plt.subplots(figsize=(3.35, 1.45))
    for y, joint in [(1, "L hip roll"), (0, "L ankle roll")]:
        selected = {r["parameter_set"]: float(r["q"]) * 1000
                    for r in rows if r["joint"] == joint}
        original, identified = selected["original_model"], selected["current_rounded"]
        ax.plot([original, identified], [y, y], color="#a6b7bd", lw=1.5, zorder=1)
        for value, color, marker, label, offset in [
            (original, GRAY, "s", "Original", 0.14),
            (identified, TEAL, "o", "Identified", -0.28),
        ]:
            ax.scatter(value, y, s=24, color=color, marker=marker, zorder=3,
                       label=label if y == 1 else None)
            ax.text(value, y + offset, f"{value:.2f}", ha="center", color=color, fontsize=7)
    ax.set(yticks=[1, 0], yticklabels=["Left hip roll", "Left ankle roll"],
           xticks=[0, 5, 10, 15], xlim=(0, 15), ylim=(-0.4, 1.4),
           xlabel="Full-record position RMSE (mrad)")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=7.5)
    ax.grid(axis="x", color="#e7edef", linewidth=0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=7.5, handletextpad=0.4, columnspacing=1.2)
    save(fig, "replay_contrast")


def paired_response():
    fig, ax = plt.subplots(figsize=(3.35, 1.8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 100), ylim=(0, 60))
    ax.axis("off")

    def box(x, y, width, height, color, face):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                   boxstyle="round,pad=0.3,rounding_size=1",
                                   linewidth=0.8, edgecolor=color, facecolor=face))

    def arrow(points):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=GRAY, linewidth=0.75)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=6, color=GRAY, linewidth=0.75))

    ax.text(50, 57, "Matched state after shared pre-roll", ha="center",
            color=NAVY, fontsize=8, weight="bold")
    for x, sign, title in [(1, "+", "Positive command"), (53, "−", "Negative command")]:
        box(x, 28, 46, 21, "#32638a", "#edf3f7")
        ax.text(x + 2.5, 43, title, color=NAVY, fontsize=8, weight="bold")
        ax.text(x + 2.5, 36.5, f"U {sign} εD", color=NAVY, fontsize=8)
        ax.text(x + 2.5, 30.5, "Physics and model rollouts", color=NAVY, fontsize=6.9)
        arrow([(50, 54), (50, 51), (x + 23, 51), (x + 23, 49)])
    box(1, 1, 98, 19, TEAL, "#eef7f4")
    ax.text(4, 14, "Response loss at each supervised step", color=TEAL,
            fontsize=8, weight="bold")
    ax.text(4, 8.3, "Form the plus/minus difference in each system.", color=NAVY, fontsize=7)
    ax.text(4, 3, "Penalize disagreement between model and physics.", color=NAVY, fontsize=7)
    for x in [24, 76]:
        arrow([(x, 28), (x, 24), (50, 24), (50, 20)])
    save(fig, "paired_response")


if __name__ == "__main__":
    replay_contrast()
    paired_response()
