"""
Draws the two self-proximity scatter plots (bridge edges/m vs jump-cell
density, and vs mean cross-sectional disagreement) from input.csv, the same
full-precision source correlations.py reads, and computes the legend
statistics from it.

Run: python analysis/A1_correlations/make_scatter_figures.py
Outputs: analysis/A1_correlations/figures/scatter_bridge_vs_jump.png
         analysis/A1_correlations/figures/scatter_bridge_vs_diff.png
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "input.csv"
OUT_DIR = HERE / "figures"

DISPLAY_LABEL = {"787": "C0787", "518": "C0518"}
CURATED_GROUP = "curated"

CURATED_COLOR = "#2a78d6"
FUSED_COLOR = "#e8642c"
TEXT_COLOR = "#444444"
GRID_COLOR = "#e3e3dc"
SPINE_COLOR = "#bdbdb4"

# Per-label (dx, dy) offsets in points, chosen so labels do not overlap markers.
LABEL_OFFSETS = {
    "jump": {"C7198": (-38, 12), "C6651": (-10, -18), "C0518": (8, -14)},
    "diff": {"C3114": (-30, 6), "C0518": (8, -10)},
}

FIGURES = [
    ("jump_pct", "Jump cells (%)", "scatter_bridge_vs_jump.png", "jump"),
    ("xsec_mean_diff_pct", "Mean cross-section difference (%)", "scatter_bridge_vs_diff.png", "diff"),
]


def load_rows():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        name = DISPLAY_LABEL.get(r["label"], r["label"])
        r["display"] = name if name.startswith("C") else f"C{int(name):04d}"
    return rows


def make_figure(rows, ykey, ylabel, filename, offset_key):
    x = [float(r["bridge_edges_per_m"]) for r in rows]
    y = [float(r[ykey]) for r in rows]
    r_val, p_val = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=200)
    for group, color, marker, name in [
        (CURATED_GROUP, CURATED_COLOR, "o", "Curated"),
        (None, FUSED_COLOR, "^", "Residue-fused"),
    ]:
        sel = [i for i, row in enumerate(rows) if (row["group"] == CURATED_GROUP) == (group is not None)]
        ax.scatter([x[i] for i in sel], [y[i] for i in sel], c=color, marker=marker, s=55, label=name, zorder=3)

    offsets = LABEL_OFFSETS[offset_key]
    for xi, yi, row in zip(x, y, rows):
        dx, dy = offsets.get(row["display"], (8, 6))
        ax.annotate(row["display"], (xi, yi), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=TEXT_COLOR)

    ax.set_xlabel("Bridge edges/m")
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
    ax.tick_params(colors="#888888")
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT_COLOR)

    stats = (f"Pearson $r$ = {r_val:.3f}, $p$ = {p_val:.3f}\n"
             f"Spearman $\\rho$ = {rho:.3f}, $p$ = {p_rho:.3f}\n"
             f"$n$ = {len(x)}")
    ax.text(0.97, 0.04, stats, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=TEXT_COLOR, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=SPINE_COLOR))

    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"{filename}: r = {r_val:.3f} (p = {p_val:.3f}), rho = {rho:.3f} (p = {p_rho:.3f})")


if __name__ == "__main__":
    rows = load_rows()
    for ykey, ylabel, filename, offset_key in FIGURES:
        make_figure(rows, ykey, ylabel, filename, offset_key)
