#!/usr/bin/env python3
"""Build the workshop multi-panel geometry figure from saved CSVs.

Panels:
  (a) Schematic of the pMHC-aligned TCR geometry (drawn, not data)
  (b) Same-peptide recovery (Same-peptide recovery AUROC)
  (c) Same-peptide neighbourhood enrichment (kNN enrichment)
  (d) Known-cognate pMHC Recall@10

Naming follows what each quantity is actually keyed on. The representation is
learned by aligning TCRs to the full pMHC, so panel (a) and the retrieval task
in (d) are pMHC-level. Grouping in (b) and (c) is by peptide sequence alone --
MHC context is not used to form the groups -- so those panels say "peptide".

VICReg models: mean over seeds 31/37/43/49/55 with error bars / band = ±1 SD.
One-hot, raw ESMC and LoRA ESMC input baselines are plotted as a single
marker/trace with no error bars.

Underlying metric CSVs and ranking calculations are not modified.

Suggested caption
-----------------
(a) Schematic: the TCR space is learned by alignment to the full pMHC, so that
TCRs sharing an associated peptide (same colour) sit closer together than TCRs
with different associated peptides. Same-peptide recovery in (b) samples an
equal number of within- and between-peptide TCR pairs per peptide so frequent
epitopes do not dominate the AUROC. The 700 evaluation pairs are drawn once and
reused for every representation; VICReg seed spread therefore reflects only the
learned spaces. (c) reports same-peptide kNN purity divided by the expectation
from peptide prevalence in the eligible candidate pool, so the dashed line at 1
marks a neighbourhood no more peptide-pure than chance; each seed re-draws the
peptide-balanced TCR subsample, so the band mixes model-seed and
evaluation-subsample variation. Input baselines are deterministic
representations and are shown as their across-seed mean. (d) retrieves
known-cognate pMHCs for multi-cognate TCRs; the dashed line is random ranking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

REPO = Path("/home/natasha/multimodal_model")
ANALYSIS = REPO / "models/outputs/workshop/paper_analysis"
REFINED = ANALYSIS / "refined_geometry"
FIG_DIR = REPO / "models/figures/workshop/paper_analysis/refined_geometry"

MODEL_ORDER = [
    "onehot_composition",
    "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool",
    "onehot_vicreg",
    "raw_esmc_vicreg",
    "finetuned_esmc_vicreg",
]
LABELS = {
    "onehot_composition": "One-hot",
    "pretrained_esmc_meanpool": "Raw ESMC",
    "finetuned_esmc_meanpool": "LoRA ESMC",
    "onehot_vicreg": "One-hot + VICReg",
    "raw_esmc_vicreg": "Raw ESMC + VICReg",
    "finetuned_esmc_vicreg": "LoRA ESMC + VICReg",
}
# Two-line, unrotated tick labels. A 30-degree rotated single line of this text
# costs about twice the vertical space, which is the figure's tallest fixed
# overhead once fonts are held constant.
TICK_LABELS = {
    "onehot_composition": "One-hot",
    "pretrained_esmc_meanpool": "Raw\nESMC",
    "finetuned_esmc_meanpool": "LoRA\nESMC",
    "onehot_vicreg": "One-hot\n+VICReg",
    "raw_esmc_vicreg": "Raw\nESMC\n+VICReg",
    "finetuned_esmc_vicreg": "LoRA\nESMC\n+VICReg",
}
# One colour family per initialisation: light = input, dark = VICReg.
COLORS = {
    "onehot_composition": "#bdbdbd",
    "onehot_vicreg": "#252525",
    "pretrained_esmc_meanpool": "#9ecae1",
    "raw_esmc_vicreg": "#08519c",
    "finetuned_esmc_meanpool": "#a1d99b",
    "finetuned_esmc_vicreg": "#006d2c",
}
LEARNED = {"onehot_vicreg", "raw_esmc_vicreg", "finetuned_esmc_vicreg"}
SEEDS = (31, 37, 43, 49, 55)

# Schematic-only palette. Deliberately disjoint from the model colours above so
# that "colour = associated peptide" in panel (a) is not read as a model identity.
PEPTIDE_A = "#e6550d"
PEPTIDE_B = "#6a51a3"


def style_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def seed_values(pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, dtype=float)


def mean_sd(pts: np.ndarray) -> tuple[float, float]:
    pts = seed_values(pts)
    mean = float(np.mean(pts))
    std = float(np.std(pts, ddof=1)) if len(pts) > 1 else 0.0
    return mean, std


def tint(color: str, amount: float) -> tuple[float, float, float]:
    """Blend `color` toward white; amount=0 keeps it, amount=1 is white."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def draw_tcr(ax: plt.Axes, x: float, y: float, color: str, s: float = 1.0) -> None:
    """Draw a minimal alpha/beta TCR cartoon centred on (x, y).

    Upper (variable) domains carry the associated-peptide colour, lower
    (constant) domains a lighter tint, and three dots stand in for the CDR loops.
    """
    edge = "0.2"
    light = tint(color, 0.55)

    def domain(cx: float, cy: float, w: float, h: float, face) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2.0, cy - h / 2.0),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=%.3f" % (0.12 * s),
                facecolor=face,
                edgecolor=edge,
                linewidth=0.9,
                zorder=5,
                mutation_aspect=1.0,
            )
        )

    w = 0.34 * s
    for dx in (-0.21 * s, 0.21 * s):
        domain(x + dx, y - 0.21 * s, w, 0.44 * s, light)
        domain(x + dx, y + 0.34 * s, w, 0.46 * s, color)
    for dx in (-0.30 * s, 0.0, 0.30 * s):
        ax.add_patch(
            Circle(
                (x + dx, y + 0.63 * s),
                radius=0.075 * s,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.7,
                zorder=6,
            )
        )


def double_arrow(
    ax: plt.Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: str,
    margin: float,
) -> None:
    """Double-headed arrow between two points, pulled back by `margin` data units."""
    v = np.array(p1, float) - np.array(p0, float)
    length = float(np.hypot(*v))
    if length <= 2 * margin:
        return
    unit = v / length
    start = np.array(p0, float) + unit * margin
    end = np.array(p1, float) - unit * margin
    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end),
            arrowstyle="<|-|>",
            mutation_scale=11,
            linewidth=1.4,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=7,
        )
    )


def panel_schematic(ax: plt.Axes) -> None:
    """Draw the pMHC-aligned TCR geometry cartoon.

    Call after `fig.tight_layout()`: the y-range is matched to the final axes
    box so that equal-aspect glyphs fill the panel instead of being letterboxed.
    """
    fig = ax.figure
    pos = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    height = 10.0 * (pos.height * fig_h) / (pos.width * fig_w)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (0.3, 0.28),
            9.4,
            height - 0.56,
            boxstyle="round,pad=0,rounding_size=0.35",
            facecolor="#fbfbfb",
            edgecolor="0.78",
            linestyle=(0, (5, 4)),
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(
        0.62,
        height - 0.48,
        "TCR embeddings",
        fontsize=8.5,
        style="italic",
        color="0.45",
        ha="left",
        va="top",
        zorder=1,
    )

    scale = min(1.0, 0.19 * height)
    y_mid = 0.60 * height
    ax_c, bx_c = 2.45, 8.50
    a1 = (1.55, y_mid + 0.52)
    a2 = (3.62, y_mid - 0.52)
    b1 = (bx_c, y_mid + 0.18)

    # Faint neighbourhoods so each highlighted TCR reads as a cluster member.
    rng = np.random.default_rng(0)
    for cx, color in ((ax_c, PEPTIDE_A), (bx_c, PEPTIDE_B)):
        pts = rng.normal(loc=(cx, y_mid), scale=(0.90, 0.40), size=(16, 2))
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=26,
            color=color,
            alpha=0.20,
            edgecolors="none",
            zorder=1,
        )

    for pos_xy, color in ((a1, PEPTIDE_A), (a2, PEPTIDE_A), (b1, PEPTIDE_B)):
        draw_tcr(ax, pos_xy[0], pos_xy[1], color, s=scale)

    margin = 0.74 * scale
    double_arrow(ax, a1, a2, "0.25", margin=margin)
    ax.text(
        2.45,
        y_mid - 1.24,
        "same associated peptide\n= smaller TCR\u2013TCR distance",
        fontsize=8.5,
        color="0.2",
        ha="center",
        va="top",
        linespacing=1.3,
        zorder=8,
    )

    double_arrow(ax, a2, b1, "0.25", margin=margin)
    ax.text(
        5.35,
        y_mid + 0.92,
        "different associated peptide\n= larger TCR\u2013TCR distance",
        fontsize=8.5,
        color="0.2",
        ha="center",
        va="bottom",
        linespacing=1.3,
        zorder=8,
    )

    ax.text(
        5.0,
        0.98,
        "TCR space learned through alignment to full pMHC",
        fontsize=8,
        style="italic",
        color="0.45",
        ha="center",
        va="center",
        zorder=8,
    )
    ax.text(
        5.0,
        0.52,
        "colour = associated peptide   \u00b7   MHC context not used for grouping",
        fontsize=8,
        style="italic",
        color="0.45",
        ha="center",
        va="center",
        zorder=8,
    )

    ax.set_title("(a) pMHC-aligned TCR geometry", fontsize=12, loc="left")


def panel_auroc(ax: plt.Axes, split: str) -> None:
    df = pd.read_csv(REFINED / f"{split}_same_peptide_auroc_protocols_by_seed.csv")
    sub = df[df["protocol"] == "peptide_balanced"]
    models = [m for m in MODEL_ORDER if m in set(sub["model_name"])]
    for i, model in enumerate(models):
        pts = sub.loc[sub["model_name"] == model, "auroc"].to_numpy(float)
        mean, std = mean_sd(pts)
        is_learned = model in LEARNED
        color = COLORS[model]
        if is_learned:
            jitter = (np.arange(len(pts)) - (len(pts) - 1) / 2.0) * 0.08
            ax.scatter(
                np.full(len(pts), i) + jitter,
                pts,
                facecolors=color,
                edgecolors=color,
                s=28,
                zorder=3,
                marker="o",
            )
            ax.errorbar(
                i,
                mean,
                yerr=std,
                fmt="none",
                ecolor=color,
                elinewidth=1.5,
                capsize=4.0,
                capthick=1.5,
                zorder=4,
            )
        else:
            ax.scatter(
                [i],
                [mean],
                facecolors="white",
                edgecolors=color,
                linewidths=1.3,
                s=42,
                zorder=3,
                marker="s",
            )
    ax.axhline(0.5, color="0.55", linestyle=":", linewidth=0.9, zorder=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [TICK_LABELS[m] for m in models], rotation=0, ha="center", fontsize=9
    )
    ax.set_ylabel("Same-peptide recovery AUROC", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(0.45, 0.85)
    ax.set_title("(b) Same-peptide recovery", fontsize=12, loc="left")
    style_spines(ax)


def panel_knn(ax: plt.Axes, split: str) -> None:
    knn = pd.read_csv(ANALYSIS / "crossreactivity" / "knn_peptide_purity.csv")
    sub = knn[knn["split"] == split]
    ks = sorted(int(k) for k in sub["k"].unique())
    for model in MODEL_ORDER:
        g = sub[sub["model_name"] == model]
        if g.empty:
            continue
        means = []
        stds = []
        for k in ks:
            pts = g.loc[g["k"] == k, "purity_enrichment"].to_numpy(float)
            mean, std = mean_sd(pts)
            means.append(mean)
            stds.append(std)
        means_a = np.asarray(means, float)
        stds_a = np.asarray(stds, float)
        is_learned = model in LEARNED
        color = COLORS[model]
        ax.plot(
            ks,
            means_a,
            color=color,
            linewidth=2.1 if is_learned else 1.5,
            linestyle="-" if is_learned else "--",
            marker="o" if is_learned else "s",
            markersize=5.5,
            markerfacecolor=color if is_learned else "white",
            markeredgecolor=color,
            markeredgewidth=1.2,
            zorder=3,
        )
        if is_learned:
            ax.fill_between(
                ks,
                means_a - stds_a,
                means_a + stds_a,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
    ax.axhline(1.0, color="0.55", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xticks(ks)
    ax.set_xlabel(r"Number of neighbours, $k$", fontsize=11)
    ax.set_ylabel("Same-peptide enrichment", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_ylim(0.8, None)
    ax.set_title("(c) Same-peptide neighbourhood enrichment", fontsize=12, loc="left")
    style_spines(ax)


def seed_metric_means(per_tcr: pd.DataFrame, model: str, metric: str) -> np.ndarray:
    g = per_tcr[per_tcr["model_name"] == model]
    means = g.groupby("seed")[metric].mean()
    return np.asarray([float(means.get(seed, np.nan)) for seed in SEEDS if seed in means.index], float)


def panel_retrieval(
    ax: plt.Axes,
    per_tcr: pd.DataFrame,
    metric: str,
    panel: str,
    ylabel: str,
    show_random_label: bool,
) -> None:
    models = [m for m in MODEL_ORDER if m in set(per_tcr["model_name"])]
    x = np.arange(len(models))
    means, stds, seed_pts = [], [], []
    for model in models:
        pts = seed_metric_means(per_tcr, model, metric)
        mean, std = mean_sd(pts)
        means.append(mean)
        stds.append(std)
        seed_pts.append(pts)

    ax.bar(
        x,
        means,
        color=[COLORS[m] for m in models],
        edgecolor="white",
        linewidth=0.6,
        alpha=0.95,
        zorder=2,
    )
    for i, model in enumerate(models):
        if model not in LEARNED:
            continue
        pts = seed_pts[i]
        jitter = (np.arange(len(pts)) - (len(pts) - 1) / 2.0) * 0.08
        ax.scatter(np.full(len(pts), i) + jitter, pts, color="0.15", s=14, zorder=4)
        ax.errorbar(
            i,
            means[i],
            yerr=stds[i],
            fmt="none",
            ecolor="0.2",
            elinewidth=1.15,
            capsize=3.0,
            capthick=1.15,
            zorder=5,
        )

    rnd = float(per_tcr[f"random_{metric}"].mean())
    ax.axhline(rnd, color="0.45", linestyle="--", linewidth=1.15, zorder=3)
    if show_random_label:
        ymin, ymax = ax.get_ylim()
        ax.text(
            0.02,
            rnd + 0.035 * max(ymax - ymin, 0.5),
            f"Random ranking = {100.0 * rnd:.1f}%",
            transform=ax.get_yaxis_transform(),
            fontsize=9,
            color="0.35",
            va="bottom",
            ha="left",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [TICK_LABELS[m] for m in models], rotation=0, ha="center", fontsize=9
    )
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(panel, fontsize=12, loc="left")
    style_spines(ax)


def shared_legend_handles():
    # Matplotlib fills legends column-major. Order so ncol=3 yields:
    #   One-hot input       Raw ESMC input       LoRA ESMC input
    #   One-hot + VICReg    Raw ESMC + VICReg    LoRA ESMC + VICReg
    entries = [
        ("onehot_composition", "One-hot input"),
        ("onehot_vicreg", "One-hot + VICReg"),
        ("pretrained_esmc_meanpool", "Raw ESMC input"),
        ("raw_esmc_vicreg", "Raw ESMC + VICReg"),
        ("finetuned_esmc_meanpool", "LoRA ESMC input"),
        ("finetuned_esmc_vicreg", "LoRA ESMC + VICReg"),
    ]
    handles = []
    for model, label in entries:
        is_learned = model in LEARNED
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[model],
                linestyle="-" if is_learned else "--",
                marker="o" if is_learned else "s",
                markerfacecolor=COLORS[model] if is_learned else "white",
                markeredgecolor=COLORS[model],
                linewidth=2.0 if is_learned else 1.6,
                markersize=6,
                label=label,
            )
        )
    return handles


def main() -> None:
    global SEEDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    SEEDS = tuple(args.seeds)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    per_tcr = pd.read_csv(REFINED / f"{args.split}_multicognate_retrieval_per_tcr_hardened.csv")

    # Column 0 and row 1 are given extra space: the panel (a) cartoon carries
    # several annotation lines set in points, so it needs a minimum physical
    # size that the data panels do not. Shrinking the figure without this
    # rebalancing makes that text overflow long before the data panels suffer.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.6, 6.5),
        gridspec_kw={"height_ratios": [1.25, 1.0], "width_ratios": [1.05, 1.0]},
    )
    axes[0, 0].axis("off")
    axes[0, 0].set_title("(a) pMHC-aligned TCR geometry", fontsize=12, loc="left")
    panel_auroc(axes[0, 1], args.split)
    panel_knn(axes[1, 0], args.split)
    panel_retrieval(
        axes[1, 1],
        per_tcr,
        "recall@10",
        "(d) Known-cognate pMHC Recall@10",
        "Recall@10",
        show_random_label=True,
    )

    handles = shared_legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        columnspacing=1.6,
        handletextpad=0.5,
        bbox_to_anchor=(0.52, -0.015),
    )
    # The previous rotated row-band labels are dropped: after removing the mAP
    # panel the rows no longer group cleanly into organisation vs retrieval.
    fig.tight_layout(rect=[0.01, 0.07, 1, 1])
    # Drawn last: the cartoon sizes itself to the settled axes box.
    panel_schematic(axes[0, 0])
    out = FIG_DIR / f"{args.split}_geometry_multipanel.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
