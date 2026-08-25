#!/usr/bin/env python3
"""Build a presentation PNG table from Boltz signal-injection logreg metrics.

Default: ImmRep test McClish partial AUC (FPR≤0.1), which is the external
benchmark that matters for the workshop claim.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path("/home/natasha/multimodal_model")
METRICS = REPO / "outputs_data/boltz_signal_injection/model_group_metrics.csv"
OUT_DIR = REPO / "models/figures/workshop/paper_analysis/boltz_signal_injection"

ORDER = [
    ("esm_only", "ESM–VICReg score only (baseline)"),
    ("confidence_only", "Boltz confidence only"),
    ("z_selected_only", "Selected z-block features only"),
    ("z_global_only", "z global block only"),
    ("z_tcr_peptide_only", "z TCR–peptide block only"),
    ("esm_plus_confidence", "ESM–VICReg + confidence"),
    ("esm_plus_confidence_score", "ESM–VICReg + confidence_score"),
    ("esm_plus_iptm", "ESM–VICReg + iPTM"),
    ("esm_plus_chain_pair_iptm", "ESM–VICReg + chain-pair iPTM"),
    ("esm_plus_z_selected", "ESM–VICReg + selected z features"),
    ("esm_plus_z_global", "ESM–VICReg + z global block"),
    ("esm_plus_z_tcr_peptide", "ESM–VICReg + z TCR–peptide"),
    ("esm_plus_z_tcra_peptide", "ESM–VICReg + z TCRα–peptide"),
    ("esm_plus_z_tcrb_peptide", "ESM–VICReg + z TCRβ–peptide"),
    ("esm_plus_z_tcr_hla", "ESM–VICReg + z TCR–HLA"),
    ("esm_plus_z_pep_hla", "ESM–VICReg + z peptide–HLA"),
    ("esm_plus_z_comparison", "ESM–VICReg + z comparison"),
    ("esm_plus_z_all", "ESM–VICReg + all z blocks"),
    ("esm_plus_confidence_z_selected", "ESM–VICReg + conf. + selected z"),
    ("esm_plus_confidence_z_all", "ESM–VICReg + conf. + all z"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS)
    # Fit on val; evaluate on ImmRep (external true-negative benchmark).
    split = df[df["eval_split"] == "immrep_test"].copy()

    rows = []
    for key, label in ORDER:
        r = split[split["model_group"] == key]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(
            {
                "Feature group": label,
                "# feats": int(r["n_features"]),
                "McClish pAUC@0.1": float(r["auc0.1_mcclish"]),
                "AUROC": float(r["auroc"]),
                "Δ pAUC vs ESM–VICReg": float(r["delta_vs_esm_only__auc0.1_mcclish"]),
                "Δ pAUC vs ESM+conf": float(r["delta_vs_esm_plus_confidence__auc0.1_mcclish"]),
            }
        )
    tab = pd.DataFrame(rows)

    csv_out = OUT_DIR / "boltz_signal_injection_logreg_immrep_table.csv"
    tab_round = tab.copy()
    for c in ["McClish pAUC@0.1", "AUROC", "Δ pAUC vs ESM–VICReg", "Δ pAUC vs ESM+conf"]:
        tab_round[c] = tab_round[c].map(
            lambda x, col=c: f"{x:+.3f}" if col.startswith("Δ") else f"{x:.3f}"
        )
    tab_round.to_csv(csv_out, index=False)

    col_labels = list(tab.columns)
    cell_text = []
    for _, r in tab.iterrows():
        cell_text.append(
            [
                r["Feature group"],
                str(int(r["# feats"])),
                f"{r['McClish pAUC@0.1']:.3f}",
                f"{r['AUROC']:.3f}",
                f"{r['Δ pAUC vs ESM–VICReg']:+.3f}",
                f"{r['Δ pAUC vs ESM+conf']:+.3f}",
            ]
        )

    n_rows = len(cell_text)
    n_cols = len(col_labels)
    fig_h = 0.65 + 0.42 * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(15.5, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.55)

    col_widths = [0.36, 0.07, 0.14, 0.10, 0.17, 0.16]
    for j, w in enumerate(col_widths):
        for i in range(n_rows + 1):
            table[(i, j)].set_width(w)

    header_color = "#1f4e79"
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold", fontsize=11)
        cell.set_edgecolor("white")

    best_pauc = float(tab["McClish pAUC@0.1"].max())
    for i, r in enumerate(tab.itertuples(index=False), start=1):
        pauc = float(r[2])
        bg = "#e6f4ea" if abs(pauc - best_pauc) < 1e-12 else ("#f3f6fa" if i % 2 else "#ffffff")
        for j in range(n_cols):
            table[(i, j)].set_facecolor(bg)
            table[(i, j)].set_edgecolor("#d0d7de")
        table[(i, 0)].set_text_props(ha="left", fontsize=10.5)
        table[(i, 0)].PAD = 0.02
        # Emphasise McClish pAUC column
        table[(i, 2)].set_text_props(fontweight="bold", fontsize=11)
        for j, d in ((4, float(r[4])), (5, float(r[5]))):
            txt = table[(i, j)].get_text()
            if d > 0.005:
                txt.set_color("#1b7a3d")
                txt.set_fontweight("bold")
            elif d < -0.005:
                txt.set_color("#a12828")
            else:
                txt.set_color("#444444")

    ax.set_title(
        "Where Boltz adds value over ESM–VICReg  |  ImmRep test (McClish pAUC@0.1)\n"
        "Logistic regression fit on validation decoys  →  evaluated on ImmRep",
        fontsize=14,
        fontweight="bold",
        pad=18,
        loc="left",
    )
    fig.text(
        0.01,
        0.01,
        "Primary metric: McClish partial AUC (FPR≤0.1). Green Δ = gain vs baseline. "
        "Green row = best McClish pAUC. Source: model_group_metrics.csv (eval_split=immrep_test).",
        fontsize=8.5,
        color="#555555",
        ha="left",
        va="bottom",
    )

    png_out = OUT_DIR / "boltz_signal_injection_logreg_immrep_table.png"
    fig.savefig(png_out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Keep the previously shared filename in sync (same ImmRep table).
    import shutil

    also_png = OUT_DIR / "boltz_signal_injection_logreg_test_table.png"
    also_csv = OUT_DIR / "boltz_signal_injection_logreg_test_table.csv"
    shutil.copy2(png_out, also_png)
    shutil.copy2(csv_out, also_csv)

    print(f"Wrote {png_out}", flush=True)
    print(f"Also updated {also_png}", flush=True)
    print(f"Wrote {csv_out}", flush=True)
    print(tab.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
