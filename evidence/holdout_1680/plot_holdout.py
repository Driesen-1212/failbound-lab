# -*- coding: utf-8 -*-
"""样本外检验的图：把 3 个旧基底（样本内）与 2 个新基底（样本外）画在同一张
「可学习性 AUC → 绝对增益」坐标里，并叠加冻结臂的 lowess 曲线与零点。

要点：曲线只用样本内 18 个点拟合（left panel 的那条线），样本外的点**不参与拟合**，
所以它们落在哪一侧就是一次真正的样本外检验。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "unzipped", "agent_lab"))

from analyze import lowess_fit, zero_crossing  # noqa: E402

OUT = os.path.join(ROOT, "results-extra")
INK = "#1F2933"
MUTED = "#7B8794"
COL = {"frozen": "#2F6F9F", "holdout": "#C2410C"}


def main():
    res = json.load(open(os.path.join(HERE, "holdout_results.json"), encoding="utf-8"))
    fr = pd.DataFrame(res["frozen"])
    ho = pd.DataFrame(res["holdout"])

    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)
    ax.axhline(0, color=MUTED, lw=0.9, zorder=1)

    # 冻结臂的 lowess 曲线（只用样本内的 18 个点）
    x = fr["auc"].values.astype(float)
    y = fr["abs_gain"].values.astype(float)
    order = np.argsort(x)
    grid, yhat = lowess_fit(x[order], y[order])
    ax.plot(grid, yhat, color=COL["frozen"], lw=1.6, alpha=0.85, zorder=2,
            label="boundary fitted on old bases only (in-sample, n=18)")
    star, _ = zero_crossing(grid, yhat)
    if not np.isnan(star):
        ax.axvline(star, color=COL["frozen"], ls="--", lw=1.0, alpha=0.7, zorder=1)
        ax.annotate(f"AUC*={star:.2f}", xy=(star, ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points",
                    fontsize=8.5, color=COL["frozen"])

    ax.scatter(fr["auc"], fr["abs_gain"], s=34, facecolors="none",
               edgecolors=COL["frozen"], linewidths=1.1, zorder=3,
               label="old bases (expt_gap / dielectric / log_gvrh)")
    ax.scatter(ho["auc"], ho["abs_gain"], s=44, marker="^",
               color=COL["holdout"], edgecolors="white", linewidths=0.5, zorder=4,
               label="new bases (log_kvrh / perovskites), never used in the fit")

    for _, r in ho.iterrows():
        if abs(r["abs_gain"]) > 18:
            ax.annotate(r["op"], xy=(r["auc"], r["abs_gain"]), fontsize=7,
                        color=COL["holdout"], xytext=(4, 3),
                        textcoords="offset points")

    ax.set_xlabel("learnability AUC (top-10% label, RF 5-fold)", fontsize=9.5, color=INK)
    ax.set_ylabel("absolute gain (BO hits - space-filling hits), budget 400", fontsize=9.5, color=INK)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C9D1D9")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Out-of-sample test of the failure boundary: new bases are not involved in the fit",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_holdout.png")
    fig.savefig(p, bbox_inches="tight")
    print(f"[save] {p}")

    b = res.get("boundary_fits", {})
    for k, v in b.items():
        print(f"  {k:<14} n={v['n_tasks']:>2} AUC*={v['auc_star']} CI={v['ci']}")


if __name__ == "__main__":
    main()
