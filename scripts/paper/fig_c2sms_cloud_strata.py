"""Fig.: flood-class IoU on C2S-MS Floods by chip cloud-and-shadow fraction (single column, IEEE 8 pt text).

Input: <results>/c2sms_all/groups.csv and groups_pairs.csv from eval_sen1_fusion_compare.py --groups-csv (groups = chip
cloud + shadow fraction bins, artifacts/results/c2sms/cloud_groups.csv).
  --kind iou (default, main text): horizontal grouped bars (strata on the y axis, zero baseline) of the joint
             multimodal TerraMind, UNet++-DEUF and TerraMind-DEUF, with the value at the end of each bar. Horizontal
             bars leave room for 8 pt value labels in one column.
  --kind ci  (appendix): paired difference fused - joint multimodal TerraMind per stratum with the 95% bootstrap
             interval over chips as a shaded band.
Writes <out>/<name>.{pdf,png,csv}; name = fig_c2sms_cloud_strata (iou) or fig_c2sms_cloud_strata_ci (ci).
Usage: python fig_c2sms_cloud_strata.py --results artifacts/results --out <dir> [--kind ci]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

p = argparse.ArgumentParser()
p.add_argument("--results", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--kind", choices=["iou", "ci"], default="iou")
a = p.parse_args()

GROUPS = ["0-25%", "25-50%", "50-75%", "75-100%"]
MODELS = [  # entry, label, colour, marker (top to bottom within a stratum)
    ("bef/fused", "TerraMind-DEUF", style.C["bef"], "o"),
    ("unet_deuf/fused", "UNet++-DEUF", style.C["unet_deuf"], "^"),
    ("multimodal", "Joint multimodal TerraMind", style.C["multimodal"], "s"),
]
DASH = "–"

g = pd.read_csv(a.results / "c2sms_all" / "groups.csv")
pr = pd.read_csv(a.results / "c2sms_all" / "groups_pairs.csv")
n = g[g.entry == "bef/fused"].set_index("group").images.reindex(GROUPS)
labels = [gr.replace("-", DASH) + f"\n($n$={k})" for gr, k in zip(GROUPS, n)]
style.apply()
rows = []

if a.kind == "iou":
    fig, ax = plt.subplots(figsize=(style.COL1, 3.0))
    yc = np.arange(len(GROUPS))[::-1]  # least clouded stratum at the top
    h = 0.27
    for k, (e, lab, c, _) in enumerate(MODELS):
        v = g[g.entry == e].set_index("group").water_iou.reindex(GROUPS).to_numpy()
        yb = yc + (1 - k) * h
        ax.barh(yb, v, height=h, color=c, edgecolor="white", linewidth=0.6, label=lab, zorder=2)
        for yi, vi in zip(yb, v):
            ax.text(vi + 0.01, yi, f"{vi:.3f}", ha="left", va="center", fontsize=8, color="#2b2b2b", zorder=3)
        rows += [dict(entry=e, group=gr, iou=x) for gr, x in zip(GROUPS, v)]
    ax.set_yticks(yc, labels)
    ax.set_ylabel("Chip cloud-and-shadow fraction")
    ax.set_xlabel("Flood-class IoU")
    ax.set_xlim(0, 1.13)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(-0.55, len(GROUPS) - 0.45)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.17), ncol=2, handlelength=1.2, columnspacing=1.0,
              borderaxespad=0.0)
    fig.subplots_adjust(top=0.85, bottom=0.14, left=0.24, right=0.98)
    name = "fig_c2sms_cloud_strata"
else:
    fig, ax = plt.subplots(figsize=(style.COL1, 2.3))
    x = np.arange(len(GROUPS))
    ax.axhline(0, color="#7a7a74", lw=0.8, zorder=1)
    for e, lab, c, m in MODELS[:2]:
        d = pr[(pr.row == e) & (pr.col == "multimodal")].set_index("group").reindex(GROUPS)
        ax.fill_between(x, d.ci_low, d.ci_high, color=c, alpha=0.15, lw=0, zorder=1)
        ax.plot(x, d["diff"], color=c, marker=m, lw=1.4, ms=4, mec="white", mew=0.6, label=lab, zorder=3)
        rows += [dict(entry=e, group=gr, diff=v, ci_low=lo, ci_high=hi)
                 for gr, v, lo, hi in zip(GROUPS, d["diff"], d.ci_low, d.ci_high)]
    ax.set_ylabel("$\\Delta$IoU vs. joint model")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Chip cloud-and-shadow fraction")
    ax.set_xlim(-0.25, len(GROUPS) - 0.75)
    ax.legend(loc="upper left", handlelength=2.0, borderaxespad=0.3)
    fig.subplots_adjust(top=0.97, bottom=0.25, left=0.17, right=0.97)
    name = "fig_c2sms_cloud_strata_ci"

a.out.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out / f"{name}.pdf")
fig.savefig(a.out / f"{name}.png")
pd.DataFrame(rows).to_csv(a.out / f"{name}.csv", index=False)
print("written", a.out / f"{name}.pdf")
