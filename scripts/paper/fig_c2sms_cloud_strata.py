"""Fig.: flood-class IoU on C2S-MS Floods by chip cloud-and-shadow fraction (and, optionally, the paired difference).

Input: <results>/c2sms_all/groups.csv and groups_pairs.csv from eval_sen1_fusion_compare.py --groups-csv (groups = chip
cloud + shadow fraction bins, artifacts/results/c2sms/cloud_groups.csv).
  --panels a  (default, main text): pooled flood-class IoU per stratum and model.
  --panels ab (appendix): adds (b), the paired difference fused - joint multimodal TerraMind with the 95% bootstrap
              interval over chips as a shaded band.
Panel (a) is a grouped bar chart (zero baseline, value labels) of the two DEUF configurations and the joint model.
Writes <out>/<name>.{pdf,png,csv}; name = fig_c2sms_cloud_strata (a) or fig_c2sms_cloud_strata_ci (ab).
Usage: python fig_c2sms_cloud_strata.py --results artifacts/results --out <dir> [--panels ab]
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
p.add_argument("--panels", choices=["a", "ab"], default="a")
a = p.parse_args()

GROUPS = ["0-25%", "25-50%", "50-75%", "75-100%"]
LINES = [  # entry, label, colour, marker, width (the sensor-specific predictors are reported in the text)
    ("multimodal", "Joint multimodal TerraMind", style.C["multimodal"], "s", 1.4),
    ("unet_deuf/fused", "UNet++-DEUF", style.C["unet_deuf"], "^", 1.4),
    ("bef/fused", "TerraMind-DEUF", style.C["bef"], "o", 1.6),
]
DIFFS = [("bef/fused", "TerraMind-DEUF", style.C["bef"], "o"),
         ("unet_deuf/fused", "UNet++-DEUF", style.C["unet_deuf"], "^")]
DASH = "–"

g = pd.read_csv(a.results / "c2sms_all" / "groups.csv")
pr = pd.read_csv(a.results / "c2sms_all" / "groups_pairs.csv")
n = g[g.entry == "bef/fused"].set_index("group").images.reindex(GROUPS)
x = np.arange(len(GROUPS))
xlabels = [gr.replace("-", DASH) + f"\n($n$={k})" for gr, k in zip(GROUPS, n)]

style.apply()
if a.panels == "ab":
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(style.COL1, 3.75), sharex=True,
                                   gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.32})
else:
    fig, ax1 = plt.subplots(figsize=(style.COL1, 2.3))
    ax2 = None

rows = []
BAR_ORDER = ["multimodal", "unet_deuf/fused", "bef/fused"]  # left to right within a stratum
width = 0.26
for k, e in enumerate(BAR_ORDER):
    lab, c = next((l_, c_) for e_, l_, c_, *_ in LINES if e_ == e)
    y = g[g.entry == e].set_index("group").water_iou.reindex(GROUPS).to_numpy()
    xb = x + (k - 1) * width
    ax1.bar(xb, y, width=width, color=c, edgecolor="white", linewidth=0.6, label=lab, zorder=2)
    for xi, yi in zip(xb, y):  # value label inside the bar top
        ax1.text(xi, yi - 0.015, f"{yi:.3f}", rotation=90, ha="center", va="top", fontsize=6,
                 color="white", zorder=3)
    rows += [dict(panel="a", entry=e, group=gr, value=v) for gr, v in zip(GROUPS, y)]
ax1.set_ylabel("Flood-class IoU")
ax1.set_ylim(0, 1.0)
ax1.set_xlim(-0.5, len(GROUPS) - 0.5)
ax1.grid(axis="x", visible=False)
hd, lb = ax1.get_legend_handles_labels()
fig.legend(hd, lb, loc="upper center", ncol=3, fontsize=6.5, handlelength=1.2,
           columnspacing=0.8, bbox_to_anchor=(0.55, 1.0))

if ax2 is not None:
    ax1.set_title("(a) Flood-class IoU", loc="left")
    ax2.axhline(0, color="#7a7a74", lw=0.8, zorder=1)
    for e, lab, c, m in DIFFS:
        d = pr[(pr.row == e) & (pr.col == "multimodal")].set_index("group").reindex(GROUPS)
        ax2.fill_between(x, d.ci_low, d.ci_high, color=c, alpha=0.15, lw=0, zorder=1)
        ax2.plot(x, d["diff"], color=c, marker=m, lw=1.4, ms=4, mec="white", mew=0.6, label=lab, zorder=3)
        rows += [dict(panel="b", entry=e, group=gr, value=v, ci_low=lo, ci_high=hi)
                 for gr, v, lo, hi in zip(GROUPS, d["diff"], d.ci_low, d.ci_high)]
    ax2.set_ylabel("$\\Delta$IoU vs. joint")
    ax2.set_title("(b) Difference to the joint model (95% CI)", loc="left")
    ax2.legend(loc="upper left", fontsize=6.5, handlelength=2.0, borderaxespad=0.3, ncol=2)
    ax2.set_xticks(x, xlabels)
    ax2.set_xlabel("Chip cloud-and-shadow fraction")
    fig.subplots_adjust(top=0.91)
    name = "fig_c2sms_cloud_strata_ci"
else:
    ax1.set_xticks(x, xlabels)
    ax1.set_xlabel("Chip cloud-and-shadow fraction")
    fig.subplots_adjust(top=0.90, bottom=0.24, left=0.15, right=0.98)
    name = "fig_c2sms_cloud_strata"

a.out.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out / f"{name}.pdf")
fig.savefig(a.out / f"{name}.png")
pd.DataFrame(rows).to_csv(a.out / f"{name}.csv", index=False)
print("written", a.out / f"{name}.pdf")
