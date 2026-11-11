"""Fig.: flood-class IoU on C2S-MS Floods by chip cloud-and-shadow fraction, and the paired DEUF - joint difference.

Input: <results>/c2sms_all/groups.csv and groups_pairs.csv from eval_sen1_fusion_compare.py --groups-csv (groups = chip
cloud + shadow fraction bins, artifacts/results/c2sms/cloud_groups.csv). (a) pooled flood-class IoU per stratum and
model; (b) paired difference fused - joint multimodal TerraMind with the 95% bootstrap interval over chips.
Writes <out>/fig_c2sms_cloud_strata.{pdf,png,csv}.
Usage: python fig_c2sms_cloud_strata.py --results artifacts/results --out <dir>
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
a = p.parse_args()

GROUPS = ["0-25%", "25-50%", "50-75%", "75-100%"]
LINES = [  # entry, label, colour, linestyle, marker, width
    ("multimodal", "Joint multimodal TerraMind", style.C["multimodal"], "-", "s", 1.4),
    ("bef/optical", "TerraMind-EDL-S2", style.C["optical"], "--", "D", 1.0),
    ("bef/sar", "TerraMind-EDL-S1", style.C["sar"], ":", "v", 1.0),
    ("unet_deuf/fused", "UNet++-DEUF", style.C["unet_deuf"], "-", "^", 1.4),
    ("bef/fused", "TerraMind-DEUF", style.C["bef"], "-", "o", 1.6),
]
DIFFS = [("bef/fused", "TerraMind-DEUF", style.C["bef"], "o", -0.08),
         ("unet_deuf/fused", "UNet++-DEUF", style.C["unet_deuf"], "^", 0.08)]

g = pd.read_csv(a.results / "c2sms_all" / "groups.csv")
pr = pd.read_csv(a.results / "c2sms_all" / "groups_pairs.csv")
n = g[g.entry == "bef/fused"].set_index("group").images.reindex(GROUPS)
x = np.arange(len(GROUPS))

style.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(style.COL1, 3.75), sharex=True,
                               gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.32})
rows = []
for e, lab, c, ls, m, lw in LINES:
    y = g[g.entry == e].set_index("group").water_iou.reindex(GROUPS).to_numpy()
    ax1.plot(x, y, color=c, ls=ls, marker=m, lw=lw, ms=4 if lw > 1.2 else 3.5, label=lab,
             mec="white" if lw > 1.2 else c, mew=0.6, zorder=3 if e == "bef/fused" else 2)
    rows += [dict(panel="a", entry=e, group=gr, value=v) for gr, v in zip(GROUPS, y)]
ax1.set_ylabel("Flood-class IoU")
ax1.set_ylim(0.6, 1.0)
fig.legend(*ax1.get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=6.5, handlelength=2.2,
           columnspacing=0.8, bbox_to_anchor=(0.55, 1.0))
ax1.set_title("(a) Flood-class IoU", loc="left")

ax2.axhline(0, color="#7a7a74", lw=0.8, zorder=1)
for e, lab, c, m, off in DIFFS:
    d = pr[(pr.row == e) & (pr.col == "multimodal")].set_index("group").reindex(GROUPS)
    ax2.errorbar(x + off, d["diff"], yerr=[d["diff"] - d.ci_low, d.ci_high - d["diff"]], fmt=m, color=c,
                 ms=4, mec="white", mew=0.6, elinewidth=1.0, capsize=2, label=lab, zorder=3)
    rows += [dict(panel="b", entry=e, group=gr, value=v, ci_low=lo, ci_high=hi)
             for gr, v, lo, hi in zip(GROUPS, d["diff"], d.ci_low, d.ci_high)]
ax2.set_ylabel("$\\Delta$IoU vs. joint")
ax2.set_title("(b) Difference to the joint model (95% CI)", loc="left")
ax2.legend(loc="upper left", fontsize=6.5, handlelength=1.2, borderaxespad=0.3, ncol=2)
ax2.set_xticks(x, [f"{gr.replace('-', '–')}\n($n$={k})" for gr, k in zip(GROUPS, n)])
ax2.set_xlabel("Chip cloud-and-shadow fraction")
ax2.set_xlim(-0.4, len(GROUPS) - 0.6)

fig.subplots_adjust(top=0.84)
a.out.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out / "fig_c2sms_cloud_strata.pdf")
fig.savefig(a.out / "fig_c2sms_cloud_strata.png")
pd.DataFrame(rows).to_csv(a.out / "fig_c2sms_cloud_strata.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())
