"""Fig.: output samples on C2S-MS Floods: inputs, reference, confusion maps of four models, fused uncertainty.

Per row (one chip): S2 true colour (B4, B3, B2; C2S-MS cloud / shadow outlined), S1 VH (dB), reference
(LabelHand: 0 land, 1 flood, -1 not scored), confusion maps (TP / FP / FN / TN) of the joint multimodal TerraMind,
TerraMind-EDL-S2, TerraMind-EDL-S1 and TerraMind-DEUF, and the DEUF uncertainty mass u = K / S from the float evidence.
The per-chip flood-class IoU is written under each confusion map.
Input: <maps>/<id>.npz of eval_sen1_fusion_compare.py --save-maps --save-evidence, and the Sen1-layout C2S-MS root.
Usage: python fig_c2sms_samples.py --maps <sen1_maps/c2sms> --layout <c2sms_all_sen1layout> --out <dir> --ids ID ...
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import style

p = argparse.ArgumentParser()
p.add_argument("--maps", type=Path, required=True)
p.add_argument("--layout", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--ids", nargs="+", required=True)
p.add_argument("--labels", nargs="+", default=None, help="row labels, e.g. (a) (b) (c)")
p.add_argument("--umax", type=float, default=0.2)
a = p.parse_args()

MODELS = [("multimodal", "Joint\nTerraMind"), ("bef/optical", "TerraMind-\nEDL-S2"),
          ("bef/sar", "TerraMind-\nEDL-S1"), ("bef/fused", "TerraMind-\nDEUF")]  # two lines: 8 pt titles
# confusion encoding of the maps: 0 not scored, 1 TN, 2 FP, 3 FN, 4 TP (palette validated with the dataviz script)
CM = ListedColormap(["#ffffff", "#e6e6e1", "#d6452f", "#c98a0b", "#2f6db5"])
REF = ListedColormap(["#ffffff", "#e6e6e1", "#2f6db5"])


def read(path, bands=None):
    with rasterio.open(path) as r:
        return r.read(bands) if bands else r.read(1)


def rgb(sid):
    x = read(a.layout / "data" / "S2L1CHand" / f"{sid}_S2Hand.tif", [4, 3, 2]).astype(np.float32)
    x = np.moveaxis(x, 0, -1)
    valid = x.sum(-1) > 0
    lo, hi = np.percentile(x[valid], [1, 99]) if valid.any() else (0, 1)
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


style.apply()
ncol = 3 + len(MODELS) + 1
n = len(a.ids)
fig, axes = plt.subplots(n, ncol, figsize=(style.COL2, 0.93 * n + 0.62), squeeze=False)
titles = ["S2\n(RGB)", "S1\nVH", "Reference\n"] + [m[1] for m in MODELS] + ["DEUF\n$u$"]
for i, sid in enumerate(a.ids):
    z = np.load(a.maps / f"{sid}.npz")
    ent, ev_ent = list(z["entries"]), list(z["evidence_entries"])
    cloud = np.isin(read(a.layout / "extra" / f"{sid}_CloudLabel.tif"), [1, 2])
    vh = read(a.layout / "data" / "S1GRDHand" / f"{sid}_S1Hand.tif", [2])[0]
    lab = read(a.layout / "data" / "LabelHand" / f"{sid}_LabelHand.tif")
    ref = np.where(lab == 1, 2, np.where(lab == 0, 1, 0))
    row = axes[i]
    row[0].imshow(rgb(sid))
    if cloud.any() and not cloud.all():
        row[0].contour(cloud.astype(float), levels=[0.5], colors="#ffd400", linewidths=0.6)
    row[1].imshow(np.clip(np.nan_to_num(vh, nan=-30), -25, -5), cmap="gray")
    row[2].imshow(ref, cmap=REF, vmin=0, vmax=2, interpolation="nearest")
    for j, (e, _) in enumerate(MODELS):
        cm = z["cm"][ent.index(e)]
        row[3 + j].imshow(cm, cmap=CM, vmin=0, vmax=4, interpolation="nearest")
        tp, fp, fn = ((cm == k).sum() for k in (4, 2, 3))
        row[3 + j].text(0.5, -0.04, f"IoU {tp / max(tp + fp + fn, 1):.3f}", transform=row[3 + j].transAxes,
                        ha="center", va="top", fontsize=8)
    ev = z["evidence"][ev_ent.index("bef/fused")].astype(np.float32)
    u = 2.0 / (ev[0] + ev[1] + 2.0)
    u[z["cm"][ent.index("bef/fused")] == 0] = np.nan
    im = row[-1].imshow(u, cmap="Purples", vmin=0, vmax=a.umax, interpolation="nearest")
    lbl = a.labels[i] if a.labels else ""
    row[0].set_ylabel(lbl, rotation=0, ha="right", va="center", labelpad=4)
    for ax in row:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.4); s.set_color("#7a7a74")
for j, t in enumerate(titles):
    axes[0, j].set_title(t, fontsize=8, pad=3)
handles = [Patch(color="#2f6db5", label="TP / flood"), Patch(color="#d6452f", label="FP"),
           Patch(color="#c98a0b", label="FN"), Patch(color="#e6e6e1", label="TN / land"),
           Patch(facecolor="#ffffff", edgecolor="#7a7a74", lw=0.4, label="Not scored"),
           plt.Line2D([], [], color="#ffd400", lw=1.0, label="Cloud / shadow")]
fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.45, -0.005), handlelength=1.2,
           columnspacing=1.0, fontsize=8)
fig.subplots_adjust(left=0.03, right=0.93, top=0.91, bottom=0.12, wspace=0.05, hspace=0.22)
cax = fig.add_axes([0.94, 0.30, 0.008, 0.5])
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=8, width=0.4, length=2)
cb.outline.set_linewidth(0.4)
cb.set_label(f"$u$ (clipped at {a.umax})", fontsize=8)
a.out.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out / "fig_c2sms_samples.pdf")
fig.savefig(a.out / "fig_c2sms_samples.png", dpi=200)
print("written", a.out / "fig_c2sms_samples.pdf")
