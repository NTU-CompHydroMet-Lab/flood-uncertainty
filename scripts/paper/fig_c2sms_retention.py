"""Fig.: retention curves on C2S-MS Floods (flood-class IoU of the most certain pixels), DEUF vs joint TerraMind.

Input: the per-chip maps of eval_sen1_fusion_compare.py --save-maps ... --save-evidence bef/optical bef/sar bef/fused
multimodal (<maps>/<id>.npz: cm per entry, raw evidence of four entries) and the C2S-MS cloud label
(<layout>/extra/<id>_CloudLabel.tif, 1 cloud, 2 shadow). The uncertainty mass u = K / S (K = 2) is computed from the
float evidence, not from the uint8 u maps, so the curve is not limited by quantisation. Pixels are pooled over all chips
and sorted by u (most certain first, ties kept in the same log-spaced u bin, 4000 bins from 1e-6 to 1); the
flood-class IoU is recomputed on the retained pixels (x axis 0.5-1: below 50% retention almost only land pixels remain). Panels: all scored pixels, and pixels under cloud or shadow.
Writes <out>/fig_c2sms_retention.{pdf,png} and fig_c2sms_retention.csv (IoU at fixed retention, normalised AUC).
Usage: python fig_c2sms_retention.py --maps <sen1_maps/c2sms> --layout <c2sms_all_sen1layout> --out <dir>
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

import style

p = argparse.ArgumentParser()
p.add_argument("--maps", type=Path, required=True)
p.add_argument("--layout", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

ENTRIES = ["multimodal", "bef/optical", "bef/sar", "bef/fused"]
NAME = {"multimodal": "Joint multimodal TerraMind", "bef/optical": "TerraMind-EDL-S2",
        "bef/sar": "TerraMind-EDL-S1", "bef/fused": "TerraMind-DEUF"}
COLOR = {"multimodal": style.C["multimodal"], "bef/optical": style.C["optical"], "bef/sar": style.C["sar"],
         "bef/fused": style.C["bef"]}
LS = {"multimodal": "-", "bef/optical": "--", "bef/sar": ":", "bef/fused": "-"}
EDGES = np.concatenate([[0.0], np.geomspace(1e-6, 1.0, 4000)])
SUBSETS = ["all", "cloud"]

ids = [x.strip() for x in (a.layout / "splits" / "flood_test_data.txt").read_text().split() if x.strip()]
hist = {(s, e): np.zeros((len(EDGES) - 1, 4), np.int64) for s in SUBSETS for e in ENTRIES}  # tp fp fn tn per u bin
for sid in ids:
    z = np.load(a.maps / f"{sid}.npz")
    ent, ev_ent = list(z["entries"]), list(z["evidence_entries"])
    with rasterio.open(a.layout / "extra" / f"{sid}_CloudLabel.tif") as r:
        cloud = np.isin(r.read(1), [1, 2])
    for e in ENTRIES:
        cm = z["cm"][ent.index(e)]
        ev = z["evidence"][ev_ent.index(e)].astype(np.float32)
        u = 2.0 / (ev[0] + ev[1] + 2.0)
        b = np.clip(np.searchsorted(EDGES, u, side="right") - 1, 0, len(EDGES) - 2)
        for s in SUBSETS:
            m = (cm > 0) if s == "all" else (cm > 0) & cloud
            for k, cls in enumerate([4, 2, 3, 1]):  # tp, fp, fn, tn in the cm encoding
                sel = m & (cm == cls)
                hist[(s, e)][:, k] += np.bincount(b[sel], minlength=len(EDGES) - 1)


def curve(h):
    c = h.cumsum(0)
    keep = c.sum(1) > 0
    c = c[keep]
    ret = c.sum(1) / c[-1].sum()
    den = c[:, 0] + c[:, 1] + c[:, 2]
    iou = np.where(den > 0, c[:, 0] / np.maximum(den, 1), np.nan)
    return ret, iou


rows = []
style.apply()
fig, axes = plt.subplots(1, 2, figsize=(style.COL1, 1.75), sharey=True)
for ax, s, title in zip(axes, SUBSETS, ["(a) All pixels", "(b) Under cloud or shadow"]):
    for e in ENTRIES:
        ret, iou = curve(hist[(s, e)])
        ax.plot(np.r_[ret[0], ret], np.r_[iou[0], iou], color=COLOR[e], ls=LS[e],
                lw=1.4 if e in ("bef/fused", "multimodal") else 1.0, label=NAME[e])
        ok = ~np.isnan(iou)
        auc = np.trapz(iou[ok], ret[ok]) / (ret[ok].max() - ret[ok].min())
        row = dict(subset=s, entry=e, pixels=int(hist[(s, e)].sum()), iou_full=float(iou[-1]), auc_norm=float(auc))
        for r in (0.5, 0.8, 0.9):
            i = min(int(np.searchsorted(ret, r - 1e-12)), len(ret) - 1)
            row[f"iou_at_{int(r * 100)}"] = float(iou[i])
            row[f"ret_at_{int(r * 100)}"] = float(ret[i])
        rows.append(row)
    ax.set_title(title, loc="left")
    ax.set_xlim(0.5, 1)
    ax.set_xlabel("Retained fraction of pixels")
axes[0].set_ylabel("Flood-class IoU")
axes[0].set_ylim(0.68, 1.005)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17), handlelength=2.2, columnspacing=1.0)
fig.tight_layout(w_pad=0.6)
a.out.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out / "fig_c2sms_retention.pdf")
fig.savefig(a.out / "fig_c2sms_retention.png")
t = pd.DataFrame(rows)
t.to_csv(a.out / "fig_c2sms_retention.csv", index=False)
print(t.round(4).to_string())
