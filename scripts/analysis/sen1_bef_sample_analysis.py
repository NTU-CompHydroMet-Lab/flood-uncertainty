"""Where does BEF (DEUF of stage-1 + Sen1 fine-tuned TerraMind) win? Per-image and per-condition analysis.

Inputs: <dir>/windows.csv (eval_sen1_fusion_compare.py) and <dir>/features.csv (sen1_window_features.py).
Prints
  1. per-image water IoU of the main entries and the BEF advantage (bef/fused - other), images sorted by it
  2. pooled water IoU per condition bin (cloud inside the eval mask, VH noise floor, speckle)
  3. Spearman correlation between per-image advantage and per-image features
Writes <dir>/per_image.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

p = argparse.ArgumentParser()
p.add_argument("dir", type=Path)
p.add_argument("--top", type=int, default=12)
a = p.parse_args()

w = pd.read_csv(a.dir / "windows.csv")
f = pd.read_csv(a.dir / "features.csv")
ENTRIES = ["bef/fused", "bef/optical", "bef/sar", "direct/fused", "direct/optical", "multimodal", "unet_deuf/fused", "unet_deuf/optical"]
OTHERS = ["multimodal", "direct/fused", "unet_deuf/fused", "bef/optical"]


def iou(d):
    return d.tp / (d.tp + d.fp + d.fn).replace(0, np.nan)


# pixel-weighted features per image (weights = evaluated pixels of the window)
f["n"] = w[w.entry == "bef/fused"].set_index("window")[["tp", "fp", "fn", "tn"]].sum(1).reindex(f.window).to_numpy()
feat_cols = ["cloud_frac_eval", "cloud_frac", "label_invalid_frac", "water_frac", "vh_floor_frac", "vv_cv_median", "vh_db_mean", "opt_bright_frac", "sar_nan_frac"]
img_feat = f.groupby("sample_id").apply(lambda g: pd.Series({c: np.average(g[c].fillna(0), weights=np.maximum(g.n, 1)) for c in feat_cols} | {"pixels": g.n.sum()}))

img = w.groupby(["sample_id", "entry"])[["tp", "fp", "fn", "tn"]].sum()
per = pd.DataFrame({e: iou(img.xs(e, level="entry")) for e in ENTRIES})
for o in OTHERS:
    per[f"adv_vs_{o}"] = per["bef/fused"] - per[o]
per = per.join(img_feat)
per.to_csv(a.dir / "per_image.csv")

pd.set_option("display.width", 250, "display.max_columns", 30)
show = ["bef/fused", "multimodal", "direct/fused", "unet_deuf/fused", "bef/optical", "bef/sar", "cloud_frac_eval", "vh_floor_frac", "vv_cv_median", "water_frac", "pixels"]
print(f"== images where bef/fused beats multimodal by the largest margin (water IoU; {len(per)} images) ==")
print(per.sort_values("adv_vs_multimodal", ascending=False)[show].head(a.top).round(3))
print("\n== images where bef/fused loses most to multimodal ==")
print(per.sort_values("adv_vs_multimodal")[show].head(a.top // 2).round(3))
valid = per.dropna(subset=["bef/fused", "multimodal"])
for o in OTHERS:
    d = valid[f"adv_vs_{o}"].dropna()
    print(f"bef/fused vs {o:16s}: wins {int((d > 0.01).sum())}, ties {int((d.abs() <= 0.01).sum())}, losses {int((d < -0.01).sum())} images (|diff| > 0.01)")

print("\n== pooled water IoU by condition (window bins) ==")
bins = {
    "cloud_frac_eval": [-0.001, 0.0, 0.05, 0.2, 1.0],
    "vh_floor_frac": [-0.001, 0.01, 0.05, 0.2, 1.0],
    "vv_cv_median": [0, 0.35, 0.45, 0.55, 2.0],
}
for col, edges in bins.items():
    b = pd.cut(f.set_index("window")[col], edges)
    ww = w.join(b.rename("bin"), on="window")
    t = ww.groupby(["bin", "entry"], observed=True)[["tp", "fp", "fn"]].sum()
    tab = iou(t).unstack("entry")[ENTRIES]
    tab["windows"] = b.value_counts().reindex(tab.index)
    tab["bef-mm"] = tab["bef/fused"] - tab["multimodal"]
    tab["bef-direct"] = tab["bef/fused"] - tab["direct/fused"]
    tab["fused-opt(bef)"] = tab["bef/fused"] - tab["bef/optical"]
    print(f"-- {col}"); print(tab.round(4))

print("\n== Spearman rho (per image) ==")
rows = []
for o in OTHERS:
    for c in ["cloud_frac_eval", "vh_floor_frac", "vv_cv_median", "water_frac", "opt_bright_frac"]:
        d = per[[f"adv_vs_{o}", c]].dropna()
        r, pv = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
        rows.append((f"bef/fused - {o}", c, round(r, 3), round(pv, 4), len(d)))
print(pd.DataFrame(rows, columns=["advantage", "feature", "rho", "p", "n"]).to_string(index=False))
