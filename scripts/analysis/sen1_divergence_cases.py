"""Case study of the Sen1Floods11 images where the SAR and optical branches of one late-fusion pair disagree most.

Selection rule (fixed): the --top images with the largest share of disagreement pixels (pair --pair-name in
fusion_evidence_windows.csv), among images with >= 10000 evaluated pixels.
Per image (full chip, stitched from the 256 windows), inside the evaluation mask (SAR valid & optical valid & GT valid):
  * branch predictions / DST uncertainty, DEUF fused prediction, which branch decides each disagreement pixel
  * physical cross-references: MNDWI = (B3 - B11) / (B3 + B11) with an Otsu threshold, and an Otsu threshold on VH dB
  * conditions: cloud (s2cloudless >= 50 %), SAR dark (VH < -25 dB), vegetation (NDVI > 0.5)
Writes <out>/cases.csv (per image), <out>/disagreement_types.csv (per image and disagreement type) and
<out>/<sample_id>.png. Usage:
  python sen1_divergence_cases.py --pair NAME SAR_CFG SAR_CKPT OPT_CFG OPT_CKPT --windows-csv CSV --out DIR [--top 8]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.colors import ListedColormap
from skimage.filters import threshold_otsu

from eval_sen1_fusion_compare import load_model, resolve_ckpt
from flood_uncertainty.data.sen1floods11 import Sen1Floods11Dataset, collect_sen1_splits

p = argparse.ArgumentParser()
p.add_argument("--pair", nargs=5, required=True, metavar=("NAME", "SAR_CFG", "SAR_CKPT", "OPT_CFG", "OPT_CKPT"))
p.add_argument("--windows-csv", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--top", type=int, default=8)
a = p.parse_args()
name, sc, sk, oc, ok = a.pair
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

w = pd.read_csv(a.windows_csv)
g = w[w.pair == name].groupby("sample_id")[["eval_px", "disagree"]].sum()
g = g[g.eval_px >= 10000]
chosen = (g.disagree / g.eval_px).sort_values(ascending=False).head(a.top).index.tolist()
print("cases:", chosen)

(scfg, sm), (ocfg, om) = (load_model(c, resolve_ckpt(k), device) for c, k in [(sc, sk), (oc, ok)])
samples = {s.sample_id: s for s in collect_sen1_splits(scfg.data_params)["test"]}
a.out.mkdir(parents=True, exist_ok=True)


def otsu_mask(x, valid):
    t = threshold_otsu(x[valid & np.isfinite(x)])
    return t


rows, type_rows = [], []
for sid in chosen:
    s = samples[sid]
    ds_s, ds_o = Sen1Floods11Dataset([s], scfg.data_params), Sen1Floods11Dataset([s], ocfg.data_params)
    with rasterio.open(s.label) as r:
        H, W = r.shape
    es, eo = np.zeros((2, H, W), np.float32), np.zeros((2, H, W), np.float32)
    ys, yo = np.zeros((H, W), np.int64), np.zeros((H, W), np.int64)
    with torch.inference_mode():
        for i, (_, win) in enumerate(ds_s.windows):
            bs, bo = ds_s[i], ds_o[i]
            sl = np.s_[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width]
            es[(slice(None),) + sl] = torch.relu(sm.network(bs["image"][None].to(device)))[0].cpu().numpy()
            eo[(slice(None),) + sl] = torch.relu(om.network(bo["image"][None].to(device)))[0].cpu().numpy()
            ys[sl], yo[sl] = bs["mask"][0].numpy(), bo["mask"][1].numpy()
    v = (ys != 0) & (yo != 0)
    gt = ys == 2
    with rasterio.open(s.sar) as r:
        vv, vh = r.read(1).astype(np.float32), r.read(2).astype(np.float32)
    with rasterio.open(s.optical) as r:
        b = r.read().astype(np.float32) / 10000  # 13 bands B1..B12 (B8A at index 8)
    cp = rasterio.open(Path(s.label).parents[1] / "S2CloudlessMask" / f"{sid}_cloudprob.tif").read(1) / 2.55
    green, red, nir, swir1 = b[2], b[3], b[7], b[11]
    mndwi = (green - swir1) / np.maximum(green + swir1, 1e-6)
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    t_mndwi, t_vh = otsu_mask(mndwi, v), otsu_mask(vh, v)
    ndwi_w, sarotsu_w = mndwi > t_mndwi, vh < t_vh
    ps, po = es[1] > es[0], eo[1] > eo[0]
    ef = es + eo
    pf = ef[1] > ef[0]
    us, uo = 2 / (es.sum(0) + 2), 2 / (eo.sum(0) + 2)
    dis = v & (ps != po)
    cloud, dark, veg = cp >= 50, np.nan_to_num(vh, nan=0) < -25, ndvi > 0.5

    def share(m, sel):
        return float(m[sel].mean()) if sel.any() else np.nan

    rows.append(dict(
        sample_id=sid, eval_px=int(v.sum()), disagree_share=share(dis, v), gt_water=share(gt, v),
        sar_right=share(ps == gt, dis), opt_right=share(po == gt, dis), fused_right=share(pf == gt, dis),
        sar_decides=share(pf == ps, dis), mndwi_otsu_right=share(ndwi_w == gt, dis), vh_otsu_right=share(sarotsu_w == gt, dis),
        mndwi_otsu_iou=float((ndwi_w & gt & v).sum() / max(((ndwi_w | gt) & v).sum(), 1)),
        vh_otsu_iou=float((sarotsu_w & gt & v).sum() / max(((sarotsu_w | gt) & v).sum(), 1)),
        t_mndwi=float(t_mndwi), t_vh_db=float(t_vh),
        cloud_in_dis=share(cloud, dis), dark_in_dis=share(dark, dis), veg_in_dis=share(veg, dis),
        cloud_in_eval=share(cloud, v), dark_in_eval=share(dark, v), veg_in_eval=share(veg, v),
        u_sar_dis_sar_wrong=float(np.median(us[dis & (ps != gt)])) if (dis & (ps != gt)).any() else np.nan,
        u_opt_dis_opt_wrong=float(np.median(uo[dis & (po != gt)])) if (dis & (po != gt)).any() else np.nan,
        u_sar_dis_sar_right=float(np.median(us[dis & (ps == gt)])) if (dis & (ps == gt)).any() else np.nan,
        u_opt_dis_opt_right=float(np.median(uo[dis & (po == gt)])) if (dis & (po == gt)).any() else np.nan,
    ))
    # disagreement types: which branch says water, and is it right
    for tname, m in [("sar_water_opt_land", dis & ps & ~po), ("opt_water_sar_land", dis & po & ~ps)]:
        type_rows.append(dict(sample_id=sid, type=tname, pixels=int(m.sum()), share_of_dis=share(m, dis),
                              gt_water=share(gt, m), fused_right=share(pf == gt, m), mndwi_otsu_water=share(ndwi_w, m),
                              vh_otsu_water=share(sarotsu_w, m), cloud=share(cloud, m), dark=share(dark, m), veg=share(veg, m),
                              vh_db_median=float(np.nanmedian(vh[m])) if m.any() else np.nan,
                              mndwi_median=float(np.nanmedian(mndwi[m])) if m.any() else np.nan))

    # figure
    rgb = np.clip(np.stack([b[3], b[2], b[1]], -1) / 0.3, 0, 1)
    def lab(pred):
        o = np.where(pred, 2, 1).astype(float); o[~v] = 0; return o
    cm3 = ListedColormap(["white", "#d9c9a3", "#2166ac"])
    decide = np.zeros((H, W)); decide[dis & (pf == gt) & (pf == ps)] = 1; decide[dis & (pf == gt) & (pf == po)] = 2
    decide[dis & (pf != gt) & (pf == ps)] = 3; decide[dis & (pf != gt) & (pf == po)] = 4
    cm5 = ListedColormap(["#eeeeee", "#1b9e77", "#7570b3", "#d95f02", "#e7298a"])
    panels = [("S2 RGB", rgb, None, None), ("s2cloudless prob (%)", cp, "gray", (0, 100)), ("VH (dB)", vh, "gray", (-30, -5)),
              ("GT (LabelHand)", lab(gt), cm3, (0, 2)), ("Optical branch", lab(po), cm3, (0, 2)), ("SAR branch", lab(ps), cm3, (0, 2)),
              ("DEUF fused", lab(pf), cm3, (0, 2)), ("Optical u (2/S)", np.where(v, uo, np.nan), "magma", (0, 1)),
              ("SAR u (2/S)", np.where(v, us, np.nan), "magma", (0, 1)),
              ("Disagreement: who decides", decide, cm5, (0, 4)),
              (f"MNDWI > Otsu ({t_mndwi:.2f})", lab(ndwi_w), cm3, (0, 2)), (f"VH < Otsu ({t_vh:.1f} dB)", lab(sarotsu_w), cm3, (0, 2))]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12.5), layout="constrained")
    for ax, (title, img, cmap, lim) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1], interpolation="nearest")
        ax.set_title(title); ax.set_axis_off()
    r = rows[-1]
    fig.suptitle(f"{sid} | {name} | disagreement {r['disagree_share']:.3f} of evaluated pixels | in disagreement: "
                 f"SAR right {r['sar_right']:.3f}, optical right {r['opt_right']:.3f}, fused right {r['fused_right']:.3f}", fontsize=13)
    fig.supxlabel("Who decides (disagreement pixels): green = SAR decides, right; purple = optical decides, right; "
                  "orange = SAR decides, wrong; pink = optical decides, wrong. Land = beige, water = blue, invalid = white.", fontsize=10)
    fig.savefig(a.out / f"{sid}.png", dpi=110)
    plt.close(fig)
    print(sid, "done", flush=True)

pd.DataFrame(rows).to_csv(a.out / "cases.csv", index=False)
pd.DataFrame(type_rows).to_csv(a.out / "disagreement_types.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(pd.DataFrame(type_rows).round(3).to_string(index=False))
