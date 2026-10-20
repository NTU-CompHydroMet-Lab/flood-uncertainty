"""Per-window cloud and SAR noise features for the Sen1Floods11 windows that eval_sen1_fusion_compare.py scores.

Windows are rebuilt with the Sen1 loader (same order as windows.csv). Features per window:
  cloud_prob_mean, cloud_frac     s2cloudless probability (S2CloudlessMask, uint8 0..255 -> percent), fraction >= 50 %
  cloud_frac_eval                 cloud fraction inside the evaluation mask; cloud_frac_invalid: inside LabelHand == -1
                                  (cloud fractions use only pixels with a cloud probability; cloud_nodata_frac = the rest)
  label_invalid_frac              LabelHand == -1 (no label, often cloud or cloud shadow)
  eval_frac                       pixels in the evaluation mask (SAR valid & optical valid & GT valid)
  water_frac                      GT water fraction of the evaluation mask
  vv_db_mean, vh_db_mean          mean backscatter over the evaluation mask
  vh_floor_frac                   VH < -25 dB (near the Sentinel-1 IW noise floor)
  vv_cv_median                    median local coefficient of variation of linear VV (7 x 7), speckle level
  sar_nan_frac                    SAR no-data
  opt_bright_frac                 B2 (blue) reflectance > 0.25 (bright cloud / haze that s2cloudless can miss)
Usage: python sen1_window_features.py --out features.csv [--splits-path DIR] [--split test]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter

from flood_uncertainty.data.sen1floods11 import Sen1Floods11Dataset, collect_sen1_splits
from flood_uncertainty.utils.config_loader import load_mode_config

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--splits-path", type=Path, default=None)
p.add_argument("--split", default="test")
a = p.parse_args()

cfg = load_mode_config(ROOT / "configurations/edl_sar_sen1.json", mode="train").data_params
if a.splits_path:
    cfg["splits_path"] = str(a.splits_path)
samples = collect_sen1_splits(cfg)[a.split]
windows = Sen1Floods11Dataset(samples, cfg).windows
cloud_root = Path(samples[0].label).parents[1] / "S2CloudlessMask"


def local_cv(lin, size=7):
    m = uniform_filter(lin, size)
    v = np.maximum(uniform_filter(lin * lin, size) - m * m, 0)
    return np.sqrt(v) / np.maximum(m, 1e-6)


def cloud_share(cp, m):
    """Fraction of pixels with cloud probability >= 50 among pixels in m that have a cloud probability."""
    m = m & np.isfinite(cp)
    return float((cp[m] >= 50).mean()) if m.any() else np.nan


rows = []
for i, (s, w) in enumerate(windows):
    with rasterio.open(s.sar) as r:
        sar = r.read([1, 2], window=w).astype(np.float32)
    with rasterio.open(s.optical) as r:
        opt = r.read(window=w).astype(np.float32)
    with rasterio.open(s.label) as r:
        lab = r.read(1, window=w)
    cp_path = cloud_root / f"{s.sample_id}_cloudprob.tif"
    cp = rasterio.open(cp_path).read(1, window=w).astype(np.float32) if cp_path.exists() else np.full(lab.shape, np.nan)
    cp = cp / 2.55  # uint8 0..255 -> percent
    sar_ok = np.isfinite(sar).all(0)
    opt_ok = np.isfinite(opt).all(0) & (opt.sum(0) > 0)
    ev = sar_ok & opt_ok & (lab != -1)
    vv, vh = sar[0], sar[1]
    cv = local_cv(np.where(sar_ok, 10 ** (vv / 10), 0))
    rows.append(dict(
        window=i, sample_id=s.sample_id, row=int(w.row_off), col=int(w.col_off),
        cloud_prob_mean=float(np.nanmean(cp)) if np.isfinite(cp).any() else np.nan,
        cloud_frac=cloud_share(cp, np.ones_like(ev)), cloud_frac_eval=cloud_share(cp, ev),
        cloud_frac_invalid=cloud_share(cp, lab == -1), cloud_nodata_frac=float(np.isnan(cp).mean()),
        label_invalid_frac=float((lab == -1).mean()), eval_frac=float(ev.mean()),
        water_frac=float((lab[ev] == 1).mean()) if ev.any() else np.nan,
        vv_db_mean=float(vv[ev].mean()) if ev.any() else np.nan, vh_db_mean=float(vh[ev].mean()) if ev.any() else np.nan,
        vh_floor_frac=float((vh[ev] < -25).mean()) if ev.any() else np.nan,
        vv_cv_median=float(np.median(cv[ev])) if ev.any() else np.nan,
        sar_nan_frac=float((~sar_ok).mean()),
        opt_bright_frac=float((opt[1][opt_ok] / 10000 > 0.25).mean()) if opt_ok.any() else np.nan,
    ))
pd.DataFrame(rows).to_csv(a.out, index=False)
print(f"{len(rows)} windows -> {a.out}")
