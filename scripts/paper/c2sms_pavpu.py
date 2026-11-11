"""PAvPU on C2S-MS Floods with the lab definition of scripts/analysis/compute_pavpu.py, from the float evidence.

Definition (compute_pavpu.py): non-overlapping P x P patches (P = 3); a patch with any not-scored pixel or with no flood
pixel in the reference and the prediction (all TN, IoU undefined) is dropped. A patch is accurate when its flood-class
IoU >= the model's pooled IoU at the retention target, and uncertain when its mean u >= the model's u threshold at the
retention target (the (1 - target) most uncertain pixels). PAvPU = (n_ac + n_iu) / n, pooled over all chips; also
p(accurate | certain) = n_ac / (n_ac + n_ic) and p(uncertain | inaccurate) = n_iu / (n_ic + n_iu).
u = K / S (K = 2) from the float evidence of the maps (eval_sen1_fusion_compare.py --save-evidence). Subsets: all
patches, and cloud patches (at least half of the patch under the C2S-MS cloud / shadow label).
Writes <out>/c2sms_pavpu.csv (one row per entry, subset and retention target).
Usage: python c2sms_pavpu.py --maps <sen1_maps/c2sms> --layout <c2sms_all_sen1layout> --out <dir> [--targets 0.8 0.9 0.95]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

p = argparse.ArgumentParser()
p.add_argument("--maps", type=Path, required=True)
p.add_argument("--layout", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--patch", type=int, default=3)
p.add_argument("--targets", type=float, nargs="+", default=[0.8, 0.9, 0.95])
a = p.parse_args()

ENTRIES = ["multimodal", "bef/optical", "bef/sar", "bef/fused"]
EDGES = np.concatenate([[0.0], np.geomspace(1e-6, 1.0, 4000)])
ids = [x.strip() for x in (a.layout / "splits" / "flood_test_data.txt").read_text().split() if x.strip()]


def load(sid):
    z = np.load(a.maps / f"{sid}.npz")
    ent, ev_ent = list(z["entries"]), list(z["evidence_entries"])
    out = {}
    for e in ENTRIES:
        ev = z["evidence"][ev_ent.index(e)].astype(np.float32)
        out[e] = (z["cm"][ent.index(e)], 2.0 / (ev[0] + ev[1] + 2.0))
    with rasterio.open(a.layout / "extra" / f"{sid}_CloudLabel.tif") as r:
        cloud = np.isin(r.read(1), [1, 2])
    return out, cloud


# pass 1: pixel-level u histogram -> per entry and target: u threshold and pooled IoU of the retained pixels
hist = {e: np.zeros((len(EDGES) - 1, 3), np.int64) for e in ENTRIES}  # tp fp fn per u bin (tn not needed for IoU)
npix = {e: np.zeros(len(EDGES) - 1, np.int64) for e in ENTRIES}
for sid in ids:
    maps, _ = load(sid)
    for e, (cm, u) in maps.items():
        b = np.clip(np.searchsorted(EDGES, u, side="right") - 1, 0, len(EDGES) - 2)
        v = cm > 0
        npix[e] += np.bincount(b[v], minlength=len(EDGES) - 1)
        for k, cls in enumerate([4, 2, 3]):
            hist[e][:, k] += np.bincount(b[v & (cm == cls)], minlength=len(EDGES) - 1)
thr = {}
for e in ENTRIES:
    ret = npix[e].cumsum() / npix[e].sum()
    c = hist[e].cumsum(0)
    for t in a.targets:
        i = min(int(np.searchsorted(ret, t - 1e-12)), len(ret) - 1)
        thr[(e, t)] = (EDGES[i + 1], c[i, 0] / c[i].sum(), ret[i])  # u threshold, accuracy threshold, retention

# pass 2: patches
P = a.patch
cnt = {}
for sid in ids:
    maps, cloud = load(sid)
    H, W = cloud.shape
    h, w = H // P * P, W // P * P
    cp = cloud[:h, :w].reshape(h // P, P, w // P, P).mean(axis=(1, 3)) >= 0.5
    for e, (cm, u) in maps.items():
        c = cm[:h, :w].reshape(h // P, P, w // P, P).transpose(0, 2, 1, 3)
        um = u[:h, :w].reshape(h // P, P, w // P, P).mean(axis=(1, 3))
        tp, fp, fn = ((c == k).sum(axis=(2, 3)) for k in (4, 2, 3))
        den = tp + fp + fn
        valid = ~(c == 0).any(axis=(2, 3)) & (den > 0)
        iou = np.where(den > 0, tp / np.maximum(den, 1), 0)
        for t in a.targets:
            ut, at, _ = thr[(e, t)]
            acc, unc = iou >= at, um >= ut
            for s, m in (("all", valid), ("cloud", valid & cp)):
                k = cnt.setdefault((e, s, t), np.zeros(4, np.int64))
                k += [(m & acc & ~unc).sum(), (m & acc & unc).sum(), (m & ~acc & ~unc).sum(), (m & ~acc & unc).sum()]

rows = []
for (e, s, t), (n_ac, n_au, n_ic, n_iu) in sorted(cnt.items()):
    n = n_ac + n_au + n_ic + n_iu
    ut, at, rr = thr[(e, t)]
    rows.append(dict(entry=e, subset=s, retention_target=t, retention=rr, u_threshold=ut, acc_threshold=at,
                     patches=int(n), inaccurate_share=n_ic / n + n_iu / n if n else np.nan,
                     pavpu=(n_ac + n_iu) / n if n else np.nan,
                     p_acc_given_certain=n_ac / (n_ac + n_ic) if n_ac + n_ic else np.nan,
                     p_unc_given_inacc=n_iu / (n_ic + n_iu) if n_ic + n_iu else np.nan,
                     n_ac=int(n_ac), n_au=int(n_au), n_ic=int(n_ic), n_iu=int(n_iu)))
t = pd.DataFrame(rows)
a.out.mkdir(parents=True, exist_ok=True)
t.to_csv(a.out / "c2sms_pavpu.csv", index=False)
pd.set_option("display.width", 250)
print(t.drop(columns=["n_ac", "n_au", "n_ic", "n_iu"]).round(4).to_string())
