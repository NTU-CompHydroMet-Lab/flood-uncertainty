"""Agreement / confidence categories of the two branches of each late-fusion pair (original DEUF sum).

Reads the per-pixel evidence caches of sen1_fusion_rules.py (cache_<pair>_<split>.npz). A branch is "sure" when its
DST uncertainty u = 2 / S <= --sure (default 0.2, i.e. S >= 10). Categories per evaluated pixel:
  agree_both_sure        same label, both sure
  agree_not_both_sure    same label, at most one sure
  dis_sar_sure           different labels, only SAR sure
  dis_opt_sure           different labels, only optical sure
  dis_both_sure          different labels, both sure (strong conflict)
  dis_neither_sure       different labels, neither sure
Per category: pixel share, accuracy of SAR / optical / DEUF fused. Writes <out>/conflict_stats.csv (all pixels) and
<out>/conflict_stats_images.csv (the same per source image; share = share of that image's evaluated pixels).
Usage: python sen1_conflict_stats.py --cache-dir artifacts/results/sen1_fusion_rules --out artifacts/results/sen1_fusion_rules
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATS = ["agree_both_sure", "agree_not_both_sure", "dis_sar_sure", "dis_opt_sure", "dis_both_sure", "dis_neither_sure"]

p = argparse.ArgumentParser()
p.add_argument("--cache-dir", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--sure", type=float, default=0.2)
a = p.parse_args()

rows, img_rows = [], []
for f in sorted(a.cache_dir.glob("cache_*_*.npz")):
    pair, split = f.stem[len("cache_"):].rsplit("_", 1)
    d = np.load(f)
    es, eo, gt = d["es"], d["eo"], d["gt"]
    ss, so = 2 / (es.sum(1) + 2) <= a.sure, 2 / (eo.sum(1) + 2) <= a.sure
    ps, po = es[:, 1] > es[:, 0], eo[:, 1] > eo[:, 0]
    ef = es + eo
    pf = ef[:, 1] > ef[:, 0]
    agree = ps == po
    masks = {
        "agree_both_sure": agree & ss & so, "agree_not_both_sure": agree & ~(ss & so),
        "dis_sar_sure": ~agree & ss & ~so, "dis_opt_sure": ~agree & so & ~ss,
        "dis_both_sure": ~agree & ss & so, "dis_neither_sure": ~agree & ~ss & ~so,
    }
    n = len(gt)
    for c in CATS:
        m = masks[c]
        k = int(m.sum())
        acc = lambda pred: float((pred[m] == gt[m]).mean()) if k else None  # noqa: E731
        rows.append(dict(pair=pair, split=split, category=c, pixels=k, share=k / n, sar_right=acc(ps), opt_right=acc(po),
                         fused_right=acc(pf), gt_water=float(gt[m].mean()) if k else None, sure_threshold=a.sure))
    # per source image
    uniq, img = np.unique(d["sample_ids"][d["window"]], return_inverse=True)
    n_img = np.bincount(img, minlength=len(uniq))
    for c in CATS:
        m = masks[c]
        cnt = np.bincount(img[m], minlength=len(uniq))
        ok = {k: np.bincount(img[m & (pred == gt)], minlength=len(uniq)) for k, pred in [("sar", ps), ("opt", po), ("fused", pf)]}
        wat = np.bincount(img[m & gt], minlength=len(uniq))
        for i, sid in enumerate(uniq):
            k = int(cnt[i])
            r = lambda x: float(x[i] / k) if k else None  # noqa: E731
            img_rows.append(dict(pair=pair, split=split, sample_id=sid, category=c, pixels=k, eval_px=int(n_img[i]),
                                 share=k / n_img[i], sar_right=r(ok["sar"]), opt_right=r(ok["opt"]),
                                 fused_right=r(ok["fused"]), gt_water=r(wat)))
t = pd.DataFrame(rows)
a.out.mkdir(parents=True, exist_ok=True)
t.to_csv(a.out / "conflict_stats.csv", index=False)
pd.DataFrame(img_rows).to_csv(a.out / "conflict_stats_images.csv", index=False)
pd.set_option("display.width", 200)
print(t[t.pair == "bef"].round(4).to_string(index=False))
