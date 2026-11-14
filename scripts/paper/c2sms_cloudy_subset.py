"""Cloudy-chip subset of C2S-MS Floods: pooled metrics and the agreement / confidence categories of the DEUF pair.

The subset is the chips whose cloud-and-shadow fraction is >= --min-cloud (default 0.25, i.e. the 25-100% strata of the
cloud-strata figure; artifacts/results/c2sms/cloud_groups.csv). --ids-csv replaces it by an explicit list (column
sample_id, e.g. the older 41-chip c2sms_cloudy_sen1layout/selected_chips.csv) to check the reproduction.
1) Pooled flood-class IoU, recall and precision per entry from <results>/c2sms_all/windows.csv.
2) Per-pixel categories as in scripts/analysis/sen1_conflict_stats.py, from the saved maps (<maps>/<id>.npz, raw
   evidence [land, water] of bef/optical, bef/sar, bef/fused): a predictor is sure when u = 2 / S <= --sure (0.2);
   label = water when e_water > e_land; fused = sum of the two evidences; reference water = cm in {FN, TP}.
Writes <out>/c2sms_cloudy_metrics.csv and c2sms_cloudy_conflict.csv (pixels, share, accuracy of S1 / S2 / DEUF).
Usage: python c2sms_cloudy_subset.py --results artifacts/results --maps <sen1_maps/c2sms> --out <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--results", type=Path, required=True)
p.add_argument("--maps", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--min-cloud", type=float, default=0.25)
p.add_argument("--ids-csv", type=Path, default=None)
p.add_argument("--sure", type=float, default=0.2)
p.add_argument("--tag", default="")
a = p.parse_args()

cg = pd.read_csv(a.results / "c2sms" / "cloud_groups.csv")
if a.ids_csv:
    s = pd.read_csv(a.ids_csv, dtype={"chip": str})
    ids = sorted(s.sample_id if "sample_id" in s else s.event.str[:8] + "_" + s.chip)
else:
    ids = sorted(cg[cg.cloud_shadow >= a.min_cloud].sample_id)
print("chips:", len(ids))

w = pd.read_csv(a.results / "c2sms_all" / "windows.csv")
w = w[w.sample_id.isin(ids)]
m = w.groupby("entry")[["tp", "fp", "fn", "tn"]].sum()
m["iou"] = m.tp / (m.tp + m.fp + m.fn)
m["recall"] = m.tp / (m.tp + m.fn)
m["precision"] = m.tp / (m.tp + m.fp)
m["chips"] = w.sample_id.nunique()

CATS = ["agree_both_sure", "agree_not_both_sure", "dis_sar_sure", "dis_opt_sure", "dis_both_sure", "dis_neither_sure"]
cnt = {c: np.zeros(4, np.int64) for c in CATS}  # pixels, SAR correct, optical correct, fused correct
for sid in ids:
    z = np.load(a.maps / f"{sid}.npz")
    ent, ev_ent = list(z["entries"]), list(z["evidence_entries"])
    cm = z["cm"][ent.index("bef/fused")]
    v = cm > 0
    gt = np.isin(cm, [3, 4])[v]
    es = z["evidence"][ev_ent.index("bef/sar")].astype(np.float32)[:, v]
    eo = z["evidence"][ev_ent.index("bef/optical")].astype(np.float32)[:, v]
    ss, so = 2 / (es.sum(0) + 2) <= a.sure, 2 / (eo.sum(0) + 2) <= a.sure
    ps, po = es[1] > es[0], eo[1] > eo[0]
    pf = (es[1] + eo[1]) > (es[0] + eo[0])
    agree = ps == po
    masks = {"agree_both_sure": agree & ss & so, "agree_not_both_sure": agree & ~(ss & so),
             "dis_sar_sure": ~agree & ss & ~so, "dis_opt_sure": ~agree & so & ~ss,
             "dis_both_sure": ~agree & ss & so, "dis_neither_sure": ~agree & ~ss & ~so}
    for c, mk in masks.items():
        cnt[c] += [mk.sum(), (mk & (ps == gt)).sum(), (mk & (po == gt)).sum(), (mk & (pf == gt)).sum()]
tot = sum(v[0] for v in cnt.values())
c = pd.DataFrame([dict(category=k, pixels=int(v[0]), share=v[0] / tot, sar_right=v[1] / max(v[0], 1),
                       opt_right=v[2] / max(v[0], 1), fused_right=v[3] / max(v[0], 1)) for k, v in cnt.items()])
a.out.mkdir(parents=True, exist_ok=True)
m.to_csv(a.out / f"c2sms_cloudy_metrics{a.tag}.csv")
c.to_csv(a.out / f"c2sms_cloudy_conflict{a.tag}.csv", index=False)
pd.set_option("display.width", 200)
print(m.round(4).to_string())
print(c.round(4).to_string())
