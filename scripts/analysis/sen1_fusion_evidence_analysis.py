"""Pixel-level mechanism of DEUF late fusion on Sen1Floods11: which modality decides the fused label, and is it right?

For each late-fusion pair (same model loading / windows / mask as eval_sen1_fusion_compare.py), per evaluated pixel:
  strength S_m = e_m,land + e_m,water + 2, prediction p_m >= 0.5, fused prediction from e_sar + e_opt.
Pixels are split by condition: cloud (s2cloudless prob >= 50) / clear, and SAR dark (VH < -25 dB) / not dark.
Per pair x condition it reports: share of pixels where SAR and optical disagree, in those pixels the share decided by
SAR (fused = SAR label), the accuracy of SAR / optical / fused there, and the median evidence strength of each
modality when it is right and when it is wrong (an overconfident wrong modality wins the evidence sum).
Usage: python sen1_fusion_evidence_analysis.py --pair NAME SAR_CFG SAR_CKPT OPT_CFG OPT_CKPT [...] --out DIR [--splits-path DIR]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader

from flood_uncertainty.data.sen1floods11 import Sen1Floods11Dataset, collect_sen1_splits

sys.path.insert(0, str(ROOT / "scripts/analysis"))
from eval_sen1_fusion_compare import load_model, resolve_ckpt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--pair", nargs=5, action="append", required=True, metavar=("NAME", "SAR_CFG", "SAR_CKPT", "OPT_CFG", "OPT_CKPT"))
p.add_argument("--out", type=Path, required=True)
p.add_argument("--splits-path", type=Path, default=None)
p.add_argument("--split", default="test")
a = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COND = ["cloud", "clear", "sar_dark", "sar_not_dark", "all"]
rows, per_window = [], []
for name, sc, sk, oc, ok in a.pair:
    (scfg, smodel), (ocfg, omodel) = (load_model(c, resolve_ckpt(k), device, a.splits_path) for c, k in [(sc, sk), (oc, ok)])
    samples = collect_sen1_splits(scfg.data_params)[a.split]
    ds_s, ds_o = Sen1Floods11Dataset(samples, scfg.data_params), Sen1Floods11Dataset(samples, ocfg.data_params)
    assert ds_s.windows == ds_o.windows
    acc = {c: dict(n=0, dis=0, sar_dec=0, sar_ok=0, opt_ok=0, fus_ok=0, S_sar_ok=[], S_sar_bad=[], S_opt_ok=[], S_opt_bad=[]) for c in COND}
    k = 0
    with torch.inference_mode():
        for bs, bo in zip(DataLoader(ds_s, batch_size=4), DataLoader(ds_o, batch_size=4)):
            es = torch.relu(smodel.network(bs["image"].to(device))).cpu().numpy()
            eo = torch.relu(omodel.network(bo["image"].to(device))).cpu().numpy()
            ys, yo = bs["mask"][:, 0].numpy(), bo["mask"][:, 1].numpy()
            for j in range(es.shape[0]):
                sample, w = ds_s.windows[k]
                cp_path = Path(sample.label).parents[1] / "S2CloudlessMask" / f"{sample.sample_id}_cloudprob.tif"
                cp = rasterio.open(cp_path).read(1, window=w).astype(np.float32) / 2.55  # uint8 0..255 -> percent
                with rasterio.open(sample.sar) as r:
                    vh = r.read(2, window=w)
                v = (ys[j] != 0) & (yo[j] != 0)
                gt = ys[j] == 2
                ps, po = es[j, 1] > es[j, 0], eo[j, 1] > eo[j, 0]  # alpha_water/S >= 0.5 <=> e_water >= e_land (ties: land)
                ef = es[j] + eo[j]
                pf = ef[1] > ef[0]
                Ss, So = es[j].sum(0) + 2, eo[j].sum(0) + 2
                masks = {"cloud": v & (cp >= 50), "clear": v & (cp < 50), "sar_dark": v & (np.nan_to_num(vh, nan=0) < -25),
                         "sar_not_dark": v & ~(np.nan_to_num(vh, nan=0) < -25), "all": v}
                for c, m in masks.items():
                    d = m & (ps != po)
                    A = acc[c]
                    A["n"] += int(m.sum()); A["dis"] += int(d.sum()); A["sar_dec"] += int((d & (pf == ps)).sum())
                    A["sar_ok"] += int((d & (ps == gt)).sum()); A["opt_ok"] += int((d & (po == gt)).sum()); A["fus_ok"] += int((d & (pf == gt)).sum())
                    rng = np.random.default_rng(k)
                    for key, sel, S in [("S_sar_ok", m & (ps == gt), Ss), ("S_sar_bad", m & (ps != gt), Ss),
                                        ("S_opt_ok", m & (po == gt), So), ("S_opt_bad", m & (po != gt), So)]:
                        vals = S[sel]
                        A[key].append(rng.choice(vals, min(len(vals), 2000), replace=False) if len(vals) else vals)
                m = v & (ps != po)
                per_window.append(dict(pair=name, window=k, sample_id=sample.sample_id, eval_px=int(v.sum()), disagree=int(m.sum()),
                                       sar_decides=int((m & (pf == ps)).sum()), fused_ok_in_disagree=int((m & (pf == gt)).sum()),
                                       opt_ok_in_disagree=int((m & (po == gt)).sum())))
                k += 1
    for c, A in acc.items():
        med = {key: float(np.median(np.concatenate(A[key]))) if sum(map(len, A[key])) else None for key in ["S_sar_ok", "S_sar_bad", "S_opt_ok", "S_opt_bad"]}
        dis = max(A["dis"], 1)
        rows.append(dict(pair=name, condition=c, pixels=A["n"], disagree_share=A["dis"] / max(A["n"], 1),
                         sar_decides=A["sar_dec"] / dis, sar_right=A["sar_ok"] / dis, opt_right=A["opt_ok"] / dis,
                         fused_right=A["fus_ok"] / dis, **{f"median_{k}": v for k, v in med.items()}))
    print(f"{name}: {k} windows", flush=True)

a.out.mkdir(parents=True, exist_ok=True)
t = pd.DataFrame(rows)
t.to_csv(a.out / "fusion_evidence.csv", index=False)
pd.DataFrame(per_window).to_csv(a.out / "fusion_evidence_windows.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(t.round(4).to_string(index=False))
