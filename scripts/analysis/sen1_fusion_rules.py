"""Alternative late-fusion rules on frozen SAR / optical EDL branches (Sen1Floods11).

Stage ``dump``: run each pair once per split and cache, for every evaluated pixel (SAR valid & optical valid & GT valid),
the raw evidence of both branches, the GT and the s2cloudless probability (uint8 0..255 -> 0..1).
Stage ``eval``: apply the fusion rules, pick each rule's parameters on ``val`` by binary cross-entropy of
p_water = alpha_water / S (the project's selection metric), then report test / Bolivia.

Rules (e = [land, water] evidence, prediction = water if fused water belief > land belief):
  optical, sar, sum            single branches; DEUF e_opt + e_sar
  wsum(w, c)                   c * (e_opt + w * e_sar); w sets the prediction, c only the calibration
  cloud(g)                     (1 - p_cloud)^g * e_opt + e_sar
  cloud_wsum(g, w, c)          c * ((1 - p_cloud)^g * e_opt + w * e_sar)
  ds                           Dempster combination of the two binomial opinions (b = e / S, u = 2 / S)
  ds_conflict(t)               ds, but where the conflict K > t keep the opinion of the branch with the lower u
  oracle                       per pixel the branch that is right (upper bound, not a method)
Outputs in --out: rules_<split>.csv (all configs), selected.csv (val-selected configs on every split),
pairs_<split>.csv (bootstrap over source images: selected rule - reference).
Usage:
  python sen1_fusion_rules.py dump --pair NAME SAR_CFG SAR_CKPT OPT_CFG OPT_CKPT [...] --split val --out DIR [--splits-path P --tag bolivia]
  python sen1_fusion_rules.py eval --out DIR --compare-dirs test=DIR1 bolivia=DIR2
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))

import numpy as np
import pandas as pd

NB = 1000


def dump(a):
    import rasterio
    import torch
    from torch.utils.data import DataLoader

    from eval_sen1_fusion_compare import load_model, resolve_ckpt
    from flood_uncertainty.data.sen1floods11 import Sen1Floods11Dataset, collect_sen1_splits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or a.split
    for name, sc, sk, oc, ok in a.pair:
        (scfg, sm), (ocfg, om) = (load_model(c, resolve_ckpt(k), device, a.splits_path) for c, k in [(sc, sk), (oc, ok)])
        samples = collect_sen1_splits(scfg.data_params)[a.split]
        ds_s, ds_o = Sen1Floods11Dataset(samples, scfg.data_params), Sen1Floods11Dataset(samples, ocfg.data_params)
        assert ds_s.windows == ds_o.windows
        out = {k: [] for k in ["window", "gt", "es", "eo", "cp"]}
        ids, k = [], 0
        with torch.inference_mode():
            for bs, bo in zip(DataLoader(ds_s, batch_size=4), DataLoader(ds_o, batch_size=4)):
                es = torch.relu(sm.network(bs["image"].to(device))).cpu().numpy()
                eo = torch.relu(om.network(bo["image"].to(device))).cpu().numpy()
                ys, yo = bs["mask"][:, 0].numpy(), bo["mask"][:, 1].numpy()
                for j in range(es.shape[0]):
                    sample, w = ds_s.windows[k]
                    cp = rasterio.open(Path(sample.label).parents[1] / "S2CloudlessMask" / f"{sample.sample_id}_cloudprob.tif").read(1, window=w)
                    v = (ys[j] != 0) & (yo[j] != 0)
                    out["window"].append(np.full(int(v.sum()), k, np.int32)); out["gt"].append((ys[j][v] == 2))
                    out["es"].append(es[j][:, v].T.astype(np.float32)); out["eo"].append(eo[j][:, v].T.astype(np.float32))
                    out["cp"].append(cp[v].astype(np.uint8)); ids.append(sample.sample_id)
                    k += 1
        np.savez(a.out / f"cache_{name}_{tag}.npz", sample_ids=np.array(ids), **{x: np.concatenate(v) for x, v in out.items()})
        print(f"{name} {tag}: {k} windows, {sum(map(len, out['gt']))} pixels", flush=True)


def opinion(e):
    S = e.sum(1) + 2
    return e[:, 0] / S, e[:, 1] / S, 2 / S  # b_land, b_water, u


def ds_combine(e1, e2):
    l1, w1, u1 = opinion(e1)
    l2, w2, u2 = opinion(e2)
    K = w1 * l2 + l1 * w2
    n = 1 - K
    return (l1 * l2 + l1 * u2 + l2 * u1) / n, (w1 * w2 + w1 * u2 + w2 * u1) / n, u1 * u2 / n, K


def from_evidence(e):
    S = e.sum(1) + 2
    return (e[:, 1] + 1) / S, 2 / S  # p_water, u


def from_opinion(bl, bw, u):
    return bw + u / 2, u  # projected probability with base rate 1/2


def scores(gt, p, u, pred, win, n_win):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    bce = float(-np.mean(np.where(gt, np.log(pc), np.log(1 - pc))))
    brier = float(np.mean((p - gt) ** 2))
    cls = [pred & gt, pred & ~gt, ~pred & gt]
    tp, fp, fn = (int(c.sum()) for c in cls)
    # retention (low uncertainty first), water IoU and accuracy of retained pixels
    b = np.clip((u * NB).astype(np.int64), 0, NB - 1)
    h = np.stack([np.bincount(b[c], minlength=NB) for c in cls + [~pred & ~gt]], 1).cumsum(0)
    ret = h.sum(1) / h[-1].sum()
    den = h[:, :3].sum(1)
    iou_r = np.where(den > 0, h[:, 0] / np.maximum(den, 1), np.nan)
    m = ~np.isnan(iou_r)
    auc = float(np.trapz(iou_r[m], ret[m]) / (ret[m].max() - ret[m].min())) if m.sum() > 1 else None
    per_win = np.stack([np.bincount(win[c], minlength=n_win) for c in cls], 1)
    return dict(water_iou=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                bce=bce, brier=brier, retention_auc_iou=auc), per_win


def rules(d):
    """Yield (rule, params, p_water, u, pred) for every config."""
    es, eo, gt, pc = d["es"], d["eo"], d["gt"], d["cp"] / 255.0
    yield "optical", {}, *from_evidence(eo), eo[:, 1] > eo[:, 0]
    yield "sar", {}, *from_evidence(es), es[:, 1] > es[:, 0]
    ef = eo + es
    yield "sum", {}, *from_evidence(ef), ef[:, 1] > ef[:, 0]
    W = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    C = [0.25, 0.5, 1.0, 2.0, 4.0]
    G = [0.5, 1.0, 2.0, 4.0]
    for w, c in itertools.product(W, C):
        ef = c * (eo + w * es)
        yield "wsum", dict(w=w, c=c), *from_evidence(ef), ef[:, 1] > ef[:, 0]
    for g in G:
        ef = ((1 - pc) ** g)[:, None] * eo + es
        yield "cloud", dict(g=g), *from_evidence(ef), ef[:, 1] > ef[:, 0]
    for g, w, c in itertools.product(G, W, C):
        ef = c * (((1 - pc) ** g)[:, None] * eo + w * es)
        yield "cloud_wsum", dict(g=g, w=w, c=c), *from_evidence(ef), ef[:, 1] > ef[:, 0]
    bl, bw, u, K = ds_combine(eo, es)
    yield "ds", {}, *from_opinion(bl, bw, u), bw > bl
    lo, wo, uo = opinion(eo)
    ls, ws, us = opinion(es)
    for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        use_o, use_s = (K > t) & (uo <= us), (K > t) & (uo > us)
        BL = np.where(use_o, lo, np.where(use_s, ls, bl))
        BW = np.where(use_o, wo, np.where(use_s, ws, bw))
        U = np.where(use_o, uo, np.where(use_s, us, u))
        yield "ds_conflict", dict(t=t), *from_opinion(BL, BW, U), BW > BL
    po, ps = eo[:, 1] > eo[:, 0], es[:, 1] > es[:, 0]
    pred = np.where(po == gt, po, ps)
    yield "oracle", {}, pred.astype(float), np.zeros(len(gt)), pred


def evaluate(a):
    caches = sorted(a.out.glob("cache_*_*.npz"))
    pairs = sorted({c.stem.split("_", 1)[1].rsplit("_", 1)[0] for c in caches})
    compare = dict(x.split("=", 1) for x in a.compare_dirs)
    rng = np.random.default_rng(0)
    sel_rows = []
    for pair in pairs:
        tables, perwin = {}, {}
        tags = [c.stem.rsplit("_", 1)[1] for c in caches if c.stem.split("_", 1)[1].rsplit("_", 1)[0] == pair]
        for tag in tags:
            d = dict(np.load(a.out / f"cache_{pair}_{tag}.npz"))
            n_win = len(d["sample_ids"])
            rows = []
            for rule, prm, p, u, pred in rules(d):
                s, pw = scores(d["gt"], p, u, pred, d["window"], n_win)
                key = (rule, json.dumps(prm, sort_keys=True))
                rows.append(dict(pair=pair, split=tag, rule=rule, params=key[1], **s))
                perwin[(tag, key)] = pw
            tables[tag] = pd.DataFrame(rows)
            tables[tag].to_csv(a.out / f"rules_{pair}_{tag}.csv", index=False)
            perwin[(tag, "ids")] = d["sample_ids"]
        if "val" not in tables:
            raise ValueError(f"{pair}: no val cache")
        val = tables["val"]
        chosen = {"optical": "{}", "sar": "{}", "sum": "{}", "ds": "{}", "oracle": "{}"}
        for rule in ["wsum", "cloud", "cloud_wsum", "ds_conflict"]:
            r = val[val.rule == rule]
            chosen[rule] = r.loc[r.bce.idxmin(), "params"]
        for tag, t in tables.items():
            for rule, prm in chosen.items():
                row = t[(t.rule == rule) & (t.params == prm)].iloc[0].to_dict()
                sel_rows.append(row)
            if tag == "val":
                continue
            # bootstrap over source images: selected rule - reference
            ids = perwin[(tag, "ids")]
            uniq, inv = np.unique(ids, return_inverse=True)
            def per_img(pw):
                return np.stack([np.bincount(inv, weights=pw[:, i], minlength=len(uniq)) for i in range(3)], 1)
            refs = {r: per_img(perwin[(tag, (r, chosen[r]))]) for r in ["optical", "sum"]}
            if tag in compare:
                wc = pd.read_csv(Path(compare[tag]) / "windows.csv")
                mm = wc[wc.entry == "multimodal"].sort_values("window")[["tp", "fp", "fn"]].to_numpy()
                assert len(mm) == len(ids)
                refs["multimodal"] = per_img(mm)
            wts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=a.boot)
            def boot_iou(c):
                s = wts @ c
                return s[:, 0] / np.maximum(s.sum(1), 1)
            out = []
            for rule, prm in chosen.items():
                x = per_img(perwin[(tag, (rule, prm))])
                bx, fx = boot_iou(x), x[:, 0].sum() / max(x.sum(), 1)
                for rname, r in refs.items():
                    dlt = bx - boot_iou(r)
                    out.append(dict(pair=pair, split=tag, rule=rule, params=prm, ref=rname,
                                    diff=fx - r[:, 0].sum() / max(r.sum(), 1),
                                    ci_low=np.percentile(dlt, 2.5), ci_high=np.percentile(dlt, 97.5)))
            pd.DataFrame(out).to_csv(a.out / f"pairs_{pair}_{tag}.csv", index=False)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(a.out / "selected.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 40)
    print(sel.round(4).to_string(index=False))


p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
pd_ = sub.add_parser("dump")
pd_.add_argument("--pair", nargs=5, action="append", required=True)
pd_.add_argument("--split", default="val")
pd_.add_argument("--splits-path", type=Path, default=None)
pd_.add_argument("--tag", default=None)
pd_.add_argument("--out", type=Path, required=True)
pe = sub.add_parser("eval")
pe.add_argument("--out", type=Path, required=True)
pe.add_argument("--compare-dirs", nargs="*", default=[])
pe.add_argument("--boot", type=int, default=10000)
a = p.parse_args()
dump(a) if a.cmd == "dump" else evaluate(a)
