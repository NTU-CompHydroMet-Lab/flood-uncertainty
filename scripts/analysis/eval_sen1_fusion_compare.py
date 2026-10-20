"""Compare DEUF late fusion and a multi-modal encoder on the same Sen1Floods11 windows and pixels.

A *system* is either
  * a late-fusion pair ``--pair NAME SAR_CFG SAR_CKPT OPT_CFG OPT_CKPT``: reported as NAME/sar, NAME/optical and
    NAME/fused (DEUF: e_f = e_sar + e_opt, alpha = e_f + 1, see ``flood_uncertainty.fusion``), or
  * a single model ``--single NAME CFG CKPT`` (e.g. the S2 + S1 TerraMind; reported as NAME).
A CKPT can be a run folder: then the checkpoint of the epoch with the lowest ``val_bce_land_water`` in
``lightning_logs/version_*/metrics.csv`` is used (rule fixed before the test).

Every model reads its own loader (its config's data_params) for the same windows (checked). All entries use one
pixel mask: SAR valid, optical valid and GT valid. Evidence = relu(network(x)), threshold 0.5 on alpha_water / S.
Outputs in --out:
  windows.csv   per window and entry: tp, fp, fn, tn
  summary.json  pooled water IoU, precision, recall, retention AUC (DST uncertainty, water IoU and accuracy)
  pairs.csv     paired bootstrap (resample the source images, --boot times) of the water IoU difference
                for every pair of entries: diff = row - col, 95% percentile CI
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from flood_uncertainty.data.sen1floods11 import Sen1Floods11Dataset, collect_sen1_splits
from flood_uncertainty.data.sen1floods11_paired import PairedSen1Floods11Dataset, split_configs
from flood_uncertainty.fusion import dirichlet_binary_stats, fuse_evidence_sum
from flood_uncertainty.utils.config_loader import load_mode_config

NB = 1000


def resolve_ckpt(path):
    if os.path.isfile(path):
        return path
    logs = sorted(glob.glob(f"{path}/lightning_logs/version_*/metrics.csv"))
    if not logs:
        raise FileNotFoundError(f"No checkpoint file and no metrics.csv under {path}")
    m = pd.read_csv(logs[-1]).dropna(subset=["val_bce_land_water"])
    epoch = int(m.loc[m.val_bce_land_water.idxmin(), "epoch"])
    found = glob.glob(f"{path}/checkpoint/epoch={epoch}-*.ckpt")
    if len(found) != 1:
        raise FileNotFoundError(f"Best epoch {epoch} checkpoint not unique / missing under {path}: {found}")
    return found[0]


def load_model(cfg_path, ckpt, device, splits_path=None):
    cfg = load_mode_config(cfg_path, mode="train")
    if splits_path:
        cfg.data_params["splits_path"] = str(splits_path)
    hp, dp = cfg.model_params.hyperparameters, cfg.data_params
    modality = dp.get("modality")
    if hp.get("model_type") == "edl_terramind":
        from flood_uncertainty.models.edl_terramind import EDL_TerraMind_ML4FloodsModel, EDL_TerraMind_SAR
        hp["backbone_pretrained"] = False  # all weights come from the checkpoint
        cls = EDL_TerraMind_SAR if modality == "sar" else EDL_TerraMind_ML4FloodsModel
        model = cls(cfg.model_params, normalized_data=modality != "sar")
    elif modality == "sar":
        from flood_uncertainty.inference.infer_sar import load_sar_model
        return cfg, load_sar_model(cfg, ckpt).to(device).eval()  # same path as the teammate's Sen1 evaluation
    else:
        from flood_uncertainty.models.edl import EDL_ML4FloodsModel
        model = EDL_ML4FloodsModel(cfg.model_params, normalized_data=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    return cfg, model.to(device).eval()


def dataset(cfg, samples):
    dp = cfg.data_params
    if dp.modality == "paired":
        o, s = split_configs(dp)
        return PairedSen1Floods11Dataset(samples, samples, o, s)
    return Sen1Floods11Dataset(samples, dp, augment=False)


def water_target(batch, modality):
    m = batch["mask"]
    return (m[:, 0] if modality == "sar" else m[:, 1]).numpy()


def norm_auc(y, x):
    m = ~np.isnan(y) & (x > 0)
    return float(np.trapz(y[m], x[m]) / (x[m].max() - x[m].min())) if m.sum() > 1 and x[m].max() > x[m].min() else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs=5, action="append", default=[], metavar=("NAME", "SAR_CFG", "SAR_CKPT", "OPT_CFG", "OPT_CKPT"))
    p.add_argument("--single", nargs=3, action="append", default=[], metavar=("NAME", "CFG", "CKPT"))
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--splits-path", type=Path, default=None,
                   help="Folder with flood_{train,valid,test}_data.txt, e.g. to test on the Bolivia hold-out")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}  # key -> (cfg, model); key = NAME/sar, NAME/optical or NAME
    for name, sc, sk, oc, ok in a.pair:
        models[f"{name}/sar"] = (sc, resolve_ckpt(sk))
        models[f"{name}/optical"] = (oc, resolve_ckpt(ok))
    for name, c, k in a.single:
        models[name] = (c, resolve_ckpt(k))
    ckpts = {key: k for key, (_, k) in models.items()}
    models = {key: load_model(c, k, device, a.splits_path) for key, (c, k) in models.items()}

    sample_lists = {key: collect_sen1_splits(cfg.data_params if cfg.data_params.modality != "paired"
                                             else split_configs(cfg.data_params)[0])[a.split]
                    for key, (cfg, _) in models.items()}
    ref_ids = [s.sample_id for s in next(iter(sample_lists.values()))]
    if any([s.sample_id for s in v] != ref_ids for v in sample_lists.values()):
        raise ValueError("Split sample IDs / order differ between models")
    datasets = {key: dataset(cfg, sample_lists[key]) for key, (cfg, _) in models.items()}
    ref_windows = [(s.sample_id, w) for s, w in next(iter(datasets.values())).windows]
    if any([(s.sample_id, w) for s, w in d.windows] != ref_windows for d in datasets.values()):
        raise ValueError("Crop windows differ between models")
    loaders = {key: DataLoader(d, batch_size=a.batch_size, shuffle=False, num_workers=4) for key, d in datasets.items()}

    entries = list(models) + [f"{name}/fused" for name, *_ in a.pair]
    hist = {e: np.zeros((NB, 4), np.int64) for e in entries}
    rows, tile = [], 0
    with torch.inference_mode():
        for bi, batches in enumerate(zip(*loaders.values())):
            if a.max_batches is not None and bi >= a.max_batches:
                break
            batches = dict(zip(loaders, batches))
            targets = [water_target(batches[k], models[k][0].data_params.modality) for k in models]
            valid = np.logical_and.reduce([t != 0 for t in targets])
            # SAR / optical masks also mark no-data image pixels invalid; the paired mask already combines them
            for t in targets[1:]:
                if not np.array_equal(t[valid], targets[0][valid]):
                    raise ValueError("GT labels differ between loaders")
            water = targets[0] == 2
            evidence = {k: torch.relu(m.network(batches[k]["image"].to(device))).cpu().numpy() for k, (_, m) in models.items()}
            for j in range(valid.shape[0]):
                v = valid[j]
                ev = {k: e[j] for k, e in evidence.items()}  # class-first (2, H, W)
                for name, *_ in a.pair:
                    ev[f"{name}/fused"] = fuse_evidence_sum(ev[f"{name}/sar"], ev[f"{name}/optical"])
                for e in entries:
                    o = dirichlet_binary_stats(ev[e])
                    pred = o["probability"] >= 0.5
                    cls = [pred & water[j] & v, pred & ~water[j] & v, ~pred & water[j] & v, ~pred & ~water[j] & v]
                    rows.append((tile, ref_windows[tile][0], e, *[int(c.sum()) for c in cls]))
                    b = np.clip((o["dst_u"] * NB).astype(np.int64), 0, NB - 1)
                    for c in range(4):
                        hist[e][:, c] += np.bincount(b[cls[c]], minlength=NB)
                tile += 1
            print(f"{tile}/{len(ref_windows)} windows", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows, columns=["window", "sample_id", "entry", "tp", "fp", "fn", "tn"])
    t.to_csv(a.out / "windows.csv", index=False)
    summary = {"split": a.split, "windows": tile, "source_images": int(t.sample_id.nunique()), "checkpoints": ckpts,
               "mask": "SAR valid & optical valid & GT valid", "fusion": "e_f = e_sar + e_opt; alpha = e_f + 1", "entries": {}}
    for e in entries:
        tp, fp, fn, tn = (int(t[t.entry == e][c].sum()) for c in ["tp", "fp", "fn", "tn"])
        cum = hist[e].cumsum(0)
        ret = cum.sum(1) / cum[-1].sum()
        den = cum[:, 0] + cum[:, 1] + cum[:, 2]
        iou = np.where(den > 0, cum[:, 0] / np.maximum(den, 1), np.nan)
        acc = (cum[:, 0] + cum[:, 3]) / np.maximum(cum.sum(1), 1)
        summary["entries"][e] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "water_iou": tp / max(tp + fp + fn, 1),
                                 "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                                 "retention_auc_iou_dst": norm_auc(iou, ret), "retention_auc_acc_dst": norm_auc(acc, ret)}

    # paired bootstrap over source images
    ids = sorted(t.sample_id.unique())
    g = t.groupby(["entry", "sample_id"])[["tp", "fp", "fn"]].sum()
    c = np.stack([g.loc[e].reindex(ids).to_numpy() for e in entries])  # (E, N, 3)
    rng = np.random.default_rng(0)
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=a.boot)  # (B, N)
    s = np.einsum("bn,enk->bek", w, c)
    boot_iou = s[..., 0] / np.maximum(s.sum(-1), 1)  # (B, E)
    full = np.array([summary["entries"][e]["water_iou"] for e in entries])
    with (a.out / "pairs.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["row", "col", "diff", "ci_low", "ci_high"])
        for i, ei in enumerate(entries):
            for k, ek in enumerate(entries):
                if i != k:
                    d = boot_iou[:, i] - boot_iou[:, k]
                    wr.writerow([ei, ek, f"{full[i] - full[k]:+.4f}", f"{np.percentile(d, 2.5):+.4f}", f"{np.percentile(d, 97.5):+.4f}"])
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({e: round(v["water_iou"], 4) for e, v in summary["entries"].items()}, indent=2))


if __name__ == "__main__":
    main()
