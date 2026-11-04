"""Run a WorldFloods-trained optical EDL model (default: unet_baseline) on C2S-MS Floods and score it.

The model is trained on WorldFloods only (no Sen1Floods11 / C2S-MS fine-tune). Inference uses the same path as
scripts/inference/run_inference.py (flood_uncertainty.inference.infer: load_model, load_inference_function,
predict; WorldFloods normalisation of the raw 13-band L1C input). Scoring uses the same rule as
scripts/eval/eval_metrics.py: water = Water_Probability > threshold, pixels with a valid GT are scored.

Input: a C2S-MS Floods folder in the Sen1Floods11 layout (scripts/analysis/c2sms_to_sen1_layout.py):
  data/S2L1CHand/<id>_S2Hand.tif   13 L1C bands B1..B12 (uint16, 0 = no data)
  data/LabelHand/<id>_LabelHand.tif   label of --label cloudaware (0 land, 1 water, -1 no label)
  extra/<id>_{S1,S2,Cloud}Label.tif   original C2S-MS labels (for --label s1 / s2 / both and the cloud strata)
  splits/flood_test_data.txt          sample ids
Mask: GT valid and S2 valid (a pixel where all 13 bands are 0 is S2 no data and is not scored).
Cloud strata: C2S-MS LabelCloud (1 cloud, 2 cloud shadow) -> "covered" (cloud or shadow) and "clear" pixels.

Outputs in --out-dir:
  pred/<id>_output_<model_type>.tif   with --save-tif: the 7-band EDL output of run_inference.py (classification
                                      0 invalid / 1 land / 2 water / 3 cloud, DST u, water probability, evidence,
                                      aleatoric, epistemic)
  chips.csv     per chip: tp, fp, fn, tn (all / clear / covered pixels), water IoU, mean DST u (clear / covered)
  summary.json  pooled metrics (water IoU, precision, recall, F1, accuracy; also for clear and covered pixels),
                average per-chip IoU, mean DST u, the run settings

Usage:
  python scripts/inference/run_inference_c2sms.py --data-root /home/NAS/house/ycchen-10014/data/c2sms_all_sen1layout \
      --out-dir artifacts/results/c2sms_unet_baseline --save-tif
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from tqdm import tqdm  # noqa: E402

from flood_uncertainty.inference.infer import (  # noqa: E402
    load_inference_function,
    load_model,
    predict,
    save_prediction_tif,
)

# unet_baseline (docs/edl_vs_terramind_comparison.md): configurations/edl.json, edl_EpochKL_20260819 epoch 8
DEFAULT_CONFIG = PROJECT_ROOT / "configurations" / "edl.json"
DEFAULT_CHECKPOINT = (
    "/home/NAS/house/ycchen-10014/repo/flood-uncertainty/artifacts/models/"
    "edl_EpochKL_20260819/checkpoint/epoch=8-step=9225.ckpt"
)
DEFAULT_DATA_ROOT = "/home/NAS/house/ycchen-10014/data/c2sms_all_sen1layout"


def read_band(path):
    with rasterio.open(path) as r:
        return r.read(1)


def read_label(root, sid, kind):
    """GT as {0 invalid, 1 land, 2 water} (the WorldFloods water band encoding)."""
    if kind == "cloudaware":
        lab = read_band(root / "data" / "LabelHand" / f"{sid}_LabelHand.tif").astype(np.int16)
        valid, water = lab != -1, lab == 1
    else:
        l1 = read_band(root / "extra" / f"{sid}_S1Label.tif")
        l2 = read_band(root / "extra" / f"{sid}_S2Label.tif")
        if kind == "s1":
            valid, water = l1 != 255, l1 == 1
        elif kind == "s2":
            valid, water = l2 != 255, l2 == 1
        else:  # both: only where the two labels agree
            valid, water = (l1 != 255) & (l2 != 255) & (l1 == l2), l1 == 1
    gt = np.zeros(valid.shape, np.uint8)
    gt[valid] = 1
    gt[valid & water] = 2
    return gt


def confusion(pred_water, gt_water, m):
    return [int((pred_water & gt_water & m).sum()), int((pred_water & ~gt_water & m).sum()),
            int((~pred_water & gt_water & m).sum()), int((~pred_water & ~gt_water & m).sum())]


def metrics(tp, fp, fn, tn):
    r = lambda x, y: x / y if y else None  # noqa: E731
    p, rc = r(tp, tp + fp), r(tp, tp + fn)
    return dict(pixels=tp + fp + fn + tn, water_iou=r(tp, tp + fp + fn), precision=p, recall=rc,
                f1=r(2 * tp, 2 * tp + fp + fn), accuracy=r(tp + tn, tp + fp + fn + tn),
                land_iou=r(tn, tn + fp + fn))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    p.add_argument("--split-file", type=Path, default=None, help="default: <data-root>/splits/flood_test_data.txt")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-type", default="EDL", choices=["EDL", "EDL-TERRAMIND"])
    p.add_argument("--max-tile-size", type=int, default=256, help="256 = the tile of the docs' WorldFloods results")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--label", choices=["cloudaware", "s1", "s2", "both"], default="cloudaware")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--save-tif", action="store_true", help="write the 7-band prediction GeoTIFF per chip")
    p.add_argument("--max-files", type=int, default=None, help="smoke test: only the first N chips")
    a = p.parse_args()

    split = a.split_file or a.data_root / "splits" / "flood_test_data.txt"
    ids = [x.strip() for x in split.read_text().splitlines() if x.strip()][: a.max_files]
    pred_dir = a.out_dir / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model, channels, config = load_model(str(a.config), a.model_type, collection_name="S2",
                                         weights_path=a.checkpoint)
    inference_fn, _ = load_inference_function(model, config, max_tile_size=a.max_tile_size,
                                              th_water=a.threshold, apply_normalization=True,
                                              used_EDL=True, collection_name="S2")
    water_index = model.water_task_index

    rows = []
    for sid in tqdm(ids, desc="c2sms"):
        s2_path = a.data_root / "data" / "S2L1CHand" / f"{sid}_S2Hand.tif"
        with rasterio.open(s2_path) as src:
            if src.count != 13:
                raise ValueError(f"Expected 13 L1C bands: {s2_path}")
            image = src.read().astype(np.float32)
            transform, crs = src.transform, src.crs
        gt = read_label(a.data_root, sid, a.label)
        if gt.shape != image.shape[1:]:
            raise ValueError(f"Label / image shape differ: {sid}")

        land_water_cloud, prob, dst_u, evidence, aleatoric, epistemic = predict(image, channels, inference_fn)
        p_water = prob[water_index].cpu().numpy()
        u = dst_u[water_index].cpu().numpy()
        s2_valid = ~np.all(image == 0, axis=0)
        m = (gt != 0) & s2_valid
        pred_water, gt_water = p_water > a.threshold, gt == 2
        cloud = read_band(a.data_root / "extra" / f"{sid}_CloudLabel.tif")
        covered = (cloud == 1) | (cloud == 2)

        c_all = confusion(pred_water, gt_water, m)
        c_clear = confusion(pred_water, gt_water, m & ~covered)
        c_cov = confusion(pred_water, gt_water, m & covered)
        rows.append(dict(
            sample_id=sid, tp=c_all[0], fp=c_all[1], fn=c_all[2], tn=c_all[3],
            tp_clear=c_clear[0], fp_clear=c_clear[1], fn_clear=c_clear[2], tn_clear=c_clear[3],
            tp_cov=c_cov[0], fp_cov=c_cov[1], fn_cov=c_cov[2], tn_cov=c_cov[3],
            covered_frac=float(covered[m].mean()) if m.any() else None,
            water_iou=metrics(*c_all)["water_iou"],
            u_sum_clear=float(u[m & ~covered].sum()), n_clear=int((m & ~covered).sum()),
            u_sum_cov=float(u[m & covered].sum()), n_cov=int((m & covered).sum()),
            pred_cloud_frac=float((land_water_cloud.numpy()[m] == 3).mean()) if m.any() else None,
        ))
        if a.save_tif:
            s2rst = type("Ref", (), {"transform": transform, "crs": crs})
            save_prediction_tif(land_water_cloud, prob, dst_u, evidence, aleatoric, epistemic, s2rst,
                                sid, a.model_type, output_dir=str(pred_dir))

    t = pd.DataFrame(rows)
    t.to_csv(a.out_dir / "chips.csv", index=False)
    s = lambda cols: [int(t[c].sum()) for c in cols]  # noqa: E731
    u_clear = t.u_sum_clear.sum() / max(t.n_clear.sum(), 1)
    u_cov = t.u_sum_cov.sum() / max(t.n_cov.sum(), 1)
    summary = dict(
        model_type=a.model_type, config=str(a.config), checkpoint=a.checkpoint, data_root=str(a.data_root),
        split_file=str(split), label=a.label, max_tile_size=a.max_tile_size, threshold=a.threshold,
        chips=len(t), mask="GT valid & S2 valid (not all 13 bands 0)",
        all=metrics(*s(["tp", "fp", "fn", "tn"])),
        clear=metrics(*s(["tp_clear", "fp_clear", "fn_clear", "tn_clear"])),
        covered=metrics(*s(["tp_cov", "fp_cov", "fn_cov", "tn_cov"])),
        average_chip_water_iou=float(t.water_iou.dropna().mean()) if len(t) else None,
        mean_dst_u_clear=float(u_clear), mean_dst_u_covered=float(u_cov),
    )
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("chips", "all", "clear", "covered")}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    main()
