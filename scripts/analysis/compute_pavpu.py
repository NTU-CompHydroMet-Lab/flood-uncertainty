import argparse
import os
import numpy as np
import pandas as pd
import rasterio
from glob import glob
from pathlib import Path
from tqdm import tqdm

from path_defaults import DEFAULT_ANALYSIS_ROOT, DEFAULT_PRED_ROOT

# ========================================
# CONFIGURATION
# ========================================
CRITERIA_MENU = {
    "EDL": {
        "Water_DST_Uncertainty": {
            "band_index": 1,
            "label": "DST Uncertainty"
        },
        "Water_Aleatoric": {
            "band_index": 5,
            "label": "Aleatoric Uncertainty"
        },
        "Water_Epistemic": {
            "band_index": 6,
            "label": "Epistemic Uncertainty"
        },
        "Water_Aleatoric_Plus_Epistemic": {
            "band_indices": [5, 6],
            "label": "Aleatoric + Epistemic Uncertainty"
        },
        "Water_Probability": {
            "band_index": 2,
            "label": "Water Probability"
        },
    },
    "ensemble": {
        "Water_Aleatoric": {
            "band_index": 4,
            "label": "Aleatoric Uncertainty"
        },
        "Water_Epistemic": {
            "band_index": 6,
            "label": "Epistemic Uncertainty"
        },
        "Water_Aleatoric_Plus_Epistemic": {
            "band_indices": [4, 6],
            "label": "Aleatoric + Epistemic Uncertainty"
        },
        "Water_Probability": {
            "band_index": 2,
            "label": "Water Probability"
        },
    },
    "mcdropout": {
        "Water_Aleatoric": {
            "band_index": 4,
            "label": "Aleatoric Uncertainty"
        },
        "Water_Epistemic": {
            "band_index": 6,
            "label": "Epistemic Uncertainty"
        },
        "Water_Aleatoric_Plus_Epistemic": {
            "band_indices": [4, 6],
            "label": "Aleatoric + Epistemic Uncertainty"
        },
        "Water_Probability": {
            "band_index": 2,
            "label": "Water Probability"
        },
    },
}

SELECTED_CRITERIA = {
    "EDL":      ["Water_DST_Uncertainty", "Water_Aleatoric", "Water_Epistemic", "Water_Aleatoric_Plus_Epistemic"],
    "ensemble": ["Water_Aleatoric", "Water_Epistemic", "Water_Aleatoric_Plus_Epistemic"],
    "mcdropout": ["Water_Aleatoric", "Water_Epistemic", "Water_Aleatoric_Plus_Epistemic"],
}

CONFIG = {
    "model_type":      "EDL",   # ← 切換此處："EDL" "ensemble" 或 "mcdropout"
    "subset":          "test",
    "pred_root":       str(DEFAULT_PRED_ROOT),
    "base_output_dir": str(DEFAULT_ANALYSIS_ROOT),
    "patch_size":      3,
    "retention_target": 0.9,
    "max_files":       None,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute PAvPU from confusion maps and prediction outputs."
    )
    parser.add_argument("--model-type", choices=["EDL", "ensemble", "mcdropout"], default=CONFIG["model_type"])
    parser.add_argument("--subset", choices=["val", "test"], default=CONFIG["subset"])
    parser.add_argument("--pred-root", default=CONFIG["pred_root"])
    parser.add_argument("--analysis-root", default=CONFIG["base_output_dir"])
    parser.add_argument("--patch-size", type=int, default=CONFIG["patch_size"])
    parser.add_argument("--retention-target", type=float, default=CONFIG["retention_target"])
    parser.add_argument("--max-files", type=int, default=CONFIG["max_files"])
    return parser.parse_args()


def get_thresholds(model_type, criteria_name, base_output_dir, retention_target):
    """
    從 retention curve CSV 讀取最接近 retention_target 的那一行，
    回傳 (unc_threshold, acc_threshold)。
    acc_threshold 使用 iou 欄位。
    """
    csv_path = os.path.join(base_output_dir, model_type, f"retention_{criteria_name}.csv")
    df = pd.read_csv(csv_path)
    idx = (df["retention_rate"] - retention_target).abs().argmin()
    row = df.iloc[idx]
    return float(row["threshold"]), float(row["iou"])


def compute_pavpu_for_image(cm_map, unc_map, patch_size, acc_threshold, unc_threshold):
    """
    以 numpy 向量化計算單張影像的 PAvPU。

    Args:
        cm_map:        2D array, 值為 0=Invalid, 1=TN, 2=FP, 3=FN, 4=TP
        unc_map:       2D array, 不確定性數值
        patch_size:    patch 邊長 (P)
        acc_threshold: IoU 準確度閾值
        unc_threshold: 不確定性閾值

    Returns:
        n_ac, n_au, n_ic, n_iu, pavpu
    """
    P = patch_size
    H, W = cm_map.shape

    H_trim = (H // P) * P
    W_trim = (W // P) * P

    cm  = cm_map[:H_trim, :W_trim]
    unc = unc_map[:H_trim, :W_trim]

    n_h = H_trim // P
    n_w = W_trim // P

    # reshape 成 (n_h, n_w, P, P)
    cm_patches  = cm.reshape(n_h, P, n_w, P).transpose(0, 2, 1, 3)
    unc_patches = unc.reshape(n_h, P, n_w, P).transpose(0, 2, 1, 3)

    # 含任意 invalid pixel 的 patch → 捨棄
    valid = ~np.any(cm_patches == 0, axis=(2, 3))

    TP = np.sum(cm_patches == 4, axis=(2, 3)).astype(np.float32)
    FP = np.sum(cm_patches == 2, axis=(2, 3)).astype(np.float32)
    FN = np.sum(cm_patches == 3, axis=(2, 3)).astype(np.float32)

    denom = TP + FP + FN
    patch_iou = np.divide(
        TP,
        denom,
        out=np.full_like(TP, np.nan, dtype=np.float32),
        where=denom > 0,
    )
    mean_unc = np.mean(unc_patches, axis=(2, 3))

    # 同時排除含 invalid pixel 的 patch 與 all-TN（IoU 未定義）的 patch
    valid = valid & (denom > 0)

    is_acc = patch_iou >= acc_threshold
    is_unc = mean_unc  >= unc_threshold

    n_ac = int(np.sum(valid &  is_acc & ~is_unc))
    n_au = int(np.sum(valid &  is_acc &  is_unc))
    n_ic = int(np.sum(valid & ~is_acc & ~is_unc))
    n_iu = int(np.sum(valid & ~is_acc &  is_unc))

    total = n_ac + n_au + n_ic + n_iu
    pavpu = (n_ac + n_iu) / total if total > 0 else np.nan

    return n_ac, n_au, n_ic, n_iu, pavpu


def process_all_images(config, criteria_name, acc_threshold, unc_threshold):
    """
    逐一讀取 cm_*.tif 與對應 output_*.tif，計算每張影像的 PAvPU。

    Returns:
        per_image_rows: list of dict，每張影像一筆
        totals:         dict，累加的 n_ac / n_au / n_ic / n_iu
    """
    model_type = config["model_type"]
    subset     = config["subset"]
    pred_root  = config["pred_root"]
    patch_size = config["patch_size"]

    crit_config = CRITERIA_MENU[model_type][criteria_name]
    band_idx = crit_config.get("band_index")
    band_indices = crit_config.get("band_indices")
    target_dir = f"{pred_root}/{subset}/{model_type}"

    cm_files = sorted(glob(f"{target_dir}/cm_*.tif"))
    max_files = config.get("max_files")
    if max_files is not None:
        cm_files = cm_files[:max_files]
        print(f"Limited to first {len(cm_files)} files (max_files={max_files})")
    print(f"Found {len(cm_files)} CM files in {target_dir}")

    per_image_rows = []
    totals = {"n_ac": 0, "n_au": 0, "n_ic": 0, "n_iu": 0}

    for cm_path in tqdm(cm_files, desc=f"Computing PAvPU for {criteria_name}"):
        cm_filename = Path(cm_path).stem

        if not cm_filename.startswith("cm_"):
            continue

        original_filename_part = cm_filename[3:]  # 去掉前綴 'cm_'
        pred_path = f"{target_dir}/{original_filename_part}.tif"

        if not os.path.exists(pred_path):
            alt_pred_path = f"{target_dir}/{original_filename_part}_output_{model_type}.tif"
            if os.path.exists(alt_pred_path):
                pred_path = alt_pred_path
            else:
                continue

        with rasterio.open(cm_path) as src:
            cm_map = src.read(1)
        with rasterio.open(pred_path) as src:
            if band_indices is not None:
                unc_map = src.read(band_indices[0] + 1).astype(np.float32)
                for extra_band_idx in band_indices[1:]:
                    unc_map = unc_map + src.read(extra_band_idx + 1).astype(np.float32)
            else:
                unc_map = src.read(band_idx + 1).astype(np.float32)  # rasterio 用 1-based index

        n_ac, n_au, n_ic, n_iu, pavpu = compute_pavpu_for_image(
            cm_map, unc_map, patch_size, acc_threshold, unc_threshold
        )

        # 取原始影像名稱（去掉 _output_{model_type} 後綴）
        suffix = "_mcdropout_output_ensemble" if model_type == "mcdropout" else f"_output_{model_type}"
        image_name = original_filename_part.replace(suffix, "")

        per_image_rows.append({
            "filename": image_name,
            "subset":   subset,
            "n_ac":     n_ac,
            "n_au":     n_au,
            "n_ic":     n_ic,
            "n_iu":     n_iu,
            "pavpu":    pavpu,
        })

        totals["n_ac"] += n_ac
        totals["n_au"] += n_au
        totals["n_ic"] += n_ic
        totals["n_iu"] += n_iu

    return per_image_rows, totals


def save_results(per_image_rows, totals, subset, output_path):
    """
    輸出 CSV，格式同 metrics_EDL.csv：
    每張影像一行，最後附上 OVERALL（所有 patch 加總）與 AVERAGE（per-image 平均）。
    """
    df = pd.DataFrame(per_image_rows)

    n_ac_tot = totals["n_ac"]
    n_au_tot = totals["n_au"]
    n_ic_tot = totals["n_ic"]
    n_iu_tot = totals["n_iu"]
    total_tot = n_ac_tot + n_au_tot + n_ic_tot + n_iu_tot
    pavpu_overall = (n_ac_tot + n_iu_tot) / total_tot if total_tot > 0 else np.nan

    overall_row = pd.DataFrame([{
        "filename": "OVERALL",
        "subset":   subset,
        "n_ac":     n_ac_tot,
        "n_au":     n_au_tot,
        "n_ic":     n_ic_tot,
        "n_iu":     n_iu_tot,
        "pavpu":    pavpu_overall,
    }])

    average_row = pd.DataFrame([{
        "filename": "AVERAGE",
        "subset":   subset,
        "n_ac":     df["n_ac"].mean(),
        "n_au":     df["n_au"].mean(),
        "n_ic":     df["n_ic"].mean(),
        "n_iu":     df["n_iu"].mean(),
        "pavpu":    df["pavpu"].mean(),
    }])

    result_df = pd.concat([df, overall_row, average_row], ignore_index=True)
    result_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()
    config = dict(CONFIG)
    config["model_type"] = args.model_type
    config["subset"] = args.subset
    config["pred_root"] = args.pred_root
    config["base_output_dir"] = args.analysis_root
    config["patch_size"] = args.patch_size
    config["retention_target"] = args.retention_target
    config["max_files"] = args.max_files

    model_type = config["model_type"]
    subset = config["subset"]
    base_output_dir = config["base_output_dir"]
    retention_target = config["retention_target"]

    output_dir = os.path.join(base_output_dir, model_type)
    os.makedirs(output_dir, exist_ok=True)

    for criteria_name in SELECTED_CRITERIA[model_type]:
        # if criteria_name not in CRITERIA_MENU[model_type]:
        #     continue

        print(f"\n{'='*50}")
        print(f"Criteria: {criteria_name}  |  Model: {model_type}")
        print(f"{'='*50}")

        unc_threshold, acc_threshold = get_thresholds(
            model_type, criteria_name, base_output_dir, retention_target
        )
        print(f"  retention_target : {retention_target}")
        print(f"  unc_threshold    : {unc_threshold:.6f}")
        print(f"  acc_threshold    : {acc_threshold:.6f}  (IoU)")
        print(f"  patch_size       : {config['patch_size']}x{config['patch_size']}")

        per_image_rows, totals = process_all_images(
            config, criteria_name, acc_threshold, unc_threshold
        )

        if not per_image_rows:
            print("No valid images found, skipping.")
            continue

        output_path = os.path.join(output_dir, f"pavpu_{criteria_name}.csv")
        save_results(per_image_rows, totals, subset, output_path)


if __name__ == "__main__":
    main()
