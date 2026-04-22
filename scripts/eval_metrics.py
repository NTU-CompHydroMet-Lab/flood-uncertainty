# ========================================
# 1. IMPORTS & SETUP
# ========================================
import argparse
import os
import sys
from glob import glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from georeader.rasterio_reader import RasterioReader

# 設定 project root 和 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from flood_uncertainty.metrics.segmentation import plot_spatial_confusion_matrix
from flood_uncertainty.utils.config_loader import load_mode_config

# ========================================
# 2. CONFIGURATION
# ========================================
CONFIG = {
    "model_type": "mcdropout", # "v2" "EDL" "ensemble" "mcdropout"
    "output_dir": "result/val_test_inference",
    "subset": "test",  # "val" "test"
    "config_mode": "infer",
    "plot_png": True,   # 是否產生 PNG 圖片
    "save_tif": True,   # 是否儲存 GeoTIFF
}

MODEL_CONFIG_PATHS = {
    "v2": "configurations/v2.json",
    "EDL": "configurations/edl.json",
    "ensemble": "configurations/ensemble.json",
    "mcdropout": "configurations/dropout.json",
}

# ========================================
# 3. MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["v2", "EDL", "ensemble", "mcdropout"], default=CONFIG["model_type"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--config_mode", choices=["infer"], default=CONFIG["config_mode"])
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default=CONFIG["output_dir"])
    parser.add_argument("--subset", choices=["val", "test"], default=CONFIG["subset"])
    parser.add_argument("--plot_png", action="store_true", default=CONFIG["plot_png"])
    parser.add_argument("--no_plot_png", action="store_false", dest="plot_png")
    parser.add_argument("--save_tif", action="store_true", default=CONFIG["save_tif"])
    parser.add_argument("--no_save_tif", action="store_false", dest="save_tif")
    args, _ = parser.parse_known_args()

    model_type = args.model_type
    config_path = args.config or MODEL_CONFIG_PATHS[model_type]
    mode_config = load_mode_config(config_path, mode=args.config_mode)
    data_root = args.data_root or mode_config.data_params.path_to_splits
    output_dir = args.output_dir
    subset = args.subset
    plot_png = args.plot_png
    save_tif = args.save_tif

    all_results = []
    total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0

    print(f"\n{'='*50}")
    print(f"Processing subset: {subset}")
    print(f"model_type={model_type}, config={config_path}")
    print(f"data_root={data_root}")

    # GT, S2, Prediction 資料夾路徑
    gt_dir = f"{data_root}/{subset}/gt"
    s2_dir = f"{data_root}/{subset}/S2"
    pred_dir = f"{output_dir}/{subset}/{model_type}"

    # 取得所有 GT 檔案
    gt_files = sorted(glob(f"{gt_dir}/*.tif"))
    print(f"Found {len(gt_files)} GT files")

    for gt_path in tqdm(gt_files, desc=f"{subset}"):
        filename = Path(gt_path).stem

        if model_type == "v2":
            pred_path = f"{pred_dir}/{filename}_prediction_{model_type}.tif"
        elif model_type == "ensemble":
            pred_path = f"{pred_dir}/{filename}_output_{model_type}.tif"
        elif model_type == "mcdropout":
            pred_path = f"{pred_dir}/{filename}_mcdropout_output_ensemble.tif"
        else:
            pred_path = f"{pred_dir}/{filename}_output_{model_type}.tif"

        # 檢查 prediction 檔案是否存在
        if not os.path.exists(pred_path):
            print(f"  [SKIP] Prediction not found: {filename}")
            continue

        # 讀取 S2 RGB
        s2_path = f"{s2_dir}/{filename}.tif"
        if not os.path.exists(s2_path):
            print(f"  [SKIP] S2 not found: {filename}")
            continue
        s2_raster = RasterioReader(s2_path).load()
        s2_rgb = s2_raster.isel({"band": [3, 2, 1]})

        # 計算 metrics 並產生圖片
        result = plot_spatial_confusion_matrix(
            gt_path=gt_path,
            pred_path=pred_path,
            s2_rgb=s2_rgb,
            model_type=model_type,
            output_dir=pred_dir,
            plot_png=plot_png,
            save_tif=save_tif
        )

        # 只保留需要的欄位
        results = {
            'filename': filename,
            'subset': subset,
            'TP': result['TP'],
            'TN': result['TN'],
            'FP': result['FP'],
            'FN': result['FN'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'iou': result['iou'],
        }
        all_results.append(results)

        # 累加 CM
        total_TP += results['TP']
        total_TN += results['TN']
        total_FP += results['FP']
        total_FN += results['FN']

    if not all_results:
        raise RuntimeError(f"No valid samples found for subset={subset}, model_type={model_type}")

    # 計算整體指標
    print(f"\n{'='*50}")
    print("Overall Results (Aggregated)")
    print(f"{'='*50}")
    print(f"Total TP: {total_TP}, TN: {total_TN}, FP: {total_FP}, FN: {total_FN}")

    overall_acc = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
    overall_prec = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_rec = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0
    overall_iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0

    print(f"Accuracy: {overall_acc:.4f}")
    print(f"Precision: {overall_prec:.4f}")
    print(f"Recall: {overall_rec:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    print(f"IoU (Water): {overall_iou:.4f}")

    # 計算平均指標
    num_files = len(all_results)
    avg_acc = sum(r['accuracy'] for r in all_results) / num_files
    avg_prec = sum(r['precision'] for r in all_results) / num_files
    avg_rec = sum(r['recall'] for r in all_results) / num_files
    avg_f1 = sum(r['f1'] for r in all_results) / num_files
    avg_iou = sum(r['iou'] for r in all_results) / num_files

    print(f"\n--- Average Metrics (per event) ---")
    print(f"Accuracy: {avg_acc:.4f}")
    print(f"Precision: {avg_prec:.4f}")
    print(f"Recall: {avg_rec:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"IoU (Water): {avg_iou:.4f}")

    # 加入 Overall 和 Average row
    overall_row = {
        'filename': 'OVERALL',
        'subset': subset,
        'TP': total_TP, 'TN': total_TN, 'FP': total_FP, 'FN': total_FN,
        'accuracy': overall_acc,
        'precision': overall_prec,
        'recall': overall_rec,
        'f1': overall_f1,
        'iou': overall_iou
    }
    average_row = {
        'filename': 'AVERAGE',
        'subset': subset,
        'TP': total_TP / num_files, 'TN': total_TN / num_files,
        'FP': total_FP / num_files, 'FN': total_FN / num_files,
        'accuracy': avg_acc,
        'precision': avg_prec,
        'recall': avg_rec,
        'f1': avg_f1,
        'iou': avg_iou
    }
    all_results.append(overall_row)
    all_results.append(average_row)

    # 儲存結果為 CSV
    df = pd.DataFrame(all_results)
    csv_path = f"{output_dir}/{subset}/{model_type}/metrics_{model_type}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
