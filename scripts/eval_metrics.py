# ========================================
# 1. IMPORTS & SETUP
# ========================================
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from georeader.rasterio_reader import RasterioReader

from matrics import plot_spatial_confusion_matrix

# ========================================
# 2. CONFIGURATION
# ========================================
CONFIG = {
    "model_type": "mcdropout", # "v2" "EDL" "ensemble" "mcdropout"
    "data_root": "/home/NAS/homes/cjchen-10025/data/worldfloods_v2/data",
    "output_dir": "result/val_test_inference",
    "subset": "test",  # "val" "test"
    "plot_png": True,   # 是否產生 PNG 圖片
    "save_tif": True,   # 是否儲存 GeoTIFF
}

# ========================================
# 3. MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    model_type = CONFIG["model_type"]
    data_root = CONFIG["data_root"]
    output_dir = CONFIG["output_dir"]
    subset = CONFIG["subset"]
    plot_png = CONFIG["plot_png"]
    save_tif = CONFIG["save_tif"]

    all_results = []
    total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0

    print(f"\n{'='*50}")
    print(f"Processing subset: {subset}")

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
