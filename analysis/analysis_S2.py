
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from tqdm import tqdm
from georeader.rasterio_reader import RasterioReader

# Add parent directory to path to import matrics
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from flood_uncertainty.metrics.segmentation import calculate_metrics_by_retention

# ========================================
# CONFIGURATION
# ========================================
CRITERIA_MENU = {
    "EDL": {
        "Water_DST_Uncertainty": {
            "band_index": 1,
            "filter_mode": "remove_above",
            "label": "DST Uncertainty"
        },
        "Water_Aleatoric": {
            "band_index": 5,
            "filter_mode": "remove_above",
            "label": "Aleatoric Uncertainty"
        },
        "Water_Epistemic": {
            "band_index": 6,
            "filter_mode": "remove_above",
            "label": "Epistemic Uncertainty"
        },
        "Water_Aleatoric_Plus_Epistemic": {
            "band_indices": [5, 6],
            "filter_mode": "remove_above",
            "label": "Aleatoric + Epistemic Uncertainty"
        },
        "Water_Probability": {
            "band_index": 2,
            "filter_mode": "remove_below",
            "label": "Water Probability"
        },
    },
    "ensemble": {
        "Water_Aleatoric": {
            "band_index": 4,             # Aleatoric_Water 在 ensemble TIF 的 index 4
            "filter_mode": "remove_above",
            "label": "Aleatoric Uncertainty"
        },
        "Water_Epistemic": {
            "band_index": 6,             # Epistemic_Water 在 ensemble TIF 的 index 6
            "filter_mode": "remove_above",
            "label": "Epistemic Uncertainty"
        },
        "Water_Aleatoric_Plus_Epistemic": {
            "band_indices": [4, 6],
            "filter_mode": "remove_above",
            "label": "Aleatoric + Epistemic Uncertainty"
        },
        "Water_Probability": {
            "band_index": 2,             # Mean_Water_Probability，與 EDL 相同
            "filter_mode": "remove_below",
            "label": "Water Probability"
        },
    },
    "mcdropout": {
        "Water_Aleatoric": {
            "band_index": 4,
            "filter_mode": "remove_above",
            "label": "Aleatoric Uncertainty"
        },
        "Water_Epistemic": {
            "band_index": 6,
            "filter_mode": "remove_above",
            "label": "Epistemic Uncertainty"
        },
        "Water_Aleatoric_Plus_Epistemic": {
            "band_indices": [4, 6],
            "filter_mode": "remove_above",
            "label": "Aleatoric + Epistemic Uncertainty"
        },
        "Water_Probability": {
            "band_index": 2,
            "filter_mode": "remove_below",
            "label": "Water Probability"
        },
    },
}

# 各 model_type 預設執行的分析項目（不含 Water_Probability 除非需要）
SELECTED_CRITERIA = {
    "EDL":      ["Water_DST_Uncertainty", "Water_Aleatoric", "Water_Epistemic", "Water_Aleatoric_Plus_Epistemic"],
    "ensemble": ["Water_Aleatoric", "Water_Epistemic", "Water_Aleatoric_Plus_Epistemic"],
    "mcdropout": ["Water_Aleatoric", "Water_Epistemic", "Water_Aleatoric_Plus_Epistemic"],
}

CONFIG = {
    "model_type":      "EDL",   # ← 切換此處："EDL" "ensemble" 或 "mcdropout"
    "subset":          "test",
    "pred_root":       "/home/NAS/homes/cjchen-10025/ML4FloodsUncertainty/result/val_test_inference",
    "base_output_dir": "/home/NAS/homes/cjchen-10025/ML4FloodsUncertainty/result/analysis_S2",
    "retention_steps": 50,
}


def load_and_prepare_vectors(config, criteria_name):
    """
    從已生成的 cm_*.tif 讀取混淆矩陣，並從 output_*.tif 讀取 Criteria
    """
    subset = config["subset"]
    pred_root = config["pred_root"]
    model_type = config["model_type"]
    
    crit_config = CRITERIA_MENU[model_type][criteria_name]
    band_idx = crit_config.get("band_index")
    band_indices = crit_config.get("band_indices")
    
    # 目標目錄: result/val_test_inference/{subset}/{model_type}
    target_dir = f"{pred_root}/{subset}/{model_type}"
    
    # 搜尋 cm_ 開頭的檔案
    # 假設檔名格式: cm_{original_filename}_output_{model_type}.tif 
    # 或者 cm_{original_filename}.tif (取決於 calculate_matrics.py 的輸出)
    # 我們先搜尋所有 cm_*.tif
    cm_files = sorted(glob(f"{target_dir}/cm_*.tif"))
    print(f"Found {len(cm_files)} CM files in {target_dir}")
    
    huge_cm_list = []
    huge_crit_list = []
    
    for cm_path in tqdm(cm_files, desc=f"Loading data for {criteria_name}"):
        cm_filename = Path(cm_path).stem
        
        # 解析原始檔名以找到對應的 prediction file
        # 假設 cm_path 是 .../cm_FILENAME.tif (由 matrics.py 的 plot_spatial_confusion_matrix 生成)
        # 則原始 filename = cm_filename[3:] (去掉 "cm_")
        if not cm_filename.startswith("cm_"):
            continue
            
        original_filename_part = cm_filename[3:] # 去掉前綴 'cm_'
        
        # 嘗試建構 prediction path
        # 根據 run_inference.py: f"{output_dir}/{filename}_output_{model_type}.tif"
        # 這裡的 original_filename_part 可能是 "FILENAME_output_EDL" 或者只是 "FILENAME"
        # 讓我們檢查一下 CM 檔案是怎麼生成的
        # matrics.py: filename = os.path.splitext(os.path.basename(pred_path))[0]
        # pred_path 是 ..._output_EDL.tif
        # 所以 cm_filename 會是 cm_FILENAME_output_EDL
        
        # Prediction file 應該就是去掉 'cm_' 的名字加上 .tif
        pred_path = f"{target_dir}/{original_filename_part}.tif"
        
        if not os.path.exists(pred_path):
            # 若找不到，嘗試另一種常見命名 (如果 CM 檔名沒有包含 output_EDL)
            alt_pred_path = f"{target_dir}/{original_filename_part}_output_{model_type}.tif"
            if os.path.exists(alt_pred_path):
                pred_path = alt_pred_path
            else:
                # print(f"Prediction file not found for {cm_filename}")
                continue
            
        try:
            # 1. Load CM Raster (Band 1 has values 0-4)
            cm_raster = RasterioReader(cm_path).load()
            cm_values = cm_raster.values[0] # Band 0
            
            # 2. Load Criteria (Uncertainty/Prob)
            # Band index is 0-based from config
            pred_raster = RasterioReader(pred_path).load()
            if band_indices is not None:
                crit_values = pred_raster.values[band_indices[0]]
                for extra_band_idx in band_indices[1:]:
                    crit_values = crit_values + pred_raster.values[extra_band_idx]
            else:
                crit_values = pred_raster.values[band_idx]
            
            # 3. Filter Valid Pixels (CM != 0)
            valid_mask = (cm_values != 0)
            
            if not np.any(valid_mask):
                continue
                
            valid_cm = cm_values[valid_mask]
            valid_crit = crit_values[valid_mask]
            
            huge_cm_list.append(valid_cm.astype(np.int8))
            huge_crit_list.append(valid_crit.astype(np.float32))
            
        except Exception as e:
            print(f"Error processing {cm_filename}: {e}")
            continue

    # Concatenate
    if not huge_cm_list:
        print("No valid data found!")
        return None, None
        
    print("Concatenating vectors...")
    cm_vector = np.concatenate(huge_cm_list)
    criteria_vector = np.concatenate(huge_crit_list)
    print(f"Total valid pixels: {len(cm_vector)}")
    
    # 簡單統計一下 CM 分布，確認是否正常
    unique, counts = np.unique(cm_vector, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"CM Distribution (1=TN, 2=FP, 3=FN, 4=TP): {dist}")
    
    return cm_vector, criteria_vector


def plot_retention_curves(results_dict, output_dir, label):
    plt.figure(figsize=(10, 6))
    
    metrics = ['accuracy', 'f1', 'iou', 'precision', 'recall']
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    df = pd.DataFrame(results_dict)
    x = df['retention_rate'] * 100 
    
    for metric, color in zip(metrics, colors):
        plt.plot(x, df[metric], marker='o', label=metric.capitalize(), color=color)
        
    plt.title(f'Retention Curve by {label}')
    plt.xlabel('Retention Rate (%)')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # plt.ylim(0, 1.05) # Fixed limit removed to allow auto-scaling
    plt.gca().invert_xaxis() 
    
    filename = f"retention_curve_{label.replace(' ', '_')}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")


def main():
    model_type = CONFIG["model_type"]
    output_dir = os.path.join(CONFIG["base_output_dir"], model_type)
    os.makedirs(output_dir, exist_ok=True)

    selected_criteria = SELECTED_CRITERIA[model_type]

    for criteria_key in selected_criteria:
        if criteria_key not in CRITERIA_MENU[model_type]:
            continue
            
        print(f"\n{'='*50}")
        print(f"Starting Analysis for: {criteria_key} (model: {model_type})")
        print(f"{'='*50}")
        
        crit_config = CRITERIA_MENU[model_type][criteria_key]
        
        cm_vec, crit_vec = load_and_prepare_vectors(CONFIG, criteria_key)
        
        if cm_vec is None:
            continue
            
        print("Calculating metrics...")
        results = calculate_metrics_by_retention(
            cm_vec, 
            crit_vec, 
            retention_steps=CONFIG["retention_steps"],
            filter_mode=crit_config["filter_mode"]
        )
        
        df = pd.DataFrame(results)
        csv_name = f"retention_{criteria_key}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")
        
        plot_retention_curves(results, output_dir, crit_config["label"])
        
        del cm_vec, crit_vec, results, df
        import gc
        gc.collect()

if __name__ == "__main__":
    main()
