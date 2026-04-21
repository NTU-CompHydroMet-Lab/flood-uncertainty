import rasterio
import numpy as np
from georeader.rasterio_reader import RasterioReader
from georeader.geotensor import GeoTensor
import torch
from torchmetrics.classification import BinaryConfusionMatrix, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryJaccardIndex
import os
import matplotlib.pyplot as plt
from georeader import plot
from georeader.save import save_cog

def calculate_metrics_bak_for_test_dont_use(gt_path, pred_path, model_type="EDL"):
    # Using RasterioReader
    gt_raster = RasterioReader(gt_path)
    gt_raster = gt_raster.load()  # Load into memory

    pred_raster = RasterioReader(pred_path)
    # Band description validation (only for EDL)
    with rasterio.open(pred_path) as src:
        if model_type == "EDL":
            band2_desc = src.descriptions[2]
        elif model_type == "v2":
            band2_desc = src.descriptions[1]
        if band2_desc != "Water_Probability":
            raise ValueError(f"Band 2 description mismatch! Expected 'Water_Probability', got '{band2_desc}'")
    pred_raster = pred_raster.load()  # Load into memory

    # Read output bands
    pred_classification = pred_raster.values[0]  # Band 0: Classification
    if model_type == "v2":
        pred_prob = pred_raster.values[1]  # v2: Band 1 是 probability
    elif model_type == "EDL":  # EDL
        pred_prob = pred_raster.values[2]  # EDL: Band 2 是 Water_Probability

    # Binary classification
    pred_binary = (pred_prob > 0.5).astype(np.int32)  # 1=water, 0=non-water

    # GT Band 1: convert to binary (2->1 water, 1->0 non-water)
    gt_values = gt_raster.values[1]
    gt_binary = (gt_values == 2).astype(np.int32)  # 1=water, 0=non-water

    # Filter out invalid (GT Band 1 == 0)
    valid_mask = gt_values != 0
    pred_valid = torch.tensor(pred_binary.flatten()[valid_mask.flatten()])
    gt_valid = torch.tensor(gt_binary.flatten()[valid_mask.flatten()])

    # Confusion Matrix
    cm = BinaryConfusionMatrix()
    cm_result = cm(pred_valid, gt_valid)

    # Calculate metrics from confusion matrix
    TN, FP = cm_result[0, 0].item(), cm_result[0, 1].item()
    FN, TP = cm_result[1, 0].item(), cm_result[1, 1].item()

    accuracy_manual = (TP + TN) / (TP + TN + FP + FN)
    precision_manual = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_manual = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_manual = 2 * precision_manual * recall_manual / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0
    iou_manual = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return {
        'cm': cm_result,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'accuracy': accuracy_manual,
        'precision': precision_manual,
        'recall': recall_manual,
        'f1': f1_manual,
        'iou': iou_manual
    }


def plot_spatial_confusion_matrix(gt_path, pred_path, s2_rgb, model_type="EDL", output_dir="result"
                    ,colors_cm = np.array([
                        [128, 128, 128],  # 0: Invalid - 灰色
                        [0, 0, 0],        # 1: TN - 黑色
                        [255, 0, 0],      # 2: FP - 紅色
                        [255, 165, 0],    # 3: FN - 橙色
                        [0, 128, 0],      # 4: TP - 綠色
                    ], dtype=np.float32) / 255, plot_png = True, save_tif = True):
    """
    繪製混淆矩陣 2D 平面分佈圖，同時儲存為 GeoTIFF 和 PNG
    
    Args:
        gt_path: GT 檔案路徑
        pred_path: 預測檔案路徑
        s2_rgb: 已選擇好的 RGB raster (3 波段，用於顯示)
        model_type: 模型類型 ("EDL" 或 "v2")
        output_dir: 輸出目錄
    
    Returns:
        dict: 包含 cm_map、metrics、輸出路徑
    """
    import matplotlib.pyplot as plt
    
    # 讀取 GT raster
    gt_raster = RasterioReader(gt_path).load()
    
    # 讀取 Prediction raster
    pred_raster = RasterioReader(pred_path)
    with rasterio.open(pred_path) as src:
        if model_type == "v2":
            band_idx, expected_desc = 1, "Water_Probability"
        elif model_type == "EDL":
            band_idx, expected_desc = 2, "Water_Probability"
        elif model_type == "ensemble":
            band_idx, expected_desc = 2, "Mean_Water_Probability"
        elif model_type == "mcdropout":
            band_idx, expected_desc = 2, "Mean_Water_Probability"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        actual_desc = src.descriptions[band_idx]
        if actual_desc != expected_desc:
            raise ValueError(f"Band description mismatch! Expected '{expected_desc}', got '{actual_desc}'")
    pred_raster = pred_raster.load()

    # 取得預測概率
    pred_prob = pred_raster.values[band_idx]
    
    # Binary classification
    pred_binary = (pred_prob > 0.5).astype(np.int32)  # 1=water, 0=non-water
    
    # GT: convert to binary (2->1 water, 1->0 non-water)
    gt_values = gt_raster.values[1]
    gt_binary = (gt_values == 2).astype(np.int32)
    
    # 建立混淆矩陣 2D map
    # 0: Invalid, 1: TN, 2: FP, 3: FN, 4: TP
    valid_mask = gt_values != 0
    cm_map = np.zeros_like(gt_values, dtype=np.int32)
    
    cm_map[~valid_mask] = 0  # Invalid
    cm_map[valid_mask & (gt_binary == 0) & (pred_binary == 0)] = 1  # TN
    cm_map[valid_mask & (gt_binary == 0) & (pred_binary == 1)] = 2  # FP
    cm_map[valid_mask & (gt_binary == 1) & (pred_binary == 0)] = 3  # FN
    cm_map[valid_mask & (gt_binary == 1) & (pred_binary == 1)] = 4  # TP
    
    # 計算指標
    TN = np.sum(cm_map == 1)
    FP = np.sum(cm_map == 2)
    FN = np.sum(cm_map == 3)
    TP = np.sum(cm_map == 4)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    
    # 建立 GeoTensor 並存檔
    cm_geotensor = GeoTensor(cm_map, transform=gt_raster.transform,
                              fill_value_default=0, crs=gt_raster.crs)
    
    # 取得檔案名稱 (不含路徑和副檔名)
    filename = os.path.splitext(os.path.basename(pred_path))[0]
    
    # 初始化路徑
    tif_path = None
    png_path = None
    
    # 儲存 GeoTIFF
    if save_tif:
        tif_path = f"{output_dir}/cm_{filename}.tif"
        band_descriptions = ['Confusion_Matrix (0=Invalid, 1=TN, 2=FP, 3=FN, 4=TP)']
        save_cog(cm_geotensor, tif_path, descriptions=band_descriptions)
    
    if plot_png:
        # 繪圖 (1x2: 左邊 RGB, 右邊 CM map)
        fig, ax = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        
        # 左邊: 原始影像
        plot.show((s2_rgb / 3500).clip(0, 1), ax=ax[0], add_scalebar=True)
        ax[0].set_title(f"Original Image: {filename}")
        
        # 右邊: 混淆矩陣分佈
        plot.plot_segmentation_mask(cm_geotensor, colors_cm, ax=ax[1],
                                    interpretation_array=["Invalid", "TN", "FP", "FN", "TP"])
        ax[1].set_title(f"Confusion Matrix Map\nIoU={iou:.4f}, F1={f1:.4f}")
        
        # 儲存 PNG
        png_path = f"{output_dir}/cm_{filename}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return {
        'cm_map': cm_map,
        'cm_geotensor': cm_geotensor,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tif_path': tif_path,
        'png_path': png_path
    }   

def calculate_metrics_by_retention(
    cm_vector,           # 向量1: 混淆矩陣標籤 (0:Invalid, 1:TN, 2:FP, 3:FN, 4:TP)
    criteria_vector,     # 向量2: 篩選基準 (例如 Uncertainty 或 Probability)
    retention_steps=100, # 百分比切分份數 (100 代表 1% 到 100%)
    filter_mode="remove_above" # 篩選模式: "remove_above" 或 "remove_below"
    ):
    """
    輸入:
        cm_vector: 1D numpy array
        criteria_vector: 1D numpy array
        filter_mode: 
            "remove_above": 視「大數值」為不佳，篩除大於動態門檻的值 (保留小值，如不確定性) -> 排序由小到大
            "remove_below": 視「小數值」為不佳，篩除小於動態門檻的值 (保留大值，如信心度) -> 排序由大到小
    輸出:
        results: 字典，包含 retention_rate 與對應的 accuracy, f1, iou 等陣列
    """

    # 1. 資料前處理：移除無效值 (Label 0)
    # 我們只計算有效的預測區域
    valid_mask = (cm_vector != 0)
    cm_valid = cm_vector[valid_mask]
    crit_valid = criteria_vector[valid_mask]

    # 2. 根據模式進行排序 (Sorting)
    # 目標是將「優先保留」的數據排在 array 的前面
    if filter_mode == "remove_above":
        # 模式：以上篩除 (Remove Above) -> 保留以下 (Keep Below)
        # 用於：Uncertainty (越小越好)
        # 排序：由小到大 (最小值在 index 0)
        sorted_indices = np.argsort(crit_valid)
        
    elif filter_mode == "remove_below":
        # 模式：以下篩除 (Remove Below) -> 保留以上 (Keep Above)
        # 用於：Probability/Confidence (越大越好)
        # 排序：由大到小 (最大值在 index 0)
        sorted_indices = np.argsort(crit_valid)[::-1] # 升序後反轉，實現降序
        
    else:
        raise ValueError("filter_mode must be 'remove_above' or 'remove_below'")

    # 套用排序
    cm_sorted = cm_valid[sorted_indices]
    
    # 3. 準備儲存結果的容器
    total_samples = len(cm_sorted)
    metrics_history = {
        "retention_rate": [], # x軸: 數據保留比例
        "threshold": [],      # 紀錄當下的切分閾值
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "iou": []
    }

    from tqdm import tqdm
    # 4. 迴圈計算每個百分比點 (Retention Curve)
    # 例如 step=100, 則計算保留 1%, 2%, ..., 100% 的數據時的指標
    for step in tqdm(range(1, retention_steps + 1), desc="Calculating Retention Metrics"):
        fraction = step / retention_steps # 例如 0.01, 0.02 ... 1.0
        
        # 計算截止索引 (Cutoff Index)
        cutoff_index = int(total_samples * fraction)
        if cutoff_index == 0:
            cutoff_index = 1 # 避免空集合
            
        # 取出「最優」的前 X% 數據
        current_cm_subset = cm_sorted[:cutoff_index]
        
        # 紀錄當下的閾值 (即被切掉的那個邊界值)
        # 注意：如果是 100%，則邊界值是最後一個
        current_threshold = crit_valid[sorted_indices[cutoff_index-1]]
        
        # 5. 統計混淆矩陣 (1:TN, 2:FP, 3:FN, 4:TP)
        TN = np.sum(current_cm_subset == 1)
        FP = np.sum(current_cm_subset == 2)
        FN = np.sum(current_cm_subset == 3)
        TP = np.sum(current_cm_subset == 4)

        # 6. 計算指標 (分母防呆)
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

        # 加入結果
        metrics_history["retention_rate"].append(fraction)
        metrics_history["threshold"].append(current_threshold)
        metrics_history["accuracy"].append(acc)
        metrics_history["precision"].append(prec)
        metrics_history["recall"].append(rec)
        metrics_history["f1"].append(f1)
        metrics_history["iou"].append(iou)

    return metrics_history

if __name__ == "__main__":
    # Define file paths
    filename= "EMSR347_07ZOMBA_DEL_v2"
    gt_path = f"/home/NAS/homes/cjchen-10025/data/worldfloods_v2/data/test/gt/{filename}.tif"
    pred_path = f"/home/NAS/homes/cjchen-10025/ML4FloodsUncertainty/result/val_test_inference/test/EDL/{filename}_output_EDL.tif"
    s2_path = f"/home/NAS/homes/cjchen-10025/data/worldfloods_v2/data/test/S2/{filename}.tif"
    # Calculate metrics
    results = calculate_metrics_bak_for_test_dont_use(gt_path, pred_path)
    
    print("\nConfusion Matrix (rows=GT, cols=Pred):")
    print("Labels: 0=non-water, 1=water")
    print(f"          Pred_0    Pred_1")
    print(f"GT_0  {results['cm'][0,0]:10d}  {results['cm'][0,1]:10d}")
    print(f"GT_1  {results['cm'][1,0]:10d}  {results['cm'][1,1]:10d}")
    
    print(f"\nTP: {results['TP']}, TN: {results['TN']}, FP: {results['FP']}, FN: {results['FN']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"IoU (Water): {results['iou']:.4f}")

    # 在呼叫前先準備好 RGB
    s2_raster = RasterioReader(s2_path).load()
    s2_rgb = s2_raster.isel({"band": [3, 2, 1]})  # 自行選擇 RGB 波段

    result = plot_spatial_confusion_matrix(
        gt_path=gt_path,
        pred_path=pred_path,
        s2_rgb=s2_rgb,  # 直接傳入已選好的 RGB
        model_type="EDL",
        output_dir="result"
    )
    
    print("\n--- Confusion Matrix 2D Map Results ---")
    print(f"TP: {result['TP']}, TN: {result['TN']}, FP: {result['FP']}, FN: {result['FN']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")
    print(f"IoU (Water): {result['iou']:.4f}")
    print(f"GeoTIFF saved: {result['tif_path']}")
    print(f"PNG saved: {result['png_path']}")
