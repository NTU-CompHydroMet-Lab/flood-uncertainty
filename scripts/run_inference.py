# ========================================
# 1. IMPORTS & SETUP
# ========================================
import sys
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm

# 設定 project root 和 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, os.path.join(project_root, 'CJ_scripts'))
sys.path.insert(0, project_root)

# 從 infer.py 匯入必要函數
from infer import (
    load_model,
    load_inference_function,
    predict,
    save_prediction_tif,
    plot_prediction,
    COLORS_PRED
)
from georeader.rasterio_reader import RasterioReader
from georeader.geotensor import GeoTensor


# ========================================
# 2. CONFIGURATION
# ========================================
CONFIG = {
    "model_type": "v2",           # "EDL" 或 "v2"
    "input_type": "S2",            # "S2" 或 "L8"
    "data_root": "/home/NAS/homes/cjchen-10025/data/worldfloods_v2/data",
    "output_dir": "result/val_test_inference",
    "subsets": ["val", "test"],    # 要處理的資料集
    "save_plot": True,            # 是否存視覺化圖片
    "save_pred_tif": True,        # 是否存預測結果
    "th_water": 0.5,               # 水體閾值
    "max_tile_size": 1024,
    "config_path_EDL": "CJ_scripts/configurations/hf_hub_download_uncertainty_infer.json",
    "config_path_v2": "CJ_scripts/configurations/hf_hub_download.json"
}

# ========================================
# 3. HELPER FUNCTIONS
# ========================================
def get_all_files(data_root: str, subset: str, input_type: str = "S2") -> list:
    """
    取得指定 subset 下所有 .tif 檔案列表
    
    Args:
        data_root: 資料根目錄
        subset: "val" 或 "test"
        input_type: "S2" 或 "L8"
    
    Returns:
        list of file paths (sorted)
    """
    pattern = f"{data_root}/{subset}/{input_type}/*.tif"
    return sorted(glob(pattern))


def process_single_file(
    filepath: str,
    inference_function,
    channels: list,
    model_type: str,
    output_dir: str,
    subset: str,
    save_plot: bool = False,
    save_pred_tif: bool = False
) -> dict:
    """
    處理單一檔案的推論流程
    
    Args:
        filepath: .tif 檔案路徑
        inference_function: 推論函數
        channels: 波段索引列表
        model_type: 模型類型
        output_dir: 輸出目錄
        subset: 資料集名稱 (用於 plot)
        save_plot: 是否存視覺化圖片
        save_pred_tif: 是否存預測結果
    
    Returns:
        dict: {"success": bool, "filename": str, "error": str or None}
    """
    filename = Path(filepath).stem
    
    # 1. 載入 tif 檔案
    s2rst = RasterioReader(filepath).isel({"band": channels})
    s2rst = s2rst.load()
    
    # 2. 執行 predict
    prediction, pred_prob, dst_u, evidence, aleatoric, epistemic = predict(
        s2rst.values,
        channels=list(range(len(channels))),
        inference_function=inference_function
    )
    
    # 3. (可選) 儲存結果
    if save_pred_tif:
        save_prediction_tif(
            prediction.cpu(), 
            pred_prob.cpu(), 
            dst_u.cpu() if dst_u is not None else None, 
            evidence.cpu() if evidence is not None else None, 
            aleatoric.cpu() if aleatoric is not None else None, 
            epistemic.cpu() if epistemic is not None else None,
            s2rst, filename, model_type, output_dir
        )
    
    # 4. (可選) 儲存視覺化圖片
    if save_plot:
        prediction_raster = GeoTensor(
            prediction.cpu().numpy(),
            transform=s2rst.transform,
            fill_value_default=0,
            crs=s2rst.crs
        )
        # 準備 RGB 順序的影像 (true color: B4, B3, B2 -> 原始索引 [3, 2, 1])
        rgb_bands = [3, 2, 1]  # true color
        rgb_indices = [channels.index(b) for b in rgb_bands if b in channels]
        s2rst_rgb = s2rst.isel({"band": rgb_indices})
        plot_prediction(s2rst_rgb, prediction_raster, subset, f"{filename}_{model_type}", output_dir)
    
    return {"success": True, "filename": filename, "error": None}




# ========================================
# 4. MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    # 解析配置
    model_type = CONFIG["model_type"]
    input_type = CONFIG["input_type"]
    data_root = CONFIG["data_root"]
    output_dir = CONFIG["output_dir"]
    subsets = CONFIG["subsets"]
    save_plot = CONFIG["save_plot"]
    save_pred_tif = CONFIG["save_pred_tif"]
    th_water = CONFIG["th_water"]
    
    

    
    # 4.1 載入模型 (只載入一次)
    print("=" * 50)
    print("Loading model...")
    if model_type == "EDL":
        config_path = CONFIG["config_path_EDL"]
    else:
        config_path = CONFIG["config_path_v2"]
    
    model, channels, config = load_model(config_path=config_path, model_type=model_type)
    inference_function, config = load_inference_function(
        model, config,
        max_tile_size=CONFIG["max_tile_size"],
        apply_normalization=True,
        used_EDL=(model_type == "EDL"),
        th_water=th_water,
        th_brightness=3500,
        distinguish_flood_traces=True
    )
    print("Model loaded successfully!")
    print("=" * 50)
    
    # 統計變數
    total_files = 0
    success_count = 0
    failed_files = []
    
    # 4.2 遍歷每個 subset (val, test)
    for subset in subsets:
        print(f"\nProcessing subset: {subset}")
        
        # 建立輸出目錄
        subset_output_dir = f"{output_dir}/{subset}/{model_type}"
        os.makedirs(subset_output_dir, exist_ok=True)

        # 儲存config
        import json
        config_save_path = f"{subset_output_dir}/run_config.json"
        with open(config_save_path, 'w') as f:
            json.dump(CONFIG, f, indent=2, ensure_ascii=False)
        print(f"Config saved to: {config_save_path}")

        # 取得所有檔案
        file_list = get_all_files(data_root, subset, input_type)
        print(f"Found {len(file_list)} files in {subset}")
        
        # 4.3 遍歷每個檔案並處理
        for filepath in tqdm(file_list, desc=f"{subset}"):
            total_files += 1
            result = process_single_file(
                filepath,
                inference_function,
                channels,
                model_type,
                subset_output_dir,
                subset,
                save_plot,
                save_pred_tif,
            )
            if result["success"]:
                success_count += 1
            else:
                failed_files.append(result["filename"])
    
    # 4.4 輸出處理統計
    print("\n" + "=" * 50)
    print(f"Inference completed!")
    print(f"Total files processed: {total_files}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_files)}")
    if failed_files:
        print(f"Failed files: {failed_files}")
    print("=" * 50)