# ========================================
# 1. IMPORTS & SETUP
# ========================================
import sys
import os
import argparse
import json
from glob import glob
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 設定 project root 和 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np

from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.models.model_setup import get_channel_configuration_bands
from georeader.rasterio_reader import RasterioReader
from georeader.geotensor import GeoTensor

# 從 infer.py 匯入 inference 與輸出函數
from flood_uncertainty.inference.infer import (
    run_ensemble_inference,
    run_mcdropout_inference,
    save_ensemble_tif,
    plot_prediction,
    COLORS_PRED,
)


# ========================================
# 2. CONFIGURATION
# ========================================
CONFIG = {
    # 模式切換
    "mode": "mcdropout",
    "n_samples": 10,

    # 資料與設定檔路徑
    "config_path": os.path.join(project_root, "configurations", "dropout.json"),
    "config_mode": "infer",
    "output_dir": os.path.join(project_root, "artifacts", "results", "val_test_inference"),

    # 執行範圍
    "subsets": ["val", "test"],   # 要處理哪些 split

    # 推理參數（對應 run_ensemble_inference 的參數）
    "th_water":                 0.5,
    "th_brightness":            3500,
    "max_tile_size":            1024,
    "distinguish_flood_traces": True,

    # 輸出控制
    "save_tif":  True,
    "save_plot": True,
}


# ========================================
# 3. HELPER FUNCTIONS
# ========================================
def get_all_files(data_root: str, subset: str, input_type: str = "S2") -> list:
    """
    取得指定 subset 下所有 .tif 檔案列表（複製自 run_inference.py L49）

    Args:
        data_root:  資料根目錄
        subset:     "val" 或 "test"
        input_type: "S2" 或 "L8"

    Returns:
        list of file paths (sorted)
    """
    pattern = f"{data_root}/{subset}/{input_type}/*.tif"
    return sorted(glob(pattern))


def resolve_checkpoint_dir(config) -> str:
    """
    解析 mcdropout 使用的 checkpoint directory。

    優先順序：
    1. config.model_params["checkpoint_dir"]
    2. config.model_params["pretrained_path"] 的所在資料夾
    """
    model_params = config.model_params

    checkpoint_dir = model_params.get("checkpoint_dir")
    if checkpoint_dir:
        return checkpoint_dir

    pretrained_path = model_params.get("pretrained_path")
    if pretrained_path:
        if os.path.isdir(pretrained_path):
            return pretrained_path
        return os.path.dirname(pretrained_path)

    raise ValueError("Cannot resolve checkpoint_dir for mcdropout from config.model_params")


def get_mode_settings(config) -> dict:
    """
    根據 CONFIG['mode'] 回傳對應的 inference 設定。
    """
    mode = CONFIG["mode"]

    if mode == "ensemble":
        ensemble_dirs = config.model_params.get("ensemble_dirs")
        if not ensemble_dirs:
            raise ValueError("ensemble mode requires config.model_params['ensemble_dirs']")

        return {
            "mode": "ensemble",
            "runner": run_ensemble_inference,
            "member_count": len(ensemble_dirs),
            "ensemble_dirs": ensemble_dirs,
        }

    if mode == "mcdropout":
        checkpoint_dir = resolve_checkpoint_dir(config)
        return {
            "mode": "mcdropout",
            "runner": run_mcdropout_inference,
            "member_count": CONFIG["n_samples"],
            "checkpoint_dir": checkpoint_dir,
        }

    raise ValueError("CONFIG['mode'] must be 'ensemble' or 'mcdropout'")


def process_inference_file(
    filepath: str,
    config,
    mode_settings: dict,
    channels: list,
    output_dir: str,
    subset: str,
    save_tif: bool = True,
    save_plot: bool = True,
) -> dict:
    """
    對單一 .tif 檔案執行 inference，支援 ensemble / mcdropout。
    """
    stem = Path(filepath).stem

    s2rst = RasterioReader(filepath).isel({"band": channels}).load()
    torch_inputs = torch.tensor(np.nan_to_num(s2rst.values.astype(np.float32)))

    common_kwargs = {
        "config": config,
        "channels": channels,
        "th_water": CONFIG["th_water"],
        "th_brightness": CONFIG["th_brightness"],
        "max_tile_size": CONFIG["max_tile_size"],
        "distinguish_flood_traces": CONFIG["distinguish_flood_traces"],
    }

    if mode_settings["mode"] == "ensemble":
        outputs = mode_settings["runner"](
            torch_inputs,
            ensemble_dirs=mode_settings["ensemble_dirs"],
            **common_kwargs,
        )
    else:
        outputs = mode_settings["runner"](
            torch_inputs,
            checkpoint_dir=mode_settings["checkpoint_dir"],
            n_samples=CONFIG["n_samples"],
            **common_kwargs,
        )

    land_water_cloud, all_probs_flat, mean_prob, aleatoric_uncertainty, epistemic_uncertainty = outputs

    if save_tif:
        save_ensemble_tif(
            prediction=land_water_cloud,
            all_probs_flat=all_probs_flat,
            mean_prob=mean_prob,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            s2rst=s2rst,
            filename=f"{stem}_{mode_settings['mode']}",
            n_models=mode_settings["member_count"],
            output_dir=output_dir,
        )

    if save_plot:
        prediction_raster = GeoTensor(
            land_water_cloud.numpy(),
            transform=s2rst.transform,
            fill_value_default=0,
            crs=s2rst.crs,
        )
        rgb_indices = [channels.index(b) for b in [3, 2, 1] if b in channels]
        s2rst_rgb = s2rst.isel({"band": rgb_indices})
        plot_prediction(
            s2rst_rgb,
            prediction_raster,
            subset,
            filename=f"{stem}_{mode_settings['mode']}",
            output_dir=output_dir,
        )

    return {"success": True, "filename": stem, "error": None}


# ========================================
# 4. MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ensemble", "mcdropout"], default=CONFIG["mode"])
    parser.add_argument("--config", default=CONFIG["config_path"])
    parser.add_argument("--config_mode", choices=["infer"], default=CONFIG["config_mode"])
    parser.add_argument("--data_root", default=None)
    args, _ = parser.parse_known_args()
    CONFIG["mode"] = args.mode
    CONFIG["config_path"] = args.config
    CONFIG["config_mode"] = args.config_mode

    # 讀取 config（只讀一次）
    config = load_mode_config(CONFIG["config_path"], mode=CONFIG["config_mode"])
    data_root = args.data_root or config.data_params.path_to_splits
    CONFIG["data_root"] = data_root
    channels = get_channel_configuration_bands(
        config.data_params.channel_configuration, collection_name="S2"
    )
    mode_settings = get_mode_settings(config)
    print(
        f"mode: {mode_settings['mode']}, "
        f"members: {mode_settings['member_count']}, "
        f"channels: {channels}"
    )

    # 統計變數
    total_files   = 0
    success_count = 0
    failed_files  = []

    # 遍歷每個 subset
    for subset in CONFIG["subsets"]:
        print(f"\n{'='*50}")
        print(f"Processing subset: {subset} | mode: {mode_settings['mode']}")

        # 建立輸出目錄
        subset_output_dir = f"{CONFIG['output_dir']}/{subset}/{mode_settings['mode']}"
        os.makedirs(subset_output_dir, exist_ok=True)

        # 3.0 寫入 run_config.json（記錄本次執行的完整參數）
        run_config = {
            **CONFIG,
            "channels":   channels,
            "timestamp":  datetime.now().isoformat(),
        }

        if mode_settings["mode"] == "ensemble":
            run_config["n_models"] = mode_settings["member_count"]
            run_config["ensemble_dirs"] = mode_settings["ensemble_dirs"]
        else:
            run_config["n_samples"] = mode_settings["member_count"]
            run_config["checkpoint_dir"] = mode_settings["checkpoint_dir"]

        config_save_path = f"{subset_output_dir}/run_config.json"
        with open(config_save_path, "w") as f:
            json.dump(run_config, f, indent=2, ensure_ascii=False)
        print(f"Config saved to: {config_save_path}")

        # 取得所有檔案
        file_list = get_all_files(data_root, subset)
        print(f"Found {len(file_list)} files to process in [{subset}]")

        # 遍歷每個檔案
        for filepath in tqdm(file_list, desc=subset):
            total_files += 1
            result = process_inference_file(
                filepath=filepath,
                config=config,
                mode_settings=mode_settings,
                channels=channels,
                output_dir=subset_output_dir,
                subset=subset,
                save_tif=CONFIG["save_tif"],
                save_plot=CONFIG["save_plot"],
            )
            if result["success"]:
                success_count += 1
            else:
                failed_files.append(result["filename"])

    # 3.2 統計輸出
    print(f"\n{'='*50}")
    print(f"Inference completed!")
    print(f"Total:   {total_files}")
    print(f"Success: {success_count}")
    print(f"Failed:  {len(failed_files)}")
    if failed_files:
        print(f"Failed files: {failed_files}")
    print("="*50)
