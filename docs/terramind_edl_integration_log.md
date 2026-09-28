# TerraMind 編碼器接入 EDL 模型：實作紀錄

> Branch: `feat/terramind-edl-backbone`（自 `dev` 分出）
> 依據：[`foundation_model_integration.md`](./foundation_model_integration.md)（下稱「計畫」）
> 目標：把 `EDL_ML4FloodsModel` 的 UNet **encoder** 換成 TerraMind v1 base，
> decoder / EDL head / loss / 資料管線 / Trainer 一律不動，並依 `train_edl.py` 的範式訓練與測試。

本文件記錄每一步「做了什麼、為什麼、驗證方式」。與計畫不同之處另列於 §4。

---

## 1. 環境盤點（動手前）

| 項目 | 結果 |
|---|---|
| Python / torch / lightning | 3.10 / 2.10.0 / pytorch-lightning 2.6.0（`pyproject.toml` 鎖定） |
| GPU | 1× RTX 3080 10 GB |
| TerraMind 權重 | HuggingFace cache 已有 `ibm-esa-geospatial/TerraMind-1.0-base/TerraMind_v1_base.pt` |
| 網路 | 可連 huggingface.co |
| 資料 | `configurations/edl.json` 指向的 `chlunchen-10030` 路徑在本機**不存在**；`cjchen-10025` 副本的 train split **沒有 S2 影像**；`cynthia-10040/Data/worldfloods_v2_smoke_source` 有完整 train S2（475 張，val/test 以 symlink 指回 cjchen-10025），但目錄唯讀。訓練腳本會把 `train_test_split_from_csv.json` 寫進 data root，因此在 `/home/NAS/house/ycchen-10014/data/worldfloods_v2` 建了一個可寫的 symlink root（train/val/test 各指向上述來源，複製 `dataset_metadata.csv`），config 指向它 |
| terratorch | 未安裝；PyPI 最新 1.2.13，硬需求 `numpy>=2.2`，與專案 `numpy==1.26.4` 衝突（計畫風險 #1 命中） |

---

## 2. 變更清單

### 新增檔案

| 檔案 | 內容 | 對應計畫 |
|---|---|---|
| `flood_uncertainty/models/backbones/__init__.py` | 子套件入口 | §4 |
| `flood_uncertainty/models/backbones/terramind_vendored/` | terratorch 1.2.11 的 TerraMind ViT encoder 原始碼（Apache-2.0，附 LICENSE 與逐檔 provenance 標頭）：`tm_utils.py`、`encoder_embeddings.py`（逐字）、`modality_info.py`、`modality_embeddings.py`（裁剪）、`terramind_vit.py`（移除 tokenizer）、`factory.py`（`build_terramind_vit` ＝ `BACKBONE_REGISTRY.build` 的等價物） | §6（見 §3.7 改採 vendoring 的原因） |
| `flood_uncertainty/models/backbones/terramind_backbone.py` | `TerraMindBackbone`：dict 進、pyramid list 出，暴露 `out_channels` / `downsample_ratios`；三個純 `nn.Module` helper（`SelectIndices`、`ReshapeTokensToImage`、`LearnedInterpolateToPyramidal`）；`InputRenormalizer`；band 名稱對照 | §3.1, §5, §6 |
| `flood_uncertainty/models/terramind_unet.py` | `TerraMindUNet`：TerraMind encoder + 與 ml4floods `UNet` 同構的 decoder（`layer_factory.double_conv`）+ 1×1 head。`forward(Tensor) -> Tensor`，與 `UNet` 介面相同 | §0, §5 |
| `flood_uncertainty/models/edl_terramind.py` | `EDL_TerraMind_ML4FloodsModel(EDL_ML4FloodsModel)`：只覆寫 `__init__`（建 `self.network`）與 `configure_optimizers`（backbone 較小 lr、跳過凍結參數） | §9 |
| `configurations/edl_terramind.json` | 以 `edl.json` 為底，加 `backbone_*` / `decoder_channels` 欄位 | §7 |
| `scripts/train/train_edl_terramind.py` | `train_edl.py` 逐字複製，只換 import、模型建構、預設 config | §8 |
| `scripts/smoke/smoke_train_edl_terramind.sh` | 包 `smoke_train.sh`；train split 缺 S2 時借一張 val 當 train（僅供管線檢查） | §8 |
| `tests/test_terramind_backbone.py` | band 對照、token→image 座標順序、pyramid stride、re-normalization 數值、`out_channels`/`downsample_ratios` 與實際輸出一致、freeze、離線 ckpt 不觸發 HF 下載 | F2, F3 |
| `tests/test_edl_terramind_smoke.py` | 隨機初始化 forward + backward、輸出形狀 `(B, 4, H, W)`、EDL 機率/不確定性範圍、optimizer 分組 | F4 |
| `docs/terramind_edl_integration_log.md` | 本文件 | — |

### 修改檔案（最小化）

| 檔案 | 修改 | 理由 |
|---|---|---|
| `scripts/smoke/smoke_train.sh` | `case` 新增 `EDL_TERRAMIND)` 分支（4 行）與註解 | 計畫 §8 指定沿用既有 dispatch |
| `flood_uncertainty/inference/infer.py` | `load_model` 新增 `model_type == "EDL-TERRAMIND"` 分支（+import） | 讓 README 4.3 的推論入口能載入 TerraMind checkpoint |
| `scripts/inference/run_inference.py` | `--model_type` 加 `EDL-TERRAMIND`、新增 `--config_edl_terramind`、`used_EDL` 判斷涵蓋新型別 | 同上 |
| `scripts/eval/eval_metrics.py` | `--model_type` 加 `EDL-TERRAMIND`、`MODEL_CONFIG_PATHS` 加對應 config | README 4.4 評估入口 |
| `flood_uncertainty/metrics/segmentation.py` | `plot_spatial_confusion_matrix` 的 band 分派把 `EDL-TERRAMIND` 視同 `EDL`（1 行） | 輸出 band 佈局與 EDL 完全相同 |
| `scripts/smoke/smoke_infer_eval.sh` | `case` 新增 `EDL-TERRAMIND)`；新增可選 `SMOKE_INFER_CHECKPOINT` 傳給 `--checkpoint` | 對齊 smoke 慣例 |
| `README.md` | 各段補上 EDL-TERRAMIND 的訓練 / 推論 / 評估 / smoke 指令，新增 §7 簡介 | 文件同步 |

### 不動的檔案

`pyproject.toml`、`uv.lock`（**未新增任何依賴**，見 §3.7）、`flood_uncertainty/models/edl.py`、`losses/`、`metrics/`、`utils/config_loader.py`、
`ml4floods/**`、`scripts/train/train_edl.py` 及其他三支訓練腳本、四份既有 config。

---

## 3. 設計決策與理由

### 3.1 為何用「子類別覆寫 `__init__`」而不是改 `configure_architecture`

`configure_architecture` 在 `ml4floods/models/worldfloods_model.py`，屬於 vendored 上游程式碼；
在那裡加 `elif architecture == 'edl_terramind'` 會把 FM 依賴塞進 ml4floods。專案內已有先例
`EDL_SAR_Unet` 用同樣方式（重寫 `__init__`、沿用其餘方法），故照做。代價是 `__init__` 有一段與父類重複；
好處是 `edl.py` 零改動、EDL 的 loss / metrics / zarr / wandb 邏輯完全共用。

### 3.2 為何 `TerraMindUNet.forward` 收 Tensor 而非 dict

父類的 `training_step` / `validation_step` / `forward` 都是 `self.network(x)`，x 是 `(B, C, H, W)`。
讓 `TerraMindUNet` 維持 Tensor 介面，dict 翻譯（`{"S2L2A": x}`）發生在它內部這一行，
就不用碰父類任何一行。這正是計畫 §5「翻譯只有一行」。

### 3.3 輸入 re-normalization（`backbone_renormalize_input`, 預設 `true`）

ml4floods dataloader 已用 `SENTINEL2_NORMALIZATION`（WorldFloods 統計）做 z-score；
TerraMind 預訓時用的是自己的 per-band mean/std（terratorch `v1_pretraining_mean/std`）。
兩者尺度接近但不同。`InputRenormalizer` 先還原 ml4floods z-score再套 TerraMind 統計，
讓預訓 patch embedding 看到它熟悉的分佈。統計值複製進本檔（避免依賴 terratorch 私有符號），
並以單元測試對照手算結果。關掉此開關即可做消融（計畫風險 #5b）。

### 3.4 波段子集

`bgriswirs` = B2, B3, B4, B8, B11, B12 → TerraMind `BLUE, GREEN, RED, NIR_BROAD, SWIR_1, SWIR_2`。
透過 terratorch factory 的 `bands=` 參數，從 12 波段預訓 patch embedding 抽出對應 6 個權重列，
不需自行動手切權重。若 config 含 B10（L2A 沒有），建構時直接報錯。

### 3.5 Pyramid 與 decoder 寬度

TerraMind 是 ViT-B/16：12 層、全部 stride 16、768 維。取第 `[2, 5, 8, 11]` 層（0-based），
經 `LearnedInterpolateToPyramidal` 變成 stride `[4, 8, 16, 32]`，再用 1×1 lateral 壓到
`[64, 128, 256, 512]`——與 ml4floods `UNet` 的 skip 寬度相同，decoder 容量對齊 baseline。
CNN UNet 的第一層 skip 在 stride 1，ViT 版最淺只到 stride 4，故 decoder 尾端多一段
bilinear ×4 + `double_conv(64, 64)` 回到全解析度。

### 3.5b 任意尺寸輸入（推論 tiling）

ml4floods 的推論 tiling 只對 `SUBSAMPLE_MODULE` 裡的 `unet` / `unet_dropout` 做 8 的倍數 padding，
不認識新 model_type，所以整張影像切出的 tile 尺寸不保證是 16 的倍數。`TerraMindUNet.forward`
自己以 reflection pad 補到 patch grid，輸出再裁回原尺寸；decoder 的上採樣改成對齊 skip 的實際尺寸
（stride-32 層是 max-pool，奇數 grid 會取 floor，×2 對不上）。有單元測試 `45×70 → 45×70`。
`ml4floods/**` 不動。

### 3.6 Optimizer

Adam 兩個 param group：decoder/head 用 `lr`，encoder 用 `lr * backbone_lr_mult`（預設 0.1）。
`backbone_freeze=true` 時 encoder 參數不進 optimizer，且 `train()` 時 encoder 維持 eval mode。
scheduler / monitor 與父類相同（`ReduceLROnPlateau` on `val_bce_land_water`）。

### 3.7 依賴：為何最終 **vendor** TerraMind 而不是 `uv add terratorch`

計畫 §6 指定「terratorch 加為 uv 依賴、只呼叫其 backbone factory」，並在風險 #1 要求動手前 dry-run。實測結果：

| 嘗試 | 結果 |
|---|---|
| `uv pip install --dry-run terratorch` | 解到 terratorch 1.2.11；需 numpy 1.26.4 → 2.2.6、safetensors 0.7 → 0.8，另加 35 個套件（torchgeo、lightning、diffusers、lightly、tensorboard…）。torch / pytorch-lightning / timm 不動 |
| 改 pin 後 `uv add terratorch==1.2.11` + `uv sync` | 安裝成功，但 `import terratorch` 失敗：`AttributeError: ResNet50_Weights.SENTINEL2_ALL_SOFTCON`。原因是它拉到 torchgeo 0.6.2（Python 3.10 能裝的最高版），而程式碼需要 torchgeo ≥ 0.7 |
| 查 PyPI 各版本需求 | terratorch 1.0（無 TerraMind）以外，所有含 TerraMind 的版本（1.0.1 ~ 1.2.13）都要求 torchgeo ≥ 0.7.0；torchgeo 0.7+ 要求 **Python ≥ 3.11**，專案 `requires-python = ">=3.10,<3.11"` |

因此在不升 Python 的前提下，terratorch 路線無解。兩個選項：

1. 把專案升到 Python 3.11 並重解 180 個 pin —— 影響所有既有管線，違反「不改專案結構」與計畫 §4「既有管線零風險」。
2. **Vendor** TerraMind encoder 原始碼（約 1.6k 行，Apache-2.0）到 `backbones/terramind_vendored/`，權重用專案既有的 `huggingface_hub` 下載/讀 cache。

採用 2。依賴檔完全不動（先前的 numpy/safetensors 改動與 `uv.lock` 已 `git checkout` 還原，環境 `uv sync` 回鎖定狀態）。
`terramind_backbone._build_terramind_encoder` 是唯一碰 factory 的地方，日後若專案升到 3.11 想改回 terratorch，只需改這一個函式。

Vendor 內容與上游差異：`terramind_vit.py` 拿掉 tokenizer（只有 LULC/NDVI 等 tokenized modality 才用）；`modality_info.py` 只留 untokenized 影像 modality；
其餘逐字。每個檔案開頭有 provenance 註解，改動處以 `# [vendored]` 標記。

---

## 4. 與計畫的差異

| 計畫 | 實作 | 原因 |
|---|---|---|
| 首例是 `FMBaselineSegModel`（單任務、CE loss） | 首例直接接 **EDL**（雙任務、EDL loss） | 使用者需求：以 `train_edl.py` 範式訓練測試 |
| `out_channels` 範例 `[96, 192, 384, 768]` | 實際 `[768, 768, 768, 768]` | 範例是 Swin 型骨幹；TerraMind 是 plain ViT，各層等寬 |
| 未提 re-normalization | 新增 `backbone_renormalize_input` | 見 §3.3 |
| `smoke_train_fm.sh` | `smoke_train_edl_terramind.sh` | 命名對齊實際 model type；額外處理 train split 缺 S2 的資料副本 |
| terratorch 為 uv 依賴 | vendor encoder 原始碼，零新依賴 | Python 3.10 與所有含 TerraMind 的 terratorch 版本不相容（§3.7） |

---

## 5. 驗證紀錄

| # | 項目 | 指令 | 結果 |
|---|---|---|---|
| 1 | 語法 | `python -m py_compile` 全部新檔 | 通過 |
| 2 | 既有模組在鎖定環境可 import | `uv run python -c "import ml4floods.models.dataset_setup, flood_uncertainty.models.edl"` | 通過 |
| 3 | 單元測試（含既有 `test_edl_annealing.py`） | `uv run --with pytest python -m pytest tests/ -q` | **21 passed**（helper 9、backbone 7、EDL smoke 5） |
| 4 | 真實權重載入（HF cache，`HF_HUB_OFFLINE=1`） | 見下 | encoder 113/113 個 key 來自 checkpoint；`encoder.0.attn.qkv.weight` 與 ckpt 完全相等；patch embedding 6 波段子集 ＝ ckpt 第 `[1,2,3,7,10,11]` 欄；256×256 輸入 → pyramid `(768,64²),(768,32²),(768,16²),(768,8²)`；可訓練參數 93.2M |
| 5 | Smoke train（GPU, 1 epoch, 1 tile/split, batch 2） | `SMOKE_TRAIN_GPUS=0 SMOKE_TRAIN_BATCH_SIZE=2 bash scripts/smoke/smoke_train_edl_terramind.sh` | **SMOKE TRAIN PASS**：600 steps / 32 s（19 it/s，RTX 3080，顯存 2.6 GB）；`epoch=0-step=600.ckpt` 與 `last.ckpt` 寫入 `artifacts/models/smoke/edl_terramind_v1_base_bgriswirs_smoke_20260928_140050/`；1 tile 訓練後 land/water recall 1.0、precision 0.018（僅管線檢查） |
| 6 | `--mode validate_only`（用第 5 項的 `last.ckpt` 當 `pretrained_path`） | `uv run python scripts/train/train_edl_terramind.py --config <smoke cfg> --mode validate_only --data_root artifacts/smoke/train/data_min` | 權重 154/154 載入、validation 跑完；**zarr 寫出失敗**：`ModuleNotFoundError: No module named 'zarr'`。`zarr` 不在 `pyproject.toml`，`train_edl.py` 走同一段繼承程式碼會一樣失敗 → 既有問題，非本次引入（見 §6） |
| 8 | 推論 + 評估 smoke（README 4.3/4.4 流程，用第 5 項的 `last.ckpt`） | `SMOKE_INFER_MODEL_TYPE=EDL-TERRAMIND SMOKE_INFER_CHECKPOINT=<last.ckpt> SMOKE_INFER_DATA_ROOT=artifacts/smoke/train/data_min bash scripts/smoke/smoke_infer_eval.sh` | **SMOKE PASS**：整張 val tile 推論（tile 非 16 倍數，由 §3.5b 的 padding 處理）、`*_output_EDL-TERRAMIND.tif` 與 `metrics_EDL-TERRAMIND.csv` 寫出。第一次跑時分別卡在 tiling 尺寸與 `segmentation.py` 的 model_type 分派，修正後通過 |
| 9 | 既有 EDL 推論 + 評估回歸（改過的 infer/eval 腳本） | `SMOKE_INFER_MODEL_TYPE=EDL ...` | SMOKE PASS |
| 7 | 既有 EDL 管線回歸（同環境、同 mini dataset） | `uv run python scripts/train/train_edl.py --config <edl smoke cfg, pretrained_path=null> --mode train --data_root artifacts/smoke/train/data_min` | 通過：600 steps / 22 s，exit 0。既有 UNet-EDL 管線在同一環境、同一 mini dataset 下正常（環境與依賴未變，屬預期） |

第 4 項腳本：建 `TerraMindBackbone(channel_configuration="bgriswirs", pretrained=True)`，
比對 `hf_hub_download("ibm-esa-geospatial/TerraMind-1.0-base", "TerraMind_v1_base.pt")` 的 state dict。


---

## 6. 已知問題與後續

| 項目 | 說明 | 建議 |
|---|---|---|
| `zarr` 套件缺失 | `validate_only` 模式的 zarr dump（`EDL_ML4FloodsModel._append_batch_to_zarr`，TerraMind 版繼承同一段）需要 `zarr`，但它不在 `pyproject.toml`。訓練與驗證指標不受影響 | `uv add zarr`（獨立於本次變更，需另行決定版本） |
| `configurations/edl.json` 路徑 | `pretrained_path` 與 `data_params` 指向 `chlunchen-10030`，本機不存在；`SMOKE_TRAIN_MODEL=EDL` 直接跑會 `FileNotFoundError` | 屬既有 config 的機器綁定問題；本次未改動它。baseline UNet 權重在 `/home/NAS/homes/cjchen-10025/flood-uncertainty/artifacts/models/WF2_unetv2_bgriswirs/model.pt` 可用 |
| 完整訓練尚未執行 | 本次只做到里程碑 F5（smoke）；F6（全量訓練、與 EDL-UNet baseline 比 mIoU / 校準）需 GPU 時數 | `uv run python scripts/train/train_edl_terramind.py --mode train`（config 預設 30 epoch、batch 16）。建議先試 `backbone_freeze: true` 與 `false` 各一組 |
| `batch_size` / 精度 | ViT-B/16 在 256² 用 fp32、batch 16 預估 < 10 GB；若 OOM，降 batch 或在 Trainer 加 `precision="bf16-mixed"`（腳本刻意與 `train_edl.py` 同步，未加此參數） | 視實際顯存調整 config |
| `scripts/analysis/*` 未接 | analysis 腳本（`analysis_S2.py`、`compute_pavpu.py`、各 `plot_*_compare.py`）每支都有自己的 model_type 字典（名稱、顏色、檔名後綴、uncertainty band 清單），未加入 `EDL-TERRAMIND` | 有完整訓練結果要做比較圖時再逐支加；輸出 band 與 EDL 相同，可直接沿用 EDL 的 band 清單 |
| `BackboneProtocol` | 仍照計畫 §3.2 暫不建立；`TerraMindBackbone` 已符合其形狀 | 有第二顆骨幹時再補 |

### 如何執行

```bash
# 單元測試
uv run --with pytest python -m pytest tests/ -q

# Smoke（1 epoch、1 tile/split）
SMOKE_TRAIN_GPUS=0 bash scripts/smoke/smoke_train_edl_terramind.sh
# 或走既有 dispatch（需 source root 有 train/S2）
SMOKE_TRAIN_MODEL=EDL_TERRAMIND SMOKE_TRAIN_SOURCE_DATA_ROOT=/home/NAS/house/ycchen-10014/data/worldfloods_v2 bash scripts/smoke/smoke_train.sh

# 完整訓練 / 驗證
uv run python scripts/train/train_edl_terramind.py --config configurations/edl_terramind.json --mode train
uv run python scripts/train/train_edl_terramind.py --config configurations/edl_terramind.json --mode validate_only
```
