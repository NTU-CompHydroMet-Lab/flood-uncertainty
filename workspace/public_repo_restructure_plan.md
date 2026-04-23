# Public Repo Restructure Plan

## Part A. 主計畫（短版）

此區聚焦「決策與執行順序」；檔案映射與細節放在 Part B。

### A.1 v1 公開版範圍（明確邊界）

`Must include`（第一版一定包含）：

- patched `ml4floods/`（維持現有 import 相容）
- `mlguess/`（目前 uncertainty loss 與部分訓練流程依賴）
- `flood_uncertainty/` 核心方法（EDL、dropout、ensemble、metrics）
- `scripts/` 的 train / infer / eval 入口
- `configurations/`（每個方法一份，內含 `shared/train/infer/validate_only` mode 區段）
- `docs/`（method summary、patch notes）

`Must exclude`（第一版一定排除）：

- checkpoints / weights / 大量推論結果
- dataset 本體與本機環境檔
- `wandb/`、`.venv/`、`.zarr`、大量 `.tif` / `.pt` / `.ckpt`
- 備份稿、過時 notebook、臨時測試圖

`Defer`（延後決策）：

- `KuroSiwo/`
- `CJ_scripts/SAR/`
- `report/meta/`
- `report/test_*`

### A.2 命名規範（含 `matrics -> metrics` 過渡）

- 對外文件、目錄與新模組一律使用 `metrics`。
- 既有檔名 `matrics` 視為 legacy 名稱，只在過渡期保留。
- Python package 名稱維持 `flood_uncertainty`，不改成 `flooduncertainty`。
- 過渡策略：
  1. 新增與搬移後的 library code 一律放到 `flood_uncertainty/metrics/`。
  2. 若舊入口仍引用 `matrics`，先保留相容層（薄轉發），再逐步替換 import。
  3. 完成條件：repo 內不再新增 `matrics*` 命名，既有引用全部轉到 `metrics*`。

### A.3 近期進度（2026-04-23）

- `done`：`configurations/{v2,edl,dropout,ensemble}.json` 已整併為單檔多 mode（`shared/train/infer/validate_only`）。
- `done`：`scripts/` 入口已從硬編碼路徑改為「config 為主 + CLI 可覆寫」。
- `done`：推論與評估預設輸出改到 `artifacts/results/`，避免寫入受 Git 管控區域。
- `done`：已完成 v2/EDL infer smoke、EDL validate_only smoke、以及小規模 infer->eval e2e 驗證。
- `done`：新增 `scripts/smoke/smoke_infer_eval.sh` 一鍵 smoke（v2/EDL infer + eval）並已驗證可執行。
- `done`：新增 `scripts/smoke/smoke_train.sh`，會建立最小 smoke dataset、生成臨時 config，並驗證 train 入口至少可完整跑完 1 epoch。
- `done`：新增 `scripts/smoke/smoke_analysis.sh`，會準備最小 prediction artifacts（EDL）並驗證 retention / PAvPU analysis 流程。
- `done`：新增 `scripts/smoke/smoke_all.sh`，可串跑 infer/eval、train、analysis；已實測整包 smoke flow 可通過。
- `done`：`scripts/smoke/` 已補非 EDL analysis 的保護邏輯，避免 `ensemble` / `mcdropout` 被 `PREPARE_INFER=1` 的預設值誤傷。
- `done`：`scripts/smoke/smoke_all.sh` 已修正 elapsed time 顯示 bug，完成時間會正確顯示秒數。
- `done`：analysis 相關腳本已收斂到 `scripts/analysis/`，輸出路徑統一到 `artifacts/results/*`，並補齊多數 CLI 參數（含 `--max-files`）。
- `done`：`plot_epistemic_fp_fn_compare.py` 已完成最小驗證並成功產圖。

## 1. 文件目的

本文件用來規劃目前 `ML4FloodsUncertainty` 專案未來整理成公開 Git repository 的方向。

本次規劃的目標不是立即重構所有程式，而是先把以下事情定清楚：

- 專案公開後要呈現的主題是什麼
- 哪些內容應該保留
- 哪些內容應該留在本機、不進公開版 repo
- 現有檔案未來應該移動到哪個位置
- 哪些部分屬於 `ml4floods` 的 bugfix，哪些部分屬於自己的 uncertainty extension
- 後續若要往 package / PyPI 方向走，架構應如何設計

本文件偏向規劃文件，而不是立即執行的重構腳本。

## 2. 專案定位

### 2.1 建議公開主題

此專案公開後的主題，建議定位為：

`flood_uncertainty`: 以 flood segmentation uncertainty 為主題的研究與實作專案，涵蓋：

- deterministic baseline (`v2`)
- evidential deep learning (`EDL`)
- deep ensemble
- MC dropout

### 2.2 與 `ml4floods` / `mlguess` 的關係

目前專案不能單純把 `ml4floods` 當外部依賴，因為本地的 `ml4floods/` 已包含實際使用中的 bugfix 與修改。

同時，`mlguess/` 在目前程式中也有實際 import（例如 uncertainty loss 相關邏輯），因此第一版公開時也應視為本地依賴一併保留。

因此目前比較合理的定位是：

- `ml4floods/` 保留在 repo 中
- `mlguess/` 保留在 repo 中
- 將其視為 patched fork / local dependency
- 自己新增的方法與分析邏輯另外整理成 `flood_uncertainty/`

這樣可以區分兩種責任：

- `ml4floods/`: 修過 bug、保留相容性的基礎框架
- `mlguess/`: uncertainty 相關共用元件與 loss / torch utilities
- `flood_uncertainty/`: 自己新增的 uncertainty methods、loss、metrics、inference、分析工具

### 2.3 名稱建議

目前建議名稱如下：

- GitHub repo name: `flood-uncertainty`
- Python package name: `flood_uncertainty`
- 不採用 package 名稱：`flooduncertainty`

暫時不建議把 repo 名稱或 package 名稱包含 `ml4floods`，因為希望公開主題更聚焦在自己的 uncertainty extension，同時避免和上游專案命名混淆。

## 3. 公開版整理原則

### 3.1 要保留的主體

公開版 repo 應以「可理解、可維護、可逐步重現」為優先，主體應保留：

- 修過 bug 的 `ml4floods/`
- 目前依賴的 `mlguess/`
- 自己的 uncertainty models / losses / metrics / inference
- train / inference / evaluation 入口 scripts
- configs
- analysis scripts
- method documentation

### 3.2 不需要放入公開主體的內容

下列內容通常不應放入公開主體：

- 訓練權重與 checkpoints
- 大量推論結果
- wandb logs
- 臨時筆記、備份檔、過時 notebook
- 本機環境檔、快取檔、資料集本體

### 3.3 目標不是一次改完

本次規劃建議採分階段方式整理：

- Phase 1: 先定義結構與檔案歸屬
- Phase 2: 再把自己的方法搬到新模組
- Phase 3: 再清理 scripts 與 configs
- Phase 4: 再補齊 README 與 docs
- Phase 5: 最後做 package 化驗證，才評估是否發佈到 PyPI

## 4. 建議目錄結構

以下是目前最適合此專案的整理方向：

```text
repo/
  README.md
  pyproject.toml
  environment.yml
  .gitignore

  ml4floods/                     # 保留你修過 bug 的 fork
  mlguess/                       # 目前流程依賴的本地套件

  flood_uncertainty/             # 你的主 package
    __init__.py
    models/
      __init__.py
      deterministic.py
      edl.py
      dropout.py
      ensemble.py
    losses/
      __init__.py
      edl_loss.py
    metrics/
      __init__.py
      segmentation.py
      retention.py
      pavpu.py
    inference/
      __init__.py
      infer.py
    utils/
      __init__.py
      io.py
      paths.py

  scripts/
    train/
      train_v2.py
      train_edl.py
      train_dropout.py
      train_ensemble.py
    inference/
      run_inference.py
      run_inference_ensemble.py
    eval/
      eval_metrics.py
    analysis/
      analysis_S2.py
      case_compare_shared.py
      compute_pavpu.py
      export_case_compare_tiles.py
      plot_epistemic_fp_fn_compare.py
      plot_worldfloods_event_map.py
      plot_case_confusion_compare.py
      plot_case_uncertainty_compare.py
      plot_retention_curve_compare.py
    smoke/
      smoke_infer_eval.sh
      smoke_train.sh
      smoke_analysis.sh
      smoke_all.sh

  configurations/
    v2.json
    edl.json
    dropout.json
    ensemble.json

  docs/
    method_summary.md
    ml4floods_patch_notes.md

  tests/
```

## 5. 各資料夾職責

### 5.1 `ml4floods/`

用途：

- 保留目前實際使用的 patched fork
- 放與上游相容但已修過 bug 的原始框架

原則：

- 先不要大幅改名或重構
- 不要再把新的 uncertainty-specific 程式持續塞進 `ml4floods/`
- 所有對上游的修補，未來應寫進 `docs/ml4floods_patch_notes.md`

### 5.2 `flood_uncertainty/`

用途：

- 放自己的研究貢獻與新增功能
- 作為未來 package 化的主要命名空間

應包含：

- EDL 相關模型封裝
- dropout / ensemble 相關方法邏輯
- uncertainty loss
- segmentation / retention / PAvPU metrics
- 推論流程中的共用邏輯

### 5.3 `scripts/`

用途：

- 放 train / inference / evaluation 的入口檔

原則：

- 只負責解析參數、讀 config、呼叫對應模組
- 不要在這裡堆積大量核心邏輯
- 盡量保持薄

### 5.4 `configurations/`

用途：

- 集中管理訓練、推論、評估設定

原則：

- 每個方法維持單一 config（例如 `edl.json`），檔內分 `shared`、`train`、`infer`、`validate_only` 區段
- `eval` 可獨立成 `metrics_eval.json`（或 `eval_metrics.json`），避免和 train/infer 混雜
- 檔名應一致且可預測
- 盡量不要再保留與本機綁死的絕對路徑

### 5.5 `analysis/`

用途：

- 放研究分析與視覺化腳本
- 目前實作位置以 `scripts/analysis/` 為主；若未來再拆 library code，才考慮把純研究腳本和可重用邏輯分開

原則：

- 和 package 主邏輯分開
- 方便保留研究性質較強的圖表分析流程
- 不要和 `scripts/` 混在一起

### 5.6 `docs/`

用途：

- 放方法說明與補充文件

建議至少包含：

- `method_summary.md`
- `ml4floods_patch_notes.md`

### 5.7 `tests/`

用途：

- 放最基本的可執行檢查

早期至少可包含：

- config 載入測試
- model 建構 smoke test
- metric 計算 smoke test

## Part B. 附錄（檔案映射與細節）

## 6. 現有檔案對照表

以下是目前較重要檔案，未來建議的新位置。

欄位說明：

- `Priority`: `P0`（第一批必要）/ `P1`（第二批）/ `P2`（可延後）
- `Status`: `todo` / `in_progress` / `done` / `defer`

### 6.1 核心方法與模型

| Current path | Suggested target | Notes | Priority | Status |
| --- | --- | --- | --- | --- |
| `CJ_scripts/model.py` | `flood_uncertainty/models/` | `defer`：目前僅 EDL 走自訂模型；v2/dropout/ensemble 仍使用 `ml4floods` 的 `get_model` | `P0` | `defer` |
| `CJ_scripts/losses_uncertainty.py` | `flood_uncertainty/losses/edl_loss.py` | 保留 EDL loss 主體 | `P0` | `done` |
| `CJ_scripts/matrics.py` | `flood_uncertainty/metrics/segmentation.py` | 依命名規範轉為 `metrics` | `P0` | `done` |
| `CJ_scripts/calculate_matrics.py` | `scripts/eval_metrics.py` 或 `flood_uncertainty/metrics/retention.py` + `scripts/eval_metrics.py` | 視內容拆成 library code + CLI script | `P0` | `done` |

### 6.2 訓練與推論入口

| Current path | Suggested target | Notes | Priority | Status |
| --- | --- | --- | --- | --- |
| `CJ_scripts/train_v2_model.py` | `scripts/train/train_v2.py` | 入口 script | `P0` | `done` |
| `CJ_scripts/train_v2_model_uncertainty.py` | `scripts/train/train_edl.py` | 入口 script | `P0` | `done` |
| `CJ_scripts/train_v2_model_dropout.py` | `scripts/train/train_dropout.py` | 入口 script | `P0` | `done` |
| `CJ_scripts/train_v2_model_ensemble.py` | `scripts/train/train_ensemble.py` | 入口 script | `P0` | `done` |
| `CJ_scripts/run_inference.py` | `scripts/inference/run_inference.py` | 一般 deterministic / EDL 推論入口 | `P0` | `done` |
| `CJ_scripts/run_inference_ensemble.py` | `scripts/inference/run_inference_ensemble.py` | 處理 ensemble / MC dropout 的入口 | `P0` | `done` |
| `n/a` | `scripts/smoke/smoke_infer_eval.sh` | infer + eval 最小 smoke 入口 | `P0` | `done` |
| `n/a` | `scripts/smoke/smoke_train.sh` | train 最小 smoke 入口 | `P0` | `done` |
| `n/a` | `scripts/smoke/smoke_analysis.sh` | analysis 最小 smoke 入口 | `P0` | `done` |
| `n/a` | `scripts/smoke/smoke_all.sh` | 串跑 infer/eval + train + analysis 的 smoke 總入口 | `P0` | `done` |

### 6.3 Configs

| Current path | Suggested target | Priority | Status |
| --- | --- | --- | --- |
| `CJ_scripts/configurations/hf_hub_download.json` | `configurations/v2.json`（整併為單檔，含 `shared/train/infer/validate_only`） | `P0` | `done` |
| `CJ_scripts/configurations/hf_hub_download_uncertainty.json` | `configurations/edl.json`（整併為單檔，含 `shared/train/infer/validate_only`） | `P0` | `done` |
| `CJ_scripts/configurations/hf_hub_download_dropout.json` | `configurations/dropout.json`（整併為單檔，含 `shared/train/infer/validate_only`） | `P0` | `done` |
| `CJ_scripts/configurations/hf_hub_download_ensemble.json` | `configurations/ensemble.json`（整併為單檔，含 `shared/train/infer/validate_only`） | `P0` | `done` |
| `CJ_scripts/configurations/hf_hub_download_uncertainty_infer.json` | `configurations/edl.json`（併入 `infer` 區段） | `P0` | `done` |
| `CJ_scripts/configurations/hf_hub_download_dropout_infer.json` | `configurations/dropout.json`（併入 `infer` 區段） | `P0` | `done` |
| `CJ_scripts/configurations/hf_hub_download_ensemble_infer.json` | `configurations/ensemble.json`（併入 `infer` 區段） | `P0` | `done` |

### 6.4 Analysis / Plotting

| Current path | Suggested target | Priority | Status |
| --- | --- | --- | --- |
| `CJ_scripts/plot/plot_case_confusion_compare.py` | `scripts/analysis/plot_case_confusion_compare.py` | `P1` | `done` |
| `CJ_scripts/plot/plot_case_uncertainty_compare.py` | `scripts/analysis/plot_case_uncertainty_compare.py` | `P1` | `done` |
| `CJ_scripts/plot/plot_retention_curve_compare.py` | `scripts/analysis/plot_retention_curve_compare.py` | `P1` | `done` |
| `CJ_scripts/plot/export_case_compare_tiles.py` | `scripts/analysis/export_case_compare_tiles.py` | `P1` | `done` |
| `CJ_scripts/plot/case_compare_shared.py` | `scripts/analysis/case_compare_shared.py` | `P1` | `done` |
| `CJ_scripts/EDA/compute_pavpu.py` | `scripts/analysis/compute_pavpu.py` 或拆一部分到 `flood_uncertainty/metrics/pavpu.py` | `P1` | `done` |
| `CJ_scripts/EDA/analysis.py` | `scripts/analysis/analysis.py` | `P2` | `defer` |
| `CJ_scripts/EDA/analysis_S2.py` | `scripts/analysis/analysis_S2.py` | `P2` | `done` |
| `CJ_scripts/EDA/plot_epistemic_fp_fn_compare.py` | `scripts/analysis/plot_epistemic_fp_fn_compare.py` | `P2` | `done` |

### 6.5 文件

| Current path | Suggested target | Priority | Status |
| --- | --- | --- | --- |
| `report/model_method_settings_summary.md` | `docs/method_summary.md` | `P1` | `todo` |
| `CJ_scripts/markdown/PAvPU.md` | `docs/pavpu_notes.md` | `P1` | `todo` |
| `CJ_scripts/markdown/ensemble_uncertainty.md` | `docs/ensemble_uncertainty.md` | `P1` | `todo` |
| `CJ_scripts/markdown/matrics_of_uncertainty.md` | `docs/metrics_of_uncertainty.md` | `P1` | `todo` |

## 7. 建議保留但不放入公開主體的內容

以下內容可以保留在本機，但不建議作為公開 repo 的主體：

### 7.1 本機保留、通常不進 Git

- `result/`
- `models/`
- `wandb/`
- `.venv/`
- dataset 本體
- checkpoints / weights
- `.zarr`
- 大量 `.tif`、`.pt`、`.ckpt`

### 7.2 研究工作區可保留，但不建議作為公開版主體

- `BAK/`
- `CJ_scripts/BAK/`
- `CJ_scripts/load_dataset/*.ipynb`
- `CJ_scripts/EDA/*.ipynb`
- 臨時測試圖
- 已過時的 notebook 與備份稿

### 7.3 視主題再決定是否公開

- `KuroSiwo/`
- `CJ_scripts/SAR/`
- `report/meta/`
- `report/test_*`

這些是否要保留在公開版，需要看未來 repo 是否要同時涵蓋 SAR 與其他額外分支研究。

如果公開主題先聚焦在 `WorldFloods v2 + optical flood uncertainty`，則上述內容可以先不進主體。

## Part A（續）. 執行策略

## 8. `ml4floods/` 的處理策略

### 8.1 目前建議

目前不建議立即改名成 `ml4flood_fixed/`，理由如下：

- 現有 import 路徑可能大量依賴 `ml4floods`
- 改名會引入額外重構成本
- 目前主要目標是先把公開版結構釐清，而不是先做大範圍 rename

因此短期建議是：

- 先保留 `ml4floods/` 原名稱
- 在文件中清楚註明這是 local patched fork

### 8.2 文件補充建議

未來應新增：

- `docs/ml4floods_patch_notes.md`

內容應至少紀錄：

- 修改了哪些檔案
- 修了哪些 bug
- 修改原因
- 是否和 uncertainty method 直接相關

## 9. package / PyPI 方向

### 9.1 目前可以往 package 化設計

如果後續要讓別人能：

```python
import flood_uncertainty
```

則目前結構是合理方向，但需要補齊：

- `pyproject.toml`
- `flood_uncertainty/__init__.py`
- 各子模組 `__init__.py`
- 將核心邏輯從 scripts 拆到 package 內

### 9.2 是否適合直接發 PyPI

目前暫時不建議立刻以現況發到 PyPI，原因如下：

- 專案還在重整中
- `ml4floods/` 仍是 local patched fork
- configs 與 scripts 尚有絕對路徑
- 公開 API 尚未穩定

比較合理的順序應是：

1. 先整理 repo
2. 先能在乾淨環境中 `pip install -e .`
3. 再決定是否上 TestPyPI
4. 最後才考慮正式發佈 PyPI

## 10. 分階段整理順序

### Phase 1: 定義邊界

目標：

- 確認 repo 主題
- 確認要保留的目錄
- 確認要排除的內容

產出：

- 本文件
- 保留 / 排除清單

完成判準（DoD）：

- `Must include / Must exclude / Defer` 清單已凍結在文件中
- 第一批搬移清單（`P0`）已完成 `priority/status` 標記
- 待排除的大檔類型與目錄已對齊 `.gitignore` 策略

### Phase 2: 把自己的方法模組化

目標：

- 將 `CJ_scripts` 中屬於自己方法的核心邏輯移到 `flood_uncertainty/`

優先處理：

- `model.py`
- `losses_uncertainty.py`
- `matrics.py`
- 與 retention / PAvPU 相關邏輯

完成判準（DoD）：

- `P0` 核心 library code 已搬到 `flood_uncertainty/`
- 核心 train / infer 流程不再直接依賴 `CJ_scripts/*` 的核心實作
- 新增程式不再使用 `matrics*` 命名

### Phase 3: 清理入口與設定

目標：

- 將 train / infer / eval scripts 收斂到 `scripts/`
- 將 configurations 重新命名為每方法單檔（內含 `shared/train/infer/validate_only` 區段）
- 移除硬編碼本機路徑

完成判準（DoD）：

- `scripts/` 入口可透過參數啟動，不含核心業務邏輯
- `configurations/` 已改為每方法單檔，且 train/infer 共用同一檔配置
- scripts 內無硬編碼資料根目錄；路徑由 config 為主，並可用 CLI 覆寫

### Phase 4: 文件化

目標：

- 重寫 `README.md`
- 將方法總結移到 `docs/`
- 補充 `ml4floods` patch notes

完成判準（DoD）：

- `README.md` 可說明安裝、訓練、推論、評估最小流程
- `docs/method_summary.md` 與 `docs/ml4floods_patch_notes.md` 已建立
- 重要方法與 patch 邊界有可追溯文件

### Phase 5: package 化與驗證

目標：

- 補 `pyproject.toml`
- 整理 import
- 建立最基本 tests
- 在乾淨環境測試安裝與執行

完成判準（DoD）：

- 乾淨環境可成功 `pip install -e .`
- 至少通過 smoke tests（config / model build / metric）
- package import 路徑穩定，`flood_uncertainty` 可直接匯入

## 11. 待確認問題

後續整理前，仍有幾個問題需要逐步確認：

1. `ml4floods/` 中實際改動過哪些檔案
2. `CJ_scripts/model.py` 先保留不拆（目前僅 EDL 使用自訂模型，其他沿用 `ml4floods`）
3. `matrics.py` 與 `calculate_matrics.py` 的邏輯邊界
4. `KuroSiwo/` 是否屬於公開主題的一部分
5. `SAR` 相關內容是否應納入第一版公開 repo
6. `report/meta/` 中哪些 metadata 值得保留
7. 哪些分析 notebook 需要轉成正式 script，哪些只保留本機

## 12. 目前最建議的下一步

在真正開始改檔前，建議先做以下確認：

1. 確認 `flood_uncertainty/` 是否就是最終 package 名稱
2. 確認 `KuroSiwo/` 與 `SAR/` 是否先排除在公開版之外
3. 確認 `docs/` 是否由 `report/` 內文件逐步搬過去
4. 確認第一批要搬的核心檔案清單

若上述確認完成，下一步即可進入：

- 建立目標目錄
- 移動第一批核心檔案
- 開始做 import 與 config 重整

---

本文件是規劃草案，重點在於先把架構、責任邊界與整理順序說清楚，再逐步執行實際重構。
