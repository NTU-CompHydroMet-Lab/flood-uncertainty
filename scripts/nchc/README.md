# 國網 Nano5（晶創25）訓練流程

設定檔中的 `/home/NAS/...` 路徑不需修改：`train.sbatch` 會設定 `FLOOD_PATH_REMAP`，由
`load_mode_config` 在載入時把實驗室路徑換成 Nano5 路徑（見 `flood_uncertainty/utils/config_loader.py`）。

| 腳本 | 執行位置 | 用途 |
|---|---|---|
| `env.sh` | 兩端 | 路徑與環境變數（被其他腳本 source） |
| `push.sh` | 實驗室 | 經資料傳輸節點（sftp, port 2222）上傳 code / data / TerraMind 權重 |
| `setup_env.sh` | Nano5 登入節點 | 安裝 uv，依 `uv.lock` 建立 `.venv`（Python 3.10） |
| `submit.sh` | Nano5 登入節點 | 送出 smoke test 或訓練鏈（依序執行，互不重疊） |
| `train.sbatch` / `smoke.sbatch` | Slurm | 單段訓練（自動從 `last.ckpt` 續訓）/ dev 分區連通性測試 |
| `pull.sh` | 實驗室 | 下載實驗結果到 `$LAB_REPO/artifacts` |

## 步驟

```bash
# 0. 實驗室端：填入帳號與計畫代號
cp scripts/nchc/nchc.env.example scripts/nchc/nchc.env   # 編輯 NCHC_USER / NCHC_ACCOUNT / NCHC_WORK

# 1. 上傳（雙因子驗證只需一次；data 約 93 GB，中斷後重跑同一指令即可續傳）
bash scripts/nchc/push.sh code
bash scripts/nchc/push.sh hf
bash scripts/nchc/push.sh data

# 2. Nano5 登入節點：nchc.env 不會隨 code 上傳，需在 Nano5 上另外建立
ssh <帳號>@nano5.nchc.org.tw
cd /work/<帳號>/flood-uncertainty
cp scripts/nchc/nchc.env.example scripts/nchc/nchc.env   # 同樣填寫
bash scripts/nchc/setup_env.sh

# 3. smoke test（dev 分區，約數分鐘）
bash scripts/nchc/submit.sh --smoke
cat artifacts/nchc_logs/flood-smoke-<jobid>.out           # 最後應出現 SMOKE TRAIN PASS

# 4. 正式訓練：依序執行，每個 config 2 段 × 48 h
bash scripts/nchc/submit.sh --chunks 2 configurations/terramind/tm_l1c13.json configurations/terramind/tm_l1c13_tim.json
squeue -u $USER

# 5. 實驗室端：取回結果並上傳 W&B
bash scripts/nchc/pull.sh edl_terramind_v1_base_s2l1c_all_nofreeze
wandb sync artifacts/models/edl_terramind_v1_base_s2l1c_all_nofreeze/wandb/offline-run-*
```

## 注意事項

- 計算節點假設無對外網路：`HF_HUB_OFFLINE`、`UV_OFFLINE`、`UV_NO_SYNC` 都已開啟，W&B 預設 offline。
- 訓練完成會寫入 `<experiment>/.nchc_done`，同一實驗後續的 chunk 會直接結束；要從頭重訓需刪除此檔與 `checkpoint/`。
- `windows_train_test_split_from_csv.json` 內含絕對路徑，不會上傳，首次訓練時會在 Nano5 重新產生。
- `push.sh code` 只新增或覆寫檔案，本機刪除的檔案不會同步刪除遠端。
- 只有 `ycchen-10014` 的 repo 與 data 路徑會被替換；指向其他成員目錄的 config（例如 KuroSiwo、
  `pretrained_path`）需先用 `push.sh dir` 上傳，並在 Nano5 的 `nchc.env` 設定 `NCHC_EXTRA_REMAP="舊路徑=新路徑;..."`。
- 未經實測、需在 Nano5 上確認的項目：工作區實際路徑（`/work` 或 `/work1`）、每張 GPU 可申請的
  CPU 數（預設 16）、資料傳輸節點是否允許 `ssh -fN` 建立 ControlMaster 連線。
