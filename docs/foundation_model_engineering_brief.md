# Foundation Model 工程實踐 Brief

> **文件定位**：1–2 頁可讀完的工程 TL;DR。給想在 3 分鐘內抓到 TerraMind 引入
> 關鍵決策、介面、風險與里程碑的讀者。
>
> 完整決策脈絡見 [`foundation_model_integration_plan.md`](./foundation_model_integration_plan.md)；
> 新元件放在專案哪個位置的流程解說見
> [`foundation_model_integration_walkthrough.md`](./foundation_model_integration_walkthrough.md)。

---

## 核心設計原則

**只借 backbone factory，不套框架全家桶**——`terratorch` 加為 uv 依賴，但**僅呼叫其 backbone factory 一個函式**，不繼承 Task、不用 registry、不用 YAML CLI。保留專案原有 `pl.LightningModule` + JSON config + script 慣例。

---

## 介面契約（唯一對外承諾）

```python
class BackboneProtocol(Protocol):
    def __call__(self, image_dict: dict[str, Tensor]) -> list[Tensor]: ...
    out_channels: list[int]         # pyramid 每層通道數
    downsample_ratios: list[int]    # 每層相對原圖下採樣倍數
```

- 輸入 dict：讓多模態（S2L2A、S1GRD、未來其他）自然擴展
- 輸出 pyramid：讓下游 decoder 用 `out_channels` 自動配置
- 屬性揭露：下游不需硬編通道數

**上下游透過此契約解耦**——換 TerraMind 為 Prithvi / SatMAE，只動一個檔案。

---

## 資料流「翻譯」只有一行

既有 `WorldFloodsDataset` 吐 `(B, C, H, W)`，TerraMind 要 dict。翻譯集中在 `FMBaselineSegModel.forward()`：

```python
image_dict = {"S2L2A": batch["image"]}   # ← 唯一翻譯層
features   = self.backbone(image_dict)
```

`WorldFloodsDataset`、`pl.Trainer`、config schema、`utils/config_loader` 全部**零改動**。

---

## 工程風險與對應

| 風險 | 對應 |
|---|---|
| 顯存壓力（尤其 BEF 雙塔） | `bf16-mixed`、小 batch、`freeze=True`（只訓 decoder）、gradient checkpointing |
| 無網路節點下載失敗 | `backbone_local_ckpt` 接口在 F2 就定義，走本地 `.ckpt` 跳過 HuggingFace |
| `terratorch` 版本衝突 | F1 前先 `uv add --dry-run` 檢查；必要時鎖 `lightning` 版本 |
| Token → image reshape 順序錯 | F2 dummy input 測試檢查空間 anchor（左上像素 ↔ 左上 token） |
| Baseline mIoU 明顯低於 v2 | 檢查三件事：波段順序、正規化統計、`select_layers` 是否合理 |

---

## 里程碑（F1–F6）

```
F1 依賴解析 ─▶ F2 骨幹包裝 ─▶ F3 離線權重接口 ─▶ F4 baseline 模型 smoke
                                                          │
                                                          ▼
                                                F5 baseline train smoke ─▶ F6 完整訓練驗證
```

- **F1–F4**：純接線工作，無需 GPU
- **F5–F6**：訓練實境驗證
- 工作量：約 2.5–3 天 + F6 訓練時間

---

## 一句話總結

TerraMind 進來只多出「**一顆骨幹、一支 script、一份 config**」，其他一律不動。這讓引入的**風險面積**壓到最小，並保留未來替換基礎模型的最大彈性。

---

> **實務原則提醒**：工程實踐的關鍵不是「引入 TerraMind」，而是**引入時只切最小介面**。實務上常見錯誤是「用什麼就繼承什麼」，導致專案結構被外部框架綁架、換掉時要動半個 codebase。此處 `BackboneProtocol` 只有兩個屬性 + 一個函式，是能想像的最小約束——介面越薄、上下游各自實作自由度就越高，換元件成本就越低。這條原則對任何要引入基礎模型的專案都通用，不只 TerraMind。
