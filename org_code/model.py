import sys
import os
# 計算到專案根目錄的相對路徑
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)
# print(f"專案根目錄: {project_root}")
from ml4floods.models.worldfloods_model import WorldFloodsModel, ML4FloodsModel, load_weights, configure_architecture, METRIC_MODE
from ml4floods.models.utils.configuration import AttrDict
from typing import Optional, Union
import torch
import numpy as np
from typing import Dict
from pytorch_lightning.loggers import WandbLogger
from ml4floods.models.utils import metrics
from ml4floods.data.worldfloods.configs import COLORS_WORLDFLOODS_INVCLEARCLOUD, COLORS_WORLDFLOODS_INVLANDWATER
import pytorch_lightning as pl
from ml4floods.models.worldfloods_model import batch_to_unnorm_rgb, mask_to_rgb
import losses_uncertainty
import xarray as xr 
class EDL_ML4FloodsModel(pl.LightningModule):
    """
    Model to do multioutput binary classification with EDL loss.
    It expects ground truths y (B, 2, H, W) tensors to be encoded as:
    - Channel 0: {0: invalid, 1: clear, 2: cloud}
    - Channel 1: {0: invalid, 1: land, 2: water}
    """
    def __init__(self, model_params: AttrDict, normalized_data:bool = True):
        super().__init__()
        self.save_hyperparameters()
        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 2)
        assert self.num_class == 2, "Expected 2 output classes"

        self.pos_weight = h_params_dict.get('pos_weight',
                                            [1 for i in range(self.num_class)])
        self.weight_problem = [1 / self.num_class for _ in range(self.num_class)]

        # EDL beta分佈需要每個任務2個evidence通道，所以總共需要 2*n 個輸出通道
        num_output_channels = 2 * self.num_class
        h_params_dict_copy = h_params_dict.copy()
        h_params_dict_copy['num_classes'] = num_output_channels
        self.network = configure_architecture(h_params_dict_copy)
        self.normalized_data = normalized_data

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 2)

        # label names setup
        self.label_names = np.array(h_params_dict['label_names'])
        assert self.label_names.shape == (2, 3), "Unexpected label names, expected: {}".format([["invalid","clear", "cloud"],
                                                                                                ["invalid", "land", "water"]])

        self.colormaps = {0 : COLORS_WORLDFLOODS_INVCLEARCLOUD, 1: COLORS_WORLDFLOODS_INVLANDWATER}
        
        # ZARR保存相關變數（只有在val_only=True時才啟用）
        self.val_only = model_params.get('val_only', False)
        self.zarr_save_path = model_params.get('zarr_save_path', None)
        self.zarr_item_count = 0
        self.zarr_initialized = False
        # self.zarr_idx = 0
        # if self.zarr_save_path:
        #     base = self.zarr_save_path.replace('.zarr', '')
        #     self.zarr_paths = [f"{base}_temp1.zarr", f"{base}_temp2.zarr"]


    def training_step(self, batch: Dict, batch_idx) -> float:
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, 2, W, H), input image
                y (torch.Tensor): (B, 2, W, H) encoded as {0: invalid, 1: neg_xxx, 2: pos_xxx}
        """
        x, y = batch['image'], batch['mask']
        logits = self.network(x)
        loss = losses_uncertainty.calc_edl_loss_multioutput_logistic_mask_invalid(logits, y,
                                                                  pos_weight_problem=self.pos_weight,
                                                                  weight_problem=self.weight_problem)

        if (batch_idx % 100) == 0:
            self.log("loss", loss)

        if (batch_idx == 0) and (self.logger is not None) and isinstance(self.logger, WandbLogger):
            with torch.no_grad():
                self.log_images(x, y, logits, prefix="train_")

        return loss

    def forward(self, x):
        """

        Args:
            x: (B, num_channels, H, W) input tensor

        Returns:
            (B, 2*n, H, W) prediction of the network, where n is the number of tasks:
            - Each task occupies 2 consecutive channels for EDL evidence
            - Task i uses channels i*2:(i+1)*2
            - For each task: channel 0 = negative class evidence, channel 1 = positive class evidence

        """
        return self.network(x)

    def image_to_logger(self, x:torch.Tensor) -> Optional[np.ndarray]:
        return batch_to_unnorm_rgb(x,
                                   self.hparams["model_params"]["hyperparameters"]['channel_configuration'],
                                   unnormalize=self.normalized_data)

    def edl_logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert EDL logits to probability predictions.
        
        Args:
            logits: Tensor shape (B, 2*n, H, W) containing EDL evidence logits for n tasks.
                   Each task has 2 channels: channel 0 = negative class, channel 1 = positive class.
        
        Returns:
            probs: Tensor shape (B, n, H, W) containing positive class probability for each task.
        """
        probs_list = []
        for i in range(self.num_class):
            # Extract task i's channels: i*2 to (i+1)*2
            task_logits = logits[:, i*2:(i+1)*2]  # (B, 2, H, W)
            
            # Calculate evidence and alpha
            evidence = torch.relu(task_logits)  # (B, 2, H, W)
            alpha = evidence + 1  # (B, 2, H, W)
            
            # Calculate sum and positive class probability
            S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)
            prob_positive = alpha[:, 1:2, :, :] / S  # (B, 1, H, W)
            prob_positive = prob_positive.squeeze(1)  # (B, H, W)
            
            probs_list.append(prob_positive)
        
        # Stack all tasks
        probs = torch.stack(probs_list, dim=1)  # (B, n, H, W)
        return probs

    def edl_logits_to_output(self, logits: torch.Tensor) -> dict:
        """從 EDL logits 計算機率和 uncertainty。
        
        Args:
            logits: Tensor shape (B, 2*n, H, W) containing EDL evidence logits for n tasks.
        
        Returns:
            dict with:
                'prob': Tensor (B, n, H, W) - 每個 task 的正類機率 (mean)
                'evidence': Tensor (B, 2*n, H, W) - 每個 task 的證據 (neg, pos 交錯)
                'dst_u': Tensor (B, n, H, W) - 每個 task 的 DST uncertainty
                'aleatoric': Tensor (B, n, H, W) - 每個 task 的 aleatoric uncertainty
                'epistemic': Tensor (B, n, H, W) - 每個 task 的 epistemic uncertainty
        """
        prob_list = []
        evidence_list = []
        dst_u_list = []
        aleatoric_list = []
        epistemic_list = []
        for i in range(self.num_class):
            task_logits = logits[:, i*2:(i+1)*2]  # (B, 2, H, W)
            evidence = torch.relu(task_logits)
            alpha = evidence + 1
            S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)
            
            # prob (mean) = alpha_positive / S
            prob = (alpha[:, 1:2] / S).squeeze(1)  # (B, H, W)
            
            # DST uncertainty = K / S
            K = 2
            u = K / S.squeeze(1)  # (B, H, W)

            # Epistemic: variance due to lack of knowledge
            epistemic = prob * (1 - prob) / (S.squeeze(1) + 1)  # (B, H, W)
            
            # Aleatoric: inherent data uncertainty
            aleatoric = prob - prob ** 2 - epistemic  # (B, H, W)
            
            prob_list.append(prob)
            evidence_list.append(evidence)
            dst_u_list.append(u)
            aleatoric_list.append(aleatoric)
            epistemic_list.append(epistemic)
        
        return {
            'prob': torch.stack(prob_list, dim=1),   # (B, n, H, W)
            'evidence': torch.cat(evidence_list, dim=1),  # (B, 2*n, H, W)
            'dst_u': torch.stack(dst_u_list, dim=1),  # (B, n, H, W)
            'aleatoric': torch.stack(aleatoric_list, dim=1),  # (B, n, H, W)
            'epistemic': torch.stack(epistemic_list, dim=1)   # (B, n, H, W)
        }

    def log_images(self, x, y, logits, prefix=""):
        import wandb
        """ Log batch images and preds using wandb """
        mask_data = y.cpu().numpy()
        pred_probs = self.edl_logits_to_probs(logits)  # (B, n, H, W)
        pred_categorical = torch.round(pred_probs).long()  # (B, n, H, W)
        pred_data = pred_categorical.cpu().numpy()
        img_data = self.image_to_logger(x)

        self.logger.experiment.log({f"{prefix}image": [wandb.Image(img) for img in img_data]})

        for i in range(self.num_class):
            problem_name = "_".join(self.label_names[i, 1:])
            self.logger.experiment.log({f"{prefix}{problem_name}_pred_cont": [wandb.Image(img[i], mode="L") for img in pred_data]})
            self.logger.experiment.log({f"{prefix}{problem_name}_pred": [wandb.Image(mask_to_rgb(img[i].round().astype(np.int64) + 1,
                                                                                                 values=[0, 1, 2], colors_cmap=self.colormaps[i])) for img in pred_data]})
            self.logger.experiment.log({f"{prefix}y_{problem_name}": [wandb.Image(mask_to_rgb(img[i], values=[0, 1, 2], colors_cmap=self.colormaps[i])) for img in mask_data]})

    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask']
        logits = self.network(x)

        bce_loss = losses_uncertainty.calc_edl_loss_multioutput_logistic_mask_invalid(logits, y)
        self.log('val_bce_loss', bce_loss)

        pred_probs = self.edl_logits_to_probs(logits)  # (B, n, H, W)
        pred_categorical = torch.round(pred_probs).long()  # (B, n, H, W)

        # cm_batch is (B, num_class, num_class)

        for i in range(len(self.label_names)):
            cm_batch = metrics.compute_confusions(y[:, i], pred_categorical[:, i],
                                                  num_class=2, remove_class_zero=True)

            problem_name = "_".join(self.label_names[i, 1:])

            cm_agg = torch.sum(cm_batch, dim=0)
            
            # Debug: 印出混淆矩陣詳細資訊
            if batch_idx == 0:
                print(f"\n=== {problem_name} Confusion Matrix ===")
                print(f"CM shape: {cm_agg.shape}")
                print(f"CM:\n{cm_agg}")
                print(f"  TP (pred=water, true=water) = cm[1,1] = {cm_agg[1,1]}")
                print(f"  FP (pred=water, true=land)  = cm[1,0] = {cm_agg[1,0]}")  
                print(f"  FN (pred=land, true=water)  = cm[0,1] = {cm_agg[0,1]}")
                print(f"  TN (pred=land, true=land)   = cm[0,0] = {cm_agg[0,0]}")
                print(f"  Precision = {cm_agg[1,1]} / ({cm_agg[1,1]} + {cm_agg[1,0]}) = {metrics.binary_precision(cm_agg):.4f}")
                print(f"  Recall = {cm_agg[1,1]} / ({cm_agg[1,1]} + {cm_agg[0,1]}) = {metrics.binary_recall(cm_agg):.4f}")
            
            self.log(f"val_Acc_{problem_name}", metrics.binary_accuracy(cm_agg))
            self.log(f"val_Precision_{problem_name}", metrics.binary_precision(cm_agg))
            self.log(f"val_Recall_{problem_name}", metrics.binary_recall(cm_agg))

            task_logits = logits[:, i*2:(i+1)*2]  # (B, 2, H, W)
            bce = losses_uncertainty.beta_EDL_loss_mask_invalid(task_logits, y[:, i], class_weight=None)
            self.log(f"val_bce_{problem_name}", bce)

            # Log IoU per class
            iou_dict = metrics.calculate_iou(cm_batch, self.label_names[i, 1:])
            for k in iou_dict.keys():
                self.log(f"val_iou_{problem_name} {k}", iou_dict[k])

        if (batch_idx == 0) and (self.logger is not None) and isinstance(self.logger, WandbLogger):
            self.log_images(x, y, logits, prefix="val_")
        
        # 只有在val_only=True時才保存ZARR
        if self.val_only and self.zarr_save_path:
            self._append_batch_to_zarr(x, y, logits, pred_probs, batch_idx)

    def on_validation_epoch_start(self):
        """在驗證epoch開始時重置ZARR計數器"""
        if self.val_only and self.zarr_save_path:
            self.zarr_item_count = 0
            self.zarr_initialized = False

    def _init_zarr_structure(self, batch_size: int, num_channels: int, height: int, width: int, zarr_path: str = None):
        """初始化ZARR文件結構（使用XArray）"""
        if zarr_path is None:
            zarr_path = self.zarr_save_path
        if os.path.exists(zarr_path):
            import shutil
            shutil.rmtree(zarr_path)
        
        # 創建目錄
        os.makedirs(os.path.dirname(zarr_path) if os.path.dirname(zarr_path) else '.', exist_ok=True)
        
        # 創建空的XArray Dataset結構
        initial_size = 0
        
        # 創建座標
        coords = {
            'item': np.arange(initial_size),
            'channel': np.arange(num_channels),
            'height': np.arange(height),
            'width': np.arange(width),
            'channel_cloud': np.arange(2),  # clear, cloud
            'channel_water': np.arange(2),  # land, water
        }
        
        # 創建空的DataArray
        data_vars = {
            'input': (['item', 'channel', 'height', 'width'], np.zeros((initial_size, num_channels, height, width), dtype=np.float32)),
            'ground_truth_cloud': (['item', 'height', 'width'], np.zeros((initial_size, height, width), dtype=np.uint8)),
            'ground_truth_water': (['item', 'height', 'width'], np.zeros((initial_size, height, width), dtype=np.uint8)),
            'logits_cloud': (['item', 'channel_cloud', 'height', 'width'], np.zeros((initial_size, 2, height, width), dtype=np.float32)),
            'logits_water': (['item', 'channel_water', 'height', 'width'], np.zeros((initial_size, 2, height, width), dtype=np.float32)),
            'pred_probs_cloud': (['item', 'channel_cloud', 'height', 'width'], np.zeros((initial_size, 2, height, width), dtype=np.float32)),
            'pred_probs_water': (['item', 'channel_water', 'height', 'width'], np.zeros((initial_size, 2, height, width), dtype=np.float32)),
        }
        
        ds = xr.Dataset(data_vars, coords=coords)
        
        # 設置chunk大小以優化I/O（XArray會自動處理壓縮）
        encoding = {
            'input': {'chunks': (100, num_channels, height, width)},
            'ground_truth_cloud': {'chunks': (100, height, width)},
            'ground_truth_water': {'chunks': (100, height, width)},
            'logits_cloud': {'chunks': (100, 2, height, width)},
            'logits_water': {'chunks': (100, 2, height, width)},
            'pred_probs_cloud': {'chunks': (100, 2, height, width)},
            'pred_probs_water': {'chunks': (100, 2, height, width)},
        }
        
        # 保存到ZARR
        ds.to_zarr(zarr_path, mode='w', encoding=encoding)
        self.zarr_initialized = True
        print(f"Initialized ZARR structure at {zarr_path}")
        
        # 保存維度信息以便後續使用
        self.zarr_num_channels = num_channels
        self.zarr_height = height
        self.zarr_width = width

    def _append_batch_to_zarr(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, pred_probs: torch.Tensor, batch_idx: int):
        """將batch數據append到ZARR文件"""
        # 轉換為numpy並移到CPU
        x_np = x.detach().cpu().numpy()  # (B, C, H, W)
        y_np = y.detach().cpu().numpy()  # (B, 2, H, W)
        logits_np = logits.detach().cpu().numpy()  # (B, 4, H, W)
        pred_probs_np = pred_probs.detach().cpu().numpy()  # (B, 2, H, W)
        
        batch_size, num_channels, height, width = x_np.shape
        
        # 如果是第一個batch，初始化ZARR結構
        if batch_idx == 0 and not self.zarr_initialized:
            self._init_zarr_structure(batch_size, num_channels, height, width, self.zarr_save_path)
        
        # 處理數據格式
        # 1. 輸入圖像: (B, C, H, W) -> (B, C, H, W)
        input_data = x_np
        
        # 2. 真實標籤: (B, 2, H, W) -> 分離為cloud和water
        gt_cloud = y_np[:, 0, :, :]  # (B, H, W)
        gt_water = y_np[:, 1, :, :]  # (B, H, W)
        
        # 3. Logits: (B, 4, H, W) -> 分離為cloud和water task
        logits_cloud = logits_np[:, 0:2, :, :]  # (B, 2, H, W)
        logits_water = logits_np[:, 2:4, :, :]  # (B, 2, H, W)
        
        # 4. 預測機率: (B, 2, H, W) -> 轉換為每個任務2個channel的格式
        # cloud task: channel 0 = clear機率 (1 - cloud_prob), channel 1 = cloud機率
        cloud_prob = pred_probs_np[:, 0, :, :]  # (B, H, W)
        pred_probs_cloud = np.stack([
            1 - cloud_prob,  # clear機率
            cloud_prob       # cloud機率
        ], axis=1)  # (B, 2, H, W)
        
        # water task: channel 0 = land機率 (1 - water_prob), channel 1 = water機率
        water_prob = pred_probs_np[:, 1, :, :]  # (B, H, W)
        pred_probs_water = np.stack([
            1 - water_prob,  # land機率
            water_prob       # water機率
        ], axis=1)  # (B, 2, H, W)
        
        # 打開現有的ZARR文件（使用XArray）
        # read_path = self.zarr_paths[self.zarr_idx] if hasattr(self, 'zarr_paths') else self.zarr_save_path
        # write_path = self.zarr_paths[1 - self.zarr_idx] if hasattr(self, 'zarr_paths') else self.zarr_save_path
        
        # ds = xr.open_zarr(read_path)
        
        # 創建新batch的Dataset
        new_coords = {
            'item': np.arange(self.zarr_item_count, self.zarr_item_count + batch_size),
            'channel': np.arange(num_channels),
            'height': np.arange(height),
            'width': np.arange(width),
            'channel_cloud': np.arange(2),
            'channel_water': np.arange(2),
        }
        
        new_data_vars = {
            'input': (['item', 'channel', 'height', 'width'], input_data),
            'ground_truth_cloud': (['item', 'height', 'width'], gt_cloud),
            'ground_truth_water': (['item', 'height', 'width'], gt_water),
            'logits_cloud': (['item', 'channel_cloud', 'height', 'width'], logits_cloud),
            'logits_water': (['item', 'channel_water', 'height', 'width'], logits_water),
            'pred_probs_cloud': (['item', 'channel_cloud', 'height', 'width'], pred_probs_cloud),
            'pred_probs_water': (['item', 'channel_water', 'height', 'width'], pred_probs_water),
        }
        
        new_ds = xr.Dataset(new_data_vars, coords=new_coords)
        
        # 寫回ZARR，使用 append_dim 在 item 維度上追加
        new_ds.to_zarr(self.zarr_save_path, mode='a', append_dim='item')
        # if hasattr(self, 'zarr_paths'):
        #     self.zarr_idx = 1 - self.zarr_idx
        
        # 更新計數器
        self.zarr_item_count += batch_size
        
        if batch_idx % 10 == 0:
            print(f"Saved batch {batch_idx} to ZARR (total items: {self.zarr_item_count})")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=METRIC_MODE[self.hparams["model_params"]["hyperparameters"]["metric_monitor"]],
                                                               factor=self.lr_decay,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": self.hparams["model_params"]["hyperparameters"]["metric_monitor"]}


class EDL_SAR_Unet(EDL_ML4FloodsModel):
    """
    SAR 專用的 EDL 模型，只處理 land/water 二元分類任務
    Input: 2 channels (VV, VH)
    Output: 1 task (land/water)
    """
    def __init__(self, model_params: AttrDict, normalized_data:bool = True):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters({'model_params': model_params, 'normalized_data': normalized_data})
        # self.save_hyperparameters()

        h_params_dict = model_params.get('hyperparameters', {})
        self.num_class = h_params_dict.get('num_classes', 2)
        self.num_channels = h_params_dict.get('num_channels', 7)
        assert self.num_class == 1, "Expected 1 output classes"

        self.pos_weight = h_params_dict.get('pos_weight',
                                            [1 for i in range(self.num_class)])
        self.weight_problem = [1 / self.num_class for _ in range(self.num_class)]

        # EDL beta分佈需要每個任務2個evidence通道，所以總共需要 2*n 個輸出通道
        num_output_channels = 2 * self.num_class
        h_params_dict_copy = h_params_dict.copy()
        h_params_dict_copy['num_classes'] = num_output_channels
        # print("------------")
        # print("------------")
        # print("------------")
        # print("------------")
        # print("num_channels: ", self.num_channels)
        # print("------------")
        # h_params_dict_copy['num_channels'] = self.num_channels
        self.network = configure_architecture(h_params_dict_copy)
        self.normalized_data = normalized_data

        # learning rate params
        self.lr = h_params_dict.get('lr', 1e-4)
        self.lr_decay = h_params_dict.get('lr_decay', 0.5)
        self.lr_patience = h_params_dict.get('lr_patience', 2)

        # label names setup
        self.label_names = np.array(h_params_dict['label_names'])
        assert self.label_names.shape == (1, 3), "Unexpected label names, expected: {}".format(["invalid", "land", "water"])
        self.colormaps = {0 : COLORS_WORLDFLOODS_INVLANDWATER}
        
        # ZARR保存相關變數（只有在val_only=True時才啟用）
        self.val_only = model_params.get('val_only', False)
        self.zarr_save_path = model_params.get('zarr_save_path', None)
        self.zarr_item_count = 0
        self.zarr_initialized = False
        # self.zarr_idx = 0
        # if self.zarr_save_path:
        #     base = self.zarr_save_path.replace('.zarr', '')
        #     self.zarr_paths = [f"{base}_temp1.zarr", f"{base}_temp2.zarr"]
    
    def log_images(self, x, y, logits, prefix=""):
        import wandb
        """ Log batch images and preds using wandb """
        mask_data = y.cpu().numpy()
        pred_probs = self.edl_logits_to_probs(logits)  # (B, n, H, W)
        pred_categorical = torch.round(pred_probs).long()  # (B, n, H, W)
        pred_data = pred_categorical.cpu().numpy()
        
        self.logger.experiment.log({f"{prefix}image_VV": [wandb.Image(np.clip((img + 2) / 8, 0, 1)) for img in x[:,0,:,:].cpu().numpy()]})
        self.logger.experiment.log({f"{prefix}image_VH": [wandb.Image(np.clip((img + 2) / 8, 0, 1)) for img in x[:,1,:,:].cpu().numpy()]})

        for i in range(self.num_class):
            problem_name = "_".join(self.label_names[i, 1:])
            self.logger.experiment.log({f"{prefix}{problem_name}_pred_cont": [wandb.Image(img[i], mode="L") for img in pred_data]})
            self.logger.experiment.log({f"{prefix}{problem_name}_pred": [wandb.Image(mask_to_rgb(img[i].round().astype(np.int64) + 1,
                                                                                                 values=[0, 1, 2], colors_cmap=self.colormaps[i])) for img in pred_data]})
            self.logger.experiment.log({f"{prefix}y_{problem_name}": [wandb.Image(mask_to_rgb(img[i], values=[0, 1, 2], colors_cmap=self.colormaps[i])) for img in mask_data]})

        # logits to output
        output_of_logits = self.edl_logits_to_output(logits)
        
        # Helper for normalization (min-max scaling per image) to visualize grayscale well
        def normalize_vis(img):
            v_min, v_max = img.min(), img.max()
            if v_max - v_min > 1e-8:
                return (img - v_min) / (v_max - v_min)
            return img

        # Pre-convert to numpy
        evidence_all = output_of_logits['evidence'].detach().cpu().numpy()
        probs_all = output_of_logits['prob'].detach().cpu().numpy()
        dst_u_all = output_of_logits['dst_u'].detach().cpu().numpy()
        aleatoric_all = output_of_logits['aleatoric'].detach().cpu().numpy()
        epistemic_all = output_of_logits['epistemic'].detach().cpu().numpy()

        for i in range(self.num_class):
            # save probs, evidence, dst_u, aleatoric, epistemic
            problem_name = "_".join(self.label_names[i, 1:])
            class0_name = self.label_names[i, 1]
            class1_name = self.label_names[i, 2]

            self.logger.experiment.log({
                f"{prefix}{problem_name}_probs_normalize_0_1": [wandb.Image(normalize_vis(img), mode="L") for img in probs_all[:, i]],
                f"{prefix}{problem_name}_evidence_{class0_name}_normalize_0_1": [wandb.Image(normalize_vis(img), mode="L") for img in evidence_all[:, i*2]],
                f"{prefix}{problem_name}_evidence_{class1_name}_normalize_0_1": [wandb.Image(normalize_vis(img), mode="L") for img in evidence_all[:, i*2+1]],
                f"{prefix}{problem_name}_dst_u_normalize_0_1": [wandb.Image(normalize_vis(img), mode="L") for img in dst_u_all[:, i]],
                f"{prefix}{problem_name}_aleatoric_normalize_0_1": [wandb.Image(normalize_vis(img), mode="L") for img in aleatoric_all[:, i]],
                f"{prefix}{problem_name}_epistemic_normalize_0_1": [wandb.Image(normalize_vis(img), mode="L") for img in epistemic_all[:, i]]
            })
        

    
    def validation_step(self, batch: Dict, batch_idx):
        """
        Args:
            batch: includes
                x (torch.Tensor): (B, C, W, H), input image
                y (torch.Tensor): (B, W, H) encoded as {0: invalid, 1: land, 2: water, 3: cloud}
        """
        x, y = batch['image'], batch['mask']
        logits = self.network(x)

        bce_loss = losses_uncertainty.calc_edl_loss_multioutput_logistic_mask_invalid(logits, y)
        self.log('val_bce_loss', bce_loss)

        pred_probs = self.edl_logits_to_probs(logits)  # (B, n, H, W)
        pred_categorical = torch.round(pred_probs).long()  # (B, n, H, W)

        # cm_batch is (B, num_class, num_class)

        for i in range(len(self.label_names)):
            cm_batch = metrics.compute_confusions(y[:, i], pred_categorical[:, i],
                                                  num_class=2, remove_class_zero=True)

            problem_name = "_".join(self.label_names[i, 1:])

            cm_agg = torch.sum(cm_batch, dim=0)

            # Accumulate CM for global epoch metrics
            if problem_name not in self.epoch_cms:
                self.epoch_cms[problem_name] = cm_agg.clone()
            else:
                self.epoch_cms[problem_name] += cm_agg
            
            # Debug: 印出混淆矩陣詳細資訊
            if batch_idx == 0:
                print(f"\n=== {problem_name} Confusion Matrix ===")
                print(f"CM shape: {cm_agg.shape}")
                print(f"CM:\n{cm_agg}")
                print(f"  TP (pred=water, true=water) = cm[1,1] = {cm_agg[1,1]}")
                print(f"  FP (pred=water, true=land)  = cm[1,0] = {cm_agg[1,0]}")  
                print(f"  FN (pred=land, true=water)  = cm[0,1] = {cm_agg[0,1]}")
                print(f"  TN (pred=land, true=land)   = cm[0,0] = {cm_agg[0,0]}")
                print(f"  Precision = {cm_agg[1,1]} / ({cm_agg[1,1]} + {cm_agg[1,0]}) = {metrics.binary_precision(cm_agg):.4f}")
                print(f"  Recall = {cm_agg[1,1]} / ({cm_agg[1,1]} + {cm_agg[0,1]}) = {metrics.binary_recall(cm_agg):.4f}")
            
            self.log(f"val_Acc_{problem_name}", metrics.binary_accuracy(cm_agg))
            self.log(f"val_Precision_{problem_name}", metrics.binary_precision(cm_agg))
            self.log(f"val_Recall_{problem_name}", metrics.binary_recall(cm_agg))

            task_logits = logits[:, i*2:(i+1)*2]  # (B, 2, H, W)
            bce = losses_uncertainty.beta_EDL_loss_mask_invalid(task_logits, y[:, i], class_weight=None)
            self.log(f"val_bce_{problem_name}", bce)

            # Log IoU per class
            iou_dict = metrics.calculate_iou(cm_batch, self.label_names[i, 1:])
            for k in iou_dict.keys():
                self.log(f"val_iou_{problem_name} {k}", iou_dict[k])

        if (batch_idx == 0) and (self.logger is not None) and isinstance(self.logger, WandbLogger):
            self.log_images(x, y, logits, prefix="val_")

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.epoch_cms = {}

    def on_validation_epoch_end(self):
        for i in range(len(self.label_names)):
            problem_name = "_".join(self.label_names[i, 1:])
            if problem_name in self.epoch_cms:
                cm_total = self.epoch_cms[problem_name]
                
                self.log(f"val_Global_Acc_{problem_name}", metrics.binary_accuracy(cm_total))
                self.log(f"val_Global_Precision_{problem_name}", metrics.binary_precision(cm_total))
                self.log(f"val_Global_Recall_{problem_name}", metrics.binary_recall(cm_total))
                
                # Calculate Global IoU
                # cm_total is (2, 2), unsqueeze to (1, 2, 2) to mimic batch for calculate_iou
                iou_dict = metrics.calculate_iou(cm_total.unsqueeze(0), self.label_names[i, 1:])
                for k, v in iou_dict.items():
                    self.log(f"val_Global_iou_{problem_name} {k}", v)
                iou_hand = cm_total[1,1] / (cm_total[1,1] + cm_total[1,0] + cm_total[0,1])
                self.log(f"val_Global_iou_hand_{problem_name}", iou_hand)
        
        # Clear memory
        self.epoch_cms = {}
        

