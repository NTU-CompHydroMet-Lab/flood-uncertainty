import sys
import os
# 計算到專案根目錄的相對路徑
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)
# print(f"專案根目錄: {project_root}")
from ml4floods.models.worldfloods_model import configure_architecture, METRIC_MODE
from ml4floods.models.utils.configuration import AttrDict
from typing import Optional
import torch
import numpy as np
from typing import Dict
from pytorch_lightning.loggers import WandbLogger
from ml4floods.models.utils import metrics
from ml4floods.data.worldfloods.configs import COLORS_WORLDFLOODS_INVCLEARCLOUD, COLORS_WORLDFLOODS_INVLANDWATER
import pytorch_lightning as pl
from ml4floods.models.worldfloods_model import batch_to_unnorm_rgb, mask_to_rgb
from flood_uncertainty.losses import edl_loss as losses_uncertainty
 
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
        self.annealing_step = h_params_dict.get('annealing_step')
        if self.annealing_step is None:
            raise ValueError("model_params.hyperparameters.annealing_step is required for EDL loss")
        if self.annealing_step <= 0:
            raise ValueError("model_params.hyperparameters.annealing_step must be > 0")

        # label names setup
        self.label_names = np.array(h_params_dict['label_names'])
        assert self.label_names.shape == (2, 3), "Unexpected label names, expected: {}".format([["invalid","clear", "cloud"],
                                                                                                ["invalid", "land", "water"]])

        self.colormaps = {0 : COLORS_WORLDFLOODS_INVCLEARCLOUD, 1: COLORS_WORLDFLOODS_INVLANDWATER}


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
                                                                  weight_problem=self.weight_problem,
                                                                  epoch_num=self.current_epoch,
                                                                  annealing_step=self.annealing_step)
        current_lr = self.get_current_lr()

        if (batch_idx % 100) == 0:
            self.log("loss", loss)
            if current_lr is not None:
                self.log("lr", current_lr)

        if (batch_idx == 0) and (self.logger is not None) and isinstance(self.logger, WandbLogger):
            with torch.no_grad():
                self.log_images(x, y, logits, prefix="train_")

        return loss

    def get_current_lr(self) -> Optional[float]:
        optimizer = self.optimizers(use_pl_optimizer=False)
        if optimizer is None:
            return None

        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            return None

        return param_groups[0].get("lr")

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

        bce_loss = losses_uncertainty.calc_edl_loss_multioutput_logistic_mask_invalid(
            logits,
            y,
            epoch_num=self.current_epoch,
            annealing_step=self.annealing_step,
        )
        self.log('val_bce_loss', bce_loss)

        pred_probs = self.edl_logits_to_probs(logits)  # (B, n, H, W)
        pred_categorical = torch.round(pred_probs).long()  # (B, n, H, W)

        # cm_batch is (B, num_class, num_class)

        for i in range(len(self.label_names)):
            cm_batch = metrics.compute_confusions(y[:, i], pred_categorical[:, i],
                                                  num_class=2, remove_class_zero=True)

            problem_name = "_".join(self.label_names[i, 1:])

            cm_agg = torch.sum(cm_batch, dim=0)
            
            self.log(f"val_Acc_{problem_name}", metrics.binary_accuracy(cm_agg))
            self.log(f"val_Precision_{problem_name}", metrics.binary_precision(cm_agg))
            self.log(f"val_Recall_{problem_name}", metrics.binary_recall(cm_agg))

            task_logits = logits[:, i*2:(i+1)*2]  # (B, 2, H, W)
            bce = losses_uncertainty.beta_EDL_loss_mask_invalid(
                task_logits,
                y[:, i],
                epoch_num=self.current_epoch,
                annealing_step=self.annealing_step,
                class_weight=None,
            )
            self.log(f"val_bce_{problem_name}", bce)

            # Log IoU per class
            iou_dict = metrics.calculate_iou(cm_batch, self.label_names[i, 1:])
            for k in iou_dict.keys():
                self.log(f"val_iou_{problem_name} {k}", iou_dict[k])

        if (batch_idx == 0) and (self.logger is not None) and isinstance(self.logger, WandbLogger):
            self.log_images(x, y, logits, prefix="val_")

    # def on_validation_epoch_start(self):
    #     super().on_validation_epoch_start()
    #     self.epoch_cms = {}

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
        self.annealing_step = h_params_dict.get('annealing_step')
        if self.annealing_step is None:
            raise ValueError("model_params.hyperparameters.annealing_step is required for EDL loss")
        if self.annealing_step <= 0:
            raise ValueError("model_params.hyperparameters.annealing_step must be > 0")

        # label names setup
        self.label_names = np.array(h_params_dict['label_names'])
        assert self.label_names.shape == (1, 3), "Unexpected label names, expected: {}".format(["invalid", "land", "water"])
        self.colormaps = {0 : COLORS_WORLDFLOODS_INVLANDWATER}
    
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

        bce_loss = losses_uncertainty.calc_edl_loss_multioutput_logistic_mask_invalid(
            logits,
            y,
            epoch_num=self.current_epoch,
            annealing_step=self.annealing_step,
        )
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
            self.log(f"val_Acc_{problem_name}", metrics.binary_accuracy(cm_agg))
            self.log(f"val_Precision_{problem_name}", metrics.binary_precision(cm_agg))
            self.log(f"val_Recall_{problem_name}", metrics.binary_recall(cm_agg))

            task_logits = logits[:, i*2:(i+1)*2]  # (B, 2, H, W)
            bce = losses_uncertainty.beta_EDL_loss_mask_invalid(
                task_logits,
                y[:, i],
                epoch_num=self.current_epoch,
                annealing_step=self.annealing_step,
                class_weight=None,
            )
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
        
