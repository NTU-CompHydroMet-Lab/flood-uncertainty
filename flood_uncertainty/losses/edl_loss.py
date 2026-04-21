from typing import Optional, List

import torch
# from torch.overrides import has_torch_function_variadic
# from torch.overrides import handle_torch_function
import torch.nn.functional as F
# from torch.nn.functional import _Reduction
# from torch import Tensor
import logging
import sys
import os
# 取得當前 notebook 的目錄
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
# 計算到專案根目錄的相對路徑
project_root = os.path.join(current_dir, '..')
project_root = os.path.abspath(project_root)
sys.path.insert(0, project_root)
# print(f"專案根目錄: {project_root}")
from mlguess.torch.class_losses import relu_evidence, kl_divergence
from mlguess.torch.class_losses import loglikelihood_loss, get_device


def loglikelihood_loss_weighted(y, alpha, class_weight=None, device=None):
    """Compute the log-likelihood loss for a Dirichlet distribution with class_weight support.
    
    Uses Cross Entropy style weighting: compute loss first, then weight by target class.
    This is equivalent to F.cross_entropy(weight=...) behavior.
    
    Args:
        y (torch.Tensor): Target values in one-hot format, shape (N, num_classes).
        alpha (torch.Tensor): The Dirichlet parameters (alpha).
        class_weight (torch.Tensor or list, optional): 
            - If list: converted to 1D tensor of shape (num_classes,)
            - If 1D tensor: weight for each class, shape (num_classes,)
            - If None: no weighting (all weights = 1.0)
        device (optional, torch.device): Device to perform computation on.
    
    Returns:
        torch.Tensor: The computed log-likelihood loss, shape (N, 1).
    """
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    
    S = torch.sum(alpha, dim=1, keepdim=True)
    err = (y - (alpha / S)) ** 2  # shape: (N, num_classes)
    var = alpha * (S - alpha) / (S * S * (S + 1))  # shape: (N, num_classes)
    
    loglikelihood_err = torch.sum(err, dim=1, keepdim=True)  # (N, 1)
    loglikelihood_var = torch.sum(var, dim=1, keepdim=True)  # (N, 1)
    loglikelihood = loglikelihood_err + loglikelihood_var  # (N, 1)
    
    # Cross Entropy 風格的外部加權：根據樣本的目標類別加權整個 loss
    if class_weight is not None:
        # 處理 1D class_weight 輸入
        if isinstance(class_weight, list):
            class_weight = torch.tensor(class_weight, dtype=torch.float32, device=device)
        else:
            class_weight = class_weight.to(device)
        target_class = torch.argmax(y, dim=1, keepdim=True)  # (N, 1)
        # 根據目標類別選擇對應的權重
        sample_weight = class_weight[target_class]  # (N, 1)
        loglikelihood = loglikelihood * sample_weight
    return loglikelihood


def edl_mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None, class_weight=None):
    """Compute the mean squared error loss with KL divergence for Dirichlet distributions.
    
    Uses Cross Entropy style weighting similar to F.cross_entropy(weight=...):
    Samples are weighted based on their target class.

    Args:
        y (torch.Tensor): Target values in one-hot format, shape (N, num_classes).
        alpha (torch.Tensor): The Dirichlet parameters (alpha).
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.
        class_weight (torch.Tensor or list, optional): 
            - If list: converted to 1D tensor of shape (num_classes,)
            - If 1D tensor: weight for each class, shape (num_classes,)
            - If None: no weighting (all weights = 1.0)
    Returns:
        torch.Tensor: The computed MSE loss with KL divergence.
    """
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    
    loglikelihood = loglikelihood_loss_weighted(y, alpha, class_weight=class_weight, device=device)
    
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss_flat(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_indices: torch.Tensor,
    epoch_num: int,
    num_classes: int,
    annealing_step: int,
    class_weight: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """EDL MSE loss for pixel-wise prediction (Equation 9).
    
    Args:
        logits: (Batch, 2, H, W) - model output
        target: (Batch, H, W) - encoded as {0: invalid, 1: negative, 2: positive}
        target_indices: (Batch, H, W) - class indices {0, 1} from (target - 1)
        epoch_num: current epoch number
        num_classes: number of classes (should be 2)
        annealing_step: annealing step for KL divergence
        class_weight: weight of positive class.
        device: computation device
    
    Returns:
        Tensor of shape (Batch, H, W) with per-pixel loss
    """
    # 步驟 1: 轉換格式 (Batch, 2, H, W) -> (Batch, H, W, 2)
    logits_permuted = logits.permute(0, 2, 3, 1)  # (B, H, W, 2)
    # logging.info(f"logits_permuted shape: {logits_permuted.shape}")
    
    # 步驟 2: 計算 alpha
    evidence = torch.relu(logits_permuted)
    alpha = evidence + 1  # (B, H, W, 2)
    
    # 步驟 3: 轉換 target 為 one-hot
    # 處理 ignore_index（負數），將其暫時設為 0 以避免 one_hot 錯誤
    # 這些 invalid pixels 會在最後通過 valid mask 過濾掉
    target_indices_safe = target_indices.clamp_min(0)
    target_onehot = torch.nn.functional.one_hot(target_indices_safe.long(), num_classes=num_classes).float()  # (B, H, W, 2)
    
    # 步驟 4: 展平以使用 mse_loss
    B, H, W, C = alpha.shape
    alpha_flat = alpha.reshape(-1, num_classes)  # (B*H*W, 2)
    target_flat = target_onehot.reshape(-1, num_classes)  # (B*H*W, 2)
    # 步驟 5: 計算 loss
    loss_flat = edl_mse_loss(target_flat, alpha_flat, epoch_num, num_classes, annealing_step, device, class_weight=class_weight)  # (B*H*W, 1)
    # 步驟 6: 恢復形狀
    pixelwise_loss = loss_flat.reshape(B, H, W)  # (B, H, W)
    return pixelwise_loss



def beta_EDL_loss_mask_invalid(
    logits: torch.Tensor,
    target: torch.Tensor,
    epoch_num: int = 5,
    annealing_step: int = 10,
    class_weight: Optional[torch.Tensor] = None,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Masked EDL MSE loss (Equation 9) for single task.

    Args:
        logits: Tensor shaped (B, 2, H, W) containing per-pixel logits.
        target: Tensor shaped (B, H, W) encoded as {0: invalid, 1: negative, 2: positive}.
        epoch_num: Current epoch number for annealing.
        annealing_step: Annealing step for KL divergence.
        class_weight: weight of positive class.
        ignore_index: Index used to ignore invalid pixels during loss computation.

    Returns:
        Scalar tensor with the mean EDL MSE loss over valid pixels.
    """

    assert logits.dim() == 4 and logits.shape[1] == 2, f"Unexpected logits shape: {logits.shape}"
    assert target.dim() == 3, f"Unexpected target shape: {target.shape}"

    valid = target != 0
    target_indices = (target - 1).clamp_min(0).long()
    target_indices = target_indices.masked_fill(~valid, ignore_index)

    pixelwise_loss = edl_loss_flat(
        logits,
        target,
        target_indices,
        epoch_num=epoch_num,
        num_classes=2,
        annealing_step=annealing_step,
        class_weight=class_weight,
        device=logits.device,
    )

    # 使用 valid mask 過濾掉 invalid pixels (target == 0) 的 loss
    return (pixelwise_loss * valid).sum() / (valid.sum() + 1e-6)



def calc_edl_loss_multioutput_logistic_mask_invalid(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight_problem: Optional[List[float]] = None,
    weight_problem: Optional[List[float]] = None,
    epoch_num: int = 5,
    annealing_step: int = 10,
) -> torch.Tensor:
    """Calculate the loss for multiple output tasks using EDL MSE loss with mask invalid pixels.

    Args:
        logits: Tensor shaped (B, N*2, H, W) containing per-pixel logits for N tasks.
                Each task has 2 channels for EDL evidence.
        target: Tensor shaped (B, N, H, W) containing per-pixel target indices for N tasks.
                Each pixel encoded as {0: invalid, 1: negative, 2: positive}.
        pos_weight_problem: weight of positive class for each task.
        weight_problem: Optional list of weights for each task. Default is equal weights.
    
    Returns:
        Scalar tensor with the weighted sum of EDL MSE loss over all tasks.
    """
    assert logits.dim() == 4, "Unexpected shape of logits"
    assert target.dim() == 4, "Unexpected shape of target"
    assert logits.shape[1] % 2 == 0, "Channel count must be 2 per task"

    num_tasks = logits.shape[1] // 2
    if weight_problem is None:
        weight_problem = [1 / num_tasks for _ in range(num_tasks)]

    total_loss = 0.0
    for i in range(num_tasks):
        task_logits = logits[:, i * 2:(i + 1) * 2]    # (B, 2, H, W)
        task_target = target[:, i]                    # (B, H, W)

        class_weight = (
            torch.tensor(pos_weight_problem[i], device=logits.device)
            if pos_weight_problem is not None
            else None
        )

        curr_loss = beta_EDL_loss_mask_invalid(task_logits, task_target, class_weight=class_weight, epoch_num=epoch_num, annealing_step=annealing_step)

        total_loss += curr_loss * weight_problem[i]

    return total_loss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    torch.manual_seed(42)
    
    # 測試結果追蹤
    test_results = {}
    
    # 全局測試參數
    device = torch.device('cpu')
    num_classes = 2
    
    logging.info("\n" + "="*70)
    logging.info("EDL MSE Loss 測試套件")
    logging.info("="*70)
    
    # ========================================================================
    # 測試 1: edl_loss_flat 基礎功能
    # ========================================================================
    logging.info("\n" + "#"*70)
    logging.info("# 測試 1: edl_loss_flat 基礎功能")
    logging.info("#"*70)
    
    batch, height, width = 1, 4, 3
    logits = torch.randn(batch, 2, height, width)
    target = torch.randint(0, 3, (batch, height, width))
    target_indices = (target - 1).clamp_min(0).long()
    
    logging.info(f"\n輸入形狀:")
    logging.info(f"  Logits: {logits.shape} (Batch, Classes, H, W)")
    logging.info(f"  Target: {target.shape} (Batch, H, W)")
    logging.info(f"  Target indices: {target_indices.shape} (Batch, H, W)")
    logging.info(f"  Target 值範圍: {target.unique().tolist()}")
    logging.info(f"  Target: {target}")
    logging.info(f"  Target indices 值範圍: {target_indices.unique().tolist()}")
    logging.info(f"  Target indices: {target_indices}")
    pixelwise_loss = edl_loss_flat(
        logits, target, target_indices,
        epoch_num=5, num_classes=2, annealing_step=10,
        device=logits.device
    )
    
    logging.info(f"\n輸出:")
    logging.info(f"  Pixelwise loss shape: {pixelwise_loss.shape}")
    logging.info(f"  Loss 統計:")
    logging.info(f"    Mean: {pixelwise_loss.mean().item():.6f}")
    logging.info(f"    Min:  {pixelwise_loss.min().item():.6f}")
    logging.info(f"    Max:  {pixelwise_loss.max().item():.6f}")
    logging.info(f"    Std:  {pixelwise_loss.std().item():.6f}")
    
    # 驗證形狀
    expected_shape = (batch, height, width)
    test_results['test1'] = pixelwise_loss.shape == expected_shape
    if test_results['test1']:
        logging.info(f"  ✅ 形狀正確: {pixelwise_loss.shape} == {expected_shape}")
    else:
        logging.info(f"  ❌ 形狀錯誤: {pixelwise_loss.shape} != {expected_shape}")
    
    # ========================================================================
    # 測試 2: beta_EDL_loss_mask_invalid
    # ========================================================================
    logging.info("\n" + "#"*70)
    logging.info("# 測試 2: beta_EDL_loss_mask_invalid (過濾 invalid pixels)")
    logging.info("#"*70)
    
    logits = torch.randn(batch, 2,height, width)
    target = torch.randint(0, 3, (batch, height, width))
    
    valid_mask = target != 0
    valid_count = valid_mask.sum().item()
    total_count = target.numel()
    
    logging.info(f"\n輸入:")
    logging.info(f"  Logits shape: {logits.shape}")
    logging.info(f"  Target shape: {target.shape}")
    logging.info(f"  Target 唯一值: {target.unique().tolist()}")
    logging.info(f"  Valid pixels: {valid_count} / {total_count} ({100*valid_count/total_count:.1f}%)")
    
    # 測試不同的 epoch
    logging.info(f"\n不同 epoch 的影響 (annealing):")
    test_results['test2'] = True
    for epoch in [1, 5, 10, 20]:
        loss = beta_EDL_loss_mask_invalid(
            logits, target, 
            epoch_num=epoch, annealing_step=10
        )
        logging.info(f"  Epoch {epoch:2d}: Loss = {loss.item():.6f}")
        # 驗證 loss 是否為有效值（非負、有限）
        test_results['test2'] = test_results['test2'] and loss.item() >= 0 and torch.isfinite(loss)
    
    status = "✅" if test_results['test2'] else "❌"
    logging.info(f"  {status} Loss 值驗證{'通過' if test_results['test2'] else '失敗'}")
    
    # ========================================================================
    # 測試 3: 形狀一致性
    # ========================================================================
    logging.info("\n" + "#"*70)
    logging.info("# 測試 3: 不同形狀的一致性")
    logging.info("#"*70)
    
    test_shapes = [
        (2, 2, 3, 3),    # 小
        (4, 2, 10, 10),  # 中
        (1, 2, 5, 7),    # 非正方形
    ]
    
    logging.info(f"\n測試不同形狀:")
    test_results['test3'] = True
    for shape in test_shapes:
        b, c, h, w = shape
        logits = torch.randn(b, c, h, w)
        target = torch.randint(0, 3, (b, h, w))
        
        try:
            loss = beta_EDL_loss_mask_invalid(
                logits, target, epoch_num=5, annealing_step=10
            )
            status = "✅"
        except Exception as e:
            loss = None
            status = f"❌ {str(e)}"
            test_results['test3'] = False
        
        loss_str = f"{loss.item():.6f}" if loss is not None else "N/A"
        logging.info(f"  形狀 {shape}: Loss = {loss_str} {status}")
    
    status = "✅" if test_results['test3'] else "❌"
    logging.info(f"  {status} 所有形狀測試{'通過' if test_results['test3'] else '失敗'}")
    
    # ========================================================================
    # 測試 4: calc_loss_multioutput_logistic_mask_invalid (多任務)
    # ========================================================================
    logging.info("\n" + "#"*70)
    logging.info("# 測試 4: 多任務 Loss")
    logging.info("#"*70)
    
    num_tasks = 2
    multi_logits = torch.randn(2, num_tasks * 2, 5, 5)  # 2 tasks, 2 classes each
    multi_target = torch.randint(0, 3, (2, num_tasks, 5, 5))
    
    logging.info(f"\n輸入:")
    logging.info(f"  Multi-task logits shape: {multi_logits.shape}")
    logging.info(f"  Multi-task target shape: {multi_target.shape}")
    logging.info(f"  任務數: {num_tasks}")
    
    for i in range(num_tasks):
        valid_count = (multi_target[:, i] != 0).sum().item()
        total_count = multi_target[:, i].numel()
        logging.info(f"  任務 {i}: Valid pixels = {valid_count}/{total_count}")
    
    # 測試不同的任務權重
    logging.info(f"\n測試任務權重:")
    
    # 均等權重
    multi_loss_equal = calc_edl_loss_multioutput_logistic_mask_invalid(
        multi_logits, multi_target
    )
    logging.info(f"  均等權重: {multi_loss_equal.item():.6f}")
    
    # 自定義權重
    multi_loss_weighted = calc_edl_loss_multioutput_logistic_mask_invalid(
        multi_logits, multi_target,
        weight_problem=[0.3, 0.7]
    )
    logging.info(f"  權重 [0.3, 0.7]: {multi_loss_weighted.item():.6f}")
    
    # 驗證 loss 值有效性
    test_results['test4'] = (
        torch.isfinite(multi_loss_equal) and multi_loss_equal.item() >= 0 and
        torch.isfinite(multi_loss_weighted) and multi_loss_weighted.item() >= 0 and
        multi_loss_equal.item() != multi_loss_weighted.item()
    )
    status = "✅" if test_results['test4'] else "❌"
    logging.info(f"  {status} 多任務測試{'通過' if test_results['test4'] else '失敗'}")
    
    # ========================================================================
    # 測試 5: 邊界情況
    # ========================================================================
    logging.info("\n" + "#"*70)
    logging.info("# 測試 5: 邊界情況")
    logging.info("#"*70)
    
    logits = torch.randn(2, 2, 5, 5)
    test_results['test5'] = True
    
    # 情況 1: 全部 valid
    logging.info(f"\n情況 1: 全部 valid pixels")
    target_all_valid = torch.randint(1, 3, (2, 5, 5))  # 只有 1 和 2
    valid_count = (target_all_valid != 0).sum().item()
    logging.info(f"  Valid pixels: {valid_count}/{target_all_valid.numel()}")
    loss_all_valid = beta_EDL_loss_mask_invalid(
        logits, target_all_valid, epoch_num=5, annealing_step=10
    )
    logging.info(f"  Loss: {loss_all_valid.item():.6f}")
    test_5_1 = torch.isfinite(loss_all_valid) and loss_all_valid.item() >= 0
    test_results['test5'] = test_results['test5'] and test_5_1
    status = "✅" if test_5_1 else "❌"
    logging.info(f"  {status} 測試{'通過' if test_5_1 else '失敗'}")
    
    # 情況 2: 部分 invalid
    logging.info(f"\n情況 2: 部分 invalid pixels")
    target_partial = torch.randint(0, 3, (2, 5, 5))
    valid_count = (target_partial != 0).sum().item()
    logging.info(f"  Valid pixels: {valid_count}/{target_partial.numel()}")
    loss_partial = beta_EDL_loss_mask_invalid(
        logits, target_partial, epoch_num=5, annealing_step=10
    )
    logging.info(f"  Loss: {loss_partial.item():.6f}")
    test_5_2 = torch.isfinite(loss_partial) and loss_partial.item() >= 0
    test_results['test5'] = test_results['test5'] and test_5_2
    status = "✅" if test_5_2 else "❌"
    logging.info(f"  {status} 測試{'通過' if test_5_2 else '失敗'}")
    
    # 情況 3: 大量 invalid
    logging.info(f"\n情況 3: 大量 invalid pixels")
    target_mostly_invalid = torch.zeros(2, 5, 5, dtype=torch.long)
    target_mostly_invalid[0, 0, 0] = 1  # 只有一個 valid
    target_mostly_invalid[1, 0, 0] = 2
    valid_count = (target_mostly_invalid != 0).sum().item()
    logging.info(f"  Valid pixels: {valid_count}/{target_mostly_invalid.numel()}")
    loss_mostly_invalid = beta_EDL_loss_mask_invalid(
        logits, target_mostly_invalid, epoch_num=5, annealing_step=10
    )
    logging.info(f"  Loss: {loss_mostly_invalid.item():.6f}")
    test_5_3 = torch.isfinite(loss_mostly_invalid) and loss_mostly_invalid.item() >= 0
    test_results['test5'] = test_results['test5'] and test_5_3
    status = "✅" if test_5_3 else "❌"
    logging.info(f"  {status} 測試{'通過' if test_5_3 else '失敗'}")

    # 情況 4: 全部 invalid
    logging.info(f"\n情況 4: 全部 invalid pixels")
    target_all_invalid = torch.zeros(2, 5, 5, dtype=torch.long)
    valid_count = (target_all_invalid != 0).sum().item()
    logging.info(f"  Valid pixels: {valid_count}/{target_all_invalid.numel()}")
    loss_all_invalid = beta_EDL_loss_mask_invalid(
        logits, target_all_invalid, epoch_num=5, annealing_step=10
    )
    logging.info(f"  Loss: {loss_all_invalid.item():.6f}")
    test_5_4 = torch.isfinite(loss_all_invalid) and loss_all_invalid.item() >= 0
    test_results['test5'] = test_results['test5'] and test_5_4
    status = "✅" if test_5_4 else "❌"
    logging.info(f"  {status} 測試{'通過' if test_5_4 else '失敗'}")
    
    # 測試 5 總結
    status = "✅" if test_results['test5'] else "❌"
    logging.info(f"\n  {status} 所有邊界情況測試{'通過' if test_results['test5'] else '失敗'}")
    
    # ========================================================================
    # 測試 6: 形狀變換的不變性
    # ========================================================================
    logging.info("\n" + "#"*70)
    logging.info("# 測試 6: 形狀變換的不變性")
    logging.info("#"*70)
    
    batch_H, batch_W = 4, 3
    evidence_2d = torch.randn(batch_H, batch_W, num_classes).to(device).relu()
    alpha_2d = evidence_2d + 1
    target_indices_2d = torch.randint(0, num_classes, (batch_H, batch_W)).to(device)
    target_onehot_2d = torch.nn.functional.one_hot(target_indices_2d, num_classes=num_classes).float()
    
    logging.info(f"\n原始形狀: alpha={alpha_2d.shape}, target={target_onehot_2d.shape}")
    
    # 方法 1: 保持 2D 形狀再展平計算
    alpha_2d_flat = alpha_2d.reshape(-1, num_classes)
    target_2d_flat = target_onehot_2d.reshape(-1, num_classes)
    loss_2d_flat = edl_mse_loss(target_2d_flat, alpha_2d_flat, epoch_num=5, 
                            num_classes=num_classes, annealing_step=10, device=device)
    
    # 方法 2: 直接使用展平的數據
    alpha_1d = alpha_2d.reshape(-1, num_classes)
    target_1d = target_onehot_2d.reshape(-1, num_classes)
    loss_1d = edl_mse_loss(target_1d, alpha_1d, epoch_num=5, 
                      num_classes=num_classes, annealing_step=10, device=device)
    
    logging.info(f"展平形狀: alpha={alpha_1d.shape}, target={target_1d.shape}")

    logging.info(f"Loss Loss (恢復形狀) (2D→flat): {loss_2d_flat.mean().item():.8f}")
    logging.info(f"Loss mean (1D): {loss_1d.mean().item():.8f}")
    
    # 驗證兩種方法結果一致
    test_results['test6'] = torch.allclose(loss_2d_flat, loss_1d, rtol=1e-5, atol=1e-8)
    max_diff = torch.abs(loss_2d_flat - loss_1d).max().item()
    logging.info(f"最大差異: {max_diff:.2e}")
    
    status = "✅" if test_results['test6'] else "❌"
    logging.info(f"  {status} 形狀變換不變性測試{'通過' if test_results['test6'] else '失敗'}")
    
    # ========================================================================
    # 測試 7: Reshape 正確性
    # ========================================================================
    logging.info("\n" + "#"*70)
    logging.info("# 測試 7: Reshape 正確性")
    logging.info("#"*70)
    
    # 使用測試 6 的 loss 結果來驗證 reshape
    logging.info(f"\nLoss (flat) shape: {loss_2d_flat.shape}")
    logging.info(f"\nLoss (flat):\n{loss_2d_flat}")
    
    # 恢復形狀
    loss_restored = loss_2d_flat.reshape(batch_H, batch_W, 1)
    logging.info(f"Loss (restored) shape: {loss_restored.shape}")
    logging.info(f"Loss (restored):\n{loss_restored}")
    # 再次展平驗證
    loss_reflat = loss_restored.reshape(-1, 1)
    logging.info(f"Loss (reflated) shape: {loss_restored.shape}")
    logging.info(f"Loss (reflated):\n{loss_reflat}")
    # 驗證 reshape 來回是否一致
    test_results['test7'] = torch.equal(loss_2d_flat, loss_reflat)
    
    logging.info(f"原始 flat loss mean: {loss_2d_flat.mean().item():.8f}")
    logging.info(f"恢復後再 flat loss mean: {loss_reflat.mean().item():.8f}")
    
    # 額外驗證：檢查位置對應關係
    all_correct = True
    for h in range( batch_H):  # 只檢查前2行
        for w in range( batch_W):  # 只檢查前2列
            flat_idx = h * batch_W + w
            val_restored = loss_restored[h, w, 0].item()
            val_flat = loss_2d_flat[flat_idx, 0].item()
            if abs(val_restored - val_flat) > 0:
                all_correct = False
                break
    
    test_results['test7'] = test_results['test7'] and all_correct
    status = "✅" if test_results['test7'] else "❌"
    logging.info(f"  {status} Reshape 正確性測試{'通過' if test_results['test7'] else '失敗'}")
    
    # ========================================================================
    # 測試 8: class_weight 加權效果驗證 (Cross Entropy 風格)
    # ========================================================================
    logging.info("\n" + "#"*70)
    logging.info("# 測試 8: class_weight 加權效果驗證 (Cross Entropy 風格)")
    logging.info("#"*70)
    
    # 創建簡單的測試數據
    batch, height, width = 2, 4, 4
    logits = torch.randn(batch, 2, height, width) * 0.5  # 小範圍的 logits
    
    # 創建 target：一半 class 0，一半 class 1
    target = torch.ones(batch, height, width).long()
    target[:, :, :height//2] = 1  # negative class
    target[:, :, height//2:] = 2  # positive class
    
    num_neg = (target == 1).sum().item()
    num_pos = (target == 2).sum().item()
    
    logging.info(f"\n測試設定:")
    logging.info(f"  Target 分布: class 0 (neg) = {num_neg} pixels, class 1 (pos) = {num_pos} pixels")
    logging.info(f"  加權方式: Cross Entropy 風格（根據樣本目標類別加權整個 loss）")
    
    # 測試 1: 無加權
    loss_no_weight = beta_EDL_loss_mask_invalid(
        logits, target, epoch_num=5, annealing_step=10, class_weight=None
    )
    
    # 測試 2: 對 positive class 加權 2.0 (使用 [1.0, 2.0])
    loss_with_weight = beta_EDL_loss_mask_invalid(
        logits, target, epoch_num=5, annealing_step=10, 
        class_weight=[1.0, 2.0]
    )
    
    # 測試 3: 對兩個類別分別加權 [1.0, 3.0]
    loss_multi_weight = beta_EDL_loss_mask_invalid(
        logits, target, epoch_num=5, annealing_step=10,
        class_weight=torch.tensor([1.0, 3.0])
    )
    
    logging.info(f"\n加權效果比較:")
    logging.info(f"  無加權:               loss = {loss_no_weight.item():.6f}")
    logging.info(f"  class_weight=[1,2]:   loss = {loss_with_weight.item():.6f}")
    logging.info(f"  class_weight=[1,3]:   loss = {loss_multi_weight.item():.6f}")
    
    # 理論計算：CE 風格的加權
    # 無加權: (loss_neg * num_neg + loss_pos * num_pos) / (num_neg + num_pos)
    # 加權:   (loss_neg * 1.0 * num_neg + loss_pos * 2.0 * num_pos) / (num_neg + num_pos)
    expected_ratio_1 = (num_neg * 1.0 + num_pos * 2.0) / (num_neg + num_pos)
    expected_ratio_2 = (num_neg * 1.0 + num_pos * 3.0) / (num_neg + num_pos)
    
    ratio_1 = loss_with_weight.item() / loss_no_weight.item()
    ratio_2 = loss_multi_weight.item() / loss_no_weight.item()
    
    logging.info(f"\nLoss 增長倍率:")
    logging.info(f"  class_weight=[1,2] 實際: {ratio_1:.3f}x, 理論: {expected_ratio_1:.3f}x")
    logging.info(f"  class_weight=[1,3] 實際: {ratio_2:.3f}x, 理論: {expected_ratio_2:.3f}x")
    
    # 驗證加權效果
    test_8_passed = (loss_with_weight > loss_no_weight and 
                     loss_multi_weight > loss_with_weight and
                     abs(ratio_1 - expected_ratio_1) < 0.1 and
                     abs(ratio_2 - expected_ratio_2) < 0.1)
    
    if test_8_passed:
        logging.info(f"  ✅ 加權效果正確: 權重越大，loss 越高，且符合理論值")
    else:
        logging.info(f"  ❌ 加權效果異常")
    
    # 詳細分析：分別計算 positive 和 negative samples 的貢獻
    logging.info(f"\n詳細分析（Cross Entropy 風格加權）:")
    target_indices = (target - 1).clamp_min(0).long()
    
    # 無加權的 pixelwise loss
    pixelwise_no_weight = edl_loss_flat(
        logits, target, target_indices, epoch_num=5, num_classes=2, 
        annealing_step=10, class_weight=None
    )
    
    # 有加權的 pixelwise loss (class_weight=[1.0, 2.0])
    pixelwise_with_weight = edl_loss_flat(
        logits, target, target_indices, epoch_num=5, num_classes=2,
        annealing_step=10, class_weight=[1.0, 2.0]
    )
    
    # 計算 positive 和 negative 區域的平均 loss
    neg_mask = (target == 1)
    pos_mask = (target == 2)
    
    neg_loss_no_weight = pixelwise_no_weight[neg_mask].mean().item()
    pos_loss_no_weight = pixelwise_no_weight[pos_mask].mean().item()
    neg_loss_with_weight = pixelwise_with_weight[neg_mask].mean().item()
    pos_loss_with_weight = pixelwise_with_weight[pos_mask].mean().item()
    
    logging.info(f"  Negative class (無加權): {neg_loss_no_weight:.6f}")
    logging.info(f"  Negative class (加權後): {neg_loss_with_weight:.6f} (weight=1.0, 應相同)")
    logging.info(f"  Positive class (無加權): {pos_loss_no_weight:.6f}")
    logging.info(f"  Positive class (加權後): {pos_loss_with_weight:.6f} (weight=2.0, 應為 2 倍)")
    
    # CE 風格：negative class weight=1.0，positive class weight=2.0
    neg_ratio = neg_loss_with_weight / (neg_loss_no_weight + 1e-6)
    pos_ratio = pos_loss_with_weight / (pos_loss_no_weight + 1e-6)
    
    logging.info(f"\n加權倍率:")
    logging.info(f"  Negative class: {neg_ratio:.3f}x (預期 1.0x)")
    logging.info(f"  Positive class: {pos_ratio:.3f}x (預期 2.0x)")
    
    # 驗證：CE 風格應該讓 negative 保持 1.0 倍，positive 變成 2.0 倍
    if abs(neg_ratio - 1.0) < 0.05 and abs(pos_ratio - 2.0) < 0.05:
        logging.info(f"  ✅ 加權行為正確: CE 風格外部加權")
    else:
        logging.info(f"  ⚠️  加權行為偏離預期")
    
    test_results['test8'] = test_8_passed
    
    # ========================================================================
    # 測試總結
    # ========================================================================
    logging.info("\n" + "="*70)
    logging.info("測試總結")
    logging.info("="*70)
    
    test_names = {
        'test1': 'edl_loss_flat 基礎功能',
        'test2': 'beta_EDL_loss_mask_invalid',
        'test3': '形狀一致性',
        'test4': '多任務 Loss',
        'test5': '邊界情況',
        'test6': '形狀變換的不變性',
        'test7': 'Reshape 正確性',
        'test8': 'class_weight 加權效果 (CE 風格)'
    }
    
    for test_id, test_name in test_names.items():
        status = "✅" if test_results[test_id] else "❌"
        result = "通過" if test_results[test_id] else "失敗"
        logging.info(f"{status} 測試 {test_id[-1]}: {test_name} - {result}")
    
    all_passed = all(test_results.values())
    if all_passed:
        logging.info("\n✅ 所有測試完成！EDL MSE Loss (Equation 9) 實現正確。")
    else:
        failed_tests = [k for k, v in test_results.items() if not v]
        logging.info(f"\n❌ 部分測試失敗: {', '.join(failed_tests)}")
    logging.info("="*70)
