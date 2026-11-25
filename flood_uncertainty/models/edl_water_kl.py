"""Water-only EDL with KL supervision on selected invalid water labels."""

import math

import torch
from pytorch_lightning.loggers import WandbLogger

from flood_uncertainty.losses.edl_loss import (
    calc_edl_loss_multioutput_logistic_mask_invalid,
    get_kl_annealing_coefficient,
)
from flood_uncertainty.models.edl import EDL_ML4FloodsModel
from ml4floods.models.utils.configuration import AttrDict
from mlguess.torch.class_losses import kl_divergence


class WaterOnlyInvalidKLEDL(EDL_ML4FloodsModel):
    def __init__(self, model_params: AttrDict, normalized_data: bool = True):
        hyperparameters = model_params.get("hyperparameters", {})
        if hyperparameters.get("task_mode") != "water_only":
            raise ValueError("WaterOnlyInvalidKLEDL requires task_mode='water_only'")
        # Defaults reproduce the original objective: invalid KL weighted by the shared beta(t).
        weight = hyperparameters.get("invalid_kl_weight", 1.0)
        anneal = hyperparameters.get("invalid_kl_anneal", True)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid_kl_weight must be a finite number >= 0, got {weight!r}")
        if not isinstance(anneal, bool):
            raise ValueError(f"invalid_kl_anneal must be true or false, got {anneal!r}")
        super().__init__(model_params, normalized_data=normalized_data)
        self.invalid_kl_weight = float(weight)
        self.invalid_kl_anneal = anneal
        self._train_kl_sum = 0.0
        self._train_kl_count = 0

    def _kl_weight(self) -> float:
        """Shared beta(t) of the built-in misleading-evidence KL on valid pixels."""
        return get_kl_annealing_coefficient(
            epoch_num=self.current_epoch,
            annealing_step=self.annealing_step,
            annealing_mode=self.annealing_mode,
            annealing_coefficient=self.annealing_coefficient,
        )

    def _invalid_kl_weight(self) -> float:
        """Invalid-pixel KL coefficient: a constant, or the constant times the shared beta(t)."""
        return self.invalid_kl_weight * (self._kl_weight() if self.invalid_kl_anneal else 1.0)

    def compute_loss(
        self, logits: torch.Tensor, batch: dict, *, accumulate_epoch: bool = False
    ) -> torch.Tensor:
        """Return mean valid EDL + invalid-KL weight * mean invalid KL. Only training collects epoch totals."""
        target = self.select_targets(batch["mask"])  # [B, 1, H, W]
        kl_mask = batch["kl_mask"]
        if kl_mask.shape != target.shape:
            raise ValueError("kl_mask must match the water target shape [B, 1, H, W]")
        water = target[:, 0]
        if logits.shape != (water.shape[0], 2, *water.shape[1:]):
            raise ValueError("Expected two aligned evidence channels [B, 2, H, W]")
        pixel_logits = logits.movedim(1, -1)

        valid = (water == 1) | (water == 2)
        kl_only = (water == 0) & (kl_mask[:, 0] == 1)
        valid_labels = water[valid]
        kl_logits = pixel_logits[kl_only].float()
        n_valid, n_kl = valid_labels.numel(), kl_logits.shape[0]

        # The empty slice contributes zero gradients even for a fully ignored batch.
        zero = logits[:0].sum()
        valid_edl = invalid_kl = zero
        if n_valid:
            # Keep lgamma/digamma and the EDL variance calculation at least fp32 under AMP.
            valid_logits = pixel_logits[valid].to(torch.promote_types(logits.dtype, torch.float32))
            valid_edl = calc_edl_loss_multioutput_logistic_mask_invalid(
                valid_logits.reshape(-1, 2, 1, 1),
                valid_labels.reshape(-1, 1, 1, 1),
                pos_weight_problem=self.pos_weight,
                weight_problem=self.weight_problem,
                epoch_num=self.current_epoch,
                annealing_step=self.annealing_step,
                annealing_mode=self.annealing_mode,
                annealing_coefficient=self.annealing_coefficient,
            )
        if n_kl:
            alpha = kl_logits.relu() + 1
            invalid_kl = kl_divergence(alpha, num_classes=2, device=logits.device).mean()

        if accumulate_epoch:
            self._train_loss_sum += valid_edl.detach().double() * n_valid
            self._train_valid_count += n_valid
            self._train_kl_sum += invalid_kl.detach().double() * n_kl
            self._train_kl_count += n_kl
        return valid_edl + self._invalid_kl_weight() * invalid_kl

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        image = batch["image"]
        logits = self(image)
        loss = self.compute_loss(logits, batch, accumulate_epoch=True)
        with torch.no_grad():
            target = self.select_targets(batch["mask"])
            predictions = self.edl_logits_to_probs(logits).round().long()
            self._accumulate_epoch_confusions("train", target, predictions)
            if batch_idx == 0 and isinstance(self.logger, WandbLogger):
                self.log_images(image, target, logits, prefix="train_")
        # Log every step: Lightning writes step metrics every log_every_n_steps, so the
        # recorded values always belong to the recorded step.
        self.log("loss", loss)
        # Shows whether a batch gave the full invalid-KL mean to very few pixels.
        kl_pixels = ((target == 0) & (batch["kl_mask"] == 1)).sum()
        self.log("train_kl_fraction", kl_pixels / target.numel())
        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._train_kl_sum = 0.0
        self._train_kl_count = 0

    def on_train_epoch_end(self):
        # Each component is weighted by its own pixel count across the epoch.
        valid_edl = self._train_loss_sum / max(self._train_valid_count, 1)
        invalid_kl = self._train_kl_sum / max(self._train_kl_count, 1)
        self.log_dict({
            "train_valid_edl_epoch": valid_edl,
            "train_invalid_kl_epoch": invalid_kl,
            "train_kl_pixels_epoch": torch.tensor(self._train_kl_count, dtype=torch.float64, device=self.device),
            "train_loss_epoch": valid_edl + self._invalid_kl_weight() * invalid_kl,
            "train_kl_weight_epoch": self._invalid_kl_weight(),  # invalid-KL coefficient
            "train_beta_epoch": self._kl_weight(),  # built-in KL beta(t)
        }, on_step=False, on_epoch=True)
        self._log_epoch_confusions("train", self.train_epoch_cms)
        self.train_epoch_cms = {}
