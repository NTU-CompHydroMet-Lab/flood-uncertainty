"""Paired optical + SAR Sen1Floods11 windows for a multi-modal (S2 + S1) model.

Wraps two :class:`Sen1Floods11Dataset` instances (one optical config, one SAR config) that share the
same samples and crop windows. Each item stacks the optical channels then VV, VH on the channel axis,
which is the order :class:`MultiModalTerraMindBackbone` expects for ``["S2L1C", "S1GRD"]``.
The water label is kept only where the optical image, the SAR image and the GT are all valid (the same
mask policy as the paired DEUF evaluation); the mask uses the optical layout (dummy cloud task, water).
"""
import copy

import torch
from torch.utils.data import DataLoader, Dataset

from flood_uncertainty.data.sen1floods11 import Sen1Floods11Dataset, collect_sen1_splits


def split_configs(config):
    """Optical and SAR loader configs from one paired config (``optical_*`` / ``sar_*`` keys)."""
    optical, sar = copy.deepcopy(dict(config)), copy.deepcopy(dict(config))
    optical.update(modality="optical", channel_configuration=config["optical_channel_configuration"])
    sar.update(modality="sar", channel_configuration="sar", num_channels=2, use_pre_event=False)
    return optical, sar


class PairedSen1Floods11Dataset(Dataset):
    def __init__(self, optical_samples, sar_samples, optical_config, sar_config):
        self.optical = Sen1Floods11Dataset(optical_samples, optical_config, augment=False)
        self.sar = Sen1Floods11Dataset(sar_samples, sar_config, augment=False)
        if self.optical.windows != self.sar.windows:
            raise ValueError("Optical and SAR crop windows must match")
        self.windows = self.optical.windows

    def __len__(self):
        return len(self.optical)

    def __getitem__(self, index):
        o, s = self.optical[index], self.sar[index]
        if o["id"] != s["id"]:
            raise ValueError("Paired sample IDs differ")
        water_o, water_s = o["mask"][1], s["mask"][0]
        both = (water_o != 0) & (water_s != 0)
        if not torch.equal(water_o[both], water_s[both]):
            raise ValueError(f"Paired GT labels differ: {o['id']}")
        mask = torch.zeros_like(o["mask"])
        mask[1][both] = water_o[both]
        return {"image": torch.cat([o["image"], s["image"]], dim=0), "mask": mask, "id": o["id"]}


def create_paired_sen1floods11_loaders(config):
    optical_config, sar_config = split_configs(config)
    o_splits, s_splits = collect_sen1_splits(optical_config), collect_sen1_splits(sar_config)
    for split in o_splits:
        if [x.sample_id for x in o_splits[split]] != [x.sample_id for x in s_splits[split]]:
            raise ValueError("Optical / SAR split IDs and order must match")
    if config.get("data_augmentations", False):
        raise NotImplementedError("Augmentation is not implemented for paired windows")
    return tuple(
        DataLoader(PairedSen1Floods11Dataset(o_splits[split], s_splits[split], optical_config, sar_config),
                   batch_size=config.get("batch_size", 16), shuffle=split == "train",
                   num_workers=config.get("num_workers", 0), pin_memory=True)
        for split in o_splits
    )
