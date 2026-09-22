"""Adapters for the external KuroSiwo dataset loaders."""

from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class KuroSiwoToEDL(Dataset):
    """Convert a KuroSiwo sample tuple to the package batch contract.

    KuroSiwo labels are mapped as ``0, 1, 2, 3`` to ``1, 2, 2, 0`` for
    land, permanent water, flood, and invalid pixels respectively.
    """

    def __init__(self, dataset: Dataset, use_pre_event: bool = False):
        self.dataset = dataset
        self.use_pre_event = use_pre_event

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        image = sample[2]
        mask = sample[3].unsqueeze(0)

        if self.use_pre_event:
            image = torch.cat((image, sample[6], sample[9]), dim=0)

        mapped_mask = torch.zeros_like(mask)
        mapped_mask[mask == 0] = 1
        mapped_mask[(mask == 1) | (mask == 2)] = 2
        return {"image": image, "mask": mapped_mask}


def wrap_kurosiwo_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    use_pre_event: bool = False,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap existing KuroSiwo loaders without importing KuroSiwo itself."""

    def make(loader: DataLoader, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            KuroSiwoToEDL(loader.dataset, use_pre_event=use_pre_event),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

    return (
        make(train_loader, shuffle=True, drop_last=True),
        make(val_loader, shuffle=False, drop_last=False),
        make(test_loader, shuffle=False, drop_last=False),
    )


def create_kurosiwo_loaders(config: Any):
    """Create wrapped loaders using the externally installed KuroSiwo package.

    The import is intentionally local so importing ``flood_uncertainty`` does
    not require KuroSiwo unless the SAR data path is actually used.
    """

    try:
        from utilities.utilities import prepare_loaders
    except ImportError as exc:
        raise ImportError(
            "KuroSiwo must be on PYTHONPATH to create SAR data loaders"
        ) from exc

    train_loader, val_loader, test_loader = prepare_loaders(config)
    return wrap_kurosiwo_loaders(
        train_loader,
        val_loader,
        test_loader,
        use_pre_event=config.get("use_pre_event", False),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
