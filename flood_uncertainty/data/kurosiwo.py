"""Adapters for the external KuroSiwo dataset loaders."""

import importlib.util
import sys
from types import ModuleType
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def _import_prepare_loaders(config: Any):
    """Import KuroSiwo while keeping terrain support optional.

    The upstream dataset module imports ``richdem`` at module import time even
    when DEM and slope channels are disabled. A minimal placeholder is supplied
    only for the VV/VH-only path; requesting terrain data still requires the
    real dependency.
    """

    code_path = config.get("kurosiwo_code_path")
    if code_path and code_path not in sys.path:
        sys.path.insert(0, code_path)

    terrain_enabled = config.get("dem", False) or config.get("slope", False)
    richdem_available = importlib.util.find_spec("richdem") is not None
    if terrain_enabled and not richdem_available:
        raise ImportError(
            "richdem is required when KuroSiwo DEM or slope inputs are enabled"
        )

    previous_richdem = sys.modules.get("richdem")
    if not richdem_available:
        sys.modules["richdem"] = ModuleType("richdem")

    try:
        from utilities.utilities import prepare_loaders
    except ImportError as exc:
        raise ImportError(
            "Could not import the KuroSiwo loader. Set data_params."
            "kurosiwo_code_path to the KuroSiwo checkout."
        ) from exc
    finally:
        if not richdem_available:
            if previous_richdem is None:
                sys.modules.pop("richdem", None)
            else:
                sys.modules["richdem"] = previous_richdem

    return prepare_loaders


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

    prepare_loaders = _import_prepare_loaders(config)
    train_loader, val_loader, test_loader = prepare_loaders(config)
    return wrap_kurosiwo_loaders(
        train_loader,
        val_loader,
        test_loader,
        use_pre_event=config.get("use_pre_event", False),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
