"""Joint optical datasets for WorldFloods and Sen1Floods11.

Both datasets expose the same binary segmentation sample contract without
requiring the model to know which dataset produced a sample.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler


S2_BANDS = 13


@dataclass(frozen=True)
class OpticalSample:
    image_path: Path
    label_path: Path
    sample_id: str
    source: str


def _read_tiff(path: Path, first_bands: int | None = None) -> np.ndarray:
    with rasterio.open(path) as source:
        if first_bands is None:
            return source.read()
        if source.count < first_bands:
            raise ValueError(f"Expected at least {first_bands} bands in {path}, got {source.count}")
        # WorldFloods may append auxiliary bands; the first 13 are the common
        # Sentinel-2 B1..B12 sequence used by both datasets.
        return source.read(list(range(1, first_bands + 1)))


def _resize(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    image_tensor = torch.from_numpy(image[None]).float()
    mask_tensor = torch.from_numpy(mask[None, None]).float()
    image_tensor = torch.nn.functional.interpolate(
        image_tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    mask_tensor = torch.nn.functional.interpolate(
        mask_tensor, size=(size, size), mode="nearest"
    )
    return image_tensor[0].numpy(), mask_tensor[0, 0].numpy().astype(np.int64)


def _normalise_s2(image: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    if image.shape[0] != S2_BANDS:
        raise ValueError(f"Expected {S2_BANDS} Sentinel-2 bands, got {image.shape[0]}")
    image = _to_common_s2_scale(image)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return (image - np.asarray(mean, dtype=np.float32)[:, None, None]) / np.asarray(
        std, dtype=np.float32
    )[:, None, None]


def _to_common_s2_scale(image: np.ndarray) -> np.ndarray:
    """Use Sentinel-2 reflectance scaled by 10000 for both sources."""
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        image = image * 10000.0
    return image


def _binary_mask(label: np.ndarray, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Return Task 1 labels and pixels eligible for the loss.

    The Task 1 encoding is 0=invalid, 1=land, 2=water. Sen1Floods11 uses
    -1/0/1 for nodata/no-water/water, while WorldFloods v2 already stores the
    water task in the second mask channel as 0/1/2.
    """
    if source == "sen1floods11":
        label = label[0] if label.ndim == 3 else label
        valid = label != -1
        target = np.zeros(label.shape, dtype=np.int64)
        target[label == 0] = 1
        target[label == 1] = 2
        return target, valid

    if label.ndim == 3:
        if label.shape[0] < 2:
            raise ValueError("WorldFloods mask must contain the water channel")
        label = label[1]
    valid = label != 0
    target = label.astype(np.int64, copy=False)
    if not np.isin(target, (0, 1, 2)).all():
        raise ValueError(
            f"WorldFloods Task 1 labels must be 0/1/2, got {np.unique(target).tolist()}"
        )
    return target, valid


class OpticalDataset(Dataset):
    """Base dataset returning normalized ``image`` and binary ``mask`` tensors."""

    def __init__(
        self,
        samples: Sequence[OpticalSample],
        mean: Sequence[float],
        std: Sequence[float],
        image_size: int = 256,
        include_metadata: bool = False,
    ) -> None:
        if len(mean) != S2_BANDS or len(std) != S2_BANDS:
            raise ValueError("Sentinel-2 mean and std must each contain 13 values")
        self.samples = list(samples)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.image_size = image_size
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = _read_tiff(sample.image_path, first_bands=S2_BANDS)
        label = _read_tiff(sample.label_path)
        image = _normalise_s2(image, self.mean, self.std)
        mask, valid_mask = _binary_mask(label, sample.source)
        image, mask = _resize(image, mask, self.image_size)
        _, valid_mask = _resize(valid_mask.astype(np.float32)[None], valid_mask, self.image_size)

        result: dict[str, torch.Tensor | str] = {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask).long(),
            "valid_mask": torch.from_numpy(valid_mask.astype(bool)),
        }
        if self.include_metadata:
            result["id"] = sample.sample_id
            result["source"] = sample.source
        return result


def _sen1_samples(root: Path, split_file: Path) -> list[OpticalSample]:
    samples = []
    for sample_id in split_file.read_text().splitlines():
        sample_id = sample_id.strip()
        if not sample_id:
            continue
        # v1.1 stores the L1C product in S2L1CHand but retains the official
        # S2Hand filename suffix.
        image = root / "data" / "S2L1CHand" / f"{sample_id}_S2Hand.tif"
        label = root / "data" / "LabelHand" / f"{sample_id}_LabelHand.tif"
        samples.append(OpticalSample(image, label, sample_id, "sen1floods11"))
    return samples


def _worldfloods_samples(split_json: Path, split: str) -> list[OpticalSample]:
    payload = json.loads(split_json.read_text())

    def resolve_path(filename: str) -> Path:
        path = Path(filename)
        if path.exists():
            return path
        parts = path.parts
        if split in parts:
            split_index = parts.index(split)
            candidate = split_json.parent / Path(*parts[split_index:])
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"WorldFloods file does not exist: {path}; "
            f"also tried relative to {split_json.parent}"
        )

    return [
        OpticalSample(resolve_path(image), resolve_path(label), Path(image).stem, "worldfloods")
        for image, label in zip(payload[split]["S2"], payload[split]["gt"])
    ]


def collect_joint_train_samples(
    worldfloods_split_json: str | Path,
    sen1floods11_root: str | Path,
    sen1floods11_splits: str | Path,
) -> list[OpticalSample]:
    """Return both train splits for one shared mean/std calculation."""
    worldfloods_split_json = Path(worldfloods_split_json)
    sen1_root = Path(sen1floods11_root)
    sen1_splits = Path(sen1floods11_splits)
    return _worldfloods_samples(worldfloods_split_json, "train") + _sen1_samples(
        sen1_root,
        sen1_splits / "flood_handlabeled" / "flood_train_data.txt",
    )


def save_joint_train_statistics(
    samples: Iterable[OpticalSample],
    output_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and save shared train statistics as a small JSON artifact."""
    mean, std = compute_train_statistics(samples)
    Path(output_path).write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2) + "\n"
    )
    return mean, std


def make_joint_optical_datasets(
    worldfloods_split_json: str | Path,
    sen1floods11_root: str | Path,
    sen1floods11_splits: str | Path,
    mean: Sequence[float],
    std: Sequence[float],
    image_size: int = 256,
    include_metadata: bool = False,
) -> dict[str, Dataset]:
    """Build train/val/test datasets using each source's existing split files."""
    worldfloods_split_json = Path(worldfloods_split_json)
    sen1_root = Path(sen1floods11_root)
    sen1_splits = Path(sen1floods11_splits)
    result = {}
    for split in ("train", "val", "test"):
        sen1_split_name = {"train": "flood_train_data.txt", "val": "flood_valid_data.txt", "test": "flood_test_data.txt"}[split]
        world = OpticalDataset(
            _worldfloods_samples(worldfloods_split_json, split), mean, std, image_size, include_metadata
        )
        sen1 = OpticalDataset(
            _sen1_samples(sen1_root, sen1_splits / "flood_handlabeled" / sen1_split_name),
            mean, std, image_size, include_metadata,
        )
        result[split] = ConcatDataset([world, sen1])
    return result


def compute_train_statistics(
    samples: Iterable[OpticalSample],
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-band mean/std from valid pixels in training samples.

    Images are processed in windows so large WorldFloods rasters do not need
    to be loaded into memory in one operation.
    """
    count = np.zeros(S2_BANDS, dtype=np.float64)
    total = np.zeros(S2_BANDS, dtype=np.float64)
    total_squared = np.zeros(S2_BANDS, dtype=np.float64)
    for sample in samples:
        with rasterio.open(sample.image_path) as image_source, rasterio.open(sample.label_path) as label_source:
            if image_source.count < S2_BANDS:
                raise ValueError(f"Expected at least {S2_BANDS} image bands, got {image_source.count}")
            if (image_source.height, image_source.width) != (label_source.height, label_source.width):
                raise ValueError(f"Image/label shape mismatch for {sample.sample_id}")
            for row in range(0, image_source.height, chunk_size):
                for col in range(0, image_source.width, chunk_size):
                    height = min(chunk_size, image_source.height - row)
                    width = min(chunk_size, image_source.width - col)
                    window = Window(col, row, width, height)
                    image = _to_common_s2_scale(
                        image_source.read(list(range(1, S2_BANDS + 1)), window=window)
                    ).astype(np.float64)
                    label = label_source.read(window=window)
                    _, valid_mask = _binary_mask(label, sample.source)
                    pixels = image.reshape(S2_BANDS, -1)
                    valid = valid_mask.reshape(-1) & np.isfinite(pixels).all(axis=0)
                    pixels = pixels[:, valid]
                    count += pixels.shape[1]
                    total += pixels.sum(axis=1)
                    total_squared += np.square(pixels).sum(axis=1)
    if np.any(count == 0):
        raise ValueError("Cannot compute statistics: at least one band has no finite pixels")
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def make_balanced_train_loader(
    train_dataset: ConcatDataset,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Sample each source equally when their train-set sizes differ."""
    source_lengths = [len(dataset) for dataset in train_dataset.datasets]
    weights = torch.cat(
        [torch.full((length,), 1.0 / length) for length in source_lengths]
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)