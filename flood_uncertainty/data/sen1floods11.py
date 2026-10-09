"""Paired Sen1Floods11 samples with native tile sizes and existing splits."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
from rasterio.windows import Window

from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
from ml4floods.preprocess.worldfloods.normalize import get_normalisation


@dataclass(frozen=True)
class Sen1Sample:
    sample_id: str
    sar: Path
    optical: Path
    label: Path


def collect_sen1_splits(config):
    root = Path(config['root_path'])
    if not (root / 'data').is_dir() and (root / 'v1.1/data').is_dir():
        root = root / 'v1.1'
    split_root = Path(config.get('splits_path', root / 'splits/flood_handlabeled'))
    result = {}
    seen = set()
    for split, filename in [('train', 'flood_train_data.txt'), ('val', 'flood_valid_data.txt'), ('test', 'flood_test_data.txt')]:
        ids = [line.strip() for line in (split_root / filename).read_text().splitlines() if line.strip()]
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f'Empty, duplicate or overlapping Sen1 split: {split}')
        seen.update(ids)
        samples = []
        for sample_id in ids:
            if Path(sample_id).name != sample_id:
                raise ValueError(f'Invalid sample ID: {sample_id}')
            sample = Sen1Sample(sample_id,
                root / 'data' / config.get('sar_folder', 'S1GRDHand') / f'{sample_id}_S1Hand.tif',
                root / 'data' / config.get('optical_folder', 'S2L1CHand') / f'{sample_id}_S2Hand.tif',
                root / 'data/LabelHand' / f'{sample_id}_LabelHand.tif')
            for path in (sample.sar, sample.optical, sample.label):
                if not path.is_file():
                    raise FileNotFoundError(f'Missing paired Sen1 file: {path}')
            samples.append(sample)
        result[split] = samples
    return result


class Sen1Floods11Dataset(Dataset):
    """Select paired SAR/optical with shared crop windows; never resize pixels."""
    def __init__(self, samples, config, *, augment=False):
        self.samples = samples
        self.config = config
        self.modality = config.get('modality', 'sar')
        if self.modality not in ('sar', 'optical'):
            raise ValueError('modality must be sar or optical')
        if self.modality == 'sar' and (config.get('num_channels', 2) != 2 or config.get('use_pre_event', False)):
            raise ValueError('Sen1 SAR provides only two event channels: VV, VH')
        self.augment = augment
        self.windows = []
        size = config.get('tile_size')
        if size is not None and (not isinstance(size, int) or size <= 0):
            raise ValueError('tile_size must be a positive integer')
        for sample in samples:
            if size is None:
                self.windows.append((sample, None))
                continue
            with rasterio.open(sample.sar) as source:
                height, width = source.shape
            if height % size or width % size:
                raise ValueError(f'Tile size must divide native shape: {sample.sample_id}')
            for row in range(0, height, size):
                for col in range(0, width, size):
                    self.windows.append((sample, Window(col, row, size, size)))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        sample, window = self.windows[index]
        path = sample.sar if self.modality == 'sar' else sample.optical
        with rasterio.open(path) as source, rasterio.open(sample.label) as target:
            if (source.shape != target.shape or source.crs != target.crs
                    or not np.allclose(tuple(source.transform), tuple(target.transform), rtol=0, atol=1e-10)):
                raise ValueError(f'Image/label grid mismatch: {sample.sample_id}')
            bands = [1, 2] if self.modality == 'sar' else [b + 1 for b in CHANNELS_CONFIGURATIONS[self.config['channel_configuration']]]
            image = source.read(bands, masked=True, window=window)
            label = target.read(1, masked=True, window=window)
        if not np.isin(label.compressed(), [-1, 0, 1]).all():
            raise ValueError(f'Unexpected Sen1 labels: {sample.sample_id}')
        valid = (~np.ma.getmaskarray(image).any(axis=0) & np.isfinite(image.data).all(axis=0)
                 & ~np.ma.getmaskarray(label) & (label.data != -1))
        mask = np.zeros((1, *label.shape), dtype=np.int64)
        mask[0, valid & (label.data == 0)] = 1
        mask[0, valid & (label.data == 1)] = 2
        values = image.astype(np.float32).filled(np.nan)
        if self.modality == 'sar':
            if self.config.get('sar_units', 'db') != 'db':
                raise ValueError('Official Sen1 SAR normalization requires dB input')
            # Official Train.ipynb: fill NaNs, clip dB, scale, then normalize VV/VH.
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            values = (np.clip(values, -50.0, 1.0) + 50.0) / 51.0
            mean = np.asarray([0.6851, 0.5235], dtype=np.float32)[:, None, None]
            std = np.asarray([0.0820, 0.1102], dtype=np.float32)[:, None, None]
            values = (values - mean) / std
        else:
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            mean, std = get_normalisation(self.config['channel_configuration'], channels_first=True)
            values = (values * self.config.get('optical_scale', 1.0) - mean) / std
            # Existing optical model selects the second (water) GT channel.
            mask = np.concatenate([np.zeros_like(mask), mask])
        if self.augment:
            k = int(torch.randint(4, ()).item())
            values, mask = np.rot90(values, k, axes=(-2, -1)), np.rot90(mask, k, axes=(-2, -1))
            for axis in (-1, -2):
                if torch.rand(()).item() < 0.5:
                    values, mask = np.flip(values, axis), np.flip(mask, axis)
        return {'image': torch.from_numpy(np.ascontiguousarray(values)).float(),
                'mask': torch.from_numpy(np.ascontiguousarray(mask)).long(), 'id': sample.sample_id}


def create_sen1floods11_loaders(config):
    splits = collect_sen1_splits(config)
    return tuple(DataLoader(Sen1Floods11Dataset(samples, config,
                      augment=split == 'train' and config.get('data_augmentations', False)),
                      batch_size=config.get('batch_size', 16), shuffle=split == 'train',
                      num_workers=config.get('num_workers', 0), pin_memory=True)
                 for split, samples in splits.items())
