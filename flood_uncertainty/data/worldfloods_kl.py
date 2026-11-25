from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import numpy as np

from ml4floods.data.worldfloods.configs import BANDS_S2
from ml4floods.data.worldfloods.dataset import WorldFloodsDatasetTiled, rasterio_read
from ml4floods.preprocess.tiling import WindowSlices


class WorldFloodsKLDatasetTiled(WorldFloodsDatasetTiled):
    """Return an aligned S2 tile, two-channel [cloud GT, water GT], and one-channel KL mask.

    Events in ``events_without_kl`` have no annotated AOI. They keep their valid labels,
    but their KL mask is all zero and no mask file is read.
    """

    def __init__(
        self,
        list_of_windows: list[WindowSlices],
        *,
        kl_mask_root: str | Path = "artifacts/kl_masks/worldfloods_v2",
        image_prefix: str = "/S2/",
        gt_prefix: str = "/gt/",
        transforms: Callable[..., dict] | None = None,
        bands: Sequence[int] = range(len(BANDS_S2)),
        mndwi_indices: list[int] | None = None,
        lock_read: bool = False,
        events_without_kl: Collection[str] = (),
    ) -> None:
        super().__init__(
            list_of_windows=list_of_windows,
            image_prefix=image_prefix,
            gt_prefix=gt_prefix,
            transforms=None,  # Apply the transform after stacking the KL mask.
            bands=list(bands),
            mndwi_indices=mndwi_indices,
            lock_read=lock_read,
        )
        self.kl_mask_root = Path(kl_mask_root)
        self.events_without_kl = frozenset(events_without_kl)  # event stems
        self.joint_transform = transforms

    def _read_kl_mask(self, tile: WindowSlices, shape: tuple[int, int]) -> np.ndarray:
        source = Path(tile.file_name)  # <split>/S2/<event>.tif or <split>/gt/<event>.tif
        if source.stem in self.events_without_kl:
            return np.zeros((1, *shape), dtype=np.uint8)  # No AOI: no pixel is selected for KL.
        kl_path = self.kl_mask_root / source.parent.parent.name / "kl_mask" / source.name
        return rasterio_read(
            str(kl_path),
            self._lock,
            channels=[1],
            kwargs_rasterio={"window": tile.window, "boundless": True, "fill_value": 0},
        )

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        kl_mask = self._read_kl_mask(self.list_of_windows[idx], sample["mask"].shape[1:])
        sample["mask"] = np.concatenate((sample["mask"], kl_mask), axis=0)
        if self.joint_transform is not None:
            sample = self.joint_transform(image=sample["image"], mask=sample["mask"])
        joint_mask = sample["mask"]
        if joint_mask.ndim != 3 or joint_mask.shape[0] != 3:
            raise ValueError(f"Expected [cloud GT, water GT, KL mask] with shape (3, H, W), got {joint_mask.shape}")
        sample["mask"] = joint_mask[:2]
        sample["kl_mask"] = joint_mask[2:3]
        return sample
