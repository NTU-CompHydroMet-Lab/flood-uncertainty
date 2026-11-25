"""Invalid-pixel KL loss and the KL dataset used by water-only EDL training."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
from rasterio.windows import Window

from flood_uncertainty.data.worldfloods_kl import WorldFloodsKLDatasetTiled
from flood_uncertainty.losses.edl_loss import calc_edl_loss_multioutput_logistic_mask_invalid
from flood_uncertainty.models.edl_water_kl import WaterOnlyInvalidKLEDL
from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
from ml4floods.models.dataset_setup import get_transformations
from ml4floods.preprocess.tiling import WindowSlices
from ml4floods.preprocess.worldfloods.normalize import get_normalisation

ROOT = Path(__file__).resolve().parents[1]
ARM_CONFIGS = ROOT / "configurations" / "edl_water_kl_ablation"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def load_config(arm="kl_002_const"):
    # A fresh AttrDict on every call; deepcopy would split its attribute and item views.
    return load_mode_config(str(ARM_CONFIGS / f"{arm}.json"), "train")


@lru_cache(maxsize=None)
def make_model(arm):
    """One CPU model per arm; compute_loss needs no forward pass and keeps no state here."""
    return WaterOnlyInvalidKLEDL(load_config(arm).model_params)


def set_epoch(monkeypatch, epoch):
    # LightningModule.current_epoch is a property that returns 0 without a Trainer.
    monkeypatch.setattr(WaterOnlyInvalidKLEDL, "current_epoch", property(lambda self: epoch))


def reference_kl(pixel_logits):
    """KL(Dir(alpha) || Dir(1, 1)) in float64, independent of mlguess; pixel_logits is [N, 2]."""
    alpha = pixel_logits.double().clamp_min(0) + 1
    total = alpha.sum(-1)
    return (torch.lgamma(total) - torch.lgamma(alpha).sum(-1)
            + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(total)[:, None])).sum(-1))


def baseline_loss(model, logits, water, epoch):
    return calc_edl_loss_multioutput_logistic_mask_invalid(
        logits, water, pos_weight_problem=model.pos_weight, weight_problem=model.weight_problem,
        epoch_num=epoch, annealing_step=model.annealing_step,
        annealing_mode=model.annealing_mode, annealing_coefficient=model.annealing_coefficient,
    )


def two_channel_mask(water):
    """Model input [B, 2, H, W]: cloud GT (all clear) and the given water GT."""
    return torch.stack([torch.ones_like(water), water], dim=1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("epoch", [0, 5, 12])
def test_without_kl_pixels_loss_equals_baseline(monkeypatch, epoch):
    model = make_model("kl_002_const")
    set_epoch(monkeypatch, epoch)
    generator = torch.Generator().manual_seed(epoch)
    logits = torch.randn(2, 2, 8, 8, generator=generator) * 3
    water = torch.randint(0, 3, (2, 8, 8), generator=generator)
    kl_mask = torch.zeros(2, 1, 8, 8, dtype=torch.long)
    loss = model.compute_loss(logits, {"mask": two_channel_mask(water), "kl_mask": kl_mask})
    torch.testing.assert_close(loss, baseline_loss(model, logits, water[:, None], epoch), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("arm, coefficient", [("kl_shared_beta", 0.5), ("kl_002_const", 0.02)])
def test_mixed_batch_adds_weighted_mean_kl_to_valid_loss(monkeypatch, arm, coefficient):
    """Unequal group sizes (3 valid, 2 KL, 1 ignored) expose a wrong denominator."""
    model = make_model(arm)
    set_epoch(monkeypatch, 5)
    water = torch.tensor([[[1, 2, 1], [0, 0, 0]]])
    kl_mask = torch.tensor([[[[0, 0, 0], [1, 1, 0]]]])
    logits = torch.tensor([[[[0.3, 0.1, 2.0], [0.5, 4.0, 2.0]],
                            [[1.2, 3.0, 0.4], [1.5, 0.2, 3.0]]]], requires_grad=True)
    loss = model.compute_loss(logits, {"mask": two_channel_mask(water), "kl_mask": kl_mask})

    kl_pixels = logits.detach()[0, :, 1, :2].T  # [2 pixels, 2 evidence channels]
    expected = baseline_loss(model, logits.detach(), water[:, None], 5) + coefficient * reference_kl(kl_pixels).mean()
    torch.testing.assert_close(loss.double(), expected.double(), rtol=1e-5, atol=1e-6)

    loss.backward()
    # The ignored pixel has positive logits, so ReLU cannot hide a wrongly included term.
    assert torch.equal(logits.grad[0, :, 1, 2], torch.zeros(2))
    assert logits.grad[0, :, 1, :2].abs().sum() > 0

    flagged = kl_mask.clone()
    flagged[0, 0, 0, 0] = 1  # A KL flag on a valid land label must not change the loss.
    toggled = model.compute_loss(logits.detach(), {"mask": two_channel_mask(water), "kl_mask": flagged})
    torch.testing.assert_close(toggled, loss.detach())


def test_fully_ignored_batch_gives_zero_loss_that_backpropagates(monkeypatch):
    model = make_model("kl_002_const")
    set_epoch(monkeypatch, 5)
    logits = torch.full((1, 2, 4, 4), 2.0, requires_grad=True)
    water = torch.zeros(1, 4, 4, dtype=torch.long)
    loss = model.compute_loss(logits, {"mask": two_channel_mask(water), "kl_mask": torch.zeros(1, 1, 4, 4, dtype=torch.long)})
    assert loss.requires_grad
    torch.testing.assert_close(loss.detach(), torch.tensor(0.0))
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_kl_only_batch_is_coefficient_times_mean_kl(monkeypatch):
    model = make_model("kl_002_const")
    set_epoch(monkeypatch, 0)
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(1, 2, 3, 5, generator=generator) * 4
    water = torch.zeros(1, 3, 5, dtype=torch.long)
    kl_mask = torch.ones(1, 1, 3, 5, dtype=torch.long)
    loss = model.compute_loss(logits, {"mask": two_channel_mask(water), "kl_mask": kl_mask})
    expected = 0.02 * reference_kl(logits[0].reshape(2, -1).T).mean()
    torch.testing.assert_close(loss.double(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("arm, epoch, invalid_weight, beta", [
    ("kl_off", 0, 0.0, 0.0), ("kl_off", 5, 0.0, 0.5),
    ("kl_shared_beta", 0, 0.0, 0.0), ("kl_shared_beta", 5, 0.5, 0.5),
    ("kl_002_beta", 0, 0.0, 0.0), ("kl_002_beta", 5, 0.01, 0.5),
    ("kl_002_const", 0, 0.02, 0.0), ("kl_002_const", 5, 0.02, 0.5),
])
def test_arm_coefficients(monkeypatch, arm, epoch, invalid_weight, beta):
    model = make_model(arm)
    set_epoch(monkeypatch, epoch)
    assert model._invalid_kl_weight() == pytest.approx(invalid_weight)
    assert model._kl_weight() == pytest.approx(beta)


def test_missing_keys_default_to_the_shared_beta_objective(monkeypatch):
    params = load_config().model_params
    del params["hyperparameters"]["invalid_kl_weight"]
    del params["hyperparameters"]["invalid_kl_anneal"]
    model = WaterOnlyInvalidKLEDL(params)
    set_epoch(monkeypatch, 5)
    assert (model.invalid_kl_weight, model.invalid_kl_anneal) == (1.0, True)
    assert model._invalid_kl_weight() == pytest.approx(0.5)


@pytest.mark.parametrize("key, value", [
    ("invalid_kl_weight", -0.1), ("invalid_kl_weight", float("nan")), ("invalid_kl_weight", float("inf")),
    ("invalid_kl_weight", True), ("invalid_kl_anneal", "false"), ("invalid_kl_anneal", 1),
])
def test_invalid_hyperparameters_are_rejected(key, value):
    params = load_config().model_params
    params["hyperparameters"][key] = value
    with pytest.raises(ValueError, match=key):
        WaterOnlyInvalidKLEDL(params)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SIZE = 8
BANDS = CHANNELS_CONFIGURATIONS["bgriswirs"]  # 0-based S2 band indices


def source_arrays():
    """Asymmetric synthetic rasters: every S2 pixel value is unique, KL is a subset of water GT 0."""
    image = (np.arange(SIZE * SIZE).reshape(SIZE, SIZE)[None] + 100 * np.arange(13)[:, None, None] + 1).astype(np.uint16)
    water = np.ones((SIZE, SIZE), dtype=np.uint8)
    water[0, :3] = 0
    water[1, 0] = 0
    water[4:6, 4:6] = 0
    water[6:, 5:] = 2
    kl = np.zeros((SIZE, SIZE), dtype=np.uint8)
    kl[0, :2] = 1
    kl[4:6, 4:6] = 1
    gt = np.stack([np.ones_like(water), water])
    return image, gt, kl


def write_event(tmp_path, name="scene", with_mask=True):
    image, gt, kl = source_arrays()
    data_root, mask_root = tmp_path / "data", tmp_path / "masks"
    profile = dict(driver="GTiff", height=SIZE, width=SIZE, crs="EPSG:4326", transform=from_origin(0, SIZE, 1, 1))
    paths = {
        "image": data_root / "train" / "S2" / f"{name}.tif",
        "gt": data_root / "train" / "gt" / f"{name}.tif",
        "kl": mask_root / "train" / "kl_mask" / f"{name}.tif",
    }
    arrays = {"image": image, "gt": gt, "kl": kl[None]}
    for key in ("image", "gt", "kl") if with_mask else ("image", "gt"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        array = arrays[key]
        with rasterio.open(paths[key], "w", count=array.shape[0], dtype=array.dtype.name, **profile) as dst:
            dst.write(array)
    return paths["image"], mask_root


def kl_dataset(image_path, mask_root, window=Window(0, 0, SIZE, SIZE), **kwargs):
    return WorldFloodsKLDatasetTiled(
        [WindowSlices(str(image_path), window)], kl_mask_root=mask_root,
        image_prefix="/S2/", gt_prefix="/gt/", bands=BANDS, **kwargs,
    )


def orient(array, turns, flip):
    rotated = np.rot90(array, turns, axes=(-2, -1))
    return np.flip(rotated, axis=-1) if flip else rotated


def test_event_without_kl_returns_zero_mask_without_a_file(tmp_path):
    image_path, mask_root = write_event(tmp_path, with_mask=False)
    _, gt, _ = source_arrays()
    sample = kl_dataset(image_path, mask_root, events_without_kl={"scene"})[0]
    np.testing.assert_array_equal(sample["mask"], gt)
    np.testing.assert_array_equal(sample["kl_mask"], np.zeros((1, SIZE, SIZE)))
    with pytest.raises(RasterioIOError):
        kl_dataset(image_path, mask_root)[0]  # Not listed: a missing mask is an error.


def test_joint_transform_keeps_image_gt_and_kl_aligned(tmp_path):
    image_path, mask_root = write_event(tmp_path)
    image, gt, kl = source_arrays()
    data = load_config().data_params
    train_transform, _ = get_transformations(data)
    mean, std = get_normalisation(data.channel_configuration, channels_first=True)
    normalized = (image[BANDS].astype(np.float64) - mean) / std
    dataset = kl_dataset(image_path, mask_root, transforms=train_transform)
    orientations = set()
    for seed in range(8):
        train_transform.set_random_seed(seed)
        sample = dataset[0]
        matches = [(turns, flip) for turns in range(4) for flip in (False, True)
                   if np.allclose(orient(normalized, turns, flip), sample["image"].numpy(), atol=1e-3)]
        assert len(matches) == 1, f"seed {seed}: image orientation is ambiguous or unmatched"
        turns, flip = matches[0]
        orientations.add(matches[0])
        np.testing.assert_array_equal(sample["mask"].numpy(), orient(gt, turns, flip))
        np.testing.assert_array_equal(sample["kl_mask"].numpy()[0], orient(kl, turns, flip))
        assert not np.any((sample["kl_mask"].numpy()[0] == 1) & (sample["mask"].numpy()[1] != 0))
    assert orientations - {(0, False)}, "no seed exercised a rotation or flip"


def test_window_past_the_edge_pads_kl_with_zero(tmp_path):
    image_path, mask_root = write_event(tmp_path)
    _, _, kl = source_arrays()
    sample = kl_dataset(image_path, mask_root, window=Window(4, 4, SIZE, SIZE))[0]
    kl_tile = sample["kl_mask"][0]
    np.testing.assert_array_equal(kl_tile[:4, :4], kl[4:, 4:])
    assert kl_tile[:4, :4].any()  # The retained part contains KL pixels.
    assert not kl_tile[4:].any() and not kl_tile[:, 4:].any()
