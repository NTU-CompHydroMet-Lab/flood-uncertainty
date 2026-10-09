"""Exercise water-only training and legacy checkpoint compatibility."""
from pathlib import Path

import numpy as np
import pytest
import torch

from flood_uncertainty.models.edl import EDL_ML4FloodsModel
from flood_uncertainty.inference.infer import load_model, load_inference_function, save_prediction_tif
from flood_uncertainty.utils.config_loader import load_mode_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(water=True):
    cfg = load_mode_config(str(ROOT / 'configurations' / ('edl_water.json' if water else 'edl.json')), 'train')
    return EDL_ML4FloodsModel(cfg.model_params), cfg


def test_water_training_uses_water_gt_and_ignores_cloud_labels():
    model, _ = make_model()
    model.log = lambda *a, **kw: None
    image = torch.randn(1, 6, 16, 16)
    target = torch.ones(1, 2, 16, 16, dtype=torch.long)
    target[:, 1, :, 8:] = 2
    target[:, 1, :2] = 0
    loss = model.training_step({'image': image, 'mask': target}, 1)
    altered = target.clone()
    altered[:, 0] = 2
    other = model.training_step({'image': image, 'mask': altered}, 1)
    torch.testing.assert_close(loss, other)
    loss.backward()
    assert model.network.conv_last.weight.grad.shape[0] == 2
    assert torch.isfinite(loss)
    model.validation_step({'image': image, 'mask': target}, 1)


def test_old_unet_water_head_can_initialize_single_task():
    old, _ = make_model(False)
    new, _ = make_model()
    assert new.load_pretrained_weights(old.state_dict()) == []
    image = torch.randn(1, 6, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(new(image), old(image)[:, 2:4])


@pytest.mark.parametrize('water', [False, True])
def test_checkpoint_settings_control_inference_architecture(tmp_path, water):
    model, cfg = make_model(water)
    path = tmp_path / 'model.ckpt'
    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams)}, path)
    # Deliberately supply the opposite config: checkpoint controls architecture.
    other = ROOT / 'configurations' / ('edl.json' if water else 'edl_water.json')
    loaded, channels, runtime = load_model(str(other), 'EDL', weights_path=str(path))
    assert loaded.num_class == (1 if water else 2)
    assert loaded.task_mode == model.task_mode


def test_water_inference_and_tiff_keep_evaluation_band_contract(tmp_path, monkeypatch):
    model, cfg = make_model()
    # Deterministic evidence favors water everywhere, without cloud or flood-trace output.
    with torch.no_grad():
        model.network.conv_last.weight.zero_()
        model.network.conv_last.bias.copy_(torch.tensor([0., 8.]))
    infer, _ = load_inference_function(model, cfg, apply_normalization=False,
                                      distinguish_flood_traces=True)
    image = torch.ones(6, 16, 16)
    image[:, 0, 0] = 0
    classification, prob, dst, evidence, aleatoric, epistemic = infer(image)
    assert set(classification.unique().tolist()) == {0, 2}
    assert prob.shape == (1, 16, 16)
    assert evidence.shape == (2, 16, 16)
    assert prob[0, 0, 0] == -1
    import flood_uncertainty.inference.infer as module
    captured = {}
    monkeypatch.setattr(module, 'save_cog', lambda raster, path, **kw: captured.update(raster=raster, **kw))
    from georeader.geotensor import GeoTensor
    from affine import Affine
    raster = GeoTensor(image.numpy(), transform=Affine.identity(), crs='EPSG:4326', fill_value_default=0)
    save_prediction_tif(classification, prob, dst, evidence, aleatoric, epistemic,
                        raster, 'sample', 'EDL', str(tmp_path))
    assert captured['raster'].values.shape == (7, 16, 16)
    np.testing.assert_allclose(captured['raster'].values[2], prob[0].numpy())


def test_water_zarr_contains_only_water_and_appends(tmp_path, monkeypatch):
    import xarray as xr
    model, cfg = make_model()
    model.zarr_save_path = str(tmp_path / 'water.zarr')
    writes = []
    monkeypatch.setattr(xr.Dataset, 'to_zarr',
                        lambda ds, path, **kw: writes.append((ds.copy(), kw)))
    x = torch.ones(1, 6, 16, 16)
    y = torch.ones(1, 1, 16, 16, dtype=torch.long)
    logits = torch.zeros(1, 2, 16, 16)
    probs = torch.full((1, 1, 16, 16), .5)
    model._append_batch_to_zarr(x, y, logits, probs, 0)
    model._append_batch_to_zarr(x, y, logits, probs, 1)
    assert set(writes[0][0].data_vars) == {'input', 'ground_truth_water', 'logits_water', 'pred_probs_water'}
    assert writes[0][1]['mode'] == 'w'
    assert writes[1][1]['append_dim'] == 'item'
    assert writes[1][0].item.values.tolist() == [1]
