import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from flood_uncertainty.data.sen1floods11 import collect_sen1_splits, create_sen1floods11_loaders


@pytest.fixture
def config(tmp_path):
    root = tmp_path / 'v1.1'
    splits = root / 'splits/flood_handlabeled'
    splits.mkdir(parents=True)
    for split, sample_id in [('train', 'A_1'), ('valid', 'B_1'), ('test', 'C_1')]:
        (splits / f'flood_{split}_data.txt').write_text(sample_id + '\n')
        for folder, suffix, values in [
            ('S1GRDHand', 'S1Hand', np.full((2, 32, 32), -10, dtype=np.float32)),
            ('S2L1CHand', 'S2Hand', np.full((13, 32, 32), 1000, dtype=np.float32)),
            ('LabelHand', 'LabelHand', np.zeros((1, 32, 32), dtype=np.float32))]:
            values[:, 0, 0] = np.nan if folder != 'LabelHand' else -1
            if folder == 'LabelHand':
                values[:, 1, 1] = 1
            path = root / 'data' / folder / f'{sample_id}_{suffix}.tif'
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, 'w', driver='GTiff', height=32, width=32,
                               count=values.shape[0], dtype='float32', transform=from_origin(0, 32, 1, 1), crs='EPSG:4326') as ds:
                ds.write(values)
    return {'root_path': str(tmp_path), 'channel_configuration': 'sar', 'batch_size': 1, 'num_workers': 0}


def test_native_tiles_official_normalization_and_labels(config):
    loaders = create_sen1floods11_loaders(config)
    batch = next(iter(loaders[1]))
    assert batch['image'].shape == (1, 2, 32, 32)
    assert torch.isfinite(batch['image']).all()
    assert batch['image'][0, 0, 1, 1].item() == pytest.approx(((40 / 51) - 0.6851) / 0.0820)
    assert batch['mask'].shape == (1, 1, 32, 32)
    assert batch['mask'][0, 0, 0, 0] == 0
    assert batch['mask'][0, 0, 1, 1] == 2
    assert batch['mask'][0, 0, 2, 2] == 1


def test_optical_uses_same_ids_and_water_channel(config):
    sar = collect_sen1_splits(config)
    config.update(modality='optical', channel_configuration='bgriswirs')
    batch = next(iter(create_sen1floods11_loaders(config)[1]))
    assert batch['id'] == [sar['val'][0].sample_id]
    assert batch['image'].shape == (1, 6, 32, 32)
    assert batch['mask'].shape == (1, 2, 32, 32)
    assert batch['mask'][0, 1, 1, 1] == 2
    assert not batch['mask'][:, 0].any()


def test_rejects_overlap_and_missing_pair(config):
    root = collect_sen1_splits(config)['train'][0].sar.parents[2]
    (root / 'splits/flood_handlabeled/flood_valid_data.txt').write_text('A_1\n')
    with pytest.raises(ValueError, match='overlapping'):
        collect_sen1_splits(config)
    (root / 'splits/flood_handlabeled/flood_valid_data.txt').write_text('B_1\n')
    collect_sen1_splits(config)['train'][0].optical.unlink()
    with pytest.raises(FileNotFoundError, match='paired'):
        collect_sen1_splits(config)


def test_sar_model_training_accepts_single_channel(config):
    from flood_uncertainty.models.edl import EDL_SAR_Unet
    from flood_uncertainty.utils.config_loader import load_mode_config
    cfg = load_mode_config('configurations/edl_sar_sen1.json')
    model = EDL_SAR_Unet(cfg.model_params, normalized_data=False)
    model.log = lambda *args, **kwargs: None
    batch = next(iter(create_sen1floods11_loaders(config)[0]))
    loss = model.training_step(batch, 1)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.network.conv_last.weight.grad is not None


def test_shared_crop_windows_preserve_split_and_labels(config):
    config['tile_size'] = 16
    loaders = create_sen1floods11_loaders(config)
    assert [len(loader.dataset) for loader in loaders] == [4, 4, 4]
    sar = loaders[1].dataset
    assert sar[0]['image'].shape == (2, 16, 16)
    assert sar[0]['mask'][0, 1, 1] == 2
    assert sar[1]['mask'][0, 1, 1] == 1
    config.update(modality='optical', channel_configuration='bgriswirs')
    optical = create_sen1floods11_loaders(config)[1].dataset
    for i in range(4):
        assert sar[i]['id'] == optical[i]['id']
        assert torch.equal(sar[i]['mask'][0], optical[i]['mask'][1])


def test_official_sar_clip_and_channel_statistics(config):
    sample = collect_sen1_splits(config)['val'][0]
    with rasterio.open(sample.sar, 'r+') as ds:
        values = ds.read()
        values[0, 2, 2] = -60
        values[1, 2, 2] = 10
        ds.write(values)
    # Legacy KuroSiwo settings must not affect this normalization.
    config.update(clamp_input=0.15, data_mean=[999, 999], data_std=[1, 1])
    batch = next(iter(create_sen1floods11_loaders(config)[1]))
    assert batch['image'][0, 0, 2, 2].item() == pytest.approx(-0.6851 / 0.0820)
    assert batch['image'][0, 1, 2, 2].item() == pytest.approx((1 - 0.5235) / 0.1102)
    assert batch['image'][0, 0, 0, 0].item() == pytest.approx((50 / 51 - 0.6851) / 0.0820)
    assert batch['mask'][0, 0, 0, 0] == 0


def test_optical_sen1_config_and_training_contract(config):
    from flood_uncertainty.models.edl import EDL_ML4FloodsModel
    from flood_uncertainty.utils.config_loader import load_mode_config
    cfg = load_mode_config('configurations/edl_optical_sen1.json')
    assert cfg.data_params.modality == 'optical'
    assert cfg.data_params.tile_size == 256
    assert cfg.model_params.hyperparameters.task_mode == 'water_only'
    assert cfg.model_params.hyperparameters.num_channels == 6
    assert cfg.wandb_enabled is True
    config.update(modality='optical', channel_configuration='bgriswirs')
    batch = next(iter(create_sen1floods11_loaders(config)[0]))
    model = EDL_ML4FloodsModel(cfg.model_params)
    model.log = lambda *args, **kwargs: None
    loss = model.training_step(batch, 1)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.network.conv_last.weight.grad is not None


def test_epoch_metrics_aggregate_empty_and_valid_batches(capsys):
    from flood_uncertainty.models.edl import EDL_ML4FloodsModel
    from flood_uncertainty.utils.config_loader import load_mode_config
    cfg = load_mode_config('configurations/edl_optical_sen1.json')
    model = EDL_ML4FloodsModel(cfg.model_params)
    logged = {}
    model.log = lambda name, value, **kwargs: logged.update({name: float(value)})
    model.on_train_epoch_start()
    model.on_validation_epoch_start()
    empty = torch.zeros((1, 1, 1, 4), dtype=torch.long)
    target = torch.tensor([[[[1, 1, 2, 2]]]])
    pred = torch.tensor([[[[0, 1, 0, 1]]]])
    for stage in ('train', 'val'):
        model._accumulate_epoch_confusions(stage, empty, pred)
        model._accumulate_epoch_confusions(stage, target, pred)
    model._train_loss_sum = 2.0
    model._train_valid_count = 4
    model.on_train_epoch_end()
    model.on_validation_epoch_end()
    for stage in ('train', 'val'):
        assert logged[f'{stage}_Global_valid_pixels_land_water'] == 4
        for metric in ('Acc', 'Precision', 'Recall', 'F1'):
            assert logged[f'{stage}_Global_{metric}_land_water'] == pytest.approx(0.5)
        assert logged[f'{stage}_Global_mIoU_land_water'] == pytest.approx(1 / 3)
    assert logged['train_loss_epoch'] == 0.5
    assert model.epoch_cms == model.train_epoch_cms == {}
    assert 'Valid pixels: 4' in capsys.readouterr().out
    model.on_train_epoch_start()
    assert model._train_valid_count == 0
