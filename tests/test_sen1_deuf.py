import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

from flood_uncertainty.data.sen1floods11 import Sen1Sample, Sen1Floods11Dataset
from flood_uncertainty.fusion import dirichlet_binary_stats
from flood_uncertainty.visualization.paired import FIGURE_TYPES, plot_paired_tile, tile_prefix

spec = importlib.util.spec_from_file_location('infer_sen1_deuf', Path(__file__).resolve().parents[1] / 'scripts/analysis/infer_sen1_deuf.py')
deuf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deuf)


@pytest.mark.parametrize('all_invalid', [False, True])
def test_five_pngs_even_for_all_land_or_invalid(tmp_path, all_invalid):
    evidence = np.stack([np.full((4, 4), 2.), np.zeros((4, 4))])
    products = {m: dirichlet_binary_stats(evidence) for m in ['sar', 'optical', 'fused']}
    valid = np.full((4, 4), not all_invalid)
    metadata = dict(sample_id='Example_1', tile_index=7, row=256, col=0)
    paths = plot_paired_tile(products, np.ones((4, 4)), valid, np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 3)), metadata, tmp_path, dpi=30)
    assert set(paths) == set(FIGURE_TYPES)
    assert len(list(tmp_path.glob('*.png'))) == 5
    for path in paths.values():
        assert 'Example_1__tile_000007__r0256_c0000' in path.name
        with Image.open(path) as image:
            image.verify()
    assert not plt.get_fignums()


def test_streaming_exports_every_crop_and_distinct_filenames(tmp_path, monkeypatch):
    paths = {}
    for name, array in [('sar', np.full((2, 16, 16), -10., np.float32)),
                        ('optical', np.full((13, 16, 16), 1000., np.float32)),
                        ('label', np.zeros((1, 16, 16), np.float32))]:
        if name == 'label':
            array[:, :8, :8] = -1  # All-invalid crop must still be exported.
            array[:, 8:, 8:] = 1
        path = tmp_path / f'{name}.tif'
        with rasterio.open(path, 'w', driver='GTiff', height=16, width=16, count=len(array), dtype='float32',
                           crs='EPSG:4326', transform=from_origin(0, 16, 1, 1)) as dest:
            dest.write(array)
        paths[name] = path
    sample = Sen1Sample('Test_1', paths['sar'], paths['optical'], paths['label'])
    datasets = {m: Sen1Floods11Dataset([sample], {'modality': m, 'tile_size': 8, 'channel_configuration': 'sar' if m == 'sar' else 'bgriswirs'}) for m in ['sar', 'optical']}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.))
        @property
        def network(self):
            return self
        def forward(self, x):
            return torch.stack([torch.ones_like(x[:, 0]), 2 * torch.ones_like(x[:, 0])], dim=1)
    seen = []
    def render(products, target, valid, vv, vh, rgb, metadata, directory, **kwargs):
        np.testing.assert_allclose(products['fused']['alpha'], np.stack([np.full((8, 8), 3.), np.full((8, 8), 5.)]))
        seen.append(metadata['tile_index'])
        prefix = tile_prefix(metadata['sample_id'], metadata['tile_index'], metadata['row'], metadata['col'])
        paths = {kind: directory / f'{prefix}__{kind}.png' for kind in FIGURE_TYPES}
        for path in paths.values():
            path.write_bytes(b'placeholder for renderer')
        return paths
    monkeypatch.setattr(deuf, 'plot_paired_tile', render)
    result = deuf.export_tiles(datasets, {m: Model() for m in ['sar', 'optical']}, list(range(4)), tmp_path / 'DEUF', batch_size=3, dpi=30, save_products=True)
    assert seen == [0, 1, 2, 3]
    assert result['completed_tiles'] == 4
    assert result['pngs_written'] == 20
    assert result['water_gt20_tiles'] == 1
    with Path(result['water_gt20_csv']).open() as handle:
        water_rows = list(csv.DictReader(handle))
    assert water_rows[0]['tile_index'] == '3'
    assert water_rows[0]['gt_water_pixels'] == '64'
    with Path(result['manifest']).open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r['valid_pixels']) for r in rows] == [0, 64, 64, 64]
    assert len({r[k] for r in rows for k in FIGURE_TYPES}) == 20
    output_dir = tmp_path / 'DEUF'
    assert not list(output_dir.glob('*.png'))
    tile_dirs = {Path(r['sar_evidence']).parent for r in rows}
    assert len(tile_dirs) == 4
    for row in rows:
        expected = tile_prefix(row['sample_id'], int(row['tile_index']), int(row['row']), int(row['col']))
        for kind in FIGURE_TYPES:
            assert Path(row[kind]).parent == Path(expected)
            assert (output_dir / row[kind]).is_file()
        assert len(list((output_dir / expected).glob('*.png'))) == 5
        assert len(list((output_dir / expected / 'products').glob('*/*.npz'))) == 3
    assert not list(output_dir.glob('*.partial'))


def test_water_index_strictly_excludes_twenty_percent_and_empty_tiles(tmp_path):
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('tile_index,valid_pixels,gt_water_fraction\n0,0,\n1,5,0.2\n2,5,0.4\n3,10,0.1\n')
    result = deuf.write_water_tile_index(manifest)
    with Path(result['water_gt20_csv']).open() as handle:
        rows = list(csv.DictReader(handle))
    assert result['water_gt20_tiles'] == 1
    assert rows[0]['tile_index'] == '2'
    assert rows[0]['gt_water_pixels'] == '2'
