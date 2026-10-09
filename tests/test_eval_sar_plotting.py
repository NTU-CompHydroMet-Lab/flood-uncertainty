import json
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.eval import eval_sar


@pytest.mark.parametrize('filtered', [False, True])
@pytest.mark.parametrize('disabled', [False, True])
def test_plot_cap_does_not_limit_evaluation(tmp_path, monkeypatch, disabled, filtered):
    class Params(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    config = SimpleNamespace(data_params=Params(scale_input='normalize', channels=['vv', 'vh']))
    target = torch.tensor([[[[1, 1], [2, 2]]]]).repeat(2, 1, 1, 1)
    image = torch.ones(2, 2, 2, 2)
    batches = [{'image': image, 'mask': target}] * 2
    if filtered:
        # First batch has no eligible tiles; second has one at the exact boundary.
        batches = [
            {'image': image, 'mask': torch.tensor([[[[0, 0], [0, 0]]], [[[1, 1], [1, 1]]]])},
            {'image': image, 'mask': torch.tensor([[[[2, 1], [0, 0]]], [[[2, 1], [1, 1]]]])},
        ]

    class Loader:
        dataset = range(4)

        def __iter__(self):
            return iter(batches)

    class Model:
        def parameters(self):
            return iter([torch.tensor(0.)])

        def network(self, image):
            return image[:, :1] * 0.5

        def edl_logits_to_probs(self, logits):
            return logits

        def edl_logits_to_output(self, logits):
            return {'dst_u': torch.ones_like(logits)}

    monkeypatch.setattr(eval_sar, 'load_mode_config', lambda *a, **kw: config)
    monkeypatch.setattr(eval_sar, 'create_kurosiwo_loaders', lambda *a: (None, None, Loader()))
    monkeypatch.setattr(eval_sar, 'load_sar_model', lambda *a: Model())
    output = tmp_path / 'metrics.json'
    argv = ['eval_sar', '--weights', 'synthetic.ckpt', '--output', str(output), '--max-plots', '3']
    if filtered:
        argv.extend(['--min-water-fraction', '0.5'])
    if disabled:
        argv.append('--no-plot-png')
    monkeypatch.setattr(sys, 'argv', argv)
    eval_sar.main()
    result = json.loads(output.read_text())
    assert result['evaluated_tiles'] == 4
    assert result['evaluated_batches'] == 2
    if filtered:
        assert result['tp'] == 2
        assert result['fp'] == 8
        assert result['miou'] == pytest.approx(0.1)
        if not disabled:
            assert result['plotted_tiles'][0]['tile_index'] == 2
            assert result['plotted_tiles'][0]['water_fraction'] == 0.5
    else:
        assert result['tp'] == result['fp'] == 8
        assert result['miou'] == 0.25
    expected = 0 if disabled else (1 if filtered else 3)
    assert result['plots_written'] == expected
    assert len(list(tmp_path.glob('metrics_plots/*.png'))) == expected
