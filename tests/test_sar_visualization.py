import numpy as np
import pytest
from PIL import Image

from flood_uncertainty.visualization.sar import plot_sar_result


def test_sar_plot_counts_threshold_and_png(tmp_path):
    target = np.array([[1, 1, 2], [2, 0, 3]])
    probability = np.array([[0.1, 0.5, 0.5], [0.1, np.nan, np.nan]])
    vv = np.full((2, 3), -15.)
    result = plot_sar_result(vv, vv, target, probability, tmp_path / 'nested/result.png',
                             input_scale='db', uncertainty=np.ones((2, 3)), dpi=30)
    np.testing.assert_array_equal(result['cm_map'], [[1, 2, 4], [3, 0, 0]])
    assert result['metrics']['miou'] == pytest.approx(1 / 3)
    assert result['metrics']['valid_pixels'] == 4
    assert result['metrics']['f1'] == 0.5
    with Image.open(result['png_path']) as image:
        assert image.format == 'PNG'
        assert image.width > 0
    np.testing.assert_array_equal(vv, np.full((2, 3), -15.))


@pytest.mark.parametrize('scale', ['linear', 'normalized'])
def test_no_valid_pixels(tmp_path, scale):
    values = np.zeros((2, 2))
    result = plot_sar_result(values, values, values, np.full((2, 2), np.nan),
                             tmp_path / 'empty.png', input_scale=scale, dpi=20)
    assert result['metrics']['miou'] is None
    assert result['metrics']['water_iou'] is None


def test_absent_class_and_bad_probability(tmp_path):
    values = np.ones((2, 2))
    result = plot_sar_result(values, values, values, values * 0,
                             tmp_path / 'land.png', input_scale='linear', dpi=20)
    assert result['metrics']['water_iou'] is None
    assert result['metrics']['miou'] == 1
    with pytest.raises(ValueError, match='probabilities'):
        plot_sar_result(values, values, values, values * np.nan,
                        tmp_path / 'bad.png', input_scale='db')
    with pytest.raises(ValueError, match='spatial shape'):
        plot_sar_result(values[:1], values, values, values,
                        tmp_path / 'bad.png', input_scale='db')
