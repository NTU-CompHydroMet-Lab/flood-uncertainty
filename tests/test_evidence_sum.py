import numpy as np
import pytest
from flood_uncertainty.fusion import fuse_evidence_sum, dirichlet_binary_stats


def test_prior_added_once_and_hand_computed_posterior():
    result = dirichlet_binary_stats(fuse_evidence_sum([2, 3], [5, 7]))
    np.testing.assert_array_equal(result['alpha'], [8, 11])
    assert result['probability'] == pytest.approx(11 / 19)
    assert result['dst_u'] == pytest.approx(2 / 19)
    assert result['aleatoric'] + result['epistemic'] == pytest.approx(11 * 8 / 19**2)


def test_no_evidence_is_uniform_and_fusion_is_symmetric():
    result = dirichlet_binary_stats([0, 0])
    assert result['probability'] == .5
    assert result['dst_u'] == 1
    np.testing.assert_array_equal(fuse_evidence_sum([0, 2], [3, 0]), fuse_evidence_sum([3, 0], [0, 2]))
    np.testing.assert_array_equal(fuse_evidence_sum([0, 2], [0, 0]), [0, 2])


def test_conflict_does_not_imply_high_vacuity():
    result = dirichlet_binary_stats(fuse_evidence_sum([100, 0], [0, 100]))
    assert result['probability'] == .5
    assert result['dst_u'] == pytest.approx(2 / 202)


@pytest.mark.parametrize('a,b', [([-1, 0], [0, 0]), ([np.nan, 0], [0, 0]), ([0, np.inf], [0, 0]), (np.zeros((2, 1)), np.zeros((2, 2))), ([0, 1, 2], [0, 1, 2])])
def test_invalid_evidence_rejected(a, b):
    with pytest.raises(ValueError):
        fuse_evidence_sum(a, b)
