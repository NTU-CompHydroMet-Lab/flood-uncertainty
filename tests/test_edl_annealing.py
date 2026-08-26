import pytest

from flood_uncertainty.losses.edl_loss import get_kl_annealing_coefficient


def test_fixed_kl_annealing_is_independent_of_epoch():
    assert get_kl_annealing_coefficient(0, 10, "fixed", 0.5) == 0.5
    assert get_kl_annealing_coefficient(100, 10, "fixed", 0.5) == 0.5


def test_linear_kl_annealing_tracks_epoch_and_is_capped():
    assert get_kl_annealing_coefficient(0, 10, "linear") == 0.0
    assert get_kl_annealing_coefficient(5, 10, "linear") == 0.5
    assert get_kl_annealing_coefficient(20, 10, "linear") == 1.0


def test_invalid_kl_annealing_mode_is_rejected():
    with pytest.raises(ValueError, match="annealing_mode"):
        get_kl_annealing_coefficient(1, 10, "unknown")
