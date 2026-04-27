import pytest
import torch

from flood_uncertainty.losses.edl_loss import (
    beta_EDL_loss_mask_invalid,
    calc_edl_loss_multioutput_logistic_mask_invalid,
)


def make_single_task_logits():
    return torch.tensor(
        [
            [
                [[1.0, 0.5], [0.2, 0.1]],
                [[0.3, 1.2], [0.7, 0.4]],
            ]
        ],
        dtype=torch.float32,
    )


def make_single_task_target_valid():
    return torch.tensor(
        [[[1, 2], [1, 2]]],
        dtype=torch.long,
    )


def make_single_task_target_with_invalid():
    return torch.tensor(
        [[[0, 2], [1, 0]]],
        dtype=torch.long,
    )


def make_single_task_target_all_invalid():
    return torch.zeros((1, 2, 2), dtype=torch.long)


def make_multioutput_logits_and_targets():
    logits = torch.tensor(
        [
            [
                [[1.0, 0.5], [0.2, 0.1]],
                [[0.3, 1.2], [0.7, 0.4]],
                [[0.9, 0.2], [0.4, 0.8]],
                [[0.1, 1.1], [0.6, 0.3]],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [
                [[1, 2], [1, 2]],
                [[2, 1], [2, 1]],
            ]
        ],
        dtype=torch.long,
    )
    return logits, target


def test_beta_edl_loss_mask_invalid_all_valid_returns_finite_scalar():
    logits = make_single_task_logits()
    target = make_single_task_target_valid()

    loss = beta_EDL_loss_mask_invalid(
        logits,
        target,
        epoch_num=1,
        annealing_step=10,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_beta_edl_loss_mask_invalid_ignores_invalid_pixels():
    logits = make_single_task_logits()
    target_valid = make_single_task_target_valid()
    target_partial_invalid = make_single_task_target_with_invalid()

    loss_valid = beta_EDL_loss_mask_invalid(
        logits,
        target_valid,
        epoch_num=1,
        annealing_step=10,
    )
    loss_partial_invalid = beta_EDL_loss_mask_invalid(
        logits,
        target_partial_invalid,
        epoch_num=1,
        annealing_step=10,
    )

    assert torch.isfinite(loss_valid)
    assert torch.isfinite(loss_partial_invalid)
    assert not torch.isclose(loss_valid, loss_partial_invalid)


def test_beta_edl_loss_mask_invalid_all_invalid_is_stable():
    logits = make_single_task_logits()
    target = make_single_task_target_all_invalid()

    loss = beta_EDL_loss_mask_invalid(
        logits,
        target,
        epoch_num=1,
        annealing_step=10,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_beta_edl_loss_mask_invalid_rejects_bad_logit_shape():
    logits = torch.randn(1, 3, 2, 2)
    target = make_single_task_target_valid()

    with pytest.raises(AssertionError):
        beta_EDL_loss_mask_invalid(
            logits,
            target,
            epoch_num=1,
            annealing_step=10,
        )


def test_beta_edl_loss_mask_invalid_rejects_bad_target_rank():
    logits = make_single_task_logits()
    target = torch.ones(1, 1, 2, 2, dtype=torch.long)

    with pytest.raises(AssertionError):
        beta_EDL_loss_mask_invalid(
            logits,
            target,
            epoch_num=1,
            annealing_step=10,
        )


def test_calc_edl_loss_multioutput_returns_scalar_for_two_tasks():
    logits, target = make_multioutput_logits_and_targets()

    loss = calc_edl_loss_multioutput_logistic_mask_invalid(
        logits,
        target,
        epoch_num=1,
        annealing_step=10,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_calc_edl_loss_multioutput_respects_weight_problem():
    logits, target = make_multioutput_logits_and_targets()

    loss_a = calc_edl_loss_multioutput_logistic_mask_invalid(
        logits,
        target,
        weight_problem=[0.5, 0.5],
        epoch_num=1,
        annealing_step=10,
    )
    loss_b = calc_edl_loss_multioutput_logistic_mask_invalid(
        logits,
        target,
        weight_problem=[0.1, 0.9],
        epoch_num=1,
        annealing_step=10,
    )

    assert torch.isfinite(loss_a)
    assert torch.isfinite(loss_b)
    assert not torch.isclose(loss_a, loss_b)


def test_calc_edl_loss_multioutput_respects_pos_weight_problem():
    logits, target = make_multioutput_logits_and_targets()

    loss_a = calc_edl_loss_multioutput_logistic_mask_invalid(
        logits,
        target,
        pos_weight_problem=None,
        epoch_num=1,
        annealing_step=10,
    )
    loss_b = calc_edl_loss_multioutput_logistic_mask_invalid(
        logits,
        target,
        pos_weight_problem=[[1.0, 3.0], [1.0, 5.0]],
        epoch_num=1,
        annealing_step=10,
    )

    assert torch.isfinite(loss_a)
    assert torch.isfinite(loss_b)
    assert not torch.isclose(loss_a, loss_b)


def test_beta_edl_loss_changes_with_epoch_num_when_annealing_active():
    logits = make_single_task_logits()
    target = make_single_task_target_valid()

    loss_epoch_1 = beta_EDL_loss_mask_invalid(
        logits,
        target,
        epoch_num=1,
        annealing_step=10,
    )
    loss_epoch_10 = beta_EDL_loss_mask_invalid(
        logits,
        target,
        epoch_num=10,
        annealing_step=10,
    )

    assert torch.isfinite(loss_epoch_1)
    assert torch.isfinite(loss_epoch_10)
    assert not torch.isclose(loss_epoch_1, loss_epoch_10)


def test_calc_edl_loss_multioutput_rejects_bad_shapes():
    logits = torch.randn(1, 3, 2, 2)
    _, target = make_multioutput_logits_and_targets()

    with pytest.raises(AssertionError):
        calc_edl_loss_multioutput_logistic_mask_invalid(
            logits,
            target,
            epoch_num=1,
            annealing_step=10,
        )
