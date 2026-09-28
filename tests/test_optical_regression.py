from pathlib import Path

import pytest
import torch

from flood_uncertainty.inference.infer import load_inference_function
from flood_uncertainty.losses.edl_loss import calc_edl_loss_multioutput_logistic_mask_invalid
from flood_uncertainty.models.edl import EDL_ML4FloodsModel
from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.models.utils.configuration import AttrDict


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "artifacts/models/edl_EpochKL_20260819/checkpoint/epoch=8-step=9225.ckpt"
)
BASELINE = ROOT / "tests/fixtures/optical_edl_epoch8_regression.pt"


def _require_artifacts() -> None:
    if not CHECKPOINT.exists():
        pytest.skip(f"trusted optical checkpoint is unavailable: {CHECKPOINT}")
    if not BASELINE.exists():
        pytest.fail(
            "optical regression baseline is missing; run "
            "python scripts/tests/generate_optical_regression_baseline.py"
        )


def _load_model_and_checkpoint():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model_params = AttrDict(checkpoint["hyper_parameters"]["model_params"])
    model = EDL_ML4FloodsModel(model_params, normalized_data=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model.cpu(), checkpoint, model_params


def _optical_input() -> torch.Tensor:
    return torch.linspace(0, 10_000, 6 * 32 * 32).reshape(6, 32, 32)


def _training_batch() -> dict[str, torch.Tensor]:
    image = torch.linspace(-2.0, 2.0, 2 * 6 * 32 * 32).reshape(2, 6, 32, 32)
    rows, cols = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    cloud = ((rows + cols) % 2 + 1).to(torch.long)
    water = ((rows // 4 + cols // 4) % 2 + 1).to(torch.long)
    mask = torch.stack((cloud, water)).unsqueeze(0).repeat(2, 1, 1, 1)
    mask[:, :, :2, :2] = 0
    return {"image": image, "mask": mask}


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_optical_edl_epoch8_checkpoint_is_the_trusted_reference():
    _require_artifacts()
    _, checkpoint, _ = _load_model_and_checkpoint()

    assert checkpoint["epoch"] == 8
    assert checkpoint["global_step"] == 9225
    callback = next(
        state
        for key, state in checkpoint["callbacks"].items()
        if "ModelCheckpoint" in str(key)
    )
    assert callback["monitor"] == "val_bce_land_water"
    assert Path(callback["best_model_path"]).name == CHECKPOINT.name


def test_optical_edl_fixed_checkpoint_inference_regression():
    _require_artifacts()
    expected = torch.load(BASELINE, map_location="cpu", weights_only=True)["inference"]
    model, _, model_params = _load_model_and_checkpoint()
    model.eval()
    config = load_mode_config(str(ROOT / "configurations/edl.json"), mode="infer")
    config["model_params"] = model_params
    predict, _ = load_inference_function(
        model,
        config,
        max_tile_size=32,
        apply_normalization=True,
        used_EDL=True,
        disable_pbar=True,
    )

    actual = predict(_optical_input())
    names = ("mask", "prob", "dst_u", "evidence", "aleatoric", "epistemic")
    for name, value in zip(names, actual):
        if name == "mask":
            assert torch.equal(value.cpu(), expected[name])
        else:
            _assert_close(value.cpu(), expected[name])


def test_optical_edl_fixed_batch_training_step_regression():
    _require_artifacts()
    expected = torch.load(BASELINE, map_location="cpu", weights_only=True)["training_step"]
    model, checkpoint, _ = _load_model_and_checkpoint()
    model.train()
    optimizer = model.configure_optimizers()["optimizer"]
    optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    batch = _training_batch()

    optimizer.zero_grad(set_to_none=True)
    logits_before = model(batch["image"])
    loss = calc_edl_loss_multioutput_logistic_mask_invalid(
        logits_before,
        batch["mask"],
        pos_weight_problem=model.pos_weight,
        weight_problem=model.weight_problem,
        epoch_num=8,
        annealing_step=model.annealing_step,
        annealing_mode=model.annealing_mode,
        annealing_coefficient=model.annealing_coefficient,
    )
    loss.backward()
    first_parameter = next(model.network.parameters())
    gradient = first_parameter.grad.detach().clone()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        logits_after = model(batch["image"])

    _assert_close(loss.detach(), expected["loss"])
    _assert_close(gradient, expected["first_parameter_gradient"])
    _assert_close(logits_after, expected["logits_after"])
