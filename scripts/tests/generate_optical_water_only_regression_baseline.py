"""Generate the initial optical water-only EDL numerical-regression baseline.

Run this script only when intentionally accepting new model behaviour.
"""

from pathlib import Path
import hashlib
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flood_uncertainty.inference.infer import load_inference_function
from flood_uncertainty.losses.edl_loss import calc_edl_loss_multioutput_logistic_mask_invalid
from flood_uncertainty.models.edl import EDL_ML4FloodsModel
from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.models.utils.configuration import AttrDict


CHECKPOINT = ROOT / "artifacts/models/edl_optical/checkpoint/epoch=8-step=9225.ckpt"
OUTPUT = ROOT / "tests/fixtures/optical_water_only_epoch8_regression.pt"


def optical_input() -> torch.Tensor:
    return torch.linspace(0, 10_000, 6 * 32 * 32).reshape(6, 32, 32)


def training_batch() -> dict[str, torch.Tensor]:
    image = torch.linspace(-2.0, 2.0, 2 * 6 * 32 * 32).reshape(2, 6, 32, 32)
    rows, cols = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    cloud = ((rows + cols) % 2 + 1).to(torch.long)
    water = ((rows // 4 + cols // 4) % 2 + 1).to(torch.long)
    mask = torch.stack((cloud, water)).unsqueeze(0).repeat(2, 1, 1, 1)
    mask[:, :, :2, :2] = 0
    return {"image": image, "mask": mask}


def main() -> None:
    torch.set_num_threads(1)
    if OUTPUT.exists():
        raise FileExistsError(f"Baseline already exists: {OUTPUT}; review changes before replacing it")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 8 or checkpoint["global_step"] != 9225:
        raise RuntimeError("unexpected trusted checkpoint metadata")
    model_params = AttrDict(checkpoint["hyper_parameters"]["model_params"])
    if model_params.hyperparameters.get("task_mode") != "water_only":
        raise RuntimeError("expected a water-only checkpoint")
    model = EDL_ML4FloodsModel(model_params, normalized_data=True).cpu()
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    config = load_mode_config(str(ROOT / "configurations/edl_water.json"), mode="infer")
    config["model_params"] = model_params
    predict, _ = load_inference_function(
        model, config, max_tile_size=32, apply_normalization=True, used_EDL=True, disable_pbar=True
    )
    names = ("mask", "prob", "dst_u", "evidence", "aleatoric", "epistemic")
    inference = {name: value.cpu() for name, value in zip(names, predict(optical_input()))}

    model.train()
    optimizer = model.configure_optimizers()["optimizer"]
    optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    batch = training_batch()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["image"])
    loss = calc_edl_loss_multioutput_logistic_mask_invalid(
        logits, model.select_targets(batch["mask"]), pos_weight_problem=model.pos_weight,
        weight_problem=model.weight_problem, epoch_num=8,
        annealing_step=model.annealing_step, annealing_mode=model.annealing_mode,
        annealing_coefficient=model.annealing_coefficient,
    )
    loss.backward()
    gradient = next(model.network.parameters()).grad.detach().clone()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        logits_after = model(batch["image"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint": {"name": CHECKPOINT.name, "epoch": 8, "global_step": 9225,
                           "sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()},
            "environment": {"torch": str(torch.__version__), "cpu_threads": 1},
            "inference": inference,
            "training_step": {
                "loss": loss.detach(),
                "first_parameter_gradient": gradient,
                "logits_after": logits_after,
            },
        },
        OUTPUT,
    )
    print(f"Wrote water-only baseline to {OUTPUT}")


if __name__ == "__main__":
    main()
