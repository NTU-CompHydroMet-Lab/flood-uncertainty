"""Train ten EDL models initialized from ten conventional UNet ensemble members."""

import argparse
import copy
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from flood_uncertainty.utils.config_loader import load_mode_config
from flood_uncertainty.utils.loss_plotting import plot_ensemble_losses


HEAD_KEYS = {"network.conv_last.weight", "network.conv_last.bias"}


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step=(\d+)", path.name)
    return int(match.group(1)) if match else -1


def find_pretraining_checkpoint(
    checkpoint_dir: Path, monitor: str = "val_bce_land_water"
) -> tuple[Path, float | None]:
    """Use inference's minimum-score rule, considering all surviving metadata.

    Old checkpoints can record files that were later removed by save_top_k.
    Resolve saved paths by basename, as inference does, and score only files
    that still exist in this member's directory.
    """
    paths = sorted(checkpoint_dir.glob("*.ckpt"))
    if not paths:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")
    available = {path.name: path for path in paths}
    scores = {}
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for key, callback in checkpoint.get("callbacks", {}).items():
            if "ModelCheckpoint" not in str(key):
                continue
            if callback.get("monitor", monitor) != monitor:
                continue
            for saved_path, score in callback.get("best_k_models", {}).items():
                candidate = available.get(Path(saved_path).name)
                if candidate is None or score is None:
                    continue
                value = float(score)
                if math.isfinite(value):
                    scores[candidate] = value
        del checkpoint
    if scores:
        best = min(scores, key=lambda path: (scores[path], path.name))
        return best, scores[best]
    fallback = max(paths, key=lambda path: (checkpoint_step(path), path.name))
    print(f"[fallback] No usable {monitor} scores; using {fallback.name}", flush=True)
    return fallback, None


def validate_pretraining_checkpoint(
    path: Path, expected_shapes: dict, channel_configuration: str
) -> dict:
    """Require a compatible backbone and a conventional two-logit output head."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    saved_h = checkpoint.get("hyper_parameters", {}).get("model_params", {}).get(
        "hyperparameters", {}
    )
    saved_channels = saved_h.get("channel_configuration")
    if saved_channels is not None and saved_channels != channel_configuration:
        raise ValueError(f"Input band configuration mismatch in {path}: {saved_channels}")
    for key, shape in expected_shapes.items():
        if key not in state:
            raise ValueError(f"Missing pretrained parameter {key} in {path}")
        source_shape = tuple(state[key].shape)
        if key in HEAD_KEYS:
            if shape[0] != 4 or source_shape != (2, *shape[1:]):
                raise ValueError(f"Expected a two-logit UNet head for {key} in {path}")
            continue
        if source_shape != shape:
            raise ValueError(
                f"Pretrained shape mismatch for {key} in {path}: {source_shape} != {shape}"
            )
    return {
        "source_epoch": checkpoint.get("epoch"),
        "source_global_step": checkpoint.get("global_step"),
        "loaded_parameter_tensors": len(expected_shapes) - len(HEAD_KEYS),
        "randomly_initialized_parameters": sorted(HEAD_KEYS),
    }


def member_config(config: dict, member: int, checkpoint: Path, data_root: Path) -> dict:
    """Serialize the resolved settings into a normal single-model EDL config."""
    shared = copy.deepcopy(config)
    shared.pop("runtime_mode", None)
    member_index = shared.pop("member_index_offset", 0) + member
    shared["experiment_name"] = f"{config['experiment_name']}_{member_index}"
    shared["seed"] = config["seed"] + member
    shared["resume_from_checkpoint"] = False
    model_params = shared["model_params"]
    for key in ("train", "test", "val_only", "ensemble_dirs"):
        model_params.pop(key, None)
    model_params["pretrained_path"] = str(checkpoint)
    shared["data_params"]["path_to_splits"] = str(data_root)
    shared["data_params"]["train_test_split_file"] = str(
        data_root / "train_test_split_from_csv.json"
    )
    return {"shared": shared, "train": {}, "infer": {}, "validate_only": {}}


def project_path(path: str) -> Path:
    return (PROJECT_ROOT / Path(path).expanduser()).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configurations/edl_ensemble.json")
    )
    parser.add_argument("--data_root", default=None)
    parser.add_argument(
        "--members", nargs="+", type=int, choices=range(10), default=None,
        help="Source member indices (0-9); output indices add member_index_offset.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Check source weights and print the plan without writing files or training.",
    )
    args = parser.parse_args()
    config = load_mode_config(args.config, mode="train")
    member_index_offset = config.get("member_index_offset", 0)
    if not isinstance(member_index_offset, int) or member_index_offset < 0:
        raise ValueError("member_index_offset must be a non-negative integer")
    hyperparameters = config.model_params.hyperparameters
    if (hyperparameters.get("task_mode", "cloud_water") != "cloud_water"
            or hyperparameters.model_type != "unet" or hyperparameters.num_classes != 2):
        raise ValueError("This entry point requires the cloud_water UNet EDL model")
    sources = [project_path(path) for path in config.model_params.get("ensemble_dirs", [])]
    if len(sources) != 10 or len(set(sources)) != 10:
        raise ValueError("Configure exactly ten distinct model_params.ensemble_dirs")
    members = list(range(10)) if args.members is None else args.members
    if len(set(members)) != len(members):
        raise ValueError("--members must not contain duplicate indices")
    data_root = project_path(args.data_root or config.data_params.path_to_splits)
    if not (data_root / "dataset_metadata.csv").is_file():
        raise FileNotFoundError(f"Missing dataset_metadata.csv in {data_root}")
    output_root = project_path(config.model_params.model_folder)
    config.model_params.model_folder = str(output_root)

    # CPU-only shape inspection; training and CUDA allocation happen in the child.
    from flood_uncertainty.models.edl import EDL_ML4FloodsModel

    template = EDL_ML4FloodsModel(config.model_params, normalized_data=True)
    expected_shapes = {key: tuple(value.shape) for key, value in template.state_dict().items()}
    del template
    plans = []
    for member in members:
        checkpoint, score = find_pretraining_checkpoint(sources[member])
        transfer = validate_pretraining_checkpoint(
            checkpoint, expected_shapes, hyperparameters.channel_configuration
        )
        settings = member_config(config, member, checkpoint, data_root)
        run_dir = output_root / settings["shared"]["experiment_name"]
        if not args.dry_run and run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Output already contains files: {run_dir}. Use a new experiment_name, "
                "--members for unstarted members, or train_edl.py --resume_ckpt to resume."
            )
        config_path = run_dir / "config.json"
        command = [
            sys.executable, "-u", str(PROJECT_ROOT / "scripts/train/train_edl.py"),
            "--config", str(config_path), "--mode", "train",
        ]
        provenance = {
            "member_index": member_index_offset + member,
            "source_member_index": member,
            "seed": settings["shared"]["seed"],
            "source_directory": str(sources[member]),
            "source_checkpoint": str(checkpoint),
            "source_metric_monitor": "val_bce_land_water",
            "source_metric_score": score,
            "selection": "minimum_available_score" if score is not None else "largest_step",
            **transfer,
            "training_command": command,
        }
        plans.append((run_dir, settings, provenance))
        print(
            f"Member {provenance['member_index']} (source {member}): "
            f"seed={provenance['seed']}, score={score}\n"
            f"  source: {checkpoint}\n  output: {run_dir}", flush=True,
        )

    if args.dry_run:
        print(f"Dry run complete: {len(plans)} members checked; no files written or training started.")
        return

    # Validate every requested member before starting any expensive training.
    # Separate sequential processes release each model's GPU memory on exit.
    for run_dir, settings, provenance in plans:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (run_dir / "pretraining.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        command = provenance["training_command"]
        print(f"Starting member {provenance['member_index']}: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=run_dir, check=True)
    # Include completed members of this batch when training only a subset.
    member_dirs = [
        output_root / f"{config.experiment_name}_{member_index_offset + member}"
        for member in range(10)
    ]
    summary_name = f"{config.experiment_name}_loss_curves.png"
    if member_index_offset:
        summary_name = (
            f"{config.experiment_name}_{member_index_offset}_"
            f"{member_index_offset + 9}_loss_curves.png"
        )
    summary_plot = plot_ensemble_losses(
        member_dirs, output_root / summary_name
    )
    if summary_plot is not None:
        print(f"Ensemble loss curves saved to: {summary_plot}", flush=True)
    print(f"Finished training {len(plans)} EDL members.", flush=True)


if __name__ == "__main__":
    main()
