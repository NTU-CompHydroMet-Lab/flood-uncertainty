"""Train water-only KL EDL with standard WorldFloods validation. Pass --skip-val to train only."""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from scipy.ndimage import binary_erosion
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT))

from flood_uncertainty.data.worldfloods_kl import WorldFloodsKLDatasetTiled
from flood_uncertainty.models.edl_water_kl import WaterOnlyInvalidKLEDL
from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.data.worldfloods.configs import (
    CHANNELS_CONFIGURATIONS, COLORS_WORLDFLOODS_INVCLEARCLOUD, COLORS_WORLDFLOODS_INVLANDWATER,
)
from ml4floods.data.worldfloods.dataset import WorldFloodsDatasetTiled
from ml4floods.models.dataset_setup import get_transformations
from ml4floods.models.utils.configuration import AttrDict
from ml4floods.models.worldfloods_model import METRIC_MODE, batch_to_unnorm_rgb
from ml4floods.preprocess.tiling import Dict_to_WindowSlices, WindowSize, WindowSlices_to_Dict, save_windows
from ml4floods.preprocess.utils import get_list_of_window_slices
from ml4floods.preprocess.worldfloods.prepare_patches import filter_windows_v2
from scripts.preprocess.generate_water_only_kl_masks import (
    EventFiles, MissingAOIError, load_aoi, read_existing_mask,
)
from scripts.train.train_edl_optical_sen1 import load_matching_weights

ENSEMBLE_WEIGHTS = ROOT / "artifacts/models/edl_ensemble/edl_from_ensemble_7/checkpoint/epoch=15-step=16400.ckpt"
# The index used by the original 0.8 run (65,582 windows). Other indices have no filter metadata.
BASELINE_INDEX_SHA256 = "bc336ce7e08d9eddc8a131243fadec0038f02bf362ade9216c88434562088a37"
PROBE = ("EMSR260_01CICOGNARA_DEL_MONIT01_v2", 512, 512)  # event, col, row
# Experiment code that is not yet committed; each run keeps a copy and its SHA256.
PROVENANCE_FILES = (
    "scripts/train/train_edl_water_kl.py",
    "flood_uncertainty/models/edl_water_kl.py",
    "flood_uncertainty/models/edl.py",
    "flood_uncertainty/losses/edl_loss.py",
    "flood_uncertainty/data/worldfloods_kl.py",
)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_code(run_dir: Path) -> dict[str, str]:
    """Copy the experiment code into the run and hash the exact bytes copied."""
    hashes = {}
    for relative in PROVENANCE_FILES:
        content = (ROOT / relative).read_bytes()
        target = run_dir / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return hashes


class WaterEDLTraining(WaterOnlyInvalidKLEDL):
    def configure_optimizers(self):
        if self.trainer.limit_val_batches == 0:
            return torch.optim.Adam(self.network.parameters(), lr=self.lr)
        baseline = super().configure_optimizers()
        return {
            "optimizer": baseline["optimizer"],
            "lr_scheduler": {
                "scheduler": baseline["lr_scheduler"], "monitor": baseline["monitor"],
                "frequency": self.trainer.check_val_every_n_epoch,
            },
        }


def split_pair(image, gt, data, split):
    source, target = Path(image), Path(gt)
    split_dir = source.parent.parent
    if (split_dir.name != split or source.parent.name != data.input_folder
            or target != split_dir / data.target_folder / source.name):
        raise ValueError(f"Unexpected {split} pair: {image}, {gt}")
    return source, target


def kl_pixel_statistics(windows, kl_masks, events_without_kl):
    """Count KL pixels in each full mask (generator tag) and in the retained training windows."""
    windows_by_event = defaultdict(list)
    for tile in windows:
        windows_by_event[Path(tile.file_name).stem].append(tile.window)

    def count_event(stem):
        # Training windows match the 256 x 256 mask tiles. Edge windows are cropped,
        # which counts the same as the zero padding used by the Dataset.
        with rasterio.open(kl_masks[stem]) as mask:
            total = int(mask.tags()["kl_pixels"])
            per_patch = [int(np.count_nonzero(mask.read(1, window=window))) for window in windows_by_event[stem]]
        if sum(per_patch) > total:
            raise ValueError(f"Retained KL pixels exceed the kl_pixels tag in {kl_masks[stem]}")
        return {"total": total, "retained": sum(per_patch), "patches_with_kl": sum(n > 0 for n in per_patch)}

    stems = list(kl_masks)
    with ThreadPoolExecutor(max_workers=8) as readers:
        counted = tqdm(readers.map(count_event, stems), total=len(stems),
                       desc="Counting retained KL pixels", unit="event")
        by_event = dict(zip(stems, counted))
    by_event.update({stem: {"total": 0, "retained": 0, "patches_with_kl": 0} for stem in events_without_kl})
    total = sum(counts["total"] for counts in by_event.values())
    retained = sum(counts["retained"] for counts in by_event.values())
    return {
        "total": total, "retained": retained, "retained_fraction": retained / total if total else None,
        "patches_with_kl": sum(counts["patches_with_kl"] for counts in by_event.values()),
        "by_event": dict(sorted(by_event.items())),
    }


def snapshot_windows(train, data, mask_root):
    split_file = Path(data.train_test_split_file)
    # The generator publishes only finished TIFFs in this directory.
    mask_dir = mask_root / "train/kl_mask"

    def classify(pair):
        """Return "kl" (mask ready), "no_aoi" (all-zero KL), or "pending" (AOI but no mask yet)."""
        image, gt = pair
        source, target = split_pair(image, gt, data, "train")
        floodmap = source.parent.parent / "floodmaps" / f"{source.stem}.geojson"
        for path in (source, target, floodmap):
            if not path.is_file():
                raise FileNotFoundError(f"Missing train S2, GT or floodmap: {path}")
        mask_path = mask_dir / source.name
        if mask_path.is_file():
            # Same headers-and-tags check as the generator's --skip-existing: format, GT grid,
            # rule version, source paths, source modification times and completion statistics.
            read_existing_mask(EventFiles(split="train", event_id=source.stem, image=source, gt=target,
                                          floodmap=floodmap, output=mask_path))
            return "kl"
        with rasterio.open(target) as labels:
            crs = labels.crs
        # Same AOI rule as the generator. Other AOI errors, such as invalid geometry, propagate.
        try:
            load_aoi(floodmap, crs)
        except MissingAOIError:
            return "no_aoi"
        return "pending"

    pairs = list(zip(train[data.input_folder], train[data.target_folder], strict=True))
    if not pairs:
        raise ValueError("The train split is empty")
    # Separate GDAL handles per event. map preserves the split order for reproducible shuffling.
    with ThreadPoolExecutor(max_workers=8) as readers:
        statuses = list(tqdm(readers.map(classify, pairs), total=len(pairs),
                             desc="Checking train GT/KL/AOI", unit="event"))
    images = [image for image, _ in pairs]
    stems = [Path(image).stem for image in images]
    pending = [stem for stem, status in zip(stems, statuses) if status == "pending"]
    if pending:
        raise ValueError(
            f"{len(pending)} train events have an AOI but no KL mask in {mask_dir}. "
            f"Finish the mask generator before training. First events: {', '.join(pending[:10])}"
        )
    events_without_kl = sorted(stem for stem, status in zip(stems, statuses) if status == "no_aoi")
    kl_masks = {stem: mask_dir / Path(image).name
                for stem, image, status in zip(stems, images, statuses) if status == "kl"}

    rules = data.get("filter_windows") or {}
    size = WindowSize(*data.window_size)
    cache = Path(data.path_to_splits) / f"windows_{split_file.name}"
    use_filter = rules.get("apply", False)
    if use_filter and rules["version"] != "v2":
        raise ValueError("Water-only KL training requires the WorldFloods v2 window filter")
    baseline_settings = use_filter and rules["threshold_clouds"] == 0.8 and size == (256, 256)
    cached = cache.read_bytes() if baseline_settings and cache.is_file() else b""
    cached_rows = []
    if cached and hashlib.sha256(cached).hexdigest() == BASELINE_INDEX_SHA256:
        cached_rows = json.loads(cached)["slices"]
    eligible_images = set(images)
    if cached_rows and eligible_images.issubset(row["file_name"] for row in cached_rows):
        windows = [Dict_to_WindowSlices(row) for row in cached_rows if row["file_name"] in eligible_images]
        filter_source = str(cache)
    else:
        windows = get_list_of_window_slices(images, size)
        filter_source = "unfiltered"
        if use_filter:
            raw = WorldFloodsDatasetTiled(windows, image_prefix=f"/{data.input_folder}/",
                                           gt_prefix=f"/{data.target_folder}/")
            windows = filter_windows_v2(raw, threshold_clouds=rules["threshold_clouds"])
            filter_source = "computed from train GT"
    if not windows:
        raise ValueError("No retained training windows")
    counts = Counter({stem: 0 for stem in stems})
    counts.update(Path(tile.file_name).stem for tile in windows)
    return windows, {
        "train_split_events": len(images), "kl_mask_events": statuses.count("kl"),
        "events_without_kl": events_without_kl,
        "train_patches": len(windows), "patches_by_event": dict(sorted(counts.items())),
        "filter_windows": rules, "window_source": filter_source,
        "kl_statistics": kl_pixel_statistics(windows, kl_masks, events_without_kl),
    }


def validation_loader(val, data, transform, dataset_args):
    """Use the full official validation split and the original valid-label metrics."""
    images = val.get(data.input_folder, [])
    if not images:
        raise ValueError("Validation is enabled but the val split is empty. Check the split or pass --skip-val.")
    for image, gt in zip(images, val[data.target_folder], strict=True):
        _, target = split_pair(image, gt, data, "val")
        if not target.is_file():
            raise FileNotFoundError(f"Missing validation GT: {target}")
    windows = get_list_of_window_slices(images, WindowSize(*data.window_size))
    dataset = WorldFloodsDatasetTiled(windows, transforms=transform, **dataset_args)
    return DataLoader(dataset, batch_size=data.batch_size, shuffle=False, num_workers=data.get("num_workers_val", 0))


def seed_worker_transform(_worker_id):
    get_worker_info().dataset.joint_transform.set_random_seed(torch.initial_seed() % 2**32)


def plot_probe(path, title, rgb, cloud, water, kl, probability, uncertainty):
    fig = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    axes[0, 0].imshow(rgb, interpolation="nearest")
    axes[0, 0].set_title("Sentinel-2 RGB")
    for axis, gt, colors, labels, name in (
        (axes[0, 1], cloud, COLORS_WORLDFLOODS_INVCLEARCLOUD, ["Invalid", "Clear", "Cloud"], "Cloud GT"),
        (axes[0, 2], water, COLORS_WORLDFLOODS_INVLANDWATER, ["Invalid", "Land", "Water"], "Water GT"),
    ):
        artist = axis.imshow(gt, cmap=ListedColormap(colors), vmin=-0.5, vmax=2.5, interpolation="nearest")
        fig.colorbar(artist, ax=axis, ticks=[0, 1, 2], shrink=0.7).set_ticklabels(labels)
        axis.set_title(name)

    visible = np.isin(water, (1, 2)) | kl
    cmap = colormaps["viridis"].with_extremes(bad="#808080")
    for axis, values, name in zip(axes[1], (probability, uncertainty, uncertainty),
                                  ("P(water)", "DST uncertainty", "DST + KL boundary (pink)")):
        artist = axis.imshow(np.ma.masked_where(~visible, values), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(name)
    boundary = kl & ~binary_erosion(kl)
    axes[1, 2].imshow(np.ma.masked_equal(boundary, False), cmap=ListedColormap(["#ff2591"]),
                      vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(artist, ax=list(axes[1]), shrink=0.7)
    for axis in axes.flat:
        axis.set_axis_off()
    fig.suptitle(title + "\nGray in bottom row = ignored water label")
    fig.savefig(path, dpi=120)
    fig.clear()


def plot_train_summary(metrics_path: Path, output: Path) -> None:
    """Update the training curves from complete epoch and fixed-patch CSV records."""
    frame = pd.read_csv(metrics_path)
    epochs = frame.groupby("epoch").last().dropna(subset=["train_loss_epoch"])
    completed = epochs.index + 1
    # The pretrained probe and the first trained probe both have Lightning epoch 0.
    probe_epoch = "train_probe/completed_epochs"
    probe = frame.groupby(probe_epoch).last() if probe_epoch in frame else pd.DataFrame()

    fig = Figure(figsize=(14, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    loss, iou, uncertainty = fig.subplots(1, 3, sharex=True)
    loss.plot(completed, epochs["train_loss_epoch"], "o-", label="Total", color="#262626")
    loss.plot(completed, epochs["train_valid_edl_epoch"], ".--", label="Valid EDL", color="#2878b8")
    # Older KL runs logged both terms of this difference, but not the annealing weight.
    weighted_kl = epochs["train_loss_epoch"] - epochs["train_valid_edl_epoch"]
    if "train_kl_weight_epoch" in epochs:
        weighted_kl = (epochs["train_kl_weight_epoch"] * epochs["train_invalid_kl_epoch"]).fillna(weighted_kl)
    loss.plot(completed, weighted_kl, ".-", label="Weighted invalid KL", color="#c22670")
    loss.set_title("Training loss")
    loss.legend()
    water_union = (epochs["train_Global_TP_land_water"] + epochs["train_Global_FP_land_water"]
                   + epochs["train_Global_FN_land_water"])
    water_iou = epochs["train_Global_iou_land_water water"].where(water_union > 0)
    iou.plot(completed, water_iou, "o-", color="#2878b8")
    iou.set(title="Water IoU on valid training labels", ylim=(-0.02, 1.02))
    for region, label, color in (("kl", "KL", "#c22670"), ("land", "Land", "#95643c"), ("water", "Water", "#2878b8")):
        means = probe.get(f"train_probe/{region}_mean_u")
        if means is None or not means.notna().any():
            continue
        count = int(probe[f"train_probe/{region}_pixels"].iloc[-1])
        uncertainty.plot(probe.index, means, "o-", color=color, label=f"{label} ({count:,} px)")
    uncertainty.set(title="Mean DST uncertainty on the fixed patch", ylim=(-0.02, 1.02))
    if uncertainty.lines:
        uncertainty.legend()
    for axis in (loss, iou, uncertainty):
        axis.set_xlabel("Completed training epochs")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
    uncertainty.set_xlabel("Completed training epochs (0 = pretrained)")
    fig.suptitle("Water-only KL training")
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=140)
    temporary.replace(output)
    fig.clear()


class TrainSummaryCSVLogger(CSVLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        if "train_loss_epoch" in metrics:
            # Lightning sends the patch and epoch aggregates in separate calls.
            self.save()
            plot_train_summary(Path(self.log_dir) / "metrics.csv", Path(self.save_dir) / "train_summary.png")


class TrainPatchMonitor(Callback):
    def __init__(self, dataset, data, run_dir):
        tile = dataset.list_of_windows[0]
        self.title = f"{Path(tile.file_name).stem} | col={tile.window.col_off}, row={tile.window.row_off}"
        sample = dataset[0]  # deterministic transform; cache this one CPU tile for the run
        self.image = sample["image"].unsqueeze(0)
        self.cloud, self.water = sample["mask"].numpy()
        self.kl = (self.water == 0) & (sample["kl_mask"][0].numpy() == 1)
        normalized = data.train_transformation.normalize
        self.rgb = batch_to_unnorm_rgb(self.image, data.channel_configuration, 3500., unnormalize=normalized)[0]
        if not normalized:
            # The local helper returns raw reflectance when unnormalize=False.
            self.rgb = np.clip(self.rgb / 3500., 0, 1)
        self.directory = run_dir / "train_probe"
        self.directory.mkdir()

    def on_train_start(self, trainer, pl_module):
        self._record_patch(trainer, pl_module, completed_epochs=0)

    def on_train_epoch_end(self, trainer, pl_module):
        self._record_patch(trainer, pl_module, completed_epochs=trainer.current_epoch + 1)

    @torch.inference_mode()
    def _record_patch(self, trainer, pl_module, completed_epochs):
        if not trainer.is_global_zero:
            return
        was_training = pl_module.training
        try:
            pl_module.eval()
            scores = pl_module.edl_logits_to_output(pl_module(self.image.to(pl_module.device)).float())
        finally:
            pl_module.train(was_training)
        prob = scores["prob"][0, 0].cpu().numpy()
        uncertainty = scores["dst_u"][0, 0].cpu().numpy()
        stage = f"epoch_{trainer.current_epoch:03d}" if completed_epochs else "pretrained"
        path = self.directory / f"{stage}.png"
        plot_probe(path, f"{stage} | {self.title}", self.rgb, self.cloud, self.water,
                   self.kl, prob, uncertainty)
        metrics = {"epoch": trainer.current_epoch, "train_probe/completed_epochs": completed_epochs}
        for name, region in (("kl", self.kl), ("land", self.water == 1), ("water", self.water == 2)):
            count = int(region.sum())
            metrics[f"train_probe/{name}_pixels"] = count
            if count:
                metrics[f"train_probe/{name}_mean_u"] = float(uncertainty[region].mean())
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                logger.log_metrics({**metrics, "train_probe/panel": [wandb.Image(str(path))]}, step=trainer.global_step)
            else:
                logger.log_metrics(metrics, step=trainer.global_step)


def main():
    # No abbreviations: the ablation wrapper rejects --config, which --conf would otherwise bypass.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT / "configurations/edl_water.json")
    parser.add_argument("--mask-root", type=Path, default=ROOT / "artifacts/kl_masks/worldfloods_v2")
    parser.add_argument("--pretrained", type=Path, default=ENSEMBLE_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/models/edl_water_kl")
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip validation and its LR scheduler, best checkpoint and early stopping")
    wandb_modes = ("disabled", "offline", "online")
    parser.add_argument("--wandb-mode", choices=wandb_modes,
                        help="Override logging mode. --skip-val defaults to local PNGs and CSV only.")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-batches", type=int, help="Maximum batches per epoch for a short check")
    args = parser.parse_args()
    default_mode = "disabled" if args.skip_val else (os.environ.get("WANDB_MODE") or "offline")
    mode = args.wandb_mode or default_mode
    mode = {"dryrun": "offline", "run": "online"}.get(mode, mode)
    if mode not in wandb_modes:
        parser.error(f"Unknown WANDB_MODE={mode!r}. Choose from {', '.join(wandb_modes)}.")
    config = load_mode_config(str(args.config), mode="train")
    data, hparams = config.data_params, config.model_params.hyperparameters
    seed_everything(config.seed, workers=True)
    if args.max_epochs is not None:
        hparams.max_epochs = args.max_epochs
    if args.batch_size is not None:
        data.batch_size = args.batch_size
    if args.num_workers is not None:
        data.num_workers = args.num_workers
    if min(hparams.max_epochs, data.batch_size) < 1 or data.num_workers < 0:
        parser.error("Expected positive max-epochs and batch-size, and nonnegative num-workers")
    if args.limit_train_batches is not None and args.limit_train_batches < 1:
        parser.error("limit-train-batches must be positive")
    if not args.skip_val and (not isinstance(hparams.val_every, int) or hparams.val_every < 1):
        parser.error("val_every must be a positive integer when validation is enabled")
    if data.get("add_mndwi_input"):
        raise ValueError("The pretrained six-band water model expects S2 bands without an additional MNDWI channel")
    config.model_params.pretrained_path = str(args.pretrained)
    model = WaterEDLTraining(config.model_params, normalized_data=data.train_transformation.normalize)
    pretrained_sha256 = file_sha256(args.pretrained)  # Hashed right before loading.
    load_matching_weights(model, str(args.pretrained))

    split_bytes = Path(data.train_test_split_file).read_bytes()  # One read: parsed and hashed bytes match.
    splits = json.loads(split_bytes)
    windows, snapshot = snapshot_windows(splits["train"], data, args.mask_root)
    probe = next((tile for tile in windows if (Path(tile.file_name).stem, tile.window.col_off,
                                               tile.window.row_off) == PROBE), None)
    if probe is None:
        probe = windows[0]
        print("The reference patch is absent. Using the first retained train patch for all epoch plots.")
    bands = CHANNELS_CONFIGURATIONS[data.channel_configuration]
    dataset_args = dict(image_prefix=f"/{data.input_folder}/", gt_prefix=f"/{data.target_folder}/", bands=bands)
    train_transform, val_transform = get_transformations(data)
    probe_transform = val_transform
    if data.train_transformation.normalize != data.test_transformation.normalize:
        probe_config = AttrDict({**data, "test_transformation": data.train_transformation})
        _, probe_transform = get_transformations(probe_config)
    train_transform.set_random_seed(config.seed)
    kl_args = dict(kl_mask_root=args.mask_root, events_without_kl=snapshot["events_without_kl"])
    dataset = WorldFloodsKLDatasetTiled(windows, transforms=train_transform, **kl_args, **dataset_args)
    probe_dataset = WorldFloodsKLDatasetTiled([probe], transforms=probe_transform, **kl_args, **dataset_args)
    # A dedicated generator keeps the shuffle and worker seeds identical across ablation arms.
    loader = DataLoader(dataset, shuffle=True, batch_size=data.batch_size, num_workers=data.num_workers,
                        worker_init_fn=seed_worker_transform, generator=torch.Generator().manual_seed(config.seed))
    val_loader = None
    if not args.skip_val:
        val_loader = validation_loader(splits.get("val", {}), data, val_transform, dataset_args)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot.update(config=str(args.config), resolved_config=config, pretrained=str(args.pretrained),
                    mask_root=str(args.mask_root), wandb_mode=mode, precision="32-true",
                    validation_enabled=not args.skip_val,
                    val_events=len(splits["val"][data.input_folder]) if val_loader is not None else 0,
                    val_patches=len(val_loader.dataset) if val_loader is not None else 0,
                    seed=config.seed, batch_size=data.batch_size, max_epochs=hparams.max_epochs, lr=model.lr,
                    limit_train_batches=args.limit_train_batches, probe=WindowSlices_to_Dict(probe),
                    code_sha256=snapshot_code(run_dir), pretrained_sha256=pretrained_sha256,
                    split_file_sha256=hashlib.sha256(split_bytes).hexdigest())
    (run_dir / "train_manifest.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    save_windows(windows, str(run_dir / "train_windows.json"))
    print(f"Train snapshot: {snapshot['train_split_events']} events ({snapshot['kl_mask_events']} with KL masks; "
          f"{len(snapshot['events_without_kl'])} without AOI use an all-zero KL mask), "
          f"{len(windows):,} patches. Val: {snapshot['val_patches']:,} patches. Output: {run_dir}", flush=True)
    kl = snapshot["kl_statistics"]
    print(f"KL pixels in retained patches: {kl['retained']:,}/{kl['total']:,} "
          f"({kl['retained'] / max(kl['total'], 1):.1%}); "
          f"{kl['patches_with_kl']:,}/{len(windows):,} patches contain KL", flush=True)

    # CSV first: the parent logs a full augmented batch when W&B is its primary logger.
    csv_logger_class = TrainSummaryCSVLogger if args.skip_val else CSVLogger
    loggers = [csv_logger_class(str(run_dir), name="logs")]
    if mode != "disabled":
        # The arm's experiment name tells the ablation runs apart on W&B.
        loggers.append(WandbLogger(name=f"{config.experiment_name}_{stamp}", project=config.wandb_project,
                                   entity=config.get("wandb_entity"), save_dir=str(run_dir), mode=mode,
                                   offline=mode == "offline", log_model=False))
    checkpoint_dir = run_dir / "checkpoint"
    latest = ModelCheckpoint(dirpath=checkpoint_dir, monitor=None, save_top_k=1,
                             save_last="link", save_on_train_epoch_end=True, every_n_epochs=1)
    # Weights of every epoch for later per-epoch evaluation. monitor="epoch" gives this callback
    # its own Lightning state key; with monitor=None it would collide with `latest`.
    epoch_weights = ModelCheckpoint(dirpath=checkpoint_dir / "epochs", monitor="epoch", mode="max", save_top_k=-1,
                                    every_n_epochs=1, save_weights_only=True, save_on_train_epoch_end=True)
    callbacks = [TrainPatchMonitor(probe_dataset, data, run_dir), LearningRateMonitor(logging_interval="epoch")]
    if not args.skip_val:
        metric = hparams.metric_monitor
        best = ModelCheckpoint(dirpath=checkpoint_dir, filename="best", monitor=metric, mode=METRIC_MODE[metric],
                               save_top_k=1, enable_version_counter=False, save_on_train_epoch_end=False)
        callbacks.extend([best, EarlyStopping(monitor=metric, mode=METRIC_MODE[metric],
                                              patience=hparams.early_stopping_patience, check_on_train_epoch_end=False)])
    callbacks.extend([latest, epoch_weights])
    trainer = Trainer(accelerator="cpu" if config.gpus is None else "gpu",
                      devices=1 if config.gpus is None else [int(config.gpus)], precision="32-true",
                      default_root_dir=run_dir, max_epochs=hparams.max_epochs, logger=loggers,
                      callbacks=callbacks, limit_train_batches=args.limit_train_batches,
                      check_val_every_n_epoch=1 if args.skip_val else hparams.val_every,
                      limit_val_batches=0 if args.skip_val else 1.0, num_sanity_val_steps=0 if args.skip_val else 2)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
