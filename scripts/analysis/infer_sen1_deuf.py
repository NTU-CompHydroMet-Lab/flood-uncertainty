"""One command: paired Sen1 test inference and five DEUF PNGs per tile."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sen1-deuf-mpl')
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Subset

from flood_uncertainty.data.sen1floods11 import collect_sen1_splits, Sen1Floods11Dataset
from flood_uncertainty.fusion import fuse_evidence_sum, dirichlet_binary_stats
from flood_uncertainty.inference.infer_sar import load_sar_model
from flood_uncertainty.models.edl import EDL_ML4FloodsModel
from flood_uncertainty.utils.config_loader import load_mode_config
from flood_uncertainty.visualization.paired import plot_paired_tile, tile_prefix


def default_checkpoint(mode):
    """Prefer the exact paired-evaluation checkpoint; never select a pretrain."""
    paired = ROOT / 'artifacts/results/sen1_paired_test/metrics.json'
    if paired.is_file():
        return json.loads(paired.read_text())['checkpoints'][mode]
    individual = ROOT / f'artifacts/results/sen1_{mode}_test/metrics_best_loss.json'
    if individual.is_file():
        return json.loads(individual.read_text())['checkpoint']
    raise ValueError(f'Supply --{mode}-weights: no recorded Sen1 checkpoint is available')


def paired_datasets(configs):
    for mode, config in configs.items():
        if config.data_params.get('dataset_type') != 'sen1floods11' or config.data_params.get('modality') != mode:
            raise ValueError(f'{mode} config must describe Sen1 {mode} data')
    if configs['optical'].model_params.hyperparameters.get('task_mode') != 'water_only':
        raise ValueError('Optical model must use water_only')
    splits = {m: collect_sen1_splits(c.data_params) for m, c in configs.items()}
    for split in ('train', 'val', 'test'):
        if [s.sample_id for s in splits['sar'][split]] != [s.sample_id for s in splits['optical'][split]]:
            raise ValueError(f'Paired {split} IDs/order differ')
    datasets = {m: Sen1Floods11Dataset(splits[m]['test'], c.data_params, augment=False) for m, c in configs.items()}
    if datasets['sar'].windows != datasets['optical'].windows:
        raise ValueError('Paired samples / crop windows differ')
    for sample in splits['sar']['test']:
        with rasterio.open(sample.sar) as sar, rasterio.open(sample.optical) as optical, rasterio.open(sample.label) as label:
            for other in (optical, label):
                if sar.shape != other.shape or sar.crs != other.crs or not np.allclose(tuple(sar.transform), tuple(other.transform), rtol=0, atol=1e-10):
                    raise ValueError(f'Paired grid mismatch: {sample.sample_id}')
    return datasets


def raw_inputs(sample, window):
    with rasterio.open(sample.sar) as sar, rasterio.open(sample.optical) as optical:
        vv, vh = np.ma.masked_invalid(sar.read([1, 2], window=window, masked=True))
        source = optical.read([4, 3, 2], window=window, masked=True)
    missing = np.ma.getmaskarray(source).any(0) | ~np.isfinite(source.data).all(0)
    rgb = np.clip(np.nan_to_num(source.filled(0).transpose(1, 2, 0)) / 3500., 0, 1)
    rgb[missing] = np.array([184, 189, 197]) / 255
    return vv, vh, rgb



def write_water_tile_index(manifest, output=None):
    """Record GT water >20% of common valid pixels; keep all figure exports."""
    manifest = Path(manifest)
    output = Path(output) if output is not None else manifest.parent / 'water_gt20_tiles.csv'
    with manifest.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not {'valid_pixels', 'gt_water_fraction'}.issubset(fields):
            raise ValueError('Manifest must contain valid_pixels and gt_water_fraction')
        rows = []
        for row in reader:
            valid_pixels = int(row['valid_pixels'])
            if valid_pixels == 0:
                continue
            fraction = float(row['gt_water_fraction'])
            if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError('Invalid GT water fraction in manifest')
            water_pixels = int(round(fraction * valid_pixels))
            # Integer comparison excludes exactly 20%, without float-boundary ambiguity.
            if water_pixels * 5 > valid_pixels:
                row['gt_water_pixels'] = water_pixels
                rows.append(row)
    if 'gt_water_pixels' not in fields:
        fields.append('gt_water_pixels')
    temporary = output.with_suffix(output.suffix + '.partial')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return dict(water_gt20_csv=str(output), water_gt20_tiles=len(rows))


def export_tiles(datasets, models, indices, directory, *, batch_size=4, num_workers=0, dpi=160, save_products=False):
    """Stream one batch at a time; no filter, aggregate retention or extra figures."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    loaders = {m: DataLoader(Subset(d, indices), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers) for m, d in datasets.items()}
    name = f'manifest_{indices[0]:06d}_{indices[-1]:06d}.csv'
    manifest = directory / name
    partial = directory / (name + '.partial')
    fields = ['sample_id', 'tile_index', 'row', 'col', 'width', 'height', 'valid_pixels', 'gt_water_fraction',
              'sar_evidence', 'optical_evidence', 'fused_evidence', 'fused_context', 'confusion_inputs']
    done = 0
    with partial.open('w', newline='') as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sar_batch, optical_batch in zip(loaders['sar'], loaders['optical']):
            if sar_batch['id'] != optical_batch['id']:
                raise ValueError('Paired batch IDs differ')
            targets = {'sar': sar_batch['mask'][:, 0].numpy(), 'optical': optical_batch['mask'][:, 1].numpy()}
            valid_batch = (targets['sar'] != 0) & (targets['optical'] != 0)
            if not np.array_equal(targets['sar'][valid_batch], targets['optical'][valid_batch]):
                raise ValueError('Paired GT differs')
            evidence = {}
            for mode, batch in (('sar', sar_batch), ('optical', optical_batch)):
                device = next(models[mode].parameters()).device
                logits = models[mode].network(batch['image'].to(device))
                if logits.shape[1] != 2:
                    raise ValueError('Expected one binary water task (two evidence channels)')
                evidence[mode] = torch.relu(logits).cpu().numpy()
            for j in range(len(sar_batch['id'])):
                index = indices[done]
                sample, window = datasets['sar'].windows[index]
                target, valid = targets['sar'][j], valid_batch[j]
                products = {m: dirichlet_binary_stats(evidence[m][j]) for m in ('sar', 'optical')}
                products['fused'] = dirichlet_binary_stats(fuse_evidence_sum(evidence['sar'][j], evidence['optical'][j]))
                metadata = dict(sample_id=sample.sample_id, tile_index=index, split='test',
                                row=int(window.row_off) if window else 0, col=int(window.col_off) if window else 0)
                vv, vh, rgb = raw_inputs(sample, window)
                tile_dir = directory / tile_prefix(sample.sample_id, index, metadata['row'], metadata['col'])
                tile_dir.mkdir(parents=True, exist_ok=True)
                paths = plot_paired_tile(products, target, valid, vv, vh, rgb, metadata, tile_dir, dpi=dpi)
                if len(paths) != 5 or not all(path.is_file() for path in paths.values()):
                    raise RuntimeError(f'Expected five PNGs for tile {index}')
                if save_products:
                    for mode, output in products.items():
                        folder = tile_dir / 'products' / mode
                        folder.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(folder / f'tile_{index:06d}.npz', **output, target=target, valid=valid)
                row = {k: metadata[k] for k in ('sample_id', 'tile_index', 'row', 'col')}
                count = int(valid.sum())
                row.update(width=valid.shape[1], height=valid.shape[0], valid_pixels=count,
                           gt_water_fraction=float((target[valid] == 2).mean()) if count else None)
                row.update({k: str(path.relative_to(directory)) for k, path in paths.items()})
                writer.writerow(row)
                handle.flush()
                done += 1
                print(f'[{done}/{len(indices)}] {sample.sample_id} tile={index:06d} row={metadata["row"]} col={metadata["col"]} | PNGs={done * 5}', flush=True)
    if done != len(indices):
        raise RuntimeError('Incomplete paired loader traversal')
    partial.replace(manifest)
    water_index = write_water_tile_index(manifest)
    return dict(completed_tiles=done, pngs_written=5 * done, manifest=str(manifest), **water_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sar-config', type=Path, default=ROOT / 'configurations/edl_sar_sen1.json')
    parser.add_argument('--optical-config', type=Path, default=ROOT / 'configurations/edl_optical_sen1.json')
    parser.add_argument('--sar-weights', type=Path, help='Defaults to recorded paired best-loss Sen1 checkpoint')
    parser.add_argument('--optical-weights', type=Path, help='Defaults to recorded paired best-loss Sen1 checkpoint')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/results/sen1_paired_test/DEUF')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=160)
    parser.add_argument('--start-tile', type=int, default=0, help='First global test tile index')
    parser.add_argument('--max-tiles', type=int, help='Optional small run; by default process all remaining test tiles')
    parser.add_argument('--save-products', action='store_true', help='Also save raw arrays under each tile folder/products')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.dpi < 1 or args.start_tile < 0 or (args.max_tiles is not None and args.max_tiles < 1):
        parser.error('Batch size, DPI, max tiles must be positive; workers/start tile nonnegative')
    configs = {m: load_mode_config(getattr(args, f'{m}_config'), mode='train') for m in ('sar', 'optical')}
    checkpoints = {m: (getattr(args, f'{m}_weights') or Path(default_checkpoint(m))).resolve() for m in ('sar', 'optical')}
    for path in checkpoints.values():
        if not path.is_file():
            parser.error(f'Checkpoint not found: {path}')
    datasets = paired_datasets(configs)
    total = len(datasets['sar'])
    if args.start_tile >= total:
        parser.error('start tile is outside test split')
    end = total if args.max_tiles is None else min(total, args.start_tile + args.max_tiles)
    indices = list(range(args.start_tile, end))
    torch.set_num_threads(4)
    models = {'sar': load_sar_model(configs['sar'], str(checkpoints['sar']))}
    optical = EDL_ML4FloodsModel(configs['optical'].model_params, normalized_data=True)
    checkpoint = torch.load(checkpoints['optical'], map_location='cpu', weights_only=False)
    optical.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
    models['optical'] = optical.to(next(models['sar'].parameters()).device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / f'run_{indices[0]:06d}_{indices[-1]:06d}.json'
    run = dict(status='running', split='test', total_test_tiles=total, requested_tiles=len(indices),
               source_images=len(datasets['sar'].samples), start_tile=indices[0], last_tile=indices[-1],
               checkpoints={m: str(p) for m, p in checkpoints.items()},
               configs={m: str(getattr(args, f'{m}_config').resolve()) for m in ('sar', 'optical')},
               dpi=args.dpi, save_products=args.save_products, threshold=.5,
               output_layout='one folder per sample_id / tile index / crop position',
               fusion='e_f = e_sar + e_optical; alpha_f = e_f + 1',
               evidence_scale='shared across five figures within each tile; log scale; floor 0.01',
               mask='intersection of valid SAR, optical and GT pixels')
    run_path.write_text(json.dumps(run, indent=2))
    print(f'{len(indices)} test tiles, {len(indices) * 5} PNGs -> {args.output}', flush=True)
    try:
        run.update(export_tiles(datasets, models, indices, args.output, batch_size=args.batch_size,
                                num_workers=args.num_workers, dpi=args.dpi, save_products=args.save_products))
        run['status'] = 'complete'
    except Exception as error:
        run.update(status='failed', error=str(error))
        raise
    finally:
        run_path.write_text(json.dumps(run, indent=2))
    print(f'Complete: {run["pngs_written"]} PNGs; manifest: {run["manifest"]}', flush=True)
    print(f'GT water >20%: {run["water_gt20_tiles"]} tiles -> {run["water_gt20_csv"]}', flush=True)


if __name__ == '__main__':
    main()
