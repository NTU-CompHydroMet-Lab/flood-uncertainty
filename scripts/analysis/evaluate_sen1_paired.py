"""Paired Sen1 evaluation, raw evidence fusion, previews and uncertainty ranking.

Use the same preprocessing and network path as eval_sar.py. PAvPU inputs
are exported without inventing test-derived accuracy/uncertainty thresholds.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sen1-paired-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

from flood_uncertainty.data.sen1floods11 import collect_sen1_splits, Sen1Floods11Dataset
from flood_uncertainty.fusion import fuse_evidence_sum, dirichlet_binary_stats
from flood_uncertainty.inference.infer_sar import load_sar_model
from flood_uncertainty.models.edl import EDL_ML4FloodsModel
from flood_uncertainty.utils.config_loader import load_mode_config

MODES = ['sar', 'optical', 'fused']
UNCERTAINTIES = ['dst_u', 'aleatoric', 'epistemic']


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(counts, brier, pixels):
    tn, fp, fn, tp = map(int, counts)
    def ratio(a, b):
        return float(a / b) if b else None
    water_iou = ratio(tp, tp + fp + fn)
    land_iou = ratio(tn, tn + fp + fn)
    defined = [x for x in [water_iou, land_iou] if x is not None]
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, valid_pixels=int(pixels),
                water_iou=water_iou, nonwater_iou=land_iou,
                miou=float(np.mean(defined)) if defined else None,
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                f1=ratio(2 * tp, 2 * tp + fp + fn), accuracy=ratio(tp + tn, pixels),
                brier=ratio(brier, pixels))


def patch_rows(target, valid, output, patch_size, metadata):
    rows = []
    height, width = valid.shape
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            sl = np.s_[row:min(row + patch_size, height), col:min(col + patch_size, width)]
            v, gt, pred = valid[sl], target[sl] == 2, output['classification'][sl] == 2
            n = int(v.sum())
            if not n:
                continue
            tp = int((gt & pred & v).sum())
            fp = int((~gt & pred & v).sum())
            fn = int((gt & ~pred & v).sum())
            rows.append(dict(**metadata, patch_row=row, patch_col=col, valid_pixels=n,
                patch_pixels=v.size, valid_fraction=n / v.size,
                patch_accuracy=float((pred[v] == gt[v]).mean()),
                patch_water_iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                **{f'mean_{k}': float(output[k][sl][v].mean()) for k in UNCERTAINTIES}))
    return rows


def retention(cm, uncertainty, steps=50):
    # Include all ties at a threshold, avoiding arbitrary selection among equal scores.
    order = np.argsort(uncertainty, kind='stable')
    sorted_u, sorted_cm = uncertainty[order], cm[order]
    cumulative = np.stack([np.cumsum(sorted_cm == c, dtype=np.int64) for c in range(1, 5)])
    rows = []
    for fraction in np.linspace(1 / steps, 1, steps):
        rank = max(1, int(len(cm) * fraction))
        threshold = sorted_u[rank - 1]
        kept = int(np.searchsorted(sorted_u, threshold, side='right'))
        counts = cumulative[:, kept - 1]
        scores = metrics(counts, 0, kept)
        scores.pop('brier')  # Brier needs retained probabilities, unavailable here.
        rows.append(dict(requested_retention_rate=float(fraction), retention_rate=kept / len(cm),
                         threshold=float(threshold), **scores))
    return rows


def preview(outputs, valid, metadata, output_dir):
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12})
    evidence_max = max(float(o['evidence'][:, valid].max()) for o in outputs.values())
    norm = LogNorm(vmin=.01, vmax=max(evidence_max, .011), clip=True)
    total_max = max(float(o['evidence'].sum(0)[valid].max()) for o in outputs.values())
    total_norm = LogNorm(vmin=.01, vmax=max(total_max, .011), clip=True)
    for mode, o in outputs.items():
        fig, axes = plt.subplots(2, 3, figsize=(12, 9), layout='constrained')
        panels = [('Prediction', o['classification'], 'Blues', None, 1, 2),
                  ('Water probability', o['probability'], 'Blues', None, 0, 1),
                  ('DST uncertainty (2 / S)', o['dst_u'], 'magma', None, 0, 1),
                  ('Non-water evidence', o['evidence'][0], 'viridis', norm, None, None),
                  ('Water evidence', o['evidence'][1], 'viridis', norm, None, None),
                  ('Total evidence (S - 2)', o['evidence'].sum(0), 'viridis', total_norm, None, None)]
        for ax, (title, values, cmap, color_norm, lo, hi) in zip(axes.flat, panels):
            masked = np.ma.masked_where(~valid, values)
            if color_norm:
                masked = np.ma.maximum(masked, .01)
            kwargs = {'norm': color_norm} if color_norm else {'vmin': lo, 'vmax': hi}
            im = ax.imshow(masked, cmap=cmap, interpolation='nearest', **kwargs)
            ax.set_title(title)
            ax.set_axis_off()
            cb = fig.colorbar(im, ax=ax, shrink=.8, pad=.02)
            if title == 'Prediction':
                cb.set_ticks([1, 2], labels=['Land', 'Water'])
            elif color_norm:
                cb.set_label('Raw evidence (log color scale)')
        name = 'SAR + Optical | Dirichlet evidence update' if mode == 'fused' else mode.upper()
        fig.suptitle(f"Sen1Floods11 test | {name}\n{metadata['sample_id']} | row={metadata['row']} col={metadata['col']} | 256 x 256", fontsize=15, fontweight='bold')
        formula = 'e = e_SAR + e_optical; alpha = e + 1; p_water = alpha_water / S' if mode == 'fused' else 'alpha = e + 1; p_water = alpha_water / S'
        fig.supxlabel(formula + '\nShared class-evidence scales across all three figures. Invalid paired pixels are white.', fontsize=9)
        fig.savefig(output_dir / f'{mode}_evidence.png', dpi=160)
        plt.close(fig)


def summary_plots(results, output_dir, split='test'):
    colors = ['#287b9c', '#db8651', '#7863ab']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [4, 1]}, layout='constrained')
    keys = ['water_iou', 'miou', 'precision', 'recall', 'f1', 'accuracy']
    x = np.arange(len(keys))
    for i, mode in enumerate(MODES):
        r = results[mode]
        bars = axes[0].bar(x + (i - 1) * .25, [r[k] for k in keys], width=.24, color=colors[i], label=mode.capitalize())
        axes[0].bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
        bar = axes[1].bar(i, r['brier'], width=.65, color=colors[i])
        axes[1].bar_label(bar, fmt='%.4f', fontsize=9, padding=4)
    axes[0].set(xticks=x, xticklabels=['Water IoU', 'mIoU', 'Precision', 'Recall', 'F1', 'Accuracy'], ylim=(0, 1.12), title='Higher is better')
    axes[0].legend(frameon=False, ncol=3, loc='upper left')
    axes[1].set(xticks=range(3), xticklabels=['SAR', 'Optical', 'Fused'], ylim=(0, max(r['brier'] for r in results.values()) * 1.4), title='Brier | lower is better')
    fig.suptitle(f'Paired Sen1Floods11 {split} | shared valid-pixel evaluation', fontsize=16, fontweight='bold')
    fig.supxlabel('SAR, optical and evidence-sum fusion | same paired samples and pixel mask | threshold = 0.5', fontsize=10)
    fig.savefig(output_dir / 'paired_metrics.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sar-weights', type=Path, help='Override best-loss SAR checkpoint')
    parser.add_argument('--optical-weights', type=Path, help='Override best-loss optical checkpoint')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/results/sen1_paired_test')
    parser.add_argument('--preview-index', type=int, default=16)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--save-products', action='store_true', help='Save raw products for every tile; preview products are always saved')
    args = parser.parse_args()
    if args.patch_size < 1 or args.batch_size < 1:
        parser.error('patch size and batch size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    configs = {m: load_mode_config(ROOT / f'configurations/edl_{m}_sen1.json', mode='train') for m in ['sar', 'optical']}
    splits = {m: collect_sen1_splits(c.data_params) for m, c in configs.items()}
    for split in ['train', 'val', 'test']:
        if [s.sample_id for s in splits['sar'][split]] != [s.sample_id for s in splits['optical'][split]]:
            raise ValueError('SAR / optical split IDs and order must match')
    datasets = {m: Sen1Floods11Dataset(splits[m][args.split], c.data_params) for m, c in configs.items()}
    if datasets['sar'].windows != datasets['optical'].windows:
        raise ValueError('Paired crop windows must match')
    if not 0 <= args.preview_index < len(datasets['sar']):
        parser.error('preview index outside split')
    for sample in splits['sar'][args.split]:
        with rasterio.open(sample.sar) as s1, rasterio.open(sample.optical) as s2:
            if s1.shape != s2.shape or s1.crs != s2.crs or not np.allclose(tuple(s1.transform), tuple(s2.transform), rtol=0, atol=1e-10):
                raise ValueError(f'Paired grid mismatch: {sample.sample_id}')
    checkpoints = {m: json.loads((ROOT / f'artifacts/results/sen1_{m}_test/metrics_best_loss.json').read_text())['checkpoint'] for m in ['sar', 'optical']}
    for mode in ['sar', 'optical']:
        override = getattr(args, f'{mode}_weights')
        if override is not None:
            checkpoints[mode] = str(override.resolve())
    models = {'sar': load_sar_model(configs['sar'], checkpoints['sar'])}
    optical = EDL_ML4FloodsModel(configs['optical'].model_params, normalized_data=True)
    optical.load_state_dict(torch.load(checkpoints['optical'], map_location='cpu', weights_only=False)['state_dict'], strict=True)
    device = next(models['sar'].parameters()).device
    models['optical'] = optical.to(device).eval()
    loaders = {m: DataLoader(d, batch_size=args.batch_size, shuffle=False, num_workers=0) for m, d in datasets.items()}
    torch.set_num_threads(4)
    totals = {m: np.zeros(4, np.int64) for m in MODES}
    brier = dict.fromkeys(MODES, 0.)
    vectors = {m: {'cm': [], **{k: [] for k in UNCERTAINTIES}} for m in MODES}
    patches, tiles = [], []
    n_pixels, tile_index = 0, 0
    with torch.inference_mode():
        for batch_sar, batch_optical in zip(loaders['sar'], loaders['optical']):
            if batch_sar['id'] != batch_optical['id']:
                raise ValueError('Batch IDs differ')
            targets = {'sar': batch_sar['mask'][:, 0].numpy(), 'optical': batch_optical['mask'][:, 1].numpy()}
            common = (targets['sar'] != 0) & (targets['optical'] != 0)
            if not np.array_equal(targets['sar'][common], targets['optical'][common]):
                raise ValueError('Paired GT labels differ')
            evidence = {}
            for mode, batch in [('sar', batch_sar), ('optical', batch_optical)]:
                logits = models[mode].network(batch['image'].to(device))
                evidence[mode] = torch.relu(logits).cpu().numpy()
                if evidence[mode].shape[1] != 2:
                    raise ValueError('Expected one binary water task')
            for j in range(len(batch_sar['id'])):
                sample, window = datasets['sar'].windows[tile_index]
                metadata = dict(tile_index=tile_index, sample_id=sample.sample_id, row=int(window.row_off), col=int(window.col_off))
                target, valid = targets['sar'][j], common[j]
                count = int(valid.sum())
                n_pixels += count
                outputs = {m: dirichlet_binary_stats(evidence[m][j]) for m in ['sar', 'optical']}
                outputs['fused'] = dirichlet_binary_stats(fuse_evidence_sum(evidence['sar'][j], evidence['optical'][j]))
                tiles.append(dict(**metadata, valid_pixels=count, invalid_pixels=int(valid.size - count)))
                for mode, o in outputs.items():
                    gt, pred = target == 2, o['classification'] == 2
                    cm = (1 + 2 * gt.astype(np.uint8) + pred.astype(np.uint8))  # TN=1 FP=2 FN=3 TP=4
                    counts = np.bincount(cm[valid], minlength=5)[1:5]
                    totals[mode] += counts
                    brier[mode] += float(((o['probability'][valid] - gt[valid])**2).sum())
                    vectors[mode]['cm'].append(cm[valid])
                    for k in UNCERTAINTIES:
                        vectors[mode][k].append(o[k][valid].astype(np.float32))
                    patches.extend(patch_rows(target, valid, o, args.patch_size, dict(**metadata, modality=mode)))
                    if args.save_products or tile_index == args.preview_index:
                        folder = args.output / 'products' / mode
                        folder.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(folder / f'tile_{tile_index:06d}.npz', **o, valid=valid, target=target)
                if tile_index == args.preview_index:
                    preview(outputs, valid, metadata, args.output)
                tile_index += 1
            print(f'Inference {tile_index}/{len(datasets["sar"])} tiles', flush=True)
    results = {m: metrics(totals[m], brier[m], n_pixels) for m in MODES}
    report = dict(split=args.split, source_samples=len(splits['sar'][args.split]), evaluated_tiles=tile_index,
        tile_size=configs['sar'].data_params.tile_size, threshold=.5,
        mask_policy='intersection of valid SAR, optical and GT pixels', checkpoints=checkpoints,
        fusion='e_fused = e_sar + e_optical; alpha_fused = e_fused + 1',
        patch_size=args.patch_size, pavpu_status='patch statistics exported; thresholds not chosen', results=results)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    write_csv(args.output / 'tile_manifest.csv', tiles)
    write_csv(args.output / 'patch_statistics.csv', patches)
    summary_plots(results, args.output, args.split)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout='constrained')
    colors = {'dst_u': '#7863ab', 'aleatoric': '#db8651', 'epistemic': '#287b9c'}
    for ax, mode in zip(axes, MODES):
        cm = np.concatenate(vectors[mode].pop('cm'))
        for k in UNCERTAINTIES:
            uncertainty = np.concatenate(vectors[mode].pop(k))
            rows = retention(cm, uncertainty)
            write_csv(args.output / f'retention_{mode}_{k}.csv', rows)
            ax.plot([r['retention_rate'] * 100 for r in rows], [r['water_iou'] if r['water_iou'] is not None else np.nan for r in rows], label=k.replace('_', ' ').upper(), color=colors[k])
            del uncertainty
        ax.axhline(results[mode]['water_iou'], color='#aaa', ls=':', lw=1)
        ax.set(title=mode.capitalize(), xlabel='Retained valid pixels (%)', xlim=(0, 100), ylim=(0, 1.02))
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        del cm
    axes[0].set_ylabel('Water IoU')
    fig.suptitle(f'Sen1Floods11 {args.split} | uncertainty retention curves', fontsize=16, fontweight='bold')
    fig.supxlabel('Retain lowest uncertainty first; ties included together. Each mode is ranked independently.\nDotted lines: full-pixel Water IoU. Curves describe uncertainty ranking, not causal information gain.', fontsize=10)
    fig.savefig(args.output / 'retention_iou.png', dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
