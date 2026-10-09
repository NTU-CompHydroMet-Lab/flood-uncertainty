"""Five Sen1 paired-result figures for one aligned tile."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np

FIGURE_TYPES = ('sar_evidence', 'optical_evidence', 'fused_evidence', 'fused_context', 'confusion_inputs')
CLASS_COLORS = ['#b8bdc5', '#20252b', '#2474ed']
CF_COLORS = ['#b8bdc5', '#20252b', '#e94b45', '#efa531', '#2ba36a']
CF_LABELS = ['Invalid / excluded', 'TN: correct land', 'FP: false water', 'FN: missed water', 'TP: correct water']


def tile_prefix(sample_id, tile_index, row, col):
    if not sample_id or Path(sample_id).name != sample_id or '\\' in sample_id:
        raise ValueError('Sample ID must be a filename component')
    return f'{sample_id}__tile_{tile_index:06d}__r{row:04d}_c{col:04d}'


def _norm(values, valid):
    maxima = [float(value[valid].max()) for value in values] if valid.any() else []
    return LogNorm(vmin=.01, vmax=max([.011, *maxima]), clip=True)


def _continuous(ax, values, valid, title, cmap, norm=None):
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad(CLASS_COLORS[0])
    displayed = np.ma.masked_where(~valid, values)
    if norm is not None:
        displayed = np.ma.maximum(displayed, .01)
    im = ax.imshow(displayed, cmap=colors, interpolation='nearest',
                   **({'norm': norm} if norm else {'vmin': 0, 'vmax': 1}))
    ax.set_title(title)
    ax.set_axis_off()
    return im


def _class(ax, values, valid, title):
    ax.imshow(np.where(valid, values, 0), cmap=ListedColormap(CLASS_COLORS),
              vmin=-.5, vmax=2.5, interpolation='nearest')
    ax.set_title(title, fontweight='bold')
    ax.set_axis_off()


def plot_paired_tile(products, target, valid, vv, vh, rgb, metadata, output_dir, *, dpi=160):
    """Write exactly five PNGs; shared evidence scales within this paired tile.

    Includes all-invalid and all-land tiles. No GT water-fraction filtering.
    Evidence panels retain the supplied 2x3 layout. Context uses total evidence,
    DST uncertainty, prediction and GT. Confusion inputs uses three CF maps
    over raw VV, VH and RGB. Raw arrays and model products are never modified.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = tile_prefix(metadata['sample_id'], metadata['tile_index'], metadata['row'], metadata['col'])
    paths = {kind: directory / f'{prefix}__{kind}.png' for kind in FIGURE_TYPES}
    split = metadata.get('split', 'test')
    subtitle = f"{metadata['sample_id']} | tile {metadata['tile_index']} | row={metadata['row']} col={metadata['col']} | {valid.shape[1]} x {valid.shape[0]}"
    class_norm = _norm([o['evidence'][k] for o in products.values() for k in (0, 1)], valid)
    total_norm = _norm([o['evidence'].sum(0) for o in products.values()], valid)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    for mode in ('sar', 'optical', 'fused'):
        output = products[mode]
        fig, axes = plt.subplots(2, 3, figsize=(12, 9), layout='constrained')
        try:
            panels = [('Prediction', output['classification'], 'Blues', None),
                      ('Water probability', output['probability'], 'Blues', None),
                      ('DST uncertainty (2 / S)', output['dst_u'], 'magma', None),
                      ('Non-water evidence', output['evidence'][0], 'viridis', class_norm),
                      ('Water evidence', output['evidence'][1], 'viridis', class_norm),
                      ('Total evidence (S - 2)', output['evidence'].sum(0), 'viridis', total_norm)]
            for ax, (title, values, cmap, norm) in zip(axes.flat, panels):
                if title == 'Prediction':
                    colors = plt.get_cmap(cmap).copy()
                    colors.set_bad(CLASS_COLORS[0])
                    im = ax.imshow(np.ma.masked_where(~valid, values), cmap=colors, vmin=1, vmax=2, interpolation='nearest')
                    ax.set_title(title)
                    ax.set_axis_off()
                else:
                    im = _continuous(ax, values, valid, title, cmap, norm)
                cb = fig.colorbar(im, ax=ax, shrink=.8, pad=.02)
                if title == 'Prediction':
                    cb.set_ticks([1, 2], labels=['Land', 'Water'])
                elif norm:
                    cb.set_label('Raw evidence (log color scale)')
            name = 'SAR + Optical | Dirichlet evidence update' if mode == 'fused' else mode.upper()
            fig.suptitle(f'Sen1Floods11 {split} | {name}\n{subtitle}', fontsize=15, fontweight='bold')
            formula = 'e = e_SAR + e_optical; alpha = e + 1; p_water = alpha_water / S' if mode == 'fused' else 'alpha = e + 1; p_water = alpha_water / S'
            fig.supxlabel(formula + '\nShared evidence scales across this tile\'s three figures; gray = excluded pixels.', fontsize=9)
            fig.savefig(paths[f'{mode}_evidence'], dpi=dpi)
        finally:
            plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    try:
        output = products['fused']
        im = _continuous(axes[0], output['evidence'].sum(0), valid, 'Total fused evidence | S - 2', 'viridis', total_norm)
        cb = fig.colorbar(im, ax=axes[0], fraction=.046, pad=.025)
        cb.set_label('Raw evidence (log scale)', fontsize=9)
        im = _continuous(axes[1], output['dst_u'], valid, 'Fused DST uncertainty | 2 / S', 'magma')
        fig.colorbar(im, ax=axes[1], fraction=.046, pad=.025)
        _class(axes[2], output['classification'], valid, 'Fused prediction')
        _class(axes[3], target, valid, 'Ground truth')
        fig.suptitle(f'Sen1Floods11 {split} | fused result\n{subtitle}', fontsize=17, fontweight='bold', y=.97)
        fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(CLASS_COLORS, ['Invalid / excluded', 'Land', 'Water'])], loc='lower center', bbox_to_anchor=(.5, .07), ncol=3, frameon=False)
        fig.text(.5, .02, 'Total evidence = sum(e_SAR + e_optical) = S - 2. Values below 0.01 use the lowest color. Water threshold = 0.5.', ha='center', fontsize=9)
        fig.subplots_adjust(left=.015, right=.985, top=.77, bottom=.2, wspace=.12)
        fig.savefig(paths['fused_context'], dpi=dpi)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    try:
        for ax, mode in zip(axes[0], ('sar', 'optical', 'fused')):
            cm = np.zeros(target.shape, np.uint8)
            cm[valid] = 1 + 2 * (target[valid] == 2).astype(np.uint8) + (products[mode]['classification'][valid] == 2).astype(np.uint8)
            ax.imshow(cm, cmap=ListedColormap(CF_COLORS), vmin=-.5, vmax=4.5, interpolation='nearest')
            tn, fp, fn, tp = [int((cm == k).sum()) for k in range(1, 5)]
            denominator = tp + fp + fn
            score = f'Water IoU={tp / denominator:.3f} | F1={2 * tp / (2 * tp + fp + fn):.3f}' if denominator else 'Water IoU / F1 undefined'
            ax.set_title(f'{mode.upper()} spatial CF', fontweight='bold')
            ax.set_axis_off()
            ax.text(.5, -.065, score, transform=ax.transAxes, ha='center', fontsize=10)
        for ax, values, title in zip(axes[1], (vv, vh, rgb), ('SAR input | VV (dB)', 'SAR input | VH (dB)', 'Optical input | RGB')):
            if values.ndim == 2:
                colors = plt.get_cmap('gray').copy()
                colors.set_bad(CLASS_COLORS[0])
                ax.imshow(values, cmap=colors, vmin=-30, vmax=0, interpolation='nearest')
            else:
                ax.imshow(values, interpolation='nearest')
            ax.set_title(title, fontweight='bold')
            ax.set_axis_off()
        fig.suptitle(f'Sen1Floods11 {split} | spatial confusion and paired inputs\n{subtitle}', fontsize=16, fontweight='bold', y=.98)
        fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(CF_COLORS, CF_LABELS)], loc='lower center', bbox_to_anchor=(.5, .055), ncol=5, frameon=False, fontsize=9)
        fig.text(.5, .018, 'SAR display: -30 to 0 dB. RGB: B4/B3/B2 divided by 3500. CF uses shared valid pixels and GT; IoU/F1 are for this tile.', ha='center', fontsize=9)
        fig.subplots_adjust(left=.015, right=.985, top=.86, bottom=.13, wspace=.08, hspace=.24)
        fig.savefig(paths['confusion_inputs'], dpi=dpi)
    finally:
        plt.close(fig)
    return paths
