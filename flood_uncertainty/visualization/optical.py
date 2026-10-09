"""Optical EDL result panels with restored Sentinel-2 RGB."""
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from ml4floods.data.worldfloods.configs import BANDS_S2, CHANNELS_CONFIGURATIONS
from ml4floods.preprocess.worldfloods.normalize import get_normalisation


def plot_optical_result(image, target, probability, output_path, *,
                        channel_configuration, uncertainty, title='Optical result'):
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    image, target, probability, uncertainty = map(array, (image, target, probability, uncertainty))
    mean, std = get_normalisation(channel_configuration, channels_first=True)
    bands = [BANDS_S2[i] for i in CHANNELS_CONFIGURATIONS[channel_configuration]]
    rgb = (image * std + mean)[[bands.index(b) for b in ('B4', 'B3', 'B2')]]
    rgb = np.clip(np.nan_to_num(rgb.transpose(1, 2, 0)) / 3500.0, 0, 1)
    valid = np.isin(target, (1, 2))
    water, predicted = target == 2, probability >= 0.5
    confusion = np.zeros(target.shape, dtype=np.uint8)
    for code, mask in enumerate((~water & ~predicted, ~water & predicted,
                                 water & ~predicted, water & predicted), 1):
        confusion[valid & mask] = code
    fig = Figure(figsize=(15, 9), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3).flatten()
    axes[0].imshow(rgb); axes[0].set_title('Sentinel-2 RGB (B4/B3/B2)')
    colors = ['#808080', 'black', '#2474ed']
    for ax, values, name in [(axes[1], np.where(valid, target, 0), 'Ground truth'),
                             (axes[2], np.where(valid, np.where(predicted, 2, 1), 0), 'Prediction (p >= 0.5)')]:
        ax.imshow(values, cmap=ListedColormap(colors), vmin=-0.5, vmax=2.5, interpolation='nearest')
        ax.set_title(name)
        ax.legend(handles=[Patch(color=c, label=l) for c,l in zip(colors,['Invalid','Land','Water'])], fontsize=8)
    cm_colors = ['#808080','black','red','orange','green']
    axes[3].imshow(confusion, cmap=ListedColormap(cm_colors), vmin=-0.5, vmax=4.5, interpolation='nearest')
    axes[3].set_title('Spatial confusion')
    axes[3].legend(handles=[Patch(color=c,label=l) for c,l in zip(cm_colors,['Invalid','TN','FP','FN','TP'])], fontsize=8)
    for ax, values, name in [(axes[4], probability, 'Water probability'), (axes[5], uncertainty, 'EDL DST uncertainty')]:
        artist=ax.imshow(np.ma.masked_where(~valid, values), cmap='viridis', vmin=0,vmax=1)
        ax.set_title(name); fig.colorbar(artist,ax=ax,shrink=0.75)
    for ax in axes:ax.set_axis_off()
    fig.suptitle(title)
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    try:fig.savefig(path,dpi=150)
    finally:fig.clear()
