# SAR result visualization

`flood_uncertainty.visualization.sar.plot_sar_result` saves a six-panel PNG:
VV, VH, ground truth, prediction, spatial confusion, and uncertainty (or water
probability if uncertainty is omitted). It does not run inference.

```python
from flood_uncertainty.visualization.sar import plot_sar_result

result = plot_sar_result(
    vv=image[0],
    vh=image[1],
    target=mask,  # (H, W), mapped labels: 1=land, 2=water, others invalid
    probability=prediction["probability"],  # (H, W), water probability
    uncertainty=prediction["dst_u"],       # optional
    uncertainty_label="EDL DST uncertainty",
    input_scale="db",  # explicitly choose db, linear, or normalized
    output_path="artifacts/results/sar_test/plots/tile_0001.png",
    title="SAR tile 0001",
)
print(result["metrics"])
```

All arrays must be spatially aligned and two-dimensional. NumPy and detached
CPU/GPU torch tensors are supported. For evaluation batches, pass one sample
at a time, with `batch["mask"][i, 0]` and the corresponding water probability.
Check the loader's channel order and preprocessing before selecting VV/VH and
`input_scale`. Raw KuroSiwo labels must first be mapped using the project's
adapter. Normalized input is displayed in normalized units; this function does
not reconstruct original backscatter. Linear power is converted to dB only for
display. Default shared dB limits are (-30, 0); override `db_limits` if needed.

The default threshold is >= 0.5, matching `eval_sar.py`. Metrics exclude invalid
targets. An IoU with zero union is `None`; mIoU averages defined class IoUs and
is `None` for an entirely invalid tile. Per-tile metrics must not be averaged
and presented as the existing aggregate test-set metrics.

The returned dictionary contains `metrics`, `cm_map` (0=invalid, 1=TN, 2=FP,
3=FN, 4=TP), and `png_path`. PNG files at the selected path are overwritten.
GeoTIFF export is not included; the array interface does not assume
georeferencing metadata. `eval_sar.py` invokes this function as described below.

## Plot during evaluation

From the project root:

```bash
python -m scripts.eval.eval_sar \
  --config configurations/edl_sar.json \
  --weights artifacts/models/edl_sar_kurosiwo_epochKL_fullrange/checkpoint/epoch=20-step=16002.ckpt \
  --output artifacts/results/sar_test/metrics_epochKL_fullrange.json \
  --max-plots 16
```

Evaluation plots the first 16 tiles with GT water covering at least 10% of valid
pixels by default, including DST uncertainty,
without limiting the evaluated dataset. PNGs go to `<output stem>_plots/` beside
metrics JSON, using zero-based test-loader indices (`tile_000000.png`). These
indices are not original dataset filenames. Existing same-named PNGs are
overwritten; older extra PNGs are not deleted. `plots_written` in JSON records
only this run's output. Metrics now include aggregate `nonwater_iou` and `miou`.

Use `--plot-dir PATH` to choose another directory, `--max-plots N` to change
the cap, and `--no-plot-png` (or `--max-plots 0`) to disable plotting.
`--max-batches 1 --max-plots 2` runs a short check instead of full evaluation.
For the current `scale_input=normalize` configuration, plots automatically use
normalized units. Other loader modes require explicit `--plot-input-scale`.
For pre-event inputs, plots display the post-event VV/VH channels.

Use `--min-water-fraction 0.1 --max-plots 16` for the default selection.
The denominator excludes invalid labels. Entirely invalid tiles are never
selected, even with a zero threshold. Fewer than 16 qualifying tiles produces
fewer PNGs; no below-threshold fallback is used. Selection follows loader order,
not a ranking of the entire test set. JSON records the selected tile indices,
water fractions and PNG paths in `plotted_tiles`. Use a new `--plot-dir` when
changing selection to avoid mixing previous runs' PNGs.
