"""Evaluate the water-only invalid-KL ablation arms on saved full-image predictions (item 13 B, step 3).

From the project root, after scripts/analysis/run_kl_ablation_eval.sh:
    uv run python -m scripts.analysis.evaluate_kl_ablation --split val --check-exact   # binned vs exact, before test
    uv run python -m scripts.analysis.evaluate_kl_ablation --split test                # all tables and the figure

Definitions (frozen in docs/water_only_kl_review_checklist.md, item 13 B):
- Scored pixels: water GT in {1, 2} and a valid prediction (classification != 0, finite P and u in [0, 1]) in every
  arm, so all arms share one pixel set. pred = P(water) > 0.5; error = pred != (water GT == 2).
- near = scored pixels within 3 px (Chebyshev) of a KL pixel, computed on the whole event; far = the other scored
  pixels. Both only on events with a KL mask (events without an AOI have none).
- Error detection uses u as the score with tie-grouped bins (10,000 uniform bins on [0, 1) plus an exact u = 1 bin):
  AUROC gives half credit within a bin; AP is grouped average precision; AURC is right-endpoint, tie-grouped.
- ECE is positive-class reliability of P(water) with 15 equal-width bins (P = 1 in the last bin).
- Event bootstrap: paired multinomial event weights shared across arms; CIs of (arm - kl_off) from recomputed
  pooled metrics; "inconclusive" when the unrounded CI includes 0.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.ndimage import binary_dilation

from scripts.preprocess.generate_water_only_kl_masks import MissingAOIError, load_aoi

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("kl_off", "kl_shared_beta", "kl_002_beta", "kl_002_const")
LABELS = {"kl_off": "kl_off", "kl_shared_beta": "A: beta(t)", "kl_002_beta": "B: 0.02*beta(t)", "kl_002_const": "C: 0.02"}
CONTROL = "kl_off"
REGIONS = ("all", "near", "far")
CLOUD_CLASSES = ("invalid", "clear", "cloud")  # cloud GT values 0, 1, 2
U_BINS, P_BINS, HALO = 10_000, 15, 3
EXACT_TOLERANCE = 1e-3  # Fixed before any test evaluation.
RANK_METRICS = ("auroc", "ap", "aurc")


def safe_div(numerator, denominator):
    numerator, denominator = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    return np.divide(numerator, denominator, out=np.full(np.broadcast(numerator, denominator).shape, np.nan),
                     where=denominator > 0)


def u_bins(u):
    """Bin index per pixel: 10,000 uniform bins on [0, 1) and a separate bin for exactly u = 1."""
    index = np.minimum((u * U_BINS).astype(np.int64), U_BINS - 1)
    index[u == 1] = U_BINS
    return index


def ranking_metrics(hist):
    """AUROC, AP and AURC from [..., 2, K] counts of (correct, error) pixels ordered by ascending u."""
    neg, pos = hist[..., 0, :].astype(np.float64), hist[..., 1, :].astype(np.float64)
    n_pos, n_neg = pos.sum(-1), neg.sum(-1)
    total = neg + pos
    auroc = safe_div((pos * (np.cumsum(neg, -1) - neg + 0.5 * neg)).sum(-1), n_pos * n_neg)
    pos_desc, total_desc = pos[..., ::-1], total[..., ::-1]
    precision = np.nan_to_num(safe_div(np.cumsum(pos_desc, -1), np.cumsum(total_desc, -1)))
    # Discrimination needs both classes: AP is NA (not 1) when every pixel is an error.
    ap = np.where(n_neg > 0, safe_div((pos_desc * precision).sum(-1), n_pos), np.nan)
    risk = np.nan_to_num(safe_div(np.cumsum(pos, -1), np.cumsum(total, -1)))
    aurc = safe_div((risk * total).sum(-1), n_pos + n_neg)
    return {"auroc": auroc, "ap": ap, "aurc": aurc}


def derived_metrics(acc):
    """Metrics from pooled accumulators; every array may carry leading bootstrap / arm / region axes."""
    tp, fp, fn, tn = (acc["confusion"][..., i].astype(np.float64) for i in range(4))
    n_valid = tp + fp + fn + tn
    metrics = {
        "iou": safe_div(tp, tp + fp + fn), "precision": safe_div(tp, tp + fp), "recall": safe_div(tp, tp + fn),
        "brier": safe_div(acc["brier"], n_valid),
        "ece": safe_div(np.abs(acc["p_sum_y"] - acc["p_sum_p"]).sum(-1), n_valid),
        "n_valid": n_valid, "n_errors": fp + fn, "prevalence": safe_div(fp + fn, n_valid), "water_pixels": tp + fn,
    }
    metrics.update(ranking_metrics(acc["u_hist"]))
    return metrics


def kl_mask_for(split, image_file, gt, mask_root, data_root):
    """KL mask, or None for an event without an AOI (the same rule as training). Any other gap is an error."""
    kl_file = mask_root / split / "kl_mask" / f"{Path(image_file).stem}.tif"
    if kl_file.is_file():
        with rasterio.open(kl_file) as mask:
            if (mask.shape, mask.crs, mask.transform) != (gt.shape, gt.crs, gt.transform):
                raise ValueError(f"KL mask grid differs from GT: {kl_file}")
            return mask.read(1) == 1
    floodmap = data_root / split / "floodmaps" / f"{Path(image_file).stem}.geojson"
    try:
        load_aoi(floodmap, gt.crs)
    except MissingAOIError:
        return None
    raise FileNotFoundError(f"Missing KL mask for an event with an AOI: {kl_file}")


PREDICTION_BANDS = ("Water_DST_Uncertainty", "Water_Probability", "Water_Evidence_Neg", "Water_Evidence_Pos")


def read_predictions(path, gt):
    with rasterio.open(path) as src:
        if (src.shape, src.crs, src.transform) != (gt.shape, gt.crs, gt.transform):
            raise ValueError(f"Prediction grid differs from GT: {path}")
        if tuple(src.descriptions[1:5]) != PREDICTION_BANDS:
            raise ValueError(f"Unexpected prediction bands {src.descriptions}: {path}")
        cls, u, p, ev_land, ev_water = (src.read(band) for band in range(1, 6))
    valid = (cls != 0) & np.isfinite(u) & np.isfinite(p) & (u >= 0) & (u <= 1) & (p >= 0) & (p <= 1)
    return dict(u=u, p=p, zero_evidence=(ev_land == 0) & (ev_water == 0), valid=valid)


def empty_accumulators(shape):
    return {
        "confusion": np.zeros((*shape, 4), np.int64), "brier": np.zeros(shape, np.float64),
        "u_hist": np.zeros((*shape, 2, U_BINS + 1), np.int64),
        "p_count": np.zeros((*shape, P_BINS), np.int64), "p_sum_p": np.zeros((*shape, P_BINS), np.float64),
        "p_sum_y": np.zeros((*shape, P_BINS), np.float64),
    }


def accumulate(acc, index, u, p, truth):
    pred = p > 0.5
    error = pred != truth
    acc["confusion"][index] = [np.count_nonzero(pred & truth), np.count_nonzero(pred & ~truth),
                               np.count_nonzero(~pred & truth), np.count_nonzero(~pred & ~truth)]
    acc["brier"][index] = float(np.square(p.astype(np.float64) - truth).sum())
    bins = u_bins(u)
    acc["u_hist"][index] = [np.bincount(bins[~error], minlength=U_BINS + 1),
                            np.bincount(bins[error], minlength=U_BINS + 1)]
    p_bins = np.minimum((p * P_BINS).astype(np.int64), P_BINS - 1)
    acc["p_count"][index] = np.bincount(p_bins, minlength=P_BINS)
    acc["p_sum_p"][index] = np.bincount(p_bins, weights=p.astype(np.float64), minlength=P_BINS)
    acc["p_sum_y"][index] = np.bincount(p_bins, weights=truth.astype(np.float64), minlength=P_BINS)


def exact_ranking(u, error):
    """Exact tie-grouped reference: one group per distinct u value."""
    values, inverse = np.unique(u, return_inverse=True)
    hist = np.stack([np.bincount(inverse, weights=~error, minlength=values.size),
                     np.bincount(inverse, weights=error, minlength=values.size)])
    return {name: float(value) for name, value in ranking_metrics(hist).items()}


def evaluate_events(args, runs, pairs):
    n_events = len(pairs)
    acc = empty_accumulators((n_events, len(ARMS), len(REGIONS)))
    kl_count = np.zeros((n_events, len(ARMS), len(CLOUD_CLASSES)), np.int64)
    kl_zero = np.zeros_like(kl_count)
    kl_hist = np.zeros((*kl_count.shape, U_BINS + 1), np.int64)
    events, exact_rows = [], []
    for e, (image_file, gt_file) in enumerate(pairs):
        event = Path(image_file).stem
        with rasterio.open(gt_file) as gt:
            cloud, water = gt.read(1), gt.read(2)
            kl = kl_mask_for(args.split, image_file, gt, args.mask_root, args.data_root)
            preds = {arm: read_predictions(runs[arm] / f"{event}_output_EDL.tif", gt) for arm in ARMS}
        labelled = (water == 1) | (water == 2)
        valid_all = np.logical_and.reduce([preds[arm]["valid"] for arm in ARMS])
        scored = labelled & valid_all
        mismatch = {arm: int(np.count_nonzero(labelled & preds[arm]["valid"] & ~scored)) for arm in ARMS}
        if any(mismatch.values()):
            print(f"WARNING {event}: prediction validity differs across arms {mismatch}", flush=True)
        truth = water == 2
        halo = binary_dilation(kl, structure=np.ones((2 * HALO + 1,) * 2, bool)) if kl is not None else None
        regions = {"all": scored}
        if halo is not None:
            regions.update(near=scored & halo, far=scored & ~halo)
        for a, arm in enumerate(ARMS):
            u, p = preds[arm]["u"], preds[arm]["p"]
            for r, region in enumerate(REGIONS):
                if region in regions:
                    mask = regions[region]
                    accumulate(acc, (e, a, r), u[mask], p[mask], truth[mask])
            if kl is not None:
                for c, _ in enumerate(CLOUD_CLASSES):
                    mask = kl & valid_all & (cloud == c)  # KL pixels with a valid prediction in every arm
                    kl_count[e, a, c] = np.count_nonzero(mask)
                    kl_zero[e, a, c] = np.count_nonzero(mask & preds[arm]["zero_evidence"])
                    kl_hist[e, a, c] = np.bincount(u_bins(u[mask]), minlength=U_BINS + 1)
            if args.check_exact:
                for r, region in enumerate(REGIONS):
                    if region not in regions or not regions[region].any():
                        continue
                    mask = regions[region]
                    exact = exact_ranking(u[mask], (p[mask] > 0.5) != truth[mask])
                    binned = ranking_metrics(acc["u_hist"][e, a, r])
                    exact_rows.extend(dict(event=event, arm=arm, region=region, metric=name, exact=exact[name],
                                           binned=float(binned[name])) for name in RANK_METRICS)
        events.append(dict(event=event, has_aoi=kl is not None, scored_pixels=int(scored.sum()),
                           kl_pixels_without_valid_prediction=int((kl & ~valid_all).sum()) if kl is not None else 0,
                           near_water_pixels=int((regions["near"] & truth).sum()) if kl is not None else 0,
                           **{f"support_mismatch_{arm}": mismatch[arm] for arm in ARMS}))
        print(f"[{e + 1}/{n_events}] {event}: {int(scored.sum()):,} scored pixels, KL mask: {kl is not None}", flush=True)
    return acc, (kl_count, kl_zero, kl_hist), pd.DataFrame(events), pd.DataFrame(exact_rows)


def pooled(acc, weights):
    """Weighted event sums: weights is [B, E]; returns accumulators with a leading bootstrap axis."""
    return {key: np.tensordot(weights, value, axes=(1, 0)) for key, value in acc.items()}


def bootstrap_table(acc, event_sets, args):
    """Pooled metrics per arm and region with paired-bootstrap CIs of (arm - kl_off)."""
    rng = np.random.default_rng(args.seed)
    control = ARMS.index(CONTROL)
    rows = []
    for r, region in enumerate(REGIONS):
        idx = event_sets[region]
        sub = {key: value[idx][:, :, r] for key, value in acc.items()}  # [E, A, ...]
        point = derived_metrics(pooled(sub, np.ones((1, len(idx)))))
        draws = {}
        for start in range(0, args.bootstrap, args.batch):
            size = min(args.batch, args.bootstrap - start)
            weights = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=size).astype(np.float64)
            for name, value in derived_metrics(pooled(sub, weights)).items():
                draws.setdefault(name, []).append(value - value[:, [control]])  # [b, A] paired differences
        draws = {name: np.concatenate(values) for name, values in draws.items()}
        for name in ("iou", "precision", "recall", "brier", "ece", *RANK_METRICS):
            for a, arm in enumerate(ARMS):
                diff = draws[name][:, a]
                defined = diff[np.isfinite(diff)]
                low, high = (np.percentile(defined, [2.5, 97.5]) if defined.size and arm != CONTROL else (np.nan, np.nan))
                rows.append(dict(
                    region=region, metric=name, arm=arm, events=len(idx), value=point[name][0, a],
                    diff_vs_control=point[name][0, a] - point[name][0, control] if arm != CONTROL else np.nan,
                    ci_low=low, ci_high=high, defined_replicates=int(defined.size) if arm != CONTROL else 0,
                    inconclusive=bool(low <= 0 <= high) if arm != CONTROL and np.isfinite(low) else None,
                    n_valid=int(point["n_valid"][0, a]), n_errors=int(point["n_errors"][0, a]),
                    prevalence=point["prevalence"][0, a], water_pixels=int(point["water_pixels"][0, a]),
                ))
    return pd.DataFrame(rows)


def event_metrics_table(acc, events):
    """Per-event metrics per arm and region with the paired difference to kl_off (near/far only with an AOI)."""
    metrics = derived_metrics(acc)  # each [E, A, R]
    control = ARMS.index(CONTROL)
    rows = []
    for e, (event, has_aoi) in enumerate(zip(events["event"], events["has_aoi"])):
        for r, region in enumerate(REGIONS):
            if region != "all" and not has_aoi:
                continue
            for a, arm in enumerate(ARMS):
                row = dict(event=event, region=region, arm=arm, n_valid=int(metrics["n_valid"][e, a, r]),
                           n_errors=int(metrics["n_errors"][e, a, r]), water_pixels=int(metrics["water_pixels"][e, a, r]))
                for name in ("iou", "precision", "recall", "brier", "ece", *RANK_METRICS):
                    row[name] = metrics[name][e, a, r]
                    row[f"{name}_diff_vs_control"] = (metrics[name][e, a, r] - metrics[name][e, control, r]
                                                      if arm != CONTROL else np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def kl_region_table(kl_count, kl_zero, kl_hist):
    rows = []
    for a, arm in enumerate(ARMS):
        for c, cloud_class in enumerate(CLOUD_CLASSES):
            count, hist = int(kl_count[:, a, c].sum()), kl_hist[:, a, c].sum(0)
            median = np.nan
            if count:
                # Mean of the two middle ranks (equal for odd counts), each located by its bin.
                cumulative = np.cumsum(hist)
                middle = np.searchsorted(cumulative, [(count + 1) // 2, count // 2 + 1])
                median = float(np.mean([1.0 if b == U_BINS else (b + 0.5) / U_BINS for b in middle]))
            rows.append(dict(arm=arm, cloud_gt=cloud_class, kl_pixels=count, median_u_approx=median,
                             fraction_u_equals_1=safe_div(kl_zero[:, a, c].sum(), count)[()]))
    return pd.DataFrame(rows)


def pavpu_table(eval_root, checkpoint):
    rows = []
    for arm in ARMS:
        base = eval_root / checkpoint / arm / "analysis" / "test" / "EDL"
        for variant in ("pavpu_from_val", "pavpu_per_image_q90", "pavpu_per_image_q90_gt_valid"):
            path = base / variant / "pavpu_Water_DST_Uncertainty.csv"
            if not path.is_file():
                print(f"WARNING missing {path}", flush=True)
                continue
            frame = pd.read_csv(path).set_index("filename")
            rows.append(dict(arm=arm, variant=variant, pavpu_overall=frame.loc["OVERALL", "pavpu"],
                             pavpu_average=frame.loc["AVERAGE", "pavpu"],
                             uncertain_fraction_overall=frame.loc["OVERALL", "uncertain_fraction"],
                             scored_patches=frame.loc["OVERALL", "n_valid_patches"],
                             mean_threshold_tie_fraction=frame.loc["AVERAGE", "threshold_tie_fraction"]))
    return pd.DataFrame(rows)


def plot_curves(acc, path, split):
    point = pooled({key: value[:, :, 0] for key, value in acc.items()}, np.ones((1, acc["brier"].shape[0])))
    figure = Figure(figsize=(11, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    risk_axis, reliability_axis = figure.subplots(1, 2)
    for a, arm in enumerate(ARMS):
        hist = point["u_hist"][0, a].astype(np.float64)
        total = hist.sum(0)
        keep = total > 0
        coverage = np.cumsum(total)[keep] / total.sum()
        risk = np.cumsum(hist[1])[keep] / np.cumsum(total)[keep]
        risk_axis.plot(coverage, risk, label=LABELS[arm], linewidth=1.6)
        count = point["p_count"][0, a]
        bins = count > 0
        reliability_axis.plot(point["p_sum_p"][0, a][bins] / count[bins], point["p_sum_y"][0, a][bins] / count[bins],
                              "o-", label=LABELS[arm], markersize=3.5)
    risk_axis.set(xlabel="Coverage (most certain first, by u)", ylabel="Error rate among covered pixels",
                  title=f"Risk-coverage, all scored {split} pixels")
    reliability_axis.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    reliability_axis.set(xlabel="Mean predicted P(water)", ylabel="Observed water fraction", xlim=(0, 1), ylim=(0, 1),
                         title="Reliability of P(water), 15 bins")
    for axis in (risk_axis, reliability_axis):
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=150)
    figure.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--eval-root", type=Path, default=ROOT / "artifacts/results/edl_water_kl_ablation")
    parser.add_argument("--checkpoint", default="best", help="Subdirectory written by run_kl_ablation_eval.sh")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--mask-root", type=Path, default=ROOT / "artifacts/kl_masks/worldfloods_v2")
    parser.add_argument("--check-exact", action="store_true", help="Compare binned and exact AUROC/AP/AURC per event")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=50, help="Bootstrap replicates per batch (memory)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.batch < 1:
        parser.error("--bootstrap and --batch must be positive")

    config = json.loads((ROOT / f"configurations/edl_water_kl_ablation/{CONTROL}.json").read_text())["shared"]
    args.data_root = Path(config["data_params"]["path_to_splits"])
    split = json.loads(Path(config["data_params"]["train_test_split_file"]).read_text())[args.split]
    pairs = sorted(zip(split["S2"], split["gt"], strict=True))
    runs = {arm: args.eval_root / args.checkpoint / arm / "predictions" / args.split / "EDL" for arm in ARMS}
    out = args.output_dir or args.eval_root / args.checkpoint / "evaluation" / args.split
    out.mkdir(parents=True, exist_ok=True)

    acc, kl_arrays, events, exact = evaluate_events(args, runs, pairs)
    events.to_csv(out / "events.csv", index=False)
    event_metrics_table(acc, events).to_csv(out / "event_metrics.csv", index=False)
    if args.check_exact:
        exact["abs_diff"] = (exact["exact"] - exact["binned"]).abs()  # NaN pairs (single class) are skipped
        exact.to_csv(out / "exact_vs_binned.csv", index=False)
        worst = exact.groupby(["metric", "region"])["abs_diff"].max()
        print(worst.to_string())
        print(f"Max abs difference {worst.max():.2e}; tolerance {EXACT_TOLERANCE:.0e}", flush=True)
        if not worst.max() <= EXACT_TOLERANCE:
            raise SystemExit(f"FAIL: binned metrics differ from the exact reference by more than {EXACT_TOLERANCE:.0e}; "
                             f"see {out / 'exact_vs_binned.csv'}")
        print("PASS", flush=True)

    aoi = np.flatnonzero(events["has_aoi"].to_numpy())
    event_sets = {"all": np.arange(len(events)), "near": aoi, "far": aoi}
    table = bootstrap_table(acc, event_sets, args)
    table.to_csv(out / "pooled_metrics.csv", index=False)
    kl_region_table(*kl_arrays).to_csv(out / "kl_region.csv", index=False)
    if args.split == "test":
        pavpu_table(args.eval_root, args.checkpoint).to_csv(out / "pavpu_summary.csv", index=False)
    plot_curves(acc, out / "risk_coverage_reliability.png", args.split)
    print(f"Outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
