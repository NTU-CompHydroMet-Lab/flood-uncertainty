"""Synthetic regression checks; fixture thresholds are not experiment recommendations."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def pavpu(monkeypatch):
    analysis_dir = Path(__file__).resolve().parents[1] / "scripts/analysis"
    monkeypatch.syspath_prepend(str(analysis_dir))
    spec = importlib.util.spec_from_file_location("pavpu_under_test", analysis_dir / "compute_pavpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tif(path, bands, descriptions=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = np.asarray(bands, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2],
        count=bands.shape[0], dtype="float32", nodata=-1,
        transform=from_origin(0, 10, 1, 1), crs="EPSG:4326",
    ) as dst:
        dst.write(bands)
        if descriptions is not None:
            dst.descriptions = descriptions


def edl_bands(uncertainty):
    bands = np.zeros((7, *uncertainty.shape), dtype=np.float32)
    bands[0] = 1
    bands[1] = uncertainty
    return bands


EDL_DESCRIPTIONS = (
    "Classification (0=invalid, 1=land, 2=water)", "Water_DST_Uncertainty",
    "Water_Probability", "Water_Evidence_Neg", "Water_Evidence_Pos",
    "Water_Aleatoric", "Water_Epistemic",
)


def test_q90_uses_all_event_pixels_and_preserves_spatial_values(pavpu):
    uncertainty = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    original = uncertainty.copy()
    valid = np.ones((4, 4), dtype=bool)

    threshold = pavpu.compute_image_uncertainty_threshold(uncertainty, valid, 0.9)

    # Pixel percentile index is 0.9 * 15 = 13.5; the only full patch mean is 5/16.
    assert threshold == pytest.approx(13.5 / 16)
    assert threshold != pytest.approx(5 / 16)
    np.testing.assert_array_equal(uncertainty, original)
    assert pavpu.compute_image_uncertainty_threshold(uncertainty, ~valid, 0.9) is None


def test_prediction_mask_keeps_zero_and_checks_both_uncertainty_sources(pavpu, tmp_path):
    bands = edl_bands(np.zeros((2, 4), dtype=np.float32))
    bands[0] = [[1, 0, 1, 1], [1, 1, 1, -1]]
    bands[5] = [[0, 0.1, -1, 0.1], [np.nan, 0.1, 0.2, 0.1]]
    bands[6] = [[0, 0.1, 0.1, -1], [0.1, np.inf, 0.3, 0.1]]
    path = tmp_path / "prediction.tif"
    write_tif(path, bands, EDL_DESCRIPTIONS)

    uncertainty, valid = pavpu.read_uncertainty_with_valid_mask(
        path, pavpu.CRITERIA_MENU["EDL"]["Water_Aleatoric_Plus_Epistemic"],
    )

    np.testing.assert_array_equal(valid, [[True, False, False, False], [False, False, True, False]])
    np.testing.assert_allclose(uncertainty[valid], [0, 0.5])
    assert pavpu.compute_image_uncertainty_threshold(uncertainty, valid, 0.9) == pytest.approx(0.45)


@pytest.mark.parametrize("invalid_source", ["prediction", "gt"])
def test_four_categories_ties_and_excluded_patches(pavpu, invalid_source):
    # Six 3x3 blocks: TP/FP/TN on top, TP/FP/invalid on the bottom.
    cm = np.repeat(np.repeat([[4, 2, 1], [4, 2, 4]], 3, axis=0), 3, axis=1)
    uncertainty = np.repeat(np.repeat([[0.25] * 3, [0.75] * 3], 3, axis=0), 3, axis=1)
    valid = np.ones(cm.shape, dtype=bool)
    if invalid_source == "prediction":
        valid[3, 6] = False
    else:
        cm[3, 6] = 0

    threshold = pavpu.compute_image_uncertainty_threshold(uncertainty, valid, 0.9)
    result = pavpu.compute_pavpu_for_image(cm, uncertainty, 3, 0.5, threshold, valid_pixels=valid)

    assert threshold == 0.75
    # Equality to Q90 is uncertain; all-TN and any-invalid patches are excluded.
    assert result == (1, 1, 1, 1, 0.5)


def test_per_image_pipeline_is_gt_independent_and_preserves_csv_statistics(pavpu, tmp_path, monkeypatch, capsys):
    pred_root = tmp_path / "predictions"
    model_dir = pred_root / "test/EDL"
    events = {
        "event_a": (np.zeros((3, 3), dtype=np.float32), np.full((3, 3), 4)),
        "event_b": (np.arange(81, dtype=np.float32).reshape(3, 27) / 100, np.full((3, 27), 4)),
        "event_empty": (np.full((3, 3), 0.25), np.ones((3, 3))),
    }
    for name, (uncertainty, cm) in events.items():
        write_tif(model_dir / f"{name}_output_EDL.tif", edl_bands(uncertainty), EDL_DESCRIPTIONS)
        write_tif(model_dir / f"cm_{name}_output_EDL.tif", cm[np.newaxis])

    monkeypatch.setitem(pavpu.SELECTED_CRITERIA, "EDL", ["Water_DST_Uncertainty"])
    monkeypatch.setattr(sys, "argv", [
        "compute_pavpu.py", "--model-type", "EDL", "--subset", "test",
        "--threshold-mode", "per-image", "--acc-threshold", "0.5",
        "--pred-root", str(pred_root), "--curve-root", str(tmp_path / "missing_curves"),
        "--output-dir", str(tmp_path / "results"),
    ])
    pavpu.main()
    path = tmp_path / "results/EDL/pavpu_per_image_q90/pavpu_Water_DST_Uncertainty.csv"
    result = pd.read_csv(path).set_index("filename")

    # A: one accurate/uncertain patch at Q90=0. B: nine accurate/certain patches.
    assert result.loc["event_a", "unc_threshold"] == 0
    assert result.loc["event_b", "unc_threshold"] == pytest.approx(0.72)
    assert result.loc["event_a", "n_au"] == 1
    assert result.loc["event_b", "n_ac"] == 9
    assert result.loc["OVERALL", "n_valid_patches"] == 10
    assert result.loc["OVERALL", "uncertain_fraction"] == pytest.approx(0.1)
    assert result.loc["OVERALL", "pavpu"] == pytest.approx(0.9)
    assert result.loc["AVERAGE", "n_valid_patches"] == pytest.approx(10 / 3)
    assert result.loc["AVERAGE", "uncertain_fraction"] == pytest.approx(0.5)
    assert result.loc["AVERAGE", "pavpu"] == pytest.approx(0.5)
    assert result.loc["event_empty", "n_valid_patches"] == 0
    assert result.loc["event_empty", ["uncertain_fraction", "pavpu"]].isna().all()
    assert result.loc[["OVERALL", "AVERAGE"], "unc_threshold"].isna().all()
    assert result["acc_threshold"].eq(0.5).all()
    assert result["threshold_mode"].eq("per-image").all()
    assert result[["curve_subset", "curve_csv"]].isna().all().all()
    assert "uncertain=N/A" in capsys.readouterr().out
    assert not (tmp_path / "missing_curves").exists()

    # Changing only GT-derived confusion changes scores, never the event's Q90.
    write_tif(model_dir / "cm_event_b_output_EDL.tif", np.full((1, 3, 27), 2))
    pavpu.main()
    changed = pd.read_csv(path).set_index("filename")
    np.testing.assert_allclose(
        changed.loc[list(events), "unc_threshold"], result.loc[list(events), "unc_threshold"],
    )
    assert changed.loc["event_b", "n_ic"] == 9
    assert changed.loc["event_b", "pavpu"] == 0


@pytest.mark.parametrize("threshold_pixels, q90, fraction, n_au", [
    ("prediction-valid", 1.0, 1 / 3, 0),
    ("gt-valid", 0.75, 1 / 2, 1),
])
def test_gt_valid_threshold_excludes_unlabelled_pixels(pavpu, tmp_path, monkeypatch, threshold_pixels, q90, fraction, n_au):
    # Three 3x3 patches: (TP, u=.25), (TP, u=.75), (unlabelled, u=1). Only the threshold source differs.
    model_dir = tmp_path / "predictions/test/EDL"
    uncertainty = np.repeat(np.repeat([[0.25, 0.75, 1.0]], 3, axis=0), 3, axis=1).astype(np.float32)
    cm = np.repeat(np.repeat([[4, 4, 0]], 3, axis=0), 3, axis=1)
    write_tif(model_dir / "event_output_EDL.tif", edl_bands(uncertainty), EDL_DESCRIPTIONS)
    write_tif(model_dir / "cm_event_output_EDL.tif", cm[np.newaxis])
    monkeypatch.setitem(pavpu.SELECTED_CRITERIA, "EDL", ["Water_DST_Uncertainty"])
    option = [] if threshold_pixels == "prediction-valid" else ["--threshold-pixels", "gt-valid"]  # default stays GT-free
    monkeypatch.setattr(sys, "argv", [
        "compute_pavpu.py", "--model-type", "EDL", "--subset", "test", "--threshold-mode", "per-image",
        "--acc-threshold", "0.5", "--pred-root", str(tmp_path / "predictions"),
        "--output-dir", str(tmp_path / "results"), *option,
    ])
    pavpu.main()
    folder = "pavpu_per_image_q90" if threshold_pixels == "prediction-valid" else "pavpu_per_image_q90_gt_valid"
    result = pd.read_csv(tmp_path / "results/EDL" / folder / "pavpu_Water_DST_Uncertainty.csv").set_index("filename")

    assert result.loc["event", "unc_threshold"] == pytest.approx(q90)
    assert result.loc["event", "threshold_tie_fraction"] == pytest.approx(fraction)
    assert result.loc["event", "threshold_flagged_pixel_fraction"] == pytest.approx(fraction)
    # Patch eligibility and accuracy never depend on the threshold population; only certainty does.
    assert result.loc["event", "n_valid_patches"] == 2
    assert result.loc["event", "n_ac"] + result.loc["event", "n_au"] == 2
    assert result.loc["event", "n_au"] == n_au
    assert result["threshold_pixels"].eq(threshold_pixels).all()
    assert result["diagnostic_pixels"].eq(threshold_pixels).all()
    assert np.isnan(result.loc["OVERALL", "threshold_tie_fraction"])
    assert result.loc["AVERAGE", "threshold_tie_fraction"] == pytest.approx(fraction)


def test_threshold_fractions_compare_an_interpolated_threshold_in_float64(pavpu):
    # Nine pixels at 0.5 and one at the next float32: a threshold strictly between them has no ties.
    upper = np.nextafter(np.float32(0.5), np.float32(1))
    values = np.array([0.5] * 9 + [upper], dtype=np.float32)
    threshold = 0.5 + 0.1 * (float(upper) - 0.5)  # Rounds to 0.5 in float32.
    assert np.float32(threshold) == np.float32(0.5)
    assert pavpu.threshold_fractions(values, threshold) == (0.0, pytest.approx(0.1))
    assert pavpu.threshold_fractions(values, threshold, chunk_size=3) == (0.0, pytest.approx(0.1))


def test_curve_threshold_must_be_finite(pavpu, tmp_path):
    curve_dir = tmp_path / "EDL"
    curve_dir.mkdir()
    pd.DataFrame({"retention_rate": [0.9], "threshold": [np.nan], "iou": [0.8]}).to_csv(
        curve_dir / "retention_Water_DST_Uncertainty.csv", index=False,
    )
    with pytest.raises(ValueError, match="Non-finite"):
        pavpu.get_thresholds("EDL", "Water_DST_Uncertainty", tmp_path, 0.9, acc_threshold=0.5)


def test_gt_valid_threshold_is_rejected_in_curve_mode(pavpu, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "compute_pavpu.py", "--threshold-mode", "curve", "--acc-threshold", "0.5", "--threshold-pixels", "gt-valid",
    ])
    with pytest.raises(SystemExit) as error:
        pavpu.parse_args()
    assert error.value.code == 2
