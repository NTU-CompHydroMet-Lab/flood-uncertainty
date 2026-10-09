import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('sen1_paired_analysis', Path(__file__).resolve().parents[1] / 'scripts/analysis/evaluate_sen1_paired.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_patch_accuracy_differs_from_iou_and_masks_invalid():
    target = np.array([[1, 1], [2, 0]])
    valid = target != 0
    output = {'classification': np.array([[1, 2], [2, 2]]),
              **{k: np.array([[.1, .2], [.3, 999.]]) for k in analysis.UNCERTAINTIES}}
    row = analysis.patch_rows(target, valid, output, 2, {'modality': 'sar'})[0]
    assert row['valid_pixels'] == 3
    assert row['valid_fraction'] == .75
    assert row['patch_accuracy'] == pytest.approx(2 / 3)
    assert row['patch_water_iou'] == .5
    assert row['mean_dst_u'] == pytest.approx(.2)


def test_all_land_patch_has_accuracy_and_undefined_water_iou():
    output = {'classification': np.ones((2, 2)), **{k: np.zeros((2, 2)) for k in analysis.UNCERTAINTIES}}
    row = analysis.patch_rows(np.ones((2, 2)), np.ones((2, 2), bool), output, 2, {})[0]
    assert row['patch_accuracy'] == 1
    assert row['patch_water_iou'] is None
    assert analysis.patch_rows(np.zeros((2, 2)), np.zeros((2, 2), bool), output, 2, {}) == []


def test_retention_includes_ties_and_full_point_matches_counts():
    cm = np.array([4, 2, 3, 1], dtype=np.uint8)
    rows = analysis.retention(cm, np.array([.1, .1, .2, .3]), steps=4)
    assert rows[0]['retention_rate'] == .5
    assert rows[0]['water_iou'] == .5
    assert rows[-1]['water_iou'] == pytest.approx(1 / 3)
    assert rows[-1]['accuracy'] == .5
