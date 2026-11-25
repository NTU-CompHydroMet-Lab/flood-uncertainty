"""Synthetic GeoTIFF integration checks for KL-only mask generation."""

import csv
import json
import os
from dataclasses import replace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box, mapping

from scripts.preprocess import generate_water_only_kl_masks as generator
from scripts.preprocess.generate_water_only_kl_masks import (
    EventFiles,
    MissingAOIError,
    collect_events,
    generate_mask,
    main,
)


def make_event(tmp_path, name="scene", *, image=None, water=None, geometry=None, aoi_class="area_of_interest"):
    data_root = tmp_path / "data"
    event = EventFiles(
        split="test",
        event_id=name,
        image=data_root / "test" / "S2" / f"{name}.tif",
        gt=data_root / "test" / "gt" / f"{name}.tif",
        floodmap=data_root / "test" / "floodmaps" / f"{name}.geojson",
        output=tmp_path / "output" / "test" / "kl_mask" / f"{name}.tif",
    )
    image = np.full((13, 6, 6), 100, dtype=np.float32) if image is None else image
    water = np.zeros((6, 6), dtype=np.uint8) if water is None else water
    geometry = box(0.1, 0.1, 3.9, 5.9) if geometry is None else geometry
    profile = dict(driver="GTiff", height=6, width=6, crs="EPSG:4326", transform=from_origin(0, 6, 1, 1))
    for path in (event.image, event.gt, event.floodmap):
        path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(event.image, "w", count=image.shape[0], dtype="float32", nodata=-9999, **profile) as dst:
        dst.write(image)
    with rasterio.open(event.gt, "w", count=2, dtype="uint8", nodata=0, **profile) as dst:
        dst.write(np.stack([np.ones_like(water), water]))
    event.floodmap.write_text(
        json.dumps({
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": [{
                "type": "Feature",
                "properties": {"w_class": aoi_class},
                "geometry": mapping(geometry),
            }],
        }),
        encoding="utf-8",
    )
    return event


@pytest.mark.parametrize("full_input_statistics", [False, True], ids=["candidate", "full"])
def test_selection_excludes_bad_input_and_outside_aoi_but_keeps_invalid_gt(tmp_path, full_input_statistics):
    image = np.full((14, 6, 6), 100, dtype=np.float32)
    image[13] = np.nan  # Appended non-spectral metadata must not veto S2 input.
    image[:13, 2, 0] = 0
    image[0, 2, 1] = np.nan
    image[1, 2, 2] = np.inf
    image[2, 2, 3] = -9999
    image[0, 3, 0] = 0  # A single zero is valid when nodata is -9999.
    water = np.zeros((6, 6), dtype=np.uint8)
    water[1, :2] = [1, 2]
    event = make_event(tmp_path, image=image, water=water)
    original_gt = event.gt.read_bytes()

    counts = generate_mask(event, block_size=2, full_input_statistics=full_input_statistics)

    expected = np.array([
        [1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
    ], dtype=np.uint8)
    with rasterio.open(event.output) as result, rasterio.open(event.gt) as gt:
        np.testing.assert_array_equal(result.read(1), expected)
        assert result.crs == gt.crs
        assert result.transform == gt.transform
        assert result.shape == gt.shape
        assert result.count == 1
        assert result.dtypes == ("uint8",)
        assert result.nodata is None
        assert result.read_masks(1).all()  # Zeros are meaningful mask values.
        assert result.tags()["kl_pixels"] == "18"
        assert result.tags()["aoi_all_touched"] == "true"
        assert result.tags()["statistics_mode"] == ("full" if full_input_statistics else "candidate")
        assert ("input_valid_pixels" in result.tags()) == full_input_statistics
    assert counts == {
        "total_pixels": 36,
        "input_valid_pixels": 32 if full_input_statistics else None,
        "aoi_pixels": 24,
        "water_invalid_pixels": 34,
        "excluded_invalid_input_pixels": 4 if full_input_statistics else None,
        "excluded_outside_aoi_pixels": 12 if full_input_statistics else None,
        "kl_pixels": 18,
        "candidate_pixels": 22,
        "candidate_invalid_input_pixels": 4,
        "water_invalid_outside_aoi_pixels": 12,
        "s2_windows_total": 9,
        "s2_windows_read": 9 if full_input_statistics else 6,
        "s2_windows_skipped": 0 if full_input_statistics else 3,
        "s2_pixels_read": 36 if full_input_statistics else 24,
    }
    assert event.gt.read_bytes() == original_gt


def test_source_validity_mask_excludes_finite_nonzero_pixels(tmp_path):
    event = make_event(tmp_path)
    validity = np.full((6, 6), 255, dtype=np.uint8)
    validity[0, 0] = 0
    with rasterio.open(event.image, "r+") as image:
        image.write_mask(validity)
    counts = generate_mask(event, block_size=4)
    with rasterio.open(event.output) as result:
        assert result.read(1)[0, 0] == 0
    assert counts["kl_pixels"] == 23
    assert counts["candidate_invalid_input_pixels"] == 1


def test_aoi_reprojection_holes_and_window_boundaries(tmp_path):
    polygon = Polygon(
        box(0.1, 0.1, 5.9, 5.9).exterior.coords,
        holes=[box(1.1, 1.1, 4.9, 4.9).exterior.coords],
    )
    event = make_event(tmp_path, geometry=polygon)
    # Store AOI in another CRS so its numeric coordinates cannot be used directly.
    projected = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    geojson = json.loads(event.floodmap.read_text(encoding="utf-8"))
    geojson["crs"]["properties"]["name"] = "EPSG:3857"
    geojson["features"][0]["geometry"] = mapping(projected)
    event.floodmap.write_text(json.dumps(geojson), encoding="utf-8")
    generate_mask(event, block_size=2)
    whole = replace(event, output=event.output.with_name("whole.tif"))
    generate_mask(whole, block_size=64)

    expected = np.ones((6, 6), dtype=np.uint8)
    expected[2:4, 2:4] = 0
    with rasterio.open(event.output) as tiled, rasterio.open(whole.output) as single:
        np.testing.assert_array_equal(tiled.read(1), expected)
        np.testing.assert_array_equal(tiled.read(1), single.read(1))


def test_zero_kl_pixels_without_candidates_does_not_read_s2(tmp_path, monkeypatch):
    def unexpected_read(*args, **kwargs):
        pytest.fail("A window without KL candidates must not read S2 pixel data")

    monkeypatch.setattr(generator, "read_input_valid", unexpected_read)
    event = make_event(tmp_path, water=np.ones((6, 6), dtype=np.uint8))
    counts = generate_mask(event)
    assert counts["kl_pixels"] == 0
    assert counts["candidate_pixels"] == 0
    assert counts["s2_windows_read"] == 0
    assert counts["s2_windows_skipped"] == counts["s2_windows_total"]
    assert counts["s2_pixels_read"] == 0
    assert counts["input_valid_pixels"] is None
    with rasterio.open(event.output) as result:
        assert not result.read(1).any()


def test_candidate_mode_skips_s2_reads_and_matches_full_mode(tmp_path, monkeypatch):
    water = np.ones((6, 6), dtype=np.uint8)
    water[0, 0] = water[1, 1] = water[1, 5] = 0
    image = np.full((13, 6, 6), 100, dtype=np.float32)
    image[0, 0, 0] = np.nan
    image[:, 1, 5] = 0  # Outside AOI: irrelevant to KL but part of full statistics.
    event = make_event(tmp_path, water=water, image=image)
    windows_read = []
    original_read = generator.read_input_valid

    def record_read(source, window):
        windows_read.append((window.col_off, window.row_off, window.width, window.height))
        return original_read(source, window)

    monkeypatch.setattr(generator, "read_input_valid", record_read)
    fast = generate_mask(event, block_size=2)
    assert windows_read == [(0, 0, 2, 2)]
    assert fast["s2_windows_read"] == 1
    assert fast["s2_windows_skipped"] == 8
    assert fast["s2_pixels_read"] == 4
    assert fast["candidate_pixels"] == 2
    assert fast["candidate_invalid_input_pixels"] == 1
    assert fast["water_invalid_outside_aoi_pixels"] == 1
    assert all(fast[key] is None for key in generator.FULL_INPUT_COUNT_FIELDS)

    windows_read.clear()
    full_event = replace(event, output=event.output.with_name("full.tif"))
    full = generate_mask(full_event, block_size=2, full_input_statistics=True)
    assert len(windows_read) == 9
    assert full["s2_windows_skipped"] == 0
    assert full["s2_pixels_read"] == 36
    assert full["input_valid_pixels"] == 34
    assert full["excluded_invalid_input_pixels"] == 2
    assert full["excluded_outside_aoi_pixels"] == 0
    for counts in (fast, full):
        assert counts["kl_pixels"] == 1
        assert counts["water_invalid_pixels"] == (
            counts["water_invalid_outside_aoi_pixels"] + counts["candidate_pixels"]
        )
        assert counts["candidate_pixels"] == counts["candidate_invalid_input_pixels"] + counts["kl_pixels"]
    expected = np.zeros((6, 6), dtype=np.uint8)
    expected[1, 1] = 1
    with rasterio.open(event.output) as fast_mask, rasterio.open(full_event.output) as full_mask:
        np.testing.assert_array_equal(fast_mask.read(1), expected)
        np.testing.assert_array_equal(fast_mask.read(1), full_mask.read(1))


@pytest.mark.parametrize("failure", ["missing_aoi", "outside_aoi", "invalid_polygon", "grid", "label"])
def test_bad_sources_do_not_publish_a_mask(tmp_path, failure):
    options = {}
    if failure == "missing_aoi":
        options["aoi_class"] = "Flooded area"
    if failure == "outside_aoi":
        options["geometry"] = box(20, 20, 21, 21)
    if failure == "invalid_polygon":
        options["geometry"] = Polygon([(0, 0), (4, 4), (0, 4), (4, 0), (0, 0)])
    if failure == "label":
        water = np.zeros((6, 6), dtype=np.uint8)
        water[5, 5] = 3  # Fail after earlier windows have already been processed.
        options["water"] = water
    event = make_event(tmp_path, **options)
    if failure == "grid":
        with rasterio.open(event.gt, "r+") as gt:
            gt.transform = from_origin(0.5, 6, 1, 1)
    messages = {
        "missing_aoi": "No area_of_interest",
        "outside_aoi": "does not cover",
        "invalid_polygon": "Invalid AOI",
        "grid": "grid mismatch",
        "label": "values outside",
    }
    with pytest.raises(ValueError, match=messages[failure]):
        generate_mask(event, block_size=2)
    assert not event.output.exists()
    assert not list(event.output.parent.glob(f".{event.event_id}.*"))


def test_overwrite_requires_opt_in_and_failed_replacement_preserves_result(tmp_path):
    event = make_event(tmp_path)
    generate_mask(event)
    original = event.output.read_bytes()
    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_mask(event)
    with rasterio.open(event.gt, "r+") as gt:
        water = gt.read(2)
        water[-1, -1] = 255
        gt.write(water, 2)
    with pytest.raises(ValueError, match="values outside"):
        generate_mask(event, block_size=2, overwrite=True)
    assert event.output.read_bytes() == original
    with pytest.raises(ValueError, match="must not replace an input"):
        generate_mask(replace(event, output=event.gt), overwrite=True)


def test_pairing_event_filter_and_missing_source(tmp_path):
    first = make_event(tmp_path, "a")
    make_event(tmp_path, "b")
    events = collect_events(tmp_path / "data", tmp_path / "output", ["test"], ["a"])
    assert events == [first]
    with pytest.raises(ValueError, match="absent"):
        collect_events(tmp_path / "data", tmp_path / "output", ["test"], ["typo"])
    first.image.unlink()
    with pytest.raises(FileNotFoundError, match="Missing paired file"):
        collect_events(tmp_path / "data", tmp_path / "output", ["test"], ["a"])


def test_cli_records_missing_aoi_and_requires_explicit_skip(tmp_path):
    missing = make_event(tmp_path, "a_missing", aoi_class="Flooded area")
    complete = make_event(tmp_path, "b_complete")
    argv = ["--data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "output"), "--splits", "test"]
    with pytest.raises(MissingAOIError):
        main(argv)
    assert not missing.output.exists()
    assert not complete.output.exists()
    main([*argv, "--skip-missing-aoi"])
    report = sorted((tmp_path / "output").glob("kl_mask_summary_*.csv"))[-1]
    with report.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == ["missing_aoi", "written"]
    assert rows[0]["mask_path"] == ""
    assert rows[1]["kl_pixels"] == "24"
    assert rows[1]["statistics_mode"] == "candidate"
    assert rows[1]["candidate_pixels"] == "24"
    assert all(rows[1][key] == "" for key in generator.FULL_INPUT_COUNT_FIELDS)
    assert float(rows[1]["elapsed_seconds"]) >= 0
    assert not missing.output.exists()
    assert complete.output.exists()


def test_cli_full_input_statistics_keeps_full_image_counts(tmp_path):
    make_event(tmp_path)
    main([
        "--data-root", str(tmp_path / "data"),
        "--output-root", str(tmp_path / "output"),
        "--splits", "test", "--full-input-statistics",
    ])
    report = next((tmp_path / "output").glob("kl_mask_summary_*.csv"))
    with report.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["statistics_mode"] == "full"
    assert row["input_valid_pixels"] == "36"
    assert row["excluded_outside_aoi_pixels"] == "12"
    assert row["candidate_pixels"] == "24"
    assert row["s2_windows_skipped"] == "0"


def test_resume_reuses_legacy_mask_and_completes_interrupted_event(tmp_path, monkeypatch):
    complete = make_event(tmp_path, "a_complete")
    pending = make_event(tmp_path, "b_pending")
    generate_mask(complete, full_input_statistics=True)
    # Recreate exactly the completion/provenance tags present before optimization.
    with rasterio.open(complete.output) as mask:
        profile, values, tags = mask.profile, mask.read(), mask.tags()
    legacy_fields = {
        "rule", "mask_version", "s2_bands", "aoi_all_touched",
        "source_s2", "source_gt", "source_floodmap",
        "total_pixels", "input_valid_pixels", "aoi_pixels", "water_invalid_pixels",
        "excluded_invalid_input_pixels", "excluded_outside_aoi_pixels", "kl_pixels",
    }
    with rasterio.open(complete.output, "w", **profile) as mask:
        mask.write(values)
        mask.update_tags(**{key: value for key, value in tags.items() if key in legacy_fields})
    completed_bytes = complete.output.read_bytes()
    partial = pending.output.parent / ".b_pending.interrupted" / pending.output.name
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"incomplete temporary output")
    generated = []
    original_generate = generator.generate_mask

    def generate_pending(event, *args, **kwargs):
        assert event.event_id != complete.event_id
        generated.append(event.event_id)
        return original_generate(event, *args, **kwargs)

    monkeypatch.setattr(generator, "generate_mask", generate_pending)
    main([
        "--data-root", str(tmp_path / "data"),
        "--output-root", str(tmp_path / "output"),
        "--splits", "test", "--skip-existing",
    ])
    assert generated == [pending.event_id]
    assert complete.output.read_bytes() == completed_bytes
    assert pending.output.exists()
    assert partial.read_bytes() == b"incomplete temporary output"
    report = next((tmp_path / "output").glob("kl_mask_summary_*.csv"))
    with report.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == ["skipped_existing", "written"]
    assert rows[0]["statistics_mode"] == "full"
    assert rows[0]["input_valid_pixels"] == "36"
    assert rows[0]["kl_pixels"] == "24"
    assert rows[0]["candidate_pixels"] == ""  # Not recorded by the legacy generator.
    assert rows[0]["s2_windows_read"] == ""  # No S2 pixel reads during reuse.
    assert rows[1]["statistics_mode"] == "candidate"


def test_resume_preserves_saved_statistics_mode(tmp_path, monkeypatch):
    event = make_event(tmp_path)
    generate_mask(event)
    original = event.output.read_bytes()

    def unexpected_generate(*args, **kwargs):
        pytest.fail("A completed mask must not be regenerated during resume")

    monkeypatch.setattr(generator, "generate_mask", unexpected_generate)
    main([
        "--data-root", str(tmp_path / "data"),
        "--output-root", str(tmp_path / "output"),
        "--splits", "test", "--skip-existing", "--full-input-statistics",
    ])
    assert event.output.read_bytes() == original
    report = next((tmp_path / "output").glob("kl_mask_summary_*.csv"))
    with report.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "skipped_existing"
    assert row["statistics_mode"] == "candidate"
    assert row["input_valid_pixels"] == ""
    assert row["candidate_pixels"] == "24"


@pytest.mark.parametrize("problem", ["provenance", "grid", "counts", "source_changed"])
def test_resume_rejects_incompatible_or_stale_outputs(tmp_path, problem):
    event = make_event(tmp_path)
    generate_mask(event)
    if problem == "source_changed":
        stamp = event.output.stat().st_mtime_ns + 1_000_000_000
        os.utime(event.floodmap, ns=(stamp, stamp))
    if problem == "provenance":
        with rasterio.open(event.output, "r+") as mask:
            mask.update_tags(source_gt=str(tmp_path / "another_event.tif"))
    if problem == "grid":
        with rasterio.open(event.output, "r+") as mask:
            mask.transform = from_origin(0.5, 6, 1, 1)
    if problem == "counts":
        with rasterio.open(event.output, "r+") as mask:
            mask.update_tags(total_pixels="35")
    original = event.output.read_bytes()
    with pytest.raises(ValueError, match="Cannot reuse existing mask"):
        main([
            "--data-root", str(tmp_path / "data"),
            "--output-root", str(tmp_path / "output"),
            "--splits", "test", "--skip-existing",
        ])
    assert event.output.read_bytes() == original


def test_resume_and_overwrite_cannot_be_combined(tmp_path):
    with pytest.raises(SystemExit) as error:
        generator.parse_args([
            "--data-root", str(tmp_path / "data"),
            "--output-root", str(tmp_path / "output"),
            "--skip-existing", "--overwrite",
        ])
    assert error.value.code == 2


def test_skip_missing_aoi_does_not_silently_reuse_a_previous_mask(tmp_path):
    event = make_event(tmp_path)
    generate_mask(event)
    original = event.output.read_bytes()
    geojson = json.loads(event.floodmap.read_text(encoding="utf-8"))
    geojson["features"][0]["properties"]["w_class"] = "Flooded area"
    event.floodmap.write_text(json.dumps(geojson), encoding="utf-8")
    with pytest.raises(ValueError, match="previous mask remains"):
        main([
            "--data-root", str(tmp_path / "data"),
            "--output-root", str(tmp_path / "output"),
            "--splits", "test", "--overwrite", "--skip-missing-aoi",
        ])
    assert event.output.read_bytes() == original


def test_preview_audit_detects_shifted_values_on_an_unchanged_grid(tmp_path):
    from scripts.preprocess.preview_water_only_kl_mask import audit_saved_mask

    event = make_event(tmp_path)
    generate_mask(event)
    report = audit_saved_mask(event, block_size=2)
    assert report["grid_matches"]
    assert report["rule_matches"]
    assert report["kl_pixels"] == 24
    with rasterio.open(event.output, "r+") as mask:
        mask.write(np.roll(mask.read(1), shift=1, axis=1), 1)
    report = audit_saved_mask(event, block_size=2)
    assert report["grid_matches"]
    assert not report["rule_matches"]
    assert report["mismatch_pixels"] == 12
    assert report["kl_outside_aoi_pixels"] == 6


def test_preview_audit_rejects_a_shifted_geotransform(tmp_path):
    from scripts.preprocess.preview_water_only_kl_mask import audit_saved_mask

    event = make_event(tmp_path)
    generate_mask(event)
    with rasterio.open(event.output, "r+") as mask:
        mask.transform = from_origin(1, 6, 1, 1)
    with pytest.raises(ValueError, match="grid mismatch"):
        audit_saved_mask(event)
