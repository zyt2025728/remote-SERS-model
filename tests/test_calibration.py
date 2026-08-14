import csv
import itertools
import math

import pytest

from calibration.calibration import (
    CalibrationDataError,
    CalibrationNotConfiguredError,
    CalibrationOutOfBoundsError,
    F_gap,
    HotspotCalibration,
)


def write_synthetic_grid(path):
    """Write an analytic test fixture, not physical COMSOL calibration data."""
    axes = ([1.0, 3.0], [20.0, 40.0], [500.0, 700.0], [0.0, 90.0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HotspotCalibration.REQUIRED_COLUMNS)
        for point in itertools.product(*axes):
            # Multilinear interpolation must reproduce this affine function.
            writer.writerow((*point, 2 + point[0] + 0.1 * point[1] + 0.01 * point[2] + 0.001 * point[3]))


@pytest.fixture
def calibration(tmp_path):
    path = tmp_path / "synthetic_test_only.csv"
    write_synthetic_grid(path)
    return HotspotCalibration.from_csv(path)


def test_exact_lookup_and_bounds(calibration):
    assert calibration.F_gap(1.0, 20.0, 500.0, 0.0) == pytest.approx(10.0)
    assert calibration.bounds["gap_nm"] == (1.0, 3.0)


def test_four_dimensional_interpolation(calibration):
    expected = 2 + 2 + 0.1 * 30 + 0.01 * 600 + 0.001 * 45
    assert calibration.F_gap(2.0, 30.0, 600.0, 45.0) == pytest.approx(expected)


@pytest.mark.parametrize("query", [
    (0.5, 30.0, 600.0, 45.0),
    (2.0, 50.0, 600.0, 45.0),
    (2.0, 30.0, 800.0, 45.0),
    (2.0, 30.0, 600.0, 100.0),
])
def test_out_of_range_queries_are_rejected(calibration, query):
    with pytest.raises(CalibrationOutOfBoundsError):
        calibration.F_gap(*query)


@pytest.mark.parametrize("query", [
    (math.nan, 30.0, 600.0, 45.0),
    (0.0, 30.0, 600.0, 45.0),
])
def test_invalid_queries_are_rejected(calibration, query):
    with pytest.raises(CalibrationDataError):
        calibration.F_gap(*query)


def test_template_is_explicitly_not_calibration_data():
    with pytest.raises(CalibrationDataError, match="template"):
        HotspotCalibration.from_csv("calibration/hotspot_lookup_template.csv")


def test_missing_file_and_unconfigured_convenience_api_are_safe(tmp_path):
    with pytest.raises(CalibrationDataError, match="does not exist"):
        HotspotCalibration.from_csv(tmp_path / "missing.csv")
    with pytest.raises(CalibrationNotConfiguredError, match="No COMSOL"):
        F_gap(1.0, 20.0, 500.0, 0.0)


def test_incomplete_grid_is_rejected(tmp_path):
    path = tmp_path / "incomplete.csv"
    write_synthetic_grid(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(CalibrationDataError, match="incomplete"):
        HotspotCalibration.from_csv(path)
