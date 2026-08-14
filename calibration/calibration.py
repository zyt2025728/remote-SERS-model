"""Interpolation of COMSOL-derived local gap-field enhancement data.

Level 1 uses a rectilinear lookup table to evaluate ``F_gap`` in

    E_hotspot = F_gap * E_drive.

The resulting local field may later be used in an electromagnetic SERS
estimate proportional to ``abs(E_hotspot)**4``.  This module does not ship
physical calibration values and does not itself make that approximation a
validated SERS model.
"""

from __future__ import annotations

import csv
import itertools
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable, Mapping


class CalibrationDataError(ValueError):
    """Raised when a lookup table is absent, incomplete, or malformed."""


class CalibrationOutOfBoundsError(ValueError):
    """Raised when interpolation would require extrapolation."""


class CalibrationNotConfiguredError(RuntimeError):
    """Raised when the convenience ``F_gap`` function has no calibration."""


@dataclass(frozen=True)
class HotspotCalibration:
    """Validated rectilinear COMSOL lookup table with linear interpolation.

    Values are multilinearly interpolated in gap, particle radius, wavelength,
    and polarization angle. Queries outside the tabulated domain are rejected;
    this class never silently extrapolates.
    """

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "gap_nm",
        "particle_radius_nm",
        "wavelength_nm",
        "polarization_angle_deg",
        "field_enhancement",
    )
    _axes: tuple[tuple[float, ...], ...]
    _values: Mapping[tuple[float, float, float, float], float]

    @classmethod
    def from_csv(cls, path: str | Path) -> "HotspotCalibration":
        """Load and validate a complete rectilinear calibration table."""
        source = Path(path)
        if not source.is_file():
            raise CalibrationDataError(f"Calibration file does not exist: {source}")

        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(cls.REQUIRED_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise CalibrationDataError(
                    f"Calibration table is missing columns: {', '.join(sorted(missing))}"
                )
            rows = list(reader)

        if not rows:
            raise CalibrationDataError(
                "Calibration table contains no data. The distributed CSV is a "
                "template and must be populated with real COMSOL results."
            )

        values: dict[tuple[float, float, float, float], float] = {}
        axes: list[set[float]] = [set() for _ in range(4)]
        for line_number, row in enumerate(rows, start=2):
            try:
                coordinates = tuple(float(row[name]) for name in cls.REQUIRED_COLUMNS[:4])
                enhancement = float(row["field_enhancement"])
            except (TypeError, ValueError) as exc:
                raise CalibrationDataError(
                    f"Non-numeric calibration value on line {line_number}"
                ) from exc
            if not all(math.isfinite(value) for value in (*coordinates, enhancement)):
                raise CalibrationDataError(f"Non-finite value on line {line_number}")
            if any(value <= 0 for value in coordinates[:3]):
                raise CalibrationDataError(
                    f"Gap, radius, and wavelength must be positive on line {line_number}"
                )
            if enhancement < 0:
                raise CalibrationDataError(
                    f"Field enhancement must be non-negative on line {line_number}"
                )
            if coordinates in values:
                raise CalibrationDataError(
                    f"Duplicate calibration point on line {line_number}: {coordinates}"
                )
            values[coordinates] = enhancement
            for axis, coordinate in zip(axes, coordinates):
                axis.add(coordinate)

        ordered_axes = tuple(tuple(sorted(axis)) for axis in axes)
        expected = math.prod(len(axis) for axis in ordered_axes)
        if len(values) != expected:
            absent = next(
                point
                for point in itertools.product(*ordered_axes)
                if point not in values
            )
            raise CalibrationDataError(
                "Calibration grid is incomplete; missing point " + repr(absent)
            )
        return cls(ordered_axes, values)

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        """Return the inclusive calibration domain for each input variable."""
        return {
            name: (axis[0], axis[-1])
            for name, axis in zip(self.REQUIRED_COLUMNS[:4], self._axes)
        }

    def F_gap(
        self,
        gap_nm: float,
        particle_radius_nm: float,
        wavelength_nm: float,
        polarization_angle_deg: float,
    ) -> float:
        """Return the multilinearly interpolated local field enhancement."""
        query = (
            gap_nm,
            particle_radius_nm,
            wavelength_nm,
            polarization_angle_deg,
        )
        numeric = self._validate_query(query)
        brackets = [self._bracket(axis, value, name) for axis, value, name in zip(
            self._axes, numeric, self.REQUIRED_COLUMNS[:4]
        )]

        result = 0.0
        choices: Iterable[tuple[tuple[float, float], ...]] = itertools.product(*brackets)
        for corner in choices:
            point = tuple(item[0] for item in corner)
            weight = math.prod(item[1] for item in corner)
            result += weight * self._values[point]  # type: ignore[index]
        return result

    @staticmethod
    def _validate_query(query: tuple[float, ...]) -> tuple[float, ...]:
        try:
            numeric = tuple(float(value) for value in query)
        except (TypeError, ValueError) as exc:
            raise CalibrationDataError("Calibration query values must be numeric") from exc
        if not all(math.isfinite(value) for value in numeric):
            raise CalibrationDataError("Calibration query values must be finite")
        if any(value <= 0 for value in numeric[:3]):
            raise CalibrationDataError("Gap, radius, and wavelength must be positive")
        return numeric

    @staticmethod
    def _bracket(
        axis: tuple[float, ...], value: float, name: str
    ) -> tuple[tuple[float, float], ...]:
        if value < axis[0] or value > axis[-1]:
            raise CalibrationOutOfBoundsError(
                f"{name}={value} is outside calibrated range [{axis[0]}, {axis[-1]}]"
            )
        index = bisect_left(axis, value)
        if index < len(axis) and axis[index] == value:
            return ((value, 1.0),)
        lower, upper = axis[index - 1], axis[index]
        upper_weight = (value - lower) / (upper - lower)
        return ((lower, 1.0 - upper_weight), (upper, upper_weight))


_default_calibration: HotspotCalibration | None = None


def configure_default_calibration(path: str | Path) -> HotspotCalibration:
    """Load the real COMSOL table used by the four-argument convenience API."""
    global _default_calibration
    _default_calibration = HotspotCalibration.from_csv(path)
    return _default_calibration


def F_gap(
    gap_nm: float,
    particle_radius_nm: float,
    wavelength_nm: float,
    polarization_angle_deg: float,
) -> float:
    """Evaluate the configured calibration as ``F_gap(g, R, wavelength, angle)``.

    A table must first be explicitly installed with
    :func:`configure_default_calibration`; the empty template is never loaded as
    scientific data automatically.
    """
    if _default_calibration is None:
        raise CalibrationNotConfiguredError(
            "No COMSOL calibration is configured; call "
            "configure_default_calibration() with a populated real-data CSV"
        )
    return _default_calibration.F_gap(
        gap_nm, particle_radius_nm, wavelength_nm, polarization_angle_deg
    )
