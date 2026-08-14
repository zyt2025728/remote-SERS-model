"""Finite-extent excitation fields."""

import numpy as np
from numpy.typing import NDArray


def gaussian_incident_field(positions_m: NDArray[np.float64], center_m: NDArray[np.float64],
                            waist_m: float, amplitude_v_per_m: complex = 1.0,
                            polarization: NDArray[np.float64] | None = None) -> NDArray[np.complex128]:
    """Return a localized Gaussian envelope with a constant complex polarization."""
    if waist_m <= 0:
        raise ValueError("waist_m must be positive")
    pol = np.array([1.0, 0.0, 0.0]) if polarization is None else np.asarray(polarization, dtype=float)
    if pol.shape != (3,) or np.linalg.norm(pol) == 0:
        raise ValueError("polarization must be a nonzero 3-vector")
    pol = pol / np.linalg.norm(pol)
    radial_squared = np.sum((np.asarray(positions_m) - np.asarray(center_m)) ** 2, axis=1)
    envelope = amplitude_v_per_m * np.exp(-radial_squared / waist_m ** 2)
    return envelope[:, None] * pol[None, :]
