"""Full retarded electric dyadic Green tensor in a homogeneous 3D medium."""

import numpy as np
from numpy.typing import NDArray

EPSILON_0 = 8.8541878128e-12


def electric_dyadic(displacement_m: NDArray[np.float64], wave_number: complex,
                    medium_relative_permittivity: complex = 1.0) -> NDArray[np.complex128]:
    """Map an electric dipole moment to its field using exp(-i omega t).

    ``displacement_m`` points from the source dipole to the observation point.
    The returned 3x3 tensor retains 1/r^3, 1/r^2, and 1/r terms.
    """
    vector = np.asarray(displacement_m, dtype=float)
    distance = float(np.linalg.norm(vector))
    if vector.shape != (3,) or distance == 0 or not np.isfinite(distance):
        raise ValueError("displacement must be a finite, nonzero 3-vector")
    direction = vector / distance
    nn = np.outer(direction, direction)
    identity = np.eye(3)
    kr = wave_number * distance
    tensor = ((wave_number ** 2 / distance) * (identity - nn)
              + (1 / distance ** 3 - 1j * wave_number / distance ** 2) * (3 * nn - identity))
    return np.exp(1j * kr) * tensor / (4 * np.pi * EPSILON_0 * medium_relative_permittivity)
