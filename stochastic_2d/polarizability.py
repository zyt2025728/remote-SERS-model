"""Complex spherical-particle polarizability for the Level 2 PDA."""

import numpy as np
from numpy.typing import NDArray

from .green_tensor import EPSILON_0


def sphere_polarizability(radii_m: NDArray[np.float64], wavelength_m: float,
                          particle_relative_permittivity: complex,
                          medium_relative_permittivity: complex = 1.0) -> NDArray[np.complex128]:
    """Return Clausius-Mossotti polarizability with radiation reaction.

    Material permittivity is an explicit input: callers must supply a sourced
    complex silver dielectric value for physical work.
    """
    radii = np.asarray(radii_m, dtype=float)
    if np.any(radii <= 0) or wavelength_m <= 0:
        raise ValueError("radii and wavelength must be positive")
    contrast = ((particle_relative_permittivity - medium_relative_permittivity)
                / (particle_relative_permittivity + 2 * medium_relative_permittivity))
    alpha_static = 4 * np.pi * EPSILON_0 * medium_relative_permittivity * radii ** 3 * contrast
    k = 2 * np.pi * np.sqrt(medium_relative_permittivity) / wavelength_m
    return alpha_static / (1 - 1j * k ** 3 * alpha_static
                           / (6 * np.pi * EPSILON_0 * medium_relative_permittivity))
